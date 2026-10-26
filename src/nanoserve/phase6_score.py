"""Fixed-window Phase 6 SLO scoring from saved completion traces and replays."""

from __future__ import annotations

import math

from .experiment import analyze_replay


TTFT_SLO_S = 1.0
TPOT_SLO_S = 0.1
SEND_LAG_P99_GATE_S = 0.05


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


def score_fixed_window(trace: dict, replay: dict) -> dict:
    """Score the offered cohort; only in-window SLO-qualified completions earn goodput.

    This does not infer token emission times from HTTP chunks. The completed
    token yield is explicitly distinct from within-window emitted throughput.
    """
    aggregate, rows, _ = analyze_replay(trace, replay)
    if replay.get("mode") != "fixed-window-bounded-drain":
        raise ValueError("Phase 6 scoring requires a bounded fixed-window replay")
    interval = replay.get("offered_interval_s")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("offered_interval_s must be finite and positive")
    if trace.get("offered_interval_s") != interval:
        raise ValueError("trace and replay offered intervals disagree")

    window_completed = 0
    slo_passed = 0
    missing_tpot = 0
    completed_tokens = 0
    missing_usage = 0
    window_ttfts: list[float] = []
    for record, row in zip(replay["records"], rows):
        if row["status"] != "completed" or record["completed_offset_s"] > interval:
            continue
        window_completed += 1
        if row["ttft_s"] is not None:
            window_ttfts.append(row["ttft_s"])
        tokens = row["completion_tokens"]
        if tokens is None:
            missing_usage += 1
        else:
            completed_tokens += tokens
        tpot = row["server_tpot_s"]
        if tokens is not None and tokens > 1 and tpot is None:
            missing_tpot += 1
        tpot_qualified = (tokens == 1 or (tpot is not None and tpot <= TPOT_SLO_S))
        if (tokens is not None and row["ttft_s"] is not None
                and row["ttft_s"] <= TTFT_SLO_S and tpot_qualified):
            slo_passed += 1

    p99_send_lag = _p99([row["send_lag_s"] for row in rows if row["send_lag_s"] is not None])
    token_window = replay.get("server_token_window")
    emitted_tokens = None
    if isinstance(token_window, dict) and not token_window.get("error"):
        if (token_window.get("clock") != "time.monotonic"
                or not math.isclose(token_window.get("end_s", 0) - token_window.get("start_s", 0), interval, abs_tol=1e-8)
                or isinstance(token_window.get("emitted_tokens"), bool)
                or not isinstance(token_window.get("emitted_tokens"), int)
                or token_window["emitted_tokens"] < 0):
            raise ValueError("invalid server_token_window")
        emitted_tokens = token_window["emitted_tokens"]
    return {
        "schema_version": 1,
        "kind": "phase6-fixed-window-score",
        "trace_sha256": trace["sha256"],
        "adapter": replay["adapter"],
        "target_offered_rate_rps": trace.get("rate_rps"),
        "offered_interval_s": interval,
        "slo": {"client_ttft_s": TTFT_SLO_S, "server_reported_tpot_s": TPOT_SLO_S},
        "cohort": aggregate["cohort"],
        "completed_within_window": window_completed,
        "completed_during_drain": aggregate["cohort"]["completed"] - window_completed,
        "slo_qualified_within_window": slo_passed,
        "slo_goodput_rps": slo_passed / interval,
        "completed_output_token_yield_within_window": completed_tokens if missing_usage == 0 else None,
        "emitted_output_tokens_within_window": emitted_tokens,
        "emitted_output_tokens_per_s": emitted_tokens / interval if emitted_tokens is not None else None,
        "server_token_window_error": token_window.get("error") if isinstance(token_window, dict) else None,
        "missing_window_usage": missing_usage,
        "missing_window_tpot": missing_tpot,
        "client_send_lag_p99_s": p99_send_lag,
        "client_send_lag_gate_s": SEND_LAG_P99_GATE_S,
        "client_send_lag_gate_passed": p99_send_lag is not None and p99_send_lag <= SEND_LAG_P99_GATE_S,
        "window_completed_client_ttft_p99_s": _p99(window_ttfts),
        "completed_client_ttft_p99_s": aggregate["client_ttft"]["p99_s"],
        "limitations": [
            "This score covers the full offered cohort, including failures and unfinished requests.",
            "Goodput counts only requests completed by the fixed measurement cutoff with both predeclared SLOs met.",
            "Server TPOT is unavailable for engines that omit per-request true-token metrics; those multi-token requests cannot qualify.",
            "Completed output-token yield is not emitted-token throughput; the latter requires nanoserve's server-side token-window counter.",
            "Per-run p99 TTFT includes completed requests that finished during drain and must be interpreted with sample count.",
        ],
    }
