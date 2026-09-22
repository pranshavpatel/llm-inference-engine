"""Phase 5 replay analysis; never infer token timings from HTTP chunks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from html import escape
from pathlib import Path

from nanoserve.replay import make_duration_completion_trace, validate_completion_trace


def _time(value, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _percentiles(values: list[float]) -> dict:
    """Empirical nearest-rank percentiles, including the maximum at p100."""
    ordered = sorted(values)
    return {
        "count": len(ordered),
        **{
            f"p{percentile}_s": ordered[math.ceil(percentile / 100 * len(ordered)) - 1]
            if ordered else None
            for percentile in (50, 90, 99)
        },
    }


def analyze_replay(trace: dict, replay: dict) -> tuple[dict, list[dict], list[dict]]:
    """Validate one full-cohort replay and return aggregate, rows, and chunk events.

    This is a diagnostic full-run summary, not a steady-state benchmark. The
    replay format has HTTP content-chunk times but no exact per-token times.
    """
    validate_completion_trace(trace)
    if not isinstance(replay, dict) or replay.get("schema_version") != 1 or replay.get("kind") != "completion-replay":
        raise ValueError("unsupported completion replay schema")
    for key, expected in (("trace_sha256", trace["sha256"]), ("model", trace["model"]), ("revision", trace["revision"])):
        if replay.get(key) != expected:
            raise ValueError(f"replay {key} does not match trace")
    if not isinstance(replay.get("adapter"), str) or not replay["adapter"]:
        raise ValueError("replay adapter must be nonempty")
    elapsed = _time(replay.get("elapsed_s"), "elapsed_s")
    if elapsed == 0:
        raise ValueError("elapsed_s must be positive")
    records = replay.get("records")
    if not isinstance(records, list) or len(records) != len(trace["requests"]):
        raise ValueError("replay must retain one record per trace request")

    rows: list[dict] = []
    events: list[dict] = []
    ttfts: list[float] = []
    completed_latencies: list[float] = []
    send_lags: list[float] = []
    inter_content_gaps: list[float] = []
    errors: Counter[str] = Counter()
    output_tokens = 0
    missing_usage = 0
    completed = 0
    timed_out = 0
    not_sent = 0
    for request, record in zip(trace["requests"], records):
        if not isinstance(record, dict) or record.get("request_id") != request["request_id"]:
            raise ValueError("replay records must match trace IDs and order")
        status = record.get("status")
        if status not in ("completed", "failed", "timed_out", "not_sent"):
            raise ValueError("unsupported replay record status")
        intended = _time(request["arrival_offset_s"], "intended arrival")
        sent = _time(record.get("actual_send_offset_s"), "actual send", optional=True)
        first = _time(record.get("first_content_offset_s"), "first content", optional=True)
        ended = _time(record.get("completed_offset_s"), "completion")
        if sent is not None and ended < sent:
            raise ValueError("completion cannot precede actual send")
        if first is not None and (sent is None or first < sent or first > ended):
            raise ValueError("first content must be between send and completion")
        if ended > elapsed + 1e-6:
            raise ValueError("record completion exceeds replay duration")
        lag = sent - intended if sent is not None else None
        if lag is not None:
            send_lags.append(lag)
        ttft = first - sent if first is not None else None
        if ttft is not None and status == "completed":
            ttfts.append(ttft)
        latency = ended - sent if sent is not None else None
        if latency is not None and status == "completed":
            completed_latencies.append(latency)
        usage = record.get("usage")
        tokens = None
        if status == "completed":
            completed += 1
            if usage is None:
                missing_usage += 1
            elif not isinstance(usage, dict) or isinstance(usage.get("completion_tokens"), bool) or not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] < 0:
                raise ValueError("completed usage must have nonnegative integer completion_tokens")
            else:
                tokens = usage["completion_tokens"]
                output_tokens += tokens
            if request.get("ignore_eos", False):
                if record.get("finish_reason") != "length" or (tokens is not None and tokens != request["max_tokens"]):
                    raise ValueError("completed fixed-output record contradicts trace token policy")
        else:
            error = str(record.get("error", "unknown"))
            if status == "timed_out":
                timed_out += 1
                errors["drain_timeout"] += 1
            elif status == "not_sent":
                not_sent += 1
                errors["not_sent"] += 1
            elif "HTTP 429" in error:
                errors["http_429"] += 1
            elif "Timeout" in error or "timed out" in error:
                errors["timeout"] += 1
            else:
                errors["other"] += 1
        chunks = record.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("record chunks must be a list")
        previous_content = None
        previous_event = None
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise ValueError("chunks must contain text strings")
            at = _time(chunk.get("at_offset_s"), "chunk time")
            if at > ended or (sent is not None and at < sent) or (previous_event is not None and at < previous_event):
                raise ValueError("chunk time must be ordered between send and completion")
            previous_event = at
            events.append({"request_id": request["request_id"], "chunk_index": index, "at_offset_s": at, "text": chunk["text"], "finish_reason": chunk.get("finish_reason")})
            if chunk["text"]:
                if previous_content is not None:
                    inter_content_gaps.append(at - previous_content)
                previous_content = at
        rows.append({
            "request_id": request["request_id"],
            "status": status,
            "intended_arrival_offset_s": intended,
            "actual_send_offset_s": sent,
            "send_lag_s": lag,
            "ttft_s": ttft,
            "end_to_end_s": latency,
            "completion_tokens": tokens,
            "finish_reason": record.get("finish_reason"),
            "error": record.get("error"),
        })
    summary = replay.get("summary")
    if not isinstance(summary, dict) or any(summary.get(key) != expected for key, expected in (
        ("requests", len(records)), ("completed", completed), ("failed", len(records) - completed), ("missing_usage", missing_usage)
    )):
        raise ValueError("replay summary disagrees with records")
    if summary.get("timed_out", timed_out) != timed_out or summary.get("not_sent", not_sent) != not_sent:
        raise ValueError("replay timeout summary disagrees with records")
    aggregate = {
        "schema_version": 1,
        "kind": "completion-replay-analysis",
        "trace_sha256": trace["sha256"],
        "adapter": replay["adapter"],
        "cohort": {
            "requests": len(records), "completed": completed, "failed": len(records) - completed,
            "timed_out": timed_out, "not_sent": not_sent,
            "missing_usage": missing_usage, "failure_types": dict(errors),
        },
        "target_offered_rate_rps": trace.get("rate_rps"),
        "offered_interval_s": replay.get("offered_interval_s"),
        "drain_deadline_offset_s": replay.get("drain_deadline_offset_s"),
        "full_run_elapsed_s": elapsed,
        "full_run_completed_output_tokens": output_tokens if missing_usage == 0 else None,
        "full_run_completed_output_tokens_per_s": output_tokens / elapsed if missing_usage == 0 else None,
        "client_ttft": _percentiles(ttfts),
        "completed_end_to_end": _percentiles(completed_latencies),
        "send_lag": _percentiles(send_lags),
        "inter_content_chunk_gap": _percentiles(inter_content_gaps),
        "limitations": [
            "Full-run cohort metrics include startup and drain; they are not steady-state throughput.",
            "HTTP chunks are not model tokens; inter-content-chunk gaps are not ITL or TPOT.",
            "Token-level TPOT SLO goodput and within-window output-token throughput are unavailable from this replay schema.",
            "A completed request with missing usage makes output-token throughput unavailable.",
        ],
    }
    return aggregate, rows, events


def write_analysis(trace_path: Path, replay_path: Path, output_dir: Path) -> dict:
    """Create an immutable-by-default analysis bundle for one saved replay."""
    trace_bytes = trace_path.read_bytes()
    replay_bytes = replay_path.read_bytes()
    trace = json.loads(trace_bytes)
    replay = json.loads(replay_bytes)
    aggregate, rows, events = analyze_replay(trace, replay)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-request.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "chunk-events.jsonl").open("w", encoding="utf-8") as output:
        for event in events:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "kind": "completion-analysis-manifest",
        "trace_file": str(trace_path),
        "trace_file_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "replay_file": str(replay_path),
        "replay_file_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "model": trace["model"],
        "revision": trace["revision"],
        "adapter": replay["adapter"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return aggregate


def write_sweep_plan(
    output_dir: Path,
    *,
    rates: list[float],
    repetitions: int,
    duration_s: float,
    base_seed: int,
    model: str,
    revision: str,
    prompts: list[str],
    max_tokens: int,
    ignore_eos: bool = False,
) -> dict:
    """Write paired, independently seeded fixed-window arrival traces."""
    if not rates or any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("rates must be finite positive numbers")
    if len(set(rates)) != len(rates):
        raise ValueError("rates must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    planned = []
    for rate_index, rate in enumerate(rates):
        for repetition in range(repetitions):
            trace = make_duration_completion_trace(
                duration_s=duration_s, rate=rate, seed=base_seed + repetition,
                model=model, revision=revision, prompts=prompts, max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
            planned.append((f"trace-rate-{rate_index:02d}-rep-{repetition:02d}.json", trace, rate_index, repetition))
    output_dir.mkdir(parents=True, exist_ok=False)
    traces = []
    for name, trace, rate_index, repetition in planned:
        (output_dir / name).write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        traces.append({
            "file": name,
            "rate_index": rate_index,
            "repetition": repetition,
            "seed": trace["seed"],
            "target_rate_rps": trace["rate_rps"],
            "offered_interval_s": trace["offered_interval_s"],
            "requests": len(trace["requests"]),
            "trace_sha256": trace["sha256"],
        })
    manifest = {
        "schema_version": 1,
        "kind": "phase5-sweep-plan",
        "model": model,
        "revision": revision,
        "rates_rps": [float(rate) for rate in rates],
        "repetitions": repetitions,
        "duration_s": float(duration_s),
        "base_seed": base_seed,
        "max_tokens": max_tokens,
        "output_policy": (
            "ignore EOS; require exactly max_tokens generated tokens"
            if ignore_eos else "normal EOS; max_tokens is a ceiling, not a fixed output length"
        ),
        "status": "arrival plan only; not a scored benchmark run",
        "traces": traces,
    }
    (output_dir / "sweep-plan.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _line_plot_svg(
    series: list[tuple[str, list[tuple[float, float, str, bool]]]],
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    """Dependency-free diagnostic SVG; hollow points indicate run failures."""
    width, height = 760, 470
    left, right, top, bottom = 88, 30, 45, 72
    inner_width, inner_height = width - left - right, height - top - bottom
    all_points = [point for _, points in series for point in points]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(0.0, min(ys)), max(ys)
    if x_min == x_max:
        x_min = max(0.0, x_min - 1)
        x_max += 1
    if y_min == y_max:
        y_max += 1
    colors = ("#2367b4", "#d35b30", "#328651", "#8456a2", "#a88921")

    def x_position(value: float) -> float:
        return left + inner_width * (value - x_min) / (x_max - x_min)

    def y_position(value: float) -> float:
        return top + inner_height * (1 - (value - y_min) / (y_max - y_min))

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-size="17">{escape(title)}</text>',
        f'<path d="M {left} {top} V {top + inner_height} H {left + inner_width}" fill="none" stroke="#333"/>',
        f'<text x="{left + inner_width / 2}" y="{height - 18}" text-anchor="middle" font-size="13">{escape(x_label)}</text>',
        f'<text x="20" y="{top + inner_height / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 20 {top + inner_height / 2})">{escape(y_label)}</text>',
    ]
    for index in range(5):
        x_value = x_min + index * (x_max - x_min) / 4
        y_value = y_min + index * (y_max - y_min) / 4
        xp, yp = x_position(x_value), y_position(y_value)
        pieces.append(f'<text x="{xp:.1f}" y="{top + inner_height + 18}" text-anchor="middle" font-size="11">{x_value:.2g}</text>')
        pieces.append(f'<text x="{left - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="11">{y_value:.2g}</text>')
    for index, (name, points) in enumerate(series):
        color = colors[index % len(colors)]
        pieces.append(f'<text x="{left + index * 155}" y="{height - 45}" fill="{color}" font-size="12">{escape(name)}</text>')
        if len(points) > 1:
            coordinates = " ".join(f"{x_position(x):.1f},{y_position(y):.1f}" for x, y, _, _ in points)
            pieces.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, label, has_failures in points:
            xp, yp = x_position(x), y_position(y)
            fill = "white" if has_failures else color
            pieces.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="5" fill="{fill}" stroke="{color}" stroke-width="2"/>')
            pieces.append(f'<text x="{xp + 7:.1f}" y="{yp - 7:.1f}" font-size="10">{escape(label)}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces) + "\n"


def write_sweep_report(plan_path: Path, replay_dirs: dict[str, Path], output_dir: Path) -> dict:
    """Regenerate paired pilot diagnostics from complete saved replay files."""
    if not replay_dirs or any(not name or not isinstance(directory, Path) for name, directory in replay_dirs.items()):
        raise ValueError("replay_dirs must map engine names to directories")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema_version") != 1 or plan.get("kind") != "phase5-sweep-plan":
        raise ValueError("unsupported sweep plan")
    entries = plan.get("traces")
    if not isinstance(entries, list) or not entries:
        raise ValueError("sweep plan must list traces")
    rates = plan.get("rates_rps")
    repetitions = plan.get("repetitions")
    if (not isinstance(rates, list) or not rates or
            any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 for rate in rates) or
            isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0 or
            len(entries) != len(rates) * repetitions):
        raise ValueError("sweep plan rates, repetitions, and trace count disagree")
    seen_pairs: set[tuple[int, int]] = set()
    rows = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("invalid sweep trace entry")
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("sweep trace file must be a simple filename")
        rate_index = entry.get("rate_index")
        repetition = entry.get("repetition")
        if (isinstance(rate_index, bool) or not isinstance(rate_index, int) or not 0 <= rate_index < len(rates) or
                isinstance(repetition, bool) or not isinstance(repetition, int) or not 0 <= repetition < repetitions or
                (rate_index, repetition) in seen_pairs):
            raise ValueError("sweep trace rate/repetition pair is invalid or duplicated")
        seen_pairs.add((rate_index, repetition))
        trace_path = plan_path.parent / entry["file"]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        validate_completion_trace(trace)
        if trace["sha256"] != entry.get("trace_sha256"):
            raise ValueError("sweep plan trace checksum mismatch")
        if (trace["model"] != plan.get("model") or trace["revision"] != plan.get("revision") or
                trace.get("seed") != entry.get("seed") or
                trace.get("rate_rps") != rates[rate_index] or
                trace.get("rate_rps") != entry.get("target_rate_rps") or
                trace.get("offered_interval_s") != plan.get("duration_s") or
                trace.get("offered_interval_s") != entry.get("offered_interval_s") or
                len(trace["requests"]) != entry.get("requests") or
                any(request["max_tokens"] != plan.get("max_tokens") for request in trace["requests"])):
            raise ValueError("sweep plan metadata disagrees with trace")
        for engine, directory in replay_dirs.items():
            replay_path = directory / entry["file"]
            replay_bytes = replay_path.read_bytes()
            replay = json.loads(replay_bytes)
            aggregate, _, _ = analyze_replay(trace, replay)
            expected_adapter = {
                "nanoserve": "nanoserve-http",
                "vllm": "vllm-http",
                "hf": "huggingface-eager",
            }.get(engine)
            if expected_adapter is not None and replay["adapter"] != expected_adapter:
                raise ValueError(f"{engine} replay has adapter {replay['adapter']!r}")
            cohort = aggregate["cohort"]
            rows.append({
                "engine": engine,
                "adapter": replay["adapter"],
                "rate_index": entry["rate_index"],
                "repetition": entry["repetition"],
                "target_rate_rps": entry["target_rate_rps"],
                "trace_sha256": trace["sha256"],
                "replay_file_sha256": hashlib.sha256(replay_bytes).hexdigest(),
                "requests": cohort["requests"],
                "completed": cohort["completed"],
                "failed": cohort["failed"],
                "timed_out": cohort["timed_out"],
                "not_sent": cohort["not_sent"],
                "missing_usage": cohort["missing_usage"],
                "p99_client_ttft_s": aggregate["client_ttft"]["p99_s"],
                "ttft_samples": aggregate["client_ttft"]["count"],
                "p99_send_lag_s": aggregate["send_lag"]["p99_s"],
                "full_run_completed_output_tokens_per_s": aggregate["full_run_completed_output_tokens_per_s"],
                "failure_fraction": cohort["failed"] / cohort["requests"],
            })

    grouped = []
    for engine in replay_dirs:
        for rate_index in sorted({row["rate_index"] for row in rows}):
            members = [row for row in rows if row["engine"] == engine and row["rate_index"] == rate_index]
            if len(members) != plan.get("repetitions"):
                raise ValueError("sweep plan repetition count disagrees with runs")
            p99_values = [row["p99_client_ttft_s"] for row in members if row["p99_client_ttft_s"] is not None]
            throughput_values = [row["full_run_completed_output_tokens_per_s"] for row in members if row["full_run_completed_output_tokens_per_s"] is not None]
            grouped.append({
                "engine": engine,
                "rate_index": rate_index,
                "target_rate_rps": members[0]["target_rate_rps"],
                "repetitions": len(members),
                "median_of_run_p99_client_ttft_s": statistics.median(p99_values) if p99_values else None,
                "min_run_p99_client_ttft_s": min(p99_values) if p99_values else None,
                "max_run_p99_client_ttft_s": max(p99_values) if p99_values else None,
                "median_full_run_completed_output_tokens_per_s": statistics.median(throughput_values) if throughput_values else None,
                "median_failure_fraction": statistics.median(row["failure_fraction"] for row in members),
                "runs_with_failures": sum(row["failed"] > 0 for row in members),
            })
    report = {
        "schema_version": 1,
        "kind": "phase5-pilot-sweep-report",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "engines": list(replay_dirs),
        "runs": rows,
        "by_rate": grouped,
        "limitations": [
            "Pilot diagnostics only: zero observed failures does not prove a stable queue or sustainable throughput.",
            "The plotted p99 value is the median of per-run p99 values, not a pooled percentile.",
            "Full-run output-token throughput includes startup and drain; it is not steady-state window throughput.",
            "No TPOT SLO goodput or sustainable frontier is inferred from HTTP chunks.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "by-rate.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(grouped[0]))
        writer.writeheader()
        writer.writerows(grouped)
    latency_series = []
    failure_series = []
    for engine in replay_dirs:
        members = [item for item in grouped if item["engine"] == engine]
        latency_series.append((engine, [
            (item["median_full_run_completed_output_tokens_per_s"], item["median_of_run_p99_client_ttft_s"], f"{item['target_rate_rps']:g} rps", item["runs_with_failures"] > 0)
            for item in members
            if item["median_full_run_completed_output_tokens_per_s"] is not None and item["median_of_run_p99_client_ttft_s"] is not None
        ]))
        failure_series.append((engine, [
            (item["target_rate_rps"], item["median_failure_fraction"], f"{item['target_rate_rps']:g} rps", item["runs_with_failures"] > 0)
            for item in members
        ]))
    if any(points for _, points in latency_series):
        (output_dir / "throughput-vs-ttft.svg").write_text(_line_plot_svg(
            [(name, points) for name, points in latency_series if points],
            x_label="Full-run completed output tokens/s (includes drain)",
            y_label="Median of run p99 client TTFT (s)",
            title="Pilot diagnostic: throughput versus TTFT",
        ), encoding="utf-8")
    (output_dir / "failure-vs-rate.svg").write_text(_line_plot_svg(
        failure_series,
        x_label="Target offered requests/s",
        y_label="Median failure fraction",
        title="Pilot diagnostic: failures versus offered rate",
    ), encoding="utf-8")
    return report
