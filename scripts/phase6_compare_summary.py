"""Validate and summarize paired Phase 6 VM replays without treating drain as throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from nanoserve.experiment import analyze_replay


ENGINES = ("reference", "flashinfer")


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(run_dir: Path) -> dict:
    plan_bytes = (run_dir / "plan" / "sweep-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    config = json.loads((run_dir / "run-config.json").read_text(encoding="utf-8"))
    saved_report = json.loads((run_dir / "report" / "report.json").read_text(encoding="utf-8"))
    if saved_report["plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError("saved report and plan checksums disagree")
    if plan["rates_rps"] != config["rates_rps"] or plan["repetitions"] != config["repetitions"]:
        raise ValueError("plan and run configuration disagree")
    if plan["duration_s"] != config["offered_interval_s"]:
        raise ValueError("plan and offered interval disagree")
    if config["backends"] != list(ENGINES) or saved_report["engines"] != list(ENGINES):
        raise ValueError("unexpected backend names")
    duration = float(config["offered_interval_s"])
    run_rows: list[dict] = []
    paired_text: dict[tuple[float, int], dict[str, int]] = {}
    for entry in plan["traces"]:
        trace_path = run_dir / "plan" / entry["file"]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        rate = float(entry["target_rate_rps"])
        repetition = entry["repetition"]
        replays = {}
        for engine in ENGINES:
            replay = json.loads((run_dir / engine / entry["file"]).read_text(encoding="utf-8"))
            aggregate, rows, _ = analyze_replay(trace, replay)
            replays[engine] = replay
            first_half = [row["ttft_s"] for row in rows
                          if row["status"] == "completed" and row["ttft_s"] is not None
                          and row["intended_arrival_offset_s"] < duration / 2]
            second_half = [row["ttft_s"] for row in rows
                           if row["status"] == "completed" and row["ttft_s"] is not None
                           and row["intended_arrival_offset_s"] >= duration / 2]
            completed_in_window = sum(
                record["status"] == "completed" and record["completed_offset_s"] <= duration
                for record in replay["records"]
            )
            cohort = aggregate["cohort"]
            run_rows.append({
                "engine": engine,
                "rate_rps": rate,
                "repetition": repetition,
                "trace_sha256": trace["sha256"],
                "requests": cohort["requests"],
                "completed": cohort["completed"],
                "completed_in_offered_window": completed_in_window,
                "completed_during_drain": cohort["completed"] - completed_in_window,
                "failed": cohort["failed"],
                "timed_out": cohort["timed_out"],
                "not_sent": cohort["not_sent"],
                "missing_usage": cohort["missing_usage"],
                "failure_types": cohort["failure_types"],
                "p99_client_send_lag_s": aggregate["send_lag"]["p99_s"],
                "send_lag_gate_passed": aggregate["send_lag"]["p99_s"] is not None
                and aggregate["send_lag"]["p99_s"] <= config["client_send_lag_p99_gate_s"],
                "p50_client_ttft_s": aggregate["client_ttft"]["p50_s"],
                "p99_client_ttft_s": aggregate["client_ttft"]["p99_s"],
                "p50_server_tpot_s": aggregate["server_reported_tpot"]["p50_s"],
                "p99_server_tpot_s": aggregate["server_reported_tpot"]["p99_s"],
                "median_first_half_client_ttft_s": median_or_none(first_half),
                "median_second_half_client_ttft_s": median_or_none(second_half),
            })
        paired = [(left, right) for left, right in zip(
            replays["reference"]["records"], replays["flashinfer"]["records"])
            if left["status"] == right["status"] == "completed"]
        paired_text[(rate, repetition)] = {
            "both_completed": len(paired),
            "different_output_text": sum(left["output_text"] != right["output_text"]
                                         for left, right in paired),
        }

    by_rate = []
    for rate in plan["rates_rps"]:
        for engine in ENGINES:
            members = [row for row in run_rows if row["rate_rps"] == rate and row["engine"] == engine]
            if len(members) != plan["repetitions"]:
                raise ValueError("missing run in sweep")
            by_rate.append({
                "engine": engine,
                "rate_rps": rate,
                "runs": len(members),
                "requests": sum(row["requests"] for row in members),
                "completed": sum(row["completed"] for row in members),
                "completed_in_offered_window": sum(row["completed_in_offered_window"] for row in members),
                "completed_during_drain": sum(row["completed_during_drain"] for row in members),
                "failed": sum(row["failed"] for row in members),
                "timed_out": sum(row["timed_out"] for row in members),
                "not_sent": sum(row["not_sent"] for row in members),
                "missing_usage": sum(row["missing_usage"] for row in members),
                "max_run_p99_client_send_lag_s": max(row["p99_client_send_lag_s"] for row in members),
                "all_send_lag_gates_passed": all(row["send_lag_gate_passed"] for row in members),
                "median_of_run_p50_client_ttft_s": median_or_none([
                    row["p50_client_ttft_s"] for row in members if row["p50_client_ttft_s"] is not None]),
                "median_of_run_p99_client_ttft_s": median_or_none([
                    row["p99_client_ttft_s"] for row in members if row["p99_client_ttft_s"] is not None]),
                "median_of_run_p50_server_tpot_s": median_or_none([
                    row["p50_server_tpot_s"] for row in members if row["p50_server_tpot_s"] is not None]),
                "median_of_run_first_half_ttft_s": median_or_none([
                    row["median_first_half_client_ttft_s"] for row in members
                    if row["median_first_half_client_ttft_s"] is not None]),
                "median_of_run_second_half_ttft_s": median_or_none([
                    row["median_second_half_client_ttft_s"] for row in members
                    if row["median_second_half_client_ttft_s"] is not None]),
                "paired_both_completed": sum(paired_text[(rate, row["repetition"])]["both_completed"]
                                             for row in members),
                "paired_different_output_text": sum(
                    paired_text[(rate, row["repetition"])]["different_output_text"]
                    for row in members),
            })
    return {
        "schema_version": 1,
        "kind": "phase6-paired-backend-diagnostic-summary",
        "engine_commit": (run_dir / "engine-commit.txt").read_text(encoding="utf-8").strip(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "run_config": config,
        "runs": run_rows,
        "by_rate": by_rate,
        "limitations": [
            "Completed-in-window counts are requests whose final response arrived by the window boundary, not exact output tokens emitted within the window.",
            "No sustainable capacity or SLO goodput is inferred automatically; inspect failures, drain, and latency drift.",
            "Per-run p99 is weak at low sample counts; do not turn median-of-run p99 into a pooled percentile.",
            "The opt-in FlashInfer path differs in greedy text from the BF16 gather path; fixed token counts do not establish output parity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(report["runs"]), "by_rate": len(report["by_rate"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
