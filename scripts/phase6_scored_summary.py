"""Validate a Phase 6 NanoServe archive and regenerate compact release evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from html import escape
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

from nanoserve.phase6_score import score_fixed_window


COLORS = {"reference": "#3566a8", "flashinfer": "#d36031"}


def _median(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.median(values) if len(values) == len(rows) and values else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resource_samples(samples: list[dict], replay: dict) -> dict:
    start = datetime.fromisoformat(replay["started_at_utc"]).timestamp()
    duration = replay["offered_interval_s"]
    window = sorted((sample["at_utc_epoch_s"] - start, sample)
                    for sample in samples if 0 <= sample["at_utc_epoch_s"] - start < duration)
    if len(window) < duration * 0.8 or any("metrics" not in sample or "gpu_memory_used_mib" not in sample for _, sample in window):
        raise ValueError("resource samples are missing or sparse")
    queue = lambda sample: sum(sample["metrics"]["states"][key] for key in ("waiting", "prefill", "decoding"))
    midpoint = min(window, key=lambda entry: abs(entry[0] - duration / 2))[1]
    first = window[0][1]["metrics"]
    last = window[-1][1]["metrics"]
    weighted_fragmentation = 0.0
    covered = 0.0
    for index, (at, sample) in enumerate(window):
        next_at = window[index + 1][0] if index + 1 < len(window) else duration
        weight = max(0.0, min(next_at, duration) - at)
        weighted_fragmentation += weight * sample["metrics"]["cache"]["internal_fragmentation"]
        covered += weight
    return {
        "resource_samples_in_window": len(window),
        "fragmentation_coverage_fraction": covered / duration,
        "time_weighted_internal_fragmentation": weighted_fragmentation / covered if covered else None,
        "queue_at_midpoint": queue(midpoint),
        "queue_at_last_window_sample": queue(window[-1][1]),
        "peak_queued_or_active_requests_sampled": max(queue(sample) for _, sample in window),
        "minimum_free_blocks_sampled": min(sample["metrics"]["cache"]["free_blocks"] for _, sample in window),
        "peak_allocated_kv_slots_sampled": max(sample["metrics"]["cache"]["allocated_slots"] for _, sample in window),
        "preemptions_window_delta_sampled": last["preemptions"] - first["preemptions"],
        "recomputed_tokens_window_delta_sampled": last["recomputed_tokens"] - first["recomputed_tokens"],
        "peak_gpu_memory_used_mib_sampled": max(sample["gpu_memory_used_mib"] for _, sample in window),
    }


def _svg_plot(rows: list[dict], *, title: str, x_label: str, y_label: str,
              x_key: str, y_key: str, log_y: bool = False) -> str:
    width, height = 760, 450
    left, right, top, bottom = 85, 35, 55, 70
    values = [(row, row[x_key], row[y_key]) for row in rows if row[x_key] is not None and row[y_key] is not None]
    xs = [point[1] for point in values]
    ys = [math.log10(point[2]) if log_y else point[2] for point in values if point[2] > 0 or not log_y]
    x_max = max(xs, default=1) * 1.1 or 1
    y_min = min(ys, default=0)
    y_max = max(ys, default=1)
    if not log_y:
        y_min = min(0, y_min)
    else:
        y_min -= 0.1
    y_max += 0.1 if log_y else max(0.05, y_max * 0.12)
    if y_max == y_min:
        y_max += 1
    X = lambda value: left + (value / x_max) * (width - left - right)
    Y = lambda value: top + (1 - ((math.log10(value) if log_y else value) - y_min) / (y_max - y_min)) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
             f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(x_label)}</text>',
             f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(y_label)}</text>']
    for tick in range(6):
        value = x_max * tick / 5
        x = X(value)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = top + (1 - tick / 5) * (height - top - bottom)
        label = f"{10**value:.2g}" if log_y else f"{value:.1f}"
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{label}</text>')
    for backend in ("reference", "flashinfer"):
        group = sorted((row for row in rows if row["backend"] == backend and row[x_key] is not None and row[y_key] is not None),
                       key=lambda row: row["rate_rps"])
        points = " ".join(f'{X(row[x_key]):.1f},{Y(row[y_key]):.1f}' for row in group if not log_y or row[y_key] > 0)
        if points:
            parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[backend]}" stroke-width="2"/>')
        for row in group:
            if log_y and row[y_key] <= 0:
                continue
            x, y = X(row[x_key]), Y(row[y_key])
            fill = "white" if row["failed_total"] else COLORS[backend]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{COLORS[backend]}" stroke-width="2"/>')
            parts.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="sans-serif" font-size="10">{row["rate_rps"]:g}</text>')
    parts.append(f'<text x="{left}" y="{height-42}" fill="{COLORS["reference"]}" font-family="sans-serif" font-size="12">reference</text>')
    parts.append(f'<text x="{left+105}" y="{height-42}" fill="{COLORS["flashinfer"]}" font-family="sans-serif" font-size="12">FlashInfer; hollow = failures</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _ablation_svg(rows: list[dict]) -> str:
    cells = {(16, 16): "reference-b16-n16-short", (16, 1): "reference-b16-n1-ablation",
             (32, 16): "reference-b32-n16-ablation", (32, 1): "reference-b32-n1-ablation"}
    lookup = {row["configuration"]: row for row in rows if row["rate_rps"] == 2}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="395" viewBox="0 0 760 395">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">Reference ablation at 2 offered requests/s</text>',
             '<text x="280" y="67" text-anchor="middle" font-family="sans-serif" font-size="14">16-token blocks</text>',
             '<text x="565" y="67" text-anchor="middle" font-family="sans-serif" font-size="14">32-token blocks</text>']
    for row_index, batch in enumerate((1, 16)):
        y = 85 + row_index * 140
        parts.append(f'<text x="60" y="{y+65}" font-family="sans-serif" font-size="13">batch cap {batch}</text>')
        for col_index, block in enumerate((16, 32)):
            x = 140 + col_index * 285
            item = lookup[cells[(block, batch)]]
            parts.extend([
                f'<rect x="{x}" y="{y}" width="265" height="120" rx="6" fill="#f2f6fb" stroke="#7993ad"/>',
                f'<text x="{x+14}" y="{y+30}" font-family="sans-serif" font-size="16">{item["emitted_tokens_per_s_median"]:.1f} emitted tokens/s</text>',
                f'<text x="{x+14}" y="{y+57}" font-family="sans-serif" font-size="13">SLO goodput {item["slo_goodput_median_rps"]:.2f} req/s</text>',
                f'<text x="{x+14}" y="{y+84}" font-family="sans-serif" font-size="13">weighted frag. {item["fragmentation_time_weighted_median"]:.1%}</text>',
                f'<text x="{x+14}" y="{y+107}" font-family="sans-serif" font-size="12">failed {item["failed_total"]}/{item["requests_total"]}</text>',
            ])
    parts.append('<text x="380" y="375" text-anchor="middle" font-family="sans-serif" font-size="12">Same 256 MiB KV budget; all four points are overloaded. This is not paging-off.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _block_svg(rows: list[dict]) -> str:
    lookup = {row["configuration"]: row for row in rows if row["rate_rps"] == 2}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="340" viewBox="0 0 700 340">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="350" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">Block size at batch cap 16, 2 offered requests/s</text>']
    for index, (block, name) in enumerate(((16, "reference-b16-n16-short"), (32, "reference-b32-n16-ablation"))):
        item = lookup[name]
        x = 110 + index * 300
        frag_width = min(220, item["fragmentation_time_weighted_median"] * 1800)
        throughput_width = min(220, item["emitted_tokens_per_s_median"] * 2)
        parts.extend([
            f'<text x="{x}" y="86" font-family="sans-serif" font-size="16">{block}-token blocks</text>',
            f'<rect x="{x}" y="111" width="{frag_width:.1f}" height="26" fill="#b486c2"/>',
            f'<text x="{x}" y="157" font-family="sans-serif" font-size="13">fragmentation {item["fragmentation_time_weighted_median"]:.1%}</text>',
            f'<rect x="{x}" y="187" width="{throughput_width:.1f}" height="26" fill="#3566a8"/>',
            f'<text x="{x}" y="233" font-family="sans-serif" font-size="13">{item["emitted_tokens_per_s_median"]:.1f} emitted tokens/s</text>',
        ])
    parts.append('<text x="350" y="300" text-anchor="middle" font-family="sans-serif" font-size="12">Bars use separate scales; compare labeled values. Both runs overloaded.</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def summarize(archive: Path, output: Path) -> dict:
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or any(".." in Path(name).parts for name in names):
            raise ValueError("archive has duplicate or unsafe paths")
        roots = {name.split("/")[0] for name in names}
        if len(roots) != 1:
            raise ValueError("archive must have one run root")
        root = roots.pop() + "/"
        if root + "run-error.txt" in names:
            raise ValueError("archive contains a run error")
        read_json = lambda name: json.loads(bundle.read(root + name))
        config = read_json("run-config.json")
        commit = bundle.read(root + "engine-commit.txt").decode().strip()
        if config["model"] != "Qwen/Qwen2.5-1.5B-Instruct" or len(commit) != 40:
            raise ValueError("unexpected run identity")
        run_rows = []
        for full_name in sorted(name for name in names if name.endswith(".score.json")):
            relative = full_name[len(root):]
            config_name, filename = relative.split("/", 1)
            if not filename.startswith("trace-rate-"):
                raise ValueError("unexpected score path")
            workload = "long" if "-long" in config_name else "short"
            trace = read_json(f"{workload}-plan/{filename.replace('.score.json', '.json')}")
            replay = read_json(relative.replace(".score.json", ".json"))
            saved_score = read_json(relative)
            score = score_fixed_window(trace, replay)
            score["configuration"] = config_name
            if score != saved_score:
                raise ValueError(f"score failed reproduction: {relative}")
            if (not score["client_send_lag_gate_passed"] or score["emitted_output_tokens_per_s"] is None
                    or score["missing_window_tpot"] or score["cohort"]["missing_usage"]):
                raise ValueError(f"data-quality gate failed: {relative}")
            samples = read_json(relative.replace(".score.json", ".metrics.json"))
            resources = _resource_samples(samples, replay)
            rate_index = int(filename.split("-")[2])
            repetition = int(filename.split("-")[4].split(".")[0])
            run_rows.append({
                "configuration": config_name, "backend": "flashinfer" if config_name.startswith("flashinfer") else "reference",
                "workload": workload, "rate_index": rate_index, "repetition": repetition,
                "rate_rps": score["target_offered_rate_rps"], "requests": score["cohort"]["requests"],
                "completed_total": score["cohort"]["completed"], "failed": score["cohort"]["failed"],
                "completed_within_window": score["completed_within_window"],
                "completed_during_drain": score["completed_during_drain"],
                "slo_goodput_rps": score["slo_goodput_rps"],
                "emitted_output_tokens_per_s": score["emitted_output_tokens_per_s"],
                "window_completed_client_ttft_p99_s": score["window_completed_client_ttft_p99_s"],
                "client_send_lag_p99_s": score["client_send_lag_p99_s"],
                **resources,
            })
    if len(run_rows) != 75:
        raise ValueError(f"expected 75 scored runs, found {len(run_rows)}")
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in run_rows:
        groups.setdefault((row["configuration"], row["rate_rps"]), []).append(row)
    by_rate = []
    for (name, rate), rows in sorted(groups.items()):
        if len(rows) != 3 or sorted(row["repetition"] for row in rows) != [0, 1, 2]:
            raise ValueError(f"missing repetition: {name} {rate}")
        by_rate.append({
            "configuration": name, "backend": rows[0]["backend"], "workload": rows[0]["workload"],
            "rate_rps": rate, "repetitions": 3,
            "requests_total": sum(row["requests"] for row in rows),
            "completed_total": sum(row["completed_total"] for row in rows),
            "completed_within_window_total": sum(row["completed_within_window"] for row in rows),
            "failed_total": sum(row["failed"] for row in rows),
            "failure_fraction": sum(row["failed"] for row in rows) / sum(row["requests"] for row in rows),
            "slo_goodput_median_rps": _median(rows, "slo_goodput_rps"),
            "slo_goodput_min_rps": min(row["slo_goodput_rps"] for row in rows),
            "slo_goodput_max_rps": max(row["slo_goodput_rps"] for row in rows),
            "emitted_tokens_per_s_median": _median(rows, "emitted_output_tokens_per_s"),
            "window_ttft_p99_median_s": _median(rows, "window_completed_client_ttft_p99_s"),
            "window_ttft_p99_sample_counts": [row["completed_within_window"] for row in rows],
            "fragmentation_time_weighted_median": _median(rows, "time_weighted_internal_fragmentation"),
            "queue_midpoint_median": _median(rows, "queue_at_midpoint"),
            "queue_end_median": _median(rows, "queue_at_last_window_sample"),
            "peak_gpu_memory_used_mib_max_sampled": max(row["peak_gpu_memory_used_mib_sampled"] for row in rows),
            "preemptions_total_sampled": sum(row["preemptions_window_delta_sampled"] for row in rows),
            "recomputed_tokens_total_sampled": sum(row["recomputed_tokens_window_delta_sampled"] for row in rows),
        })
    summary = {
        "schema_version": 1, "kind": "phase6-scored-nanoserve-summary",
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "engine_commit": commit, "run_config": config,
        "quality": {"recomputed_scores": len(run_rows), "send_lag_gate_failures": 0,
                    "missing_token_windows": 0, "missing_window_tpot": 0,
                    "missing_usage": 0, "resource_sample_errors": 0},
        "total_requests": sum(row["requests"] for row in run_rows),
        "total_failed": sum(row["failed"] for row in run_rows),
        "by_rate": by_rate,
        "limitations": [
            "Synthetic repetitive prompts are not a production trace.",
            "Each repetition offers load for 60 seconds, then drains for at most 30; drain completions earn no goodput.",
            "p99 values are median-of-run p99s, with per-run sample counts shown; they are not pooled p99s.",
            "Resource peaks and fragmentation are sampled approximately once per second, not device-wide continuous extrema.",
            "FlashInfer adds planning workspace beyond the equal 256 MiB KV pools; total device-memory use differs.",
            "A 2x2 block-size/batching-limit ablation is not a paging/no-paging ablation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "by-rate.csv", by_rate)
    short = [row for row in by_rate if row["configuration"] in ("reference-b16-n16-short", "flashinfer-b16-n16-short")]
    (output / "throughput-vs-ttft.svg").write_text(_svg_plot(
        short, title="Short synthetic workload: throughput vs p99 TTFT",
        x_label="Output tokens emitted per offered-window second", y_label="Median of run p99 client TTFT (s, log scale)",
        x_key="emitted_tokens_per_s_median", y_key="window_ttft_p99_median_s", log_y=True), encoding="utf-8")
    (output / "goodput-vs-rate.svg").write_text(_svg_plot(
        short, title="Short synthetic workload: SLO goodput vs offered rate",
        x_label="Offered requests/s", y_label="Median SLO-qualified completions/s",
        x_key="rate_rps", y_key="slo_goodput_median_rps"), encoding="utf-8")
    (output / "failure-vs-rate.svg").write_text(_svg_plot(
        short, title="Short synthetic workload: failures vs offered rate",
        x_label="Offered requests/s", y_label="Full-cohort failure fraction",
        x_key="rate_rps", y_key="failure_fraction"), encoding="utf-8")
    (output / "ablation-2x2.svg").write_text(_ablation_svg(by_rate), encoding="utf-8")
    (output / "block-size.svg").write_text(_block_svg(by_rate), encoding="utf-8")
    long_rows = [row for row in by_rate if row["configuration"] in ("reference-b16-n16-long", "flashinfer-b16-n16-long")]
    (output / "long-goodput-vs-rate.svg").write_text(_svg_plot(
        long_rows, title="Long synthetic prefill: SLO goodput vs offered rate",
        x_label="Offered requests/s", y_label="Median SLO-qualified completions/s",
        x_key="rate_rps", y_key="slo_goodput_median_rps"), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.archive, args.output_dir)
    print(json.dumps({"archive_sha256": report["source_archive_sha256"],
                      "validated_runs": report["quality"]["recomputed_scores"],
                      "by_rate_rows": len(report["by_rate"])}, indent=2))


if __name__ == "__main__":
    main()
