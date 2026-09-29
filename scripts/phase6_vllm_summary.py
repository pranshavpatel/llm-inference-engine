"""Validate the saved vLLM sweep against NanoServe's exact Phase 6 traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from nanoserve.phase6_score import score_fixed_window


ENGINE_CONFIGS = {
    "reference": "reference-b16-n16",
    "flashinfer": "flashinfer-b16-n16",
    "vllm": "vllm",
}
COLORS = {"reference": "#3566a8", "flashinfer": "#d36031", "vllm": "#39824b"}


def _root(bundle: zipfile.ZipFile) -> str:
    names = bundle.namelist()
    if len(names) != len(set(names)) or any(".." in Path(name).parts for name in names):
        raise ValueError("duplicate or unsafe archive path")
    roots = {name.split("/")[0] for name in names}
    if len(roots) != 1:
        raise ValueError("archive must have exactly one root")
    root = roots.pop() + "/"
    if root + "run-error.txt" in names:
        raise ValueError("archive contains run-error.txt")
    return root


def _median(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows]
    return statistics.median(values) if values and all(value is not None for value in values) else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _comparison_svg(rows: list[dict], workload: str) -> str:
    selected = [row for row in rows if row["workload"] == workload]
    max_rate = max(row["rate_rps"] for row in selected)
    max_goodput = max(row["slo_goodput_median_rps"] for row in selected) * 1.12
    width, height = 760, 430
    left, top, right, bottom = 80, 55, 30, 75
    X = lambda value: left + value / max_rate * (width - left - right)
    Y = lambda value: top + (1 - value / max_goodput) * (height - top - bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{workload.title()} synthetic workload: SLO goodput</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
             f'<text x="380" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="13">Offered requests/s</text>',
             f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Median SLO-qualified completions/s</text>']
    for index in range(7):
        x = X(max_rate * index / 6)
        y = Y(max_goodput * index / 6)
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="11">{max_rate*index/6:.1f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-7}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{max_goodput*index/6:.1f}</text>')
    for engine in ("reference", "flashinfer", "vllm"):
        group = sorted((row for row in selected if row["engine"] == engine), key=lambda row: row["rate_rps"])
        points = " ".join(f'{X(row["rate_rps"]):.1f},{Y(row["slo_goodput_median_rps"]):.1f}' for row in group)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[engine]}" stroke-width="2"/>')
        for row in group:
            parts.append(f'<circle cx="{X(row["rate_rps"]):.1f}" cy="{Y(row["slo_goodput_median_rps"]):.1f}" r="4" fill="{COLORS[engine]}"/>')
    for index, engine in enumerate(("reference", "flashinfer", "vllm")):
        parts.append(f'<text x="{left+index*145}" y="{height-43}" fill="{COLORS[engine]}" font-family="sans-serif" font-size="12">{engine}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def summarize(vllm_archive: Path, nano_archive: Path, output: Path) -> dict:
    with zipfile.ZipFile(vllm_archive) as vllm, zipfile.ZipFile(nano_archive) as nano:
        vr, nr = _root(vllm), _root(nano)
        names = vllm.namelist()
        commit = vllm.read(vr + "engine-commit.txt").decode().strip()
        packages = vllm.read(vr + "packages.txt").decode().strip().splitlines()
        if len(commit) != 40 or "vllm 0.30.0" not in packages or vllm.read(vr + "git-tracked-status.txt").strip():
            raise ValueError("unexpected or dirty vLLM environment")
        for workload in ("short", "long"):
            plan = f"{workload}-plan/sweep-plan.json"
            if vllm.read(vr + plan) != nano.read(nr + plan):
                raise ValueError(f"{workload} plan differs from NanoServe")
        rows = []
        for name in sorted(n for n in names if n.endswith(".score.json")):
            relative = name[len(vr):]
            parts = relative.split("/")
            if len(parts) != 2 or parts[0] not in ("short", "long") or not parts[1].startswith("trace-rate-"):
                raise ValueError(f"unexpected score: {relative}")
            workload = parts[0]
            replay_name = relative.replace(".score.json", ".json")
            trace_name = f"{workload}-plan/{parts[1].replace('.score.json', '.json')}"
            if vllm.read(vr + trace_name) != nano.read(nr + trace_name):
                raise ValueError(f"trace differs from NanoServe: {trace_name}")
            trace = json.loads(vllm.read(vr + trace_name))
            replay = json.loads(vllm.read(vr + replay_name))
            score = json.loads(vllm.read(name))
            recomputed = score_fixed_window(trace, replay)
            recomputed["configuration"] = f"vllm-{workload}"
            if recomputed != score:
                raise ValueError(f"score does not reproduce: {relative}")
            if (not score["client_send_lag_gate_passed"] or score["missing_window_tpot"]
                    or score["cohort"]["missing_usage"] or score["cohort"]["failed"]
                    or score["cohort"]["requests"] != len(replay["records"])):
                raise ValueError(f"quality gate failed: {relative}")
            if score["emitted_output_tokens_per_s"] is not None:
                raise ValueError("vLLM must not claim NanoServe's exact token-window count")
            filename = parts[1]
            rows.append({
                "engine": "vllm", "workload": workload,
                "rate_index": int(filename.split("-")[2]),
                "repetition": int(filename.split("-")[4].split(".")[0]),
                "rate_rps": score["target_offered_rate_rps"],
                "requests": score["cohort"]["requests"],
                "completed_total": score["cohort"]["completed"],
                "completed_within_window": score["completed_within_window"],
                "completed_during_drain": score["completed_during_drain"],
                "failed": score["cohort"]["failed"],
                "slo_goodput_rps": score["slo_goodput_rps"],
                "window_completed_client_ttft_p99_s": score["window_completed_client_ttft_p99_s"],
                "client_send_lag_p99_s": score["client_send_lag_p99_s"],
            })
    if len(rows) != 33 or sum(row["requests"] for row in rows) != 3691:
        raise ValueError("incomplete or changed vLLM workload")
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["workload"], row["rate_rps"]), []).append(row)
    vllm_rates = []
    for (workload, rate), group in sorted(groups.items()):
        if sorted(row["repetition"] for row in group) != [0, 1, 2]:
            raise ValueError(f"missing repetition: {workload} at {rate}")
        vllm_rates.append({
            "engine": "vllm", "workload": workload, "rate_rps": rate,
            "requests_total": sum(row["requests"] for row in group),
            "completed_within_window_total": sum(row["completed_within_window"] for row in group),
            "failed_total": sum(row["failed"] for row in group),
            "slo_goodput_median_rps": _median(group, "slo_goodput_rps"),
            "window_ttft_p99_median_s": _median(group, "window_completed_client_ttft_p99_s"),
            "window_ttft_p99_sample_counts": [row["completed_within_window"] for row in sorted(group, key=lambda r: r["repetition"])],
        })
    nano_summary = json.loads(Path("environment/phase6-vm-scored/report/summary.json").read_text(encoding="utf-8"))
    if nano_summary["source_archive_sha256"] != hashlib.sha256(nano_archive.read_bytes()).hexdigest():
        raise ValueError("NanoServe summary does not match source archive")
    comparison = []
    for row in nano_summary["by_rate"]:
        for engine in ("reference", "flashinfer"):
            if row["configuration"] == f"{ENGINE_CONFIGS[engine]}-{row['workload']}":
                comparison.append({
                    "engine": engine, "workload": row["workload"], "rate_rps": row["rate_rps"],
                    "requests_total": row["requests_total"],
                    "completed_within_window_total": row["completed_within_window_total"],
                    "failed_total": row["failed_total"],
                    "slo_goodput_median_rps": row["slo_goodput_median_rps"],
                    "window_ttft_p99_median_s": row["window_ttft_p99_median_s"],
                    "window_ttft_p99_sample_counts": row["window_ttft_p99_sample_counts"],
                })
    comparison.extend(vllm_rates)
    for row in vllm_rates:
        for engine in ("reference", "flashinfer"):
            paired = next((item for item in comparison if item["engine"] == engine
                           and item["workload"] == row["workload"] and item["rate_rps"] == row["rate_rps"]), None)
            if paired is None or paired["requests_total"] != row["requests_total"]:
                raise ValueError(f"unpaired run: {row['workload']} at {row['rate_rps']}")
    result = {
        "schema_version": 1, "kind": "phase6-scored-cross-engine-summary",
        "vllm_archive_sha256": hashlib.sha256(vllm_archive.read_bytes()).hexdigest(),
        "nanoserve_archive_sha256": nano_summary["source_archive_sha256"],
        "vllm_engine_commit": commit, "vllm_packages": packages,
        "quality": {"recomputed_vllm_scores": len(rows), "identical_plans_and_traces": True,
                    "vllm_requests": sum(row["requests"] for row in rows),
                    "vllm_failed": sum(row["failed"] for row in rows),
                    "vllm_completed_during_drain": sum(row["completed_during_drain"] for row in rows),
                    "send_lag_gate_failures": 0, "missing_usage": 0, "missing_window_tpot": 0},
        "by_rate": sorted(comparison, key=lambda row: (row["workload"], row["rate_rps"], row["engine"])),
        "limitations": [
            "These are synthetic repetitive prompts, not a public or production trace.",
            "p99 values are medians of three run-level p99s, not pooled p99s; low-rate samples are small.",
            "vLLM had no overload knee in the tested range; its maximum sustainable rate was not measured.",
            "vLLM has no exact in-window emitted-token counter; do not compare its completed-token yield to NanoServe's emitted-token throughput.",
            "The KV pools were 256 MiB, but total GPU memory and engine optimizations were not feature- or footprint-identical.",
            "vLLM's per-request metrics mode may add CPU overhead.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "vllm-runs.csv", rows)
    _write_csv(output / "by-rate.csv", comparison)
    for workload in ("short", "long"):
        (output / f"{workload}-goodput-vs-rate.svg").write_text(_comparison_svg(comparison, workload), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-archive", type=Path, required=True)
    parser.add_argument("--nano-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.vllm_archive, args.nano_archive, args.output_dir)
    print(json.dumps({"vllm_archive_sha256": result["vllm_archive_sha256"],
                      "recomputed_scores": result["quality"]["recomputed_vllm_scores"],
                      "paired_rates": len(result["by_rate"])}, indent=2))


if __name__ == "__main__":
    main()
