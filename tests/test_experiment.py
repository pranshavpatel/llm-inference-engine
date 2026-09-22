import csv
import json
import tempfile
import unittest
from pathlib import Path

from nanoserve.experiment import analyze_replay, write_analysis, write_sweep_plan
from nanoserve.replay import make_completion_trace, validate_completion_trace


class ExperimentTests(unittest.TestCase):
    def make_pair(self):
        trace = make_completion_trace(
            count=2, rate=10, seed=4, model="test-model", revision="revision",
            prompts=["prompt"], max_tokens=2,
        )
        records = []
        for index, request in enumerate(trace["requests"]):
            sent = request["arrival_offset_s"] + 0.01
            records.append({
                "request_id": request["request_id"],
                "status": "completed",
                "actual_send_offset_s": sent,
                "first_content_offset_s": sent + 0.2,
                "completed_offset_s": sent + 0.5,
                "finish_reason": "length",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "chunks": [
                    {"at_offset_s": sent + 0.2, "text": "one", "finish_reason": None},
                    {"at_offset_s": sent + 0.4, "text": " two", "finish_reason": "length"},
                ],
            })
        replay = {
            "schema_version": 1, "kind": "completion-replay",
            "trace_sha256": trace["sha256"], "model": trace["model"],
            "revision": trace["revision"], "adapter": "fake",
            "elapsed_s": max(record["completed_offset_s"] for record in records) + 0.01,
            "summary": {"requests": 2, "completed": 2, "failed": 0, "missing_usage": 0},
            "records": records,
        }
        return trace, replay

    def test_full_cohort_metrics_do_not_confuse_chunks_with_tokens(self):
        trace, replay = self.make_pair()
        aggregate, rows, events = analyze_replay(trace, replay)
        self.assertEqual(aggregate["cohort"]["completed"], 2)
        self.assertEqual(aggregate["full_run_completed_output_tokens"], 4)
        self.assertEqual(aggregate["client_ttft"]["p99_s"], 0.2)
        self.assertEqual(aggregate["inter_content_chunk_gap"]["count"], 2)
        self.assertEqual(len(events), 4)
        self.assertEqual(rows[0]["completion_tokens"], 2)
        self.assertNotIn("tpot", aggregate)
        self.assertNotIn("goodput", aggregate)

    def test_failure_and_missing_usage_remain_visible(self):
        trace, replay = self.make_pair()
        replay["records"][0]["usage"] = None
        replay["records"][1] = {
            "request_id": trace["requests"][1]["request_id"],
            "status": "failed", "actual_send_offset_s": 0.2,
            "completed_offset_s": 0.21, "error": "HTTP 429: overloaded",
        }
        replay["summary"] = {"requests": 2, "completed": 1, "failed": 1, "missing_usage": 1}
        aggregate, _, _ = analyze_replay(trace, replay)
        self.assertEqual(aggregate["cohort"]["failure_types"], {"http_429": 1})
        self.assertEqual(aggregate["cohort"]["missing_usage"], 1)
        self.assertIsNone(aggregate["full_run_completed_output_tokens_per_s"])

    def test_mismatched_trace_and_corrupt_timing_are_rejected(self):
        trace, replay = self.make_pair()
        replay["trace_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "trace_sha256"):
            analyze_replay(trace, replay)
        replay["trace_sha256"] = trace["sha256"]
        replay["records"][0]["first_content_offset_s"] = 0.0
        with self.assertRaisesRegex(ValueError, "first content"):
            analyze_replay(trace, replay)

    def test_export_writes_manifest_csv_events_and_aggregate(self):
        trace, replay = self.make_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.json"
            replay_path = root / "replay.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            output_dir = root / "analysis"
            write_analysis(trace_path, replay_path, output_dir)
            self.assertEqual(json.loads((output_dir / "aggregate.json").read_text())["cohort"]["requests"], 2)
            self.assertEqual(json.loads((output_dir / "manifest.json").read_text())["model"], "test-model")
            with (output_dir / "per-request.csv").open(newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)
            self.assertEqual(len((output_dir / "chunk-events.jsonl").read_text().splitlines()), 4)
            with self.assertRaises(FileExistsError):
                write_analysis(trace_path, replay_path, output_dir)

    def test_committed_phase4_records_are_accepted_without_performance_claims(self):
        root = Path(__file__).resolve().parents[1] / "environment"
        trace = json.loads((root / "phase4-debug-trace.json").read_text(encoding="utf-8"))
        for adapter in ("hf", "nanoserve", "vllm"):
            with self.subTest(adapter=adapter):
                replay = json.loads((root / f"phase4-{adapter}-smoke.json").read_text(encoding="utf-8"))
                aggregate, _, _ = analyze_replay(trace, replay)
                self.assertEqual(aggregate["cohort"]["completed"], 2)
                self.assertEqual(aggregate["full_run_completed_output_tokens"], 4)

    def test_sweep_plan_is_fixed_window_reproducible_and_paired(self):
        arguments = dict(
            rates=[2, 4], repetitions=3, duration_s=5, base_seed=10,
            model="test-model", revision="revision", prompts=["a", "b"], max_tokens=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            plan = write_sweep_plan(first, **arguments)
            self.assertEqual(plan["status"], "arrival plan only; not a scored benchmark run")
            self.assertEqual(len(plan["traces"]), 6)
            self.assertEqual([item["seed"] for item in plan["traces"]], [10, 11, 12, 10, 11, 12])
            for item in plan["traces"]:
                trace = json.loads((first / item["file"]).read_text(encoding="utf-8"))
                validate_completion_trace(trace)
                self.assertLessEqual(trace["requests"][-1]["arrival_offset_s"], 5)
            write_sweep_plan(second, **arguments)
            self.assertEqual(
                (first / "sweep-plan.json").read_bytes(),
                (second / "sweep-plan.json").read_bytes(),
            )

    def test_sweep_plan_rejects_bad_rates_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with self.assertRaisesRegex(ValueError, "rates"):
                write_sweep_plan(
                    output, rates=[1, 1], repetitions=3, duration_s=5, base_seed=0,
                    model="m", revision="r", prompts=["p"], max_tokens=2,
                )
            self.assertFalse(output.exists())

    def test_sweep_plan_fixed_output_policy_reaches_each_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixed"
            manifest = write_sweep_plan(
                output, rates=[10], repetitions=1, duration_s=1, base_seed=1,
                model="m", revision="r", prompts=["p"], max_tokens=4,
                ignore_eos=True,
            )
            trace = json.loads((output / manifest["traces"][0]["file"]).read_text(encoding="utf-8"))
            self.assertIn("require exactly", manifest["output_policy"])
            self.assertTrue(all(request["ignore_eos"] for request in trace["requests"]))


if __name__ == "__main__":
    unittest.main()
