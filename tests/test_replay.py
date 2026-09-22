import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nanoserve.replay import (
    HTTPCompletionsAdapter,
    make_completion_trace,
    make_duration_completion_trace,
    replay_bounded_http_trace,
    replay_completion_trace,
    validate_completion_trace,
)
from nanoserve.experiment import analyze_replay


class ReplayTests(unittest.TestCase):
    def make_trace(self, count=2):
        return make_completion_trace(
            count=count,
            rate=1000,
            seed=7,
            model="test-model",
            revision="test-revision",
            prompts=["hello", "world"],
            max_tokens=2,
        )

    def test_trace_is_reproducible_and_tampering_is_rejected(self):
        first = self.make_trace()
        self.assertEqual(first, self.make_trace())
        validate_completion_trace(first)
        first["requests"][0]["prompt"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_completion_trace(first)

    def test_saved_trace_reproduces_with_equivalent_rate_types(self):
        saved_path = Path(__file__).resolve().parents[1] / "environment" / "phase4-debug-trace.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        arguments = dict(
            count=2,
            seed=7,
            model=saved["model"],
            revision=saved["revision"],
            prompts=["The capital of France is", "Two plus two equals"],
            max_tokens=2,
        )
        self.assertEqual(saved, make_completion_trace(rate=100, **arguments))
        self.assertEqual(saved, make_completion_trace(rate=100.0, **arguments))

    def test_replay_retains_every_completion_and_failure_in_trace_order(self):
        class FakeAdapter:
            name = "fake"

            def run(self, request, trace_start):
                if request["request_id"] == "r000001":
                    raise RuntimeError("injected failure")
                sent = time.monotonic() - trace_start
                return {
                    "request_id": request["request_id"],
                    "status": "completed",
                    "actual_send_offset_s": sent,
                    "usage": {"completion_tokens": 2},
                }

        result = replay_completion_trace(self.make_trace(), FakeAdapter())
        self.assertEqual([item["request_id"] for item in result["records"]], ["r000000", "r000001"])
        self.assertEqual(result["summary"], {"requests": 2, "completed": 1, "failed": 1, "missing_usage": 0})
        self.assertIn("injected failure", result["records"][1]["error"])
        self.assertIn("send_lag_s", result["records"][0])

    def test_http_adapter_consumes_usage_chunk_and_does_not_count_chunks_as_tokens(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                data = self.rfile.read(int(self.headers["Content-Length"]))
                captured.append(json.loads(data))
                events = [
                    {"id": "remote-1", "choices": [{"text": "hello", "finish_reason": None}]},
                    {"id": "remote-1", "choices": [{"text": " world", "finish_reason": "length"}]},
                    {"id": "remote-1", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}},
                ]
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                body += "data: [DONE]\n\n"
                encoded = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            adapter = HTTPCompletionsAdapter(
                f"http://{host}:{port}/v1/completions", "test-model"
            )
            result = replay_completion_trace(self.make_trace(count=1), adapter)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

        record = result["records"][0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["output_text"], "hello world")
        self.assertEqual(len(record["chunks"]), 2)
        self.assertEqual(record["usage"]["completion_tokens"], 3)
        self.assertEqual(result["summary"]["missing_usage"], 0)
        self.assertTrue(captured[0]["stream_options"]["include_usage"])

    def test_http_rejection_retains_actual_send_time(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                self.send_response(429)
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            adapter = HTTPCompletionsAdapter(
                f"http://{host}:{port}/v1/completions", "test-model"
            )
            result = replay_completion_trace(self.make_trace(count=1), adapter)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

        record = result["records"][0]
        self.assertEqual(record["status"], "failed")
        self.assertIn("HTTP 429", record["error"])
        self.assertIn("actual_send_offset_s", record)
        self.assertIn("send_lag_s", record)

    def test_bounded_replay_completes_fast_fixed_window_requests(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                events = [
                    {"id": "remote", "choices": [{"text": " ok", "finish_reason": "length"}]},
                    {"id": "remote", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                ]
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
                encoded = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        trace = make_duration_completion_trace(
            duration_s=0.05, rate=100, seed=7, model="test-model",
            revision="revision", prompts=["p"], max_tokens=1,
        )
        try:
            host, port = server.server_address
            adapter = HTTPCompletionsAdapter(f"http://{host}:{port}/v1/completions", "test-model")
            replay = replay_bounded_http_trace(trace, adapter, drain_s=1, max_workers=4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertEqual(replay["summary"]["completed"], len(trace["requests"]))
        self.assertEqual(replay["summary"]["timed_out"], 0)
        self.assertEqual(replay["summary"]["not_sent"], 0)
        aggregate, _, _ = analyze_replay(trace, replay)
        self.assertEqual(aggregate["cohort"]["completed"], len(trace["requests"]))

    def test_bounded_replay_retains_inflight_and_queued_at_cutoff(self):
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                release.wait(2)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        trace = make_duration_completion_trace(
            duration_s=0.2, rate=100, seed=7, model="test-model",
            revision="revision", prompts=["p"], max_tokens=1,
        )
        self.assertGreater(len(trace["requests"]), 1)
        try:
            host, port = server.server_address
            adapter = HTTPCompletionsAdapter(f"http://{host}:{port}/v1/completions", "test-model")
            started = time.monotonic()
            replay = replay_bounded_http_trace(trace, adapter, drain_s=0.1, max_workers=1)
            self.assertLess(time.monotonic() - started, 1.5)
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(2)
        self.assertEqual(replay["summary"]["completed"], 0)
        self.assertEqual(replay["summary"]["timed_out"], 1)
        self.assertEqual(replay["summary"]["not_sent"], len(trace["requests"]) - 1)
        self.assertEqual([record["request_id"] for record in replay["records"]], [request["request_id"] for request in trace["requests"]])
        aggregate, _, _ = analyze_replay(trace, replay)
        self.assertEqual(aggregate["cohort"]["failure_types"]["drain_timeout"], 1)

    def test_bounded_replay_requires_fixed_window(self):
        adapter = HTTPCompletionsAdapter("http://127.0.0.1:8000/v1/completions", "test-model")
        with self.assertRaisesRegex(ValueError, "offered_interval"):
            replay_bounded_http_trace(self.make_trace(), adapter, drain_s=1)


if __name__ == "__main__":
    unittest.main()
