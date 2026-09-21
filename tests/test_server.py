import http.client
import json
import socket
import struct
import threading
import time
import unittest

from nanoserve.server import CompletionService, make_server
from nanoserve.types import FinishReason, OutputEvent
from nanoserve.worker import RequestHandle, WorkerQueueFull


class CharacterCodec:
    eos_token_id = None

    def encode(self, text):
        return [ord(character) for character in text]

    def decode_token(self, token_id):
        return chr(token_id)


class StubWorker:
    def __init__(self):
        self.is_running = True
        self.overloaded = False
        self.cancelled = []
        self.counter = 0

    def submit(self, prompt_token_ids, max_new_tokens, *, eos_token_id=None, request_id=None):
        if self.overloaded:
            raise WorkerQueueFull("test capacity reached")
        self.counter += 1
        handle = RequestHandle(
            request_id or f"cmpl-test-{self.counter}",
            len(prompt_token_ids),
            max_new_tokens,
            1_700_000_000,
        )
        for index in range(max_new_tokens):
            finished = index == max_new_tokens - 1
            handle._events.put(
                OutputEvent(
                    handle.request_id,
                    ord("A") + index,
                    finished,
                    FinishReason.LENGTH if finished else None,
                    emitted_at=1_700_000_000 + index,
                )
            )
        return handle

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        return None

    def stats(self):
        return {"worker": {"running": self.is_running, "submitted": self.counter}}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.worker = StubWorker()
        self.server = make_server(
            self.worker,
            CharacterCodec(),
            model="test-model",
            port=0,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
        body = None if payload is None else json.dumps(payload)
        request_headers = headers or {}
        if payload is not None and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        result = response.status, response.getheader("Content-Type"), data
        connection.close()
        return result

    def test_health_readiness_and_metrics(self):
        status, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})

        status, _, body = self.request("GET", "/ready")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ready"})

        status, _, body = self.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["worker"]["running"])

    def test_nonstreaming_completion_has_usage_and_finish_reason(self):
        status, content_type, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "hi", "max_tokens": 3},
        )
        response = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(response["choices"][0]["text"], "ABC")
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertEqual(
            response["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    def test_streaming_completion_uses_sse_and_done_marker(self):
        status, content_type, body = self.request(
            "POST",
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 2,
                "stream": True,
                "temperature": 0,
            },
        )
        text = body.decode()

        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/event-stream")
        self.assertEqual(text.count("data: {"), 2)
        self.assertIn('"text":"A"', text)
        self.assertIn('"finish_reason":"length"', text)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_eos_maps_to_openai_stop_reason(self):
        self.assertEqual(CompletionService.finish_reason(FinishReason.EOS), "stop")

    def test_stream_disconnect_requests_cancellation(self):
        release = threading.Event()

        class SlowWorker(StubWorker):
            def submit(
                self,
                prompt_token_ids,
                max_new_tokens,
                *,
                eos_token_id=None,
                request_id=None,
            ):
                handle = RequestHandle(
                    "cmpl-disconnect", len(prompt_token_ids), max_new_tokens, 1
                )

                def feed():
                    handle._events.put(OutputEvent(handle.request_id, ord("A"), False))
                    release.wait(2)
                    for index in range(20):
                        handle._events.put(
                            OutputEvent(
                                handle.request_id,
                                ord("B"),
                                index == 19,
                                FinishReason.LENGTH if index == 19 else None,
                            )
                        )
                        time.sleep(0.01)

                threading.Thread(target=feed, daemon=True).start()
                return handle

        slow_worker = SlowWorker()
        self.server.service.worker = slow_worker
        payload = json.dumps(
            {
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 21,
                "stream": True,
            }
        ).encode()
        request = (
            b"POST /v1/completions HTTP/1.1\r\n"
            + f"Host: {self.host}:{self.port}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        connection = socket.create_connection((self.host, self.port), timeout=2)
        connection.sendall(request)
        received = b""
        while b"data: {" not in received:
            received += connection.recv(4096)
        connection.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("hh", 1, 0),
        )
        connection.close()
        release.set()

        deadline = time.time() + 2
        while not slow_worker.cancelled and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(slow_worker.cancelled, ["cmpl-disconnect"])

    def test_validation_overload_and_unknown_routes_are_explicit(self):
        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "x"},
        )
        self.assertEqual(status, 400)
        self.assertIn("model", json.loads(body)["error"]["message"])

        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "wrong", "prompt": "x"},
        )
        self.assertEqual(status, 400)
        self.assertIn("only model", json.loads(body)["error"]["message"])

        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "x", "temperature": 0.5},
        )
        self.assertEqual(status, 400)
        self.assertIn("greedy", json.loads(body)["error"]["message"])

        self.worker.overloaded = True
        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "x"},
        )
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body)["error"]["type"], "overloaded")

        status, _, _ = self.request("GET", "/missing")
        self.assertEqual(status, 404)

    def test_rejects_wrong_content_type_and_unsupported_parameters(self):
        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "x"},
            {"Content-Type": "text/plain"},
        )
        self.assertEqual(status, 400)
        self.assertIn("Content-Type", json.loads(body)["error"]["message"])

        status, _, body = self.request(
            "POST",
            "/v1/completions",
            {"model": "test-model", "prompt": "x", "top_p": 1},
        )
        self.assertEqual(status, 400)
        self.assertIn("top_p", json.loads(body)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
