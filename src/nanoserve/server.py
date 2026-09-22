"""Dependency-free HTTP completion server with SSE streaming."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from nanoserve.scheduler import QueueFull
from nanoserve.types import FinishReason, OutputEvent
from nanoserve.worker import InferenceWorker, TokenCodec, WorkerClosed


MAX_REQUEST_BYTES = 1_048_576


def _error(message: str, error_type: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": error_type}}


class CompletionService:
    def __init__(self, worker: InferenceWorker, codec: TokenCodec, *, model: str) -> None:
        if not model:
            raise ValueError("model must be nonempty")
        self.worker = worker
        self.codec = codec
        self.model = model

    def submit(self, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        requested_model = payload.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            raise ValueError("model must be a nonempty string")
        if requested_model != self.model:
            raise ValueError(f"only model '{self.model}' is available")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        max_tokens = payload.get("max_tokens", 16)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        ignore_eos = payload.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        n = payload.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int) or n != 1:
            raise ValueError("only n=1 is supported")
        temperature = payload.get("temperature", 0)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature != 0
        ):
            raise ValueError("only greedy temperature=0 is supported")
        stream_options = payload.get("stream_options", {})
        if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
            raise ValueError("stream_options only supports include_usage")
        include_usage = stream_options.get("include_usage", False)
        if not isinstance(include_usage, bool) or (include_usage and not stream):
            raise ValueError("include_usage requires stream=true and a boolean value")
        unsupported = set(payload) - {
            "model",
            "prompt",
            "max_tokens",
            "ignore_eos",
            "stream",
            "n",
            "temperature",
            "stream_options",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported parameters: {names}")
        prompt_ids = tuple(self.codec.encode(prompt))
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        handle = self.worker.submit(
            prompt_ids,
            max_tokens,
            eos_token_id=None if ignore_eos else self.codec.eos_token_id,
        )
        return handle, stream, include_usage

    def event_payload(self, handle, event: OutputEvent, *, text: str) -> dict:
        finish_reason = self.finish_reason(event.finish_reason)
        return {
            "id": handle.request_id,
            "object": "text_completion.chunk",
            "created": int(handle.created_at),
            "model": self.model,
            "choices": [
                {"index": 0, "text": text, "finish_reason": finish_reason}
            ],
        }

    @staticmethod
    def finish_reason(reason: Optional[FinishReason]) -> Optional[str]:
        if reason == FinishReason.EOS:
            return "stop"
        return reason.value if reason is not None else None

    def collect(self, handle) -> dict:
        token_ids = []
        final_event: Optional[OutputEvent] = None
        for event in handle.iter_events():
            if event.error:
                raise RuntimeError(event.error)
            if event.token_id is not None:
                token_ids.append(event.token_id)
            final_event = event
        # Token count is exact because every non-error event carries one token.
        completion_tokens = len(token_ids)
        return {
            "id": handle.request_id,
            "object": "text_completion",
            "created": int(handle.created_at),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "text": self.codec.decode_tokens(token_ids),
                    "finish_reason": (
                        self.finish_reason(final_event.finish_reason)
                        if final_event is not None and final_event.finish_reason is not None
                        else None
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": handle.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": handle.prompt_tokens + completion_tokens,
            },
        }


class CompletionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service: CompletionService):
        self.service = service
        super().__init__(address, CompletionRequestHandler)


class CompletionRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "nanoserve/0.1"

    @property
    def service(self) -> CompletionService:
        return self.server.service

    def log_message(self, format, *args):
        return

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
        elif self.path == "/ready":
            ready = self.service.worker.is_running
            self._write_json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "ready" if ready else "not_ready"},
            )
        elif self.path == "/metrics":
            self._write_json(HTTPStatus.OK, self.service.worker.stats())
        else:
            self._write_json(HTTPStatus.NOT_FOUND, _error("route not found"))

    def _read_payload(self) -> dict:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("request body must be valid JSON") from error

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self._write_json(HTTPStatus.NOT_FOUND, _error("route not found"))
            return
        try:
            payload = self._read_payload()
            handle, stream, include_usage = self.service.submit(payload)
        except ValueError as error:
            self._write_json(HTTPStatus.BAD_REQUEST, _error(str(error)))
            return
        except QueueFull as error:
            self._write_json(HTTPStatus.TOO_MANY_REQUESTS, _error(str(error), "overloaded"))
            return
        except WorkerClosed as error:
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _error(str(error), "service_unavailable"),
            )
            return

        if not stream:
            try:
                response = self.service.collect(handle)
            except RuntimeError as error:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _error(str(error), "engine_error"),
                )
            else:
                self._write_json(HTTPStatus.OK, response)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        token_ids: list[int] = []
        first_token_at: float | None = None
        last_token_at: float | None = None
        token_times_available = True
        emitted_text = ""
        errored = False
        try:
            for event in handle.iter_events():
                if event.error:
                    errored = True
                    chunk = _error(event.error, "engine_error")
                else:
                    if event.token_id is not None:
                        token_ids.append(event.token_id)
                        if event.emitted_at is None:
                            token_times_available = False
                        else:
                            if first_token_at is None:
                                first_token_at = event.emitted_at
                            last_token_at = event.emitted_at
                    decoded = self.service.codec.decode_tokens(token_ids)
                    if not decoded.startswith(emitted_text):
                        raise RuntimeError("tokenizer output changed text already streamed")
                    # A byte-level tokenizer may show the Unicode replacement
                    # character until later tokens finish a multibyte codepoint.
                    stable = decoded if event.finished else decoded.rstrip("\ufffd")
                    text = stable[len(emitted_text) :]
                    emitted_text = stable
                    chunk = self.service.event_payload(handle, event, text=text)
                data = f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()
            if include_usage and not errored:
                generation_time_ms = None
                mean_itl_ms = None
                if token_times_available and first_token_at is not None and last_token_at is not None:
                    generation_time_ms = (last_token_at - first_token_at) * 1000
                    if len(token_ids) > 1:
                        mean_itl_ms = generation_time_ms / (len(token_ids) - 1)
                usage_chunk = {
                    "id": handle.request_id,
                    "object": "text_completion.chunk",
                    "created": int(handle.created_at),
                    "model": self.service.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": handle.prompt_tokens,
                        "completion_tokens": len(token_ids),
                        "total_tokens": handle.prompt_tokens + len(token_ids),
                    },
                    "metrics": {
                        "generation_time_ms": generation_time_ms,
                        "mean_itl_ms": mean_itl_ms,
                    },
                }
                data = f"data: {json.dumps(usage_chunk, separators=(',', ':'))}\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except RuntimeError as error:
            chunk = _error(str(error), "engine_error")
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                self.service.worker.cancel(handle.request_id)
            except WorkerClosed:
                pass
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.service.worker.cancel(handle.request_id)
            except (WorkerClosed, TimeoutError):
                pass


def make_server(
    worker: InferenceWorker,
    codec: TokenCodec,
    *,
    model: str,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> CompletionHTTPServer:
    return CompletionHTTPServer((host, port), CompletionService(worker, codec, model=model))
