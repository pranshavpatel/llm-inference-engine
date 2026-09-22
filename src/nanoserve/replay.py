"""Checksummed open-loop completion traces and comparable request records."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import queue
import random
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit

from nanoserve.bench import make_trace


def _checksum(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_completion_trace(
    *,
    count: int,
    rate: float,
    seed: int,
    model: str,
    revision: str,
    prompts: list[str],
    max_tokens: int,
    ignore_eos: bool = False,
) -> dict:
    """Create a saved text workload with arrival offsets fixed before replay."""
    if not model or not revision:
        raise ValueError("model and revision must be nonempty")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must contain nonempty strings")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(ignore_eos, bool):
        raise ValueError("ignore_eos must be a boolean")
    plan = make_trace(count, rate, seed, prompt_tokens=1, output_tokens=max_tokens)
    rng = random.Random(seed)
    payload = {
        "schema_version": 1,
        "kind": "completion-request-trace",
        "model": model,
        "revision": revision,
        "seed": seed,
        "rate_rps": float(rate),
        "requests": [
            {
                "request_id": item["request_id"],
                "arrival_offset_s": item["arrival_offset_s"],
                "prompt": rng.choice(prompts),
                "max_tokens": max_tokens,
                **({"ignore_eos": True} if ignore_eos else {}),
            }
            for item in plan["requests"]
        ],
    }
    return {**payload, "sha256": _checksum(payload)}


def make_duration_completion_trace(
    *,
    duration_s: float,
    rate: float,
    seed: int,
    model: str,
    revision: str,
    prompts: list[str],
    max_tokens: int,
    ignore_eos: bool = False,
) -> dict:
    """Save every Poisson arrival in a fixed offered-load interval.

    This plans arrivals only. Scored replays must opt into bounded drain
    and retain unfinished requests in their result records.
    """
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not model or not revision or not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("model, revision, and prompts must be nonempty")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(ignore_eos, bool):
        raise ValueError("ignore_eos must be a boolean")
    arrival_rng = random.Random(seed)
    prompt_rng = random.Random(seed)
    arrival = 0.0
    requests = []
    while True:
        arrival += arrival_rng.expovariate(rate)
        if arrival > duration_s:
            break
        requests.append({
            "request_id": f"r{len(requests):06d}",
            "arrival_offset_s": arrival,
            "prompt": prompt_rng.choice(prompts),
            "max_tokens": max_tokens,
            **({"ignore_eos": True} if ignore_eos else {}),
        })
    if not requests:
        raise ValueError("seeded interval has no arrivals; increase rate or duration_s")
    payload = {
        "schema_version": 1,
        "kind": "completion-request-trace",
        "model": model,
        "revision": revision,
        "seed": seed,
        "rate_rps": float(rate),
        "offered_interval_s": float(duration_s),
        "arrival_process": "poisson",
        "requests": requests,
    }
    return {**payload, "sha256": _checksum(payload)}


def validate_completion_trace(trace: dict) -> None:
    if not isinstance(trace, dict):
        raise ValueError("trace must be a JSON object")
    if trace.get("schema_version") != 1 or trace.get("kind") != "completion-request-trace":
        raise ValueError("unsupported completion trace schema")
    if not isinstance(trace.get("model"), str) or not trace["model"]:
        raise ValueError("trace model must be nonempty")
    if not isinstance(trace.get("revision"), str) or not trace["revision"]:
        raise ValueError("trace revision must be nonempty")
    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("trace requests must be a nonempty list")
    offered_interval = trace.get("offered_interval_s")
    if offered_interval is not None and (
        isinstance(offered_interval, bool)
        or not isinstance(offered_interval, (int, float))
        or not math.isfinite(offered_interval)
        or offered_interval <= 0
    ):
        raise ValueError("offered_interval_s must be finite and positive")
    seen: set[str] = set()
    previous = -1.0
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("each trace request must be an object")
        request_id = request.get("request_id")
        offset = request.get("arrival_offset_s")
        prompt = request.get("prompt")
        max_tokens = request.get("max_tokens")
        ignore_eos = request.get("ignore_eos", False)
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("trace request IDs must be unique nonempty strings")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or offset < 0
            or offset < previous
            or (offered_interval is not None and offset > offered_interval)
        ):
            raise ValueError("trace arrival offsets must be finite and nondecreasing")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("trace prompts must be nonempty strings")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("trace max_tokens must be positive integers")
        if not isinstance(ignore_eos, bool):
            raise ValueError("trace ignore_eos must be a boolean")
        seen.add(request_id)
        previous = offset
    payload = {key: value for key, value in trace.items() if key != "sha256"}
    if trace.get("sha256") != _checksum(payload):
        raise ValueError("trace checksum mismatch")


class CompletionAdapter(Protocol):
    name: str

    def run(self, request: dict, trace_start: float) -> dict: ...


class HTTPCompletionsAdapter:
    """Replay against nanoserve or vLLM's `/v1/completions` stream."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout_s: float = 60,
        name: str = "http-completions",
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("endpoint must be an HTTP or HTTPS URL")
        if parsed.path != "/v1/completions" or parsed.query or parsed.fragment:
            raise ValueError("endpoint path must be /v1/completions")
        if not model:
            raise ValueError("model must be nonempty")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.name = name
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self._url = parsed
        self._active_sockets: set[socket.socket] = set()
        self._active_lock = threading.Lock()

    def abort_active(self) -> None:
        """Interrupt in-flight HTTP reads after a bounded replay deadline."""
        with self._active_lock:
            sockets = tuple(self._active_sockets)
        for active in sockets:
            try:
                active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active.close()
            except OSError:
                pass

    def run(
        self,
        request: dict,
        trace_start: float,
        *,
        deadline_monotonic: float | None = None,
        on_sent: Callable[[float], None] | None = None,
    ) -> dict:
        def remaining_timeout() -> float:
            if deadline_monotonic is None:
                return self.timeout_s
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("bounded replay drain deadline exceeded")
            return min(self.timeout_s, remaining)

        connection_type = (
            http.client.HTTPSConnection
            if self._url.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self._url.hostname,
            self._url.port,
            timeout=remaining_timeout(),
        )
        payload = {
            "model": self.model,
            "prompt": request["prompt"],
            "max_tokens": request["max_tokens"],
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.get("ignore_eos", False):
            payload["ignore_eos"] = True
        body = json.dumps(payload)
        sent = None
        active_socket = None
        try:
            connection.connect()
            active_socket = connection.sock
            with self._active_lock:
                self._active_sockets.add(active_socket)
            active_socket.settimeout(remaining_timeout())
            sent = time.monotonic() - trace_start
            if on_sent is not None:
                on_sent(sent)
            connection.request(
                "POST",
                "/v1/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            active_socket.settimeout(remaining_timeout())
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read(2048).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status}: {detail}")
            chunks = []
            text_parts = []
            first_content = None
            finish_reason = None
            usage = None
            server_request_id = None
            done = False
            while True:
                active_socket.settimeout(remaining_timeout())
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                item = json.loads(data)
                if "error" in item:
                    raise RuntimeError(f"stream error: {item['error']}")
                server_request_id = item.get("id", server_request_id)
                if item.get("usage") is not None:
                    usage = item["usage"]
                for choice in item.get("choices", ()):
                    piece = choice.get("text") or ""
                    at = time.monotonic() - trace_start
                    if piece:
                        text_parts.append(piece)
                        if first_content is None:
                            first_content = at
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    chunks.append(
                        {"at_offset_s": at, "text": piece, "finish_reason": choice.get("finish_reason")}
                    )
            if not done or finish_reason is None:
                raise RuntimeError("completion stream ended without [DONE] and finish reason")
            record = {
                "request_id": request["request_id"],
                "status": "completed",
                "actual_send_offset_s": sent,
                "first_content_offset_s": first_content,
                "completed_offset_s": time.monotonic() - trace_start,
                "server_request_id": server_request_id,
                "output_text": "".join(text_parts),
                "finish_reason": finish_reason,
                "usage": usage,
                "chunks": chunks,
            }
            if request.get("ignore_eos", False):
                observed = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if observed != request["max_tokens"] or finish_reason != "length":
                    record["status"] = "failed"
                    record["error"] = (
                        "fixed-output policy mismatch: expected "
                        f"{request['max_tokens']} tokens and length finish, "
                        f"got {observed} tokens and {finish_reason!r}"
                    )
            return record
        except Exception as error:
            at = time.monotonic()
            drain_timeout = (
                deadline_monotonic is not None
                and isinstance(error, TimeoutError)
                and at >= deadline_monotonic - 0.01
            )
            return {
                "request_id": request["request_id"],
                "status": "timed_out" if drain_timeout else "failed",
                "actual_send_offset_s": sent,
                "completed_offset_s": at - trace_start,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            if active_socket is not None:
                with self._active_lock:
                    self._active_sockets.discard(active_socket)
            connection.close()


class HuggingFaceAdapter:
    """Single-model greedy baseline with serialized request execution."""

    def __init__(self, model_dir: str | Path, *, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU Hugging Face replay requires float32")
        self.name = "huggingface-eager"
        self.device = torch.device(device)
        self._torch = torch
        local = str(Path(model_dir).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local,
            local_files_only=True,
            dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype],
            attn_implementation="eager",
        ).to(self.device).eval()
        self._lock = threading.Lock()

    def run(self, request: dict, trace_start: float) -> dict:
        torch = self._torch
        token_ids = self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        if not token_ids:
            raise ValueError("prompt encoded to zero tokens")
        generated: list[int] = []
        chunks = []
        first_content = None
        emitted_text = ""
        finish_reason = "length"
        eos = self.tokenizer.eos_token_id
        with self._lock, torch.inference_mode():
            inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            sent = time.monotonic() - trace_start
            output = self.model(inputs, use_cache=True)
            cache = output.past_key_values
            for step in range(request["max_tokens"]):
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                if token == eos and not request.get("ignore_eos", False):
                    finish_reason = "stop"
                decoded = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not decoded.startswith(emitted_text):
                    raise RuntimeError("tokenizer output changed text already emitted")
                stable = decoded if finish_reason == "stop" or step + 1 == request["max_tokens"] else decoded.rstrip("\ufffd")
                piece = stable[len(emitted_text) :]
                emitted_text = stable
                at = time.monotonic() - trace_start
                if piece and first_content is None:
                    first_content = at
                chunks.append({"at_offset_s": at, "text": piece, "finish_reason": None})
                if finish_reason == "stop" or step + 1 == request["max_tokens"]:
                    chunks[-1]["finish_reason"] = finish_reason
                    break
                next_input = torch.tensor([[token]], dtype=torch.long, device=self.device)
                output = self.model(next_input, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
        return {
            "request_id": request["request_id"],
            "status": "completed",
            "actual_send_offset_s": sent,
            "first_content_offset_s": first_content,
            "completed_offset_s": time.monotonic() - trace_start,
            "server_request_id": None,
            "output_text": emitted_text,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": len(token_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(token_ids) + len(generated),
            },
            "chunks": chunks,
        }


def replay_completion_trace(
    trace: dict,
    adapter: CompletionAdapter,
    *,
    max_workers: int = 32,
) -> dict:
    validate_completion_trace(trace)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()

    def run_one(request: dict) -> dict:
        dispatched = time.monotonic() - start
        try:
            return adapter.run(request, start)
        except Exception as error:
            return {
                "request_id": request["request_id"],
                "status": "failed",
                "dispatch_offset_s": dispatched,
                "completed_offset_s": time.monotonic() - start,
                "error": f"{type(error).__name__}: {error}",
            }

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for request in trace["requests"]:
            delay = start + request["arrival_offset_s"] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(run_one, request))
        records = []
        for request, future in zip(trace["requests"], futures):
            record = future.result()
            record["intended_arrival_offset_s"] = request["arrival_offset_s"]
            if record.get("actual_send_offset_s") is not None:
                record["send_lag_s"] = (
                    record["actual_send_offset_s"] - request["arrival_offset_s"]
                )
            records.append(record)
    completed = sum(record["status"] == "completed" for record in records)
    return {
        "schema_version": 1,
        "kind": "completion-replay",
        "trace_sha256": trace["sha256"],
        "model": trace["model"],
        "revision": trace["revision"],
        "adapter": adapter.name,
        "started_at_utc": started_at,
        "elapsed_s": time.monotonic() - start,
        "summary": {
            "requests": len(records),
            "completed": completed,
            "failed": len(records) - completed,
            "missing_usage": sum(
                record["status"] == "completed" and record.get("usage") is None
                for record in records
            ),
        },
        "records": records,
    }


def replay_bounded_http_trace(
    trace: dict,
    adapter: HTTPCompletionsAdapter,
    *,
    drain_s: float,
    max_workers: int = 32,
) -> dict:
    """Offer a fixed-window trace and retain queued/in-flight work at cutoff.

    This mode is HTTP-only. Active sockets are interrupted at the drain
    deadline; it refuses to return a report if worker threads do not stop.
    """
    validate_completion_trace(trace)
    offered_s = trace.get("offered_interval_s")
    if offered_s is None:
        raise ValueError("bounded replay requires a fixed offered_interval_s trace")
    if not isinstance(adapter, HTTPCompletionsAdapter):
        raise ValueError("bounded replay requires an HTTP completions adapter")
    if isinstance(drain_s, bool) or not isinstance(drain_s, (int, float)) or not math.isfinite(drain_s) or drain_s < 0:
        raise ValueError("drain_s must be finite and nonnegative")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    deadline = start + offered_s + drain_s
    pending: queue.Queue[dict] = queue.Queue()
    stopped = threading.Event()
    lock = threading.Lock()
    started: dict[str, float] = {}
    sent_offsets: dict[str, float] = {}
    results: dict[str, dict] = {}

    def consume() -> None:
        while not stopped.is_set():
            try:
                request = pending.get(timeout=0.02)
            except queue.Empty:
                continue
            if stopped.is_set() or time.monotonic() >= deadline:
                return
            request_id = request["request_id"]
            with lock:
                started[request_id] = time.monotonic() - start
            def record_send(offset: float) -> None:
                with lock:
                    sent_offsets[request_id] = offset

            try:
                result = adapter.run(
                    request, start, deadline_monotonic=deadline, on_sent=record_send
                )
            except Exception as error:
                result = {
                    "request_id": request_id,
                    "status": "failed",
                    "completed_offset_s": time.monotonic() - start,
                    "error": f"{type(error).__name__}: {error}",
                }
            with lock:
                if time.monotonic() <= deadline:
                    results[request_id] = result

    threads = [
        threading.Thread(target=consume, name=f"bounded-replay-{index}", daemon=True)
        for index in range(min(max_workers, len(trace["requests"])))
    ]
    for thread in threads:
        thread.start()
    try:
        for request in trace["requests"]:
            delay = start + request["arrival_offset_s"] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            pending.put_nowait(request)
        while time.monotonic() < deadline:
            with lock:
                finished = len(results) == len(trace["requests"])
            if finished and time.monotonic() >= start + offered_s:
                break
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
    finally:
        stopped.set()
        adapter.abort_active()
        shutdown_deadline = time.monotonic() + 2.0
        for thread in threads:
            thread.join(max(0, shutdown_deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("bounded replay could not stop all HTTP workers")

    cutoff_offset = offered_s + drain_s
    records = []
    for request in trace["requests"]:
        request_id = request["request_id"]
        with lock:
            record = results.get(request_id)
            dispatch = started.get(request_id)
            sent = sent_offsets.get(request_id)
        if record is None:
            status = "timed_out" if dispatch is not None else "not_sent"
            record = {
                "request_id": request_id,
                "status": status,
                "dispatch_offset_s": dispatch,
                "actual_send_offset_s": sent,
                "completed_offset_s": cutoff_offset,
                "error": "drain deadline exceeded" if dispatch is not None else "worker capacity exhausted before drain deadline",
            }
        record["intended_arrival_offset_s"] = request["arrival_offset_s"]
        if record.get("actual_send_offset_s") is not None:
            record["send_lag_s"] = record["actual_send_offset_s"] - request["arrival_offset_s"]
        records.append(record)
    completed = sum(record["status"] == "completed" for record in records)
    return {
        "schema_version": 1,
        "kind": "completion-replay",
        "mode": "fixed-window-bounded-drain",
        "trace_sha256": trace["sha256"],
        "model": trace["model"],
        "revision": trace["revision"],
        "adapter": adapter.name,
        "started_at_utc": started_at,
        "offered_interval_s": offered_s,
        "drain_s": drain_s,
        "drain_deadline_offset_s": cutoff_offset,
        "elapsed_s": time.monotonic() - start,
        "summary": {
            "requests": len(records),
            "completed": completed,
            "failed": len(records) - completed,
            "timed_out": sum(record["status"] == "timed_out" for record in records),
            "not_sent": sum(record["status"] == "not_sent" for record in records),
            "missing_usage": sum(record["status"] == "completed" and record.get("usage") is None for record in records),
        },
        "records": records,
    }
