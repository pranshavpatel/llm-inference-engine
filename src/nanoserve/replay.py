"""Checksummed open-loop completion traces and comparable request records."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
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
) -> dict:
    """Create a saved text workload with arrival offsets fixed before replay."""
    if not model or not revision:
        raise ValueError("model and revision must be nonempty")
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must contain nonempty strings")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    plan = make_trace(count, rate, seed, prompt_tokens=1, output_tokens=max_tokens)
    rng = random.Random(seed)
    payload = {
        "schema_version": 1,
        "kind": "completion-request-trace",
        "model": model,
        "revision": revision,
        "seed": seed,
        "rate_rps": rate,
        "requests": [
            {
                "request_id": item["request_id"],
                "arrival_offset_s": item["arrival_offset_s"],
                "prompt": rng.choice(prompts),
                "max_tokens": max_tokens,
            }
            for item in plan["requests"]
        ],
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
    seen: set[str] = set()
    previous = -1.0
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("each trace request must be an object")
        request_id = request.get("request_id")
        offset = request.get("arrival_offset_s")
        prompt = request.get("prompt")
        max_tokens = request.get("max_tokens")
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("trace request IDs must be unique nonempty strings")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
            or offset < 0
            or offset < previous
        ):
            raise ValueError("trace arrival offsets must be finite and nondecreasing")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("trace prompts must be nonempty strings")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("trace max_tokens must be positive integers")
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

    def run(self, request: dict, trace_start: float) -> dict:
        connection_type = (
            http.client.HTTPSConnection
            if self._url.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self._url.hostname,
            self._url.port,
            timeout=self.timeout_s,
        )
        body = json.dumps(
            {
                "model": self.model,
                "prompt": request["prompt"],
                "max_tokens": request["max_tokens"],
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        sent = time.monotonic() - trace_start
        try:
            connection.request(
                "POST",
                "/v1/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
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
            for raw_line in response:
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
            return {
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
        except Exception as error:
            return {
                "request_id": request["request_id"],
                "status": "failed",
                "actual_send_offset_s": sent,
                "completed_offset_s": time.monotonic() - trace_start,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
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
        sent = time.monotonic() - trace_start
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
            output = self.model(inputs, use_cache=True)
            cache = output.past_key_values
            for step in range(request["max_tokens"]):
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                if token == eos:
                    finish_reason = "stop"
                decoded = self.tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if not decoded.startswith(emitted_text):
                    raise RuntimeError("tokenizer output changed text already emitted")
                stable = decoded if finish_reason == "stop" or step + 1 == request["max_tokens"] else decoded.split("\ufffd", 1)[0]
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
