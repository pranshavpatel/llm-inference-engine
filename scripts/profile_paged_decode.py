"""Profile one fixed nanoserve prefill/decode batch on the pinned L40S model.

This is an operator-level diagnostic, not an HTTP benchmark or throughput claim.
It deliberately excludes tokenization, admission, networking, and client delay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from nanoserve.runtime import ServingConfig, build_serving_runtime
from nanoserve.types import FinishReason


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPTS = ("The capital of France is", "Two plus two equals")


def run_batch(engine, prompt_ids: tuple[tuple[int, ...], ...], *, tokens: int,
              prefix: str, profiler=None) -> dict:
    """Drive a fixed active batch, labelling the first step as prefill."""
    if not prompt_ids or tokens <= 0:
        raise ValueError("a nonempty batch and positive token count are required")
    request_ids = tuple(f"{prefix}-{index}" for index in range(len(prompt_ids)))
    for request_id, prompt in zip(request_ids, prompt_ids):
        engine.add_request(request_id, prompt, tokens, eos_token_id=None)
    step_wall_ms: list[float] = []
    for index in range(tokens):
        phase = "prefill" if index == 0 else "decode"
        if engine.scheduler.cache_manager.cache.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with record_function(f"nanoserve.{phase}_step"):
            events = engine.step()
            if engine.scheduler.cache_manager.cache.device.type == "cuda":
                torch.cuda.synchronize()
        step_wall_ms.append((time.perf_counter() - started) * 1000)
        if len(events) != len(request_ids) or {event.request_id for event in events} != set(request_ids):
            raise RuntimeError("profile batch did not progress every request in one step")
        if any(event.finished != (index == tokens - 1) for event in events):
            raise RuntimeError("profile batch finished at an unexpected step")
        if profiler is not None:
            profiler.step()
    for request_id in request_ids:
        snapshot = engine.scheduler.snapshot(request_id)
        if len(snapshot.generated_token_ids) != tokens or snapshot.finish_reason != FinishReason.LENGTH:
            raise RuntimeError("profile batch did not finish with the fixed-output policy")
    return {
        "requests": len(request_ids),
        "tokens_per_request": tokens,
        "prefill_step_wall_ms": step_wall_ms[0],
        "decode_step_wall_ms": step_wall_ms[1:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    if model_dir.parent.name != "snapshots" or model_dir.name != REVISION:
        raise ValueError("model-dir must be the pinned Hugging Face snapshot")
    if not 1 <= args.batch_size <= 8:
        raise ValueError("batch-size must be in [1, 8] for the 64-token batch budget")
    if not 1 <= args.tokens <= 16:
        raise ValueError("tokens must be in [1, 16]")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    if args.output_dir.exists():
        raise FileExistsError(f"profile output already exists: {args.output_dir}")

    runtime = build_serving_runtime(ServingConfig(
        model_dir=model_dir, model_name=MODEL, device="cuda", dtype="bfloat16",
        kv_pool_mib=128, block_size=16, max_context_tokens=64,
        max_num_sequences=16, watermark=0.05,
    ))
    prompt_ids = tuple(
        tuple(runtime.codec.encode(PROMPTS[index % len(PROMPTS)]))
        for index in range(args.batch_size)
    )
    if sum(map(len, prompt_ids)) > 64 or max(map(len, prompt_ids)) + args.tokens > 64:
        raise ValueError("the prompt batch exceeds the pilot's token/context budget")
    engine = runtime.worker.engine
    run_batch(engine, prompt_ids, tokens=args.tokens, prefix="warmup")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        measured = run_batch(engine, prompt_ids, tokens=args.tokens,
                             prefix="profile", profiler=profiler)

    operations = profiler.key_averages()
    ranked = sorted(operations, key=lambda item: item.self_device_time_total, reverse=True)
    result = {
        "schema_version": 1,
        "kind": "nanoserve-paged-operator-profile",
        "performance_claim": False,
        "scope": "single-process fixed batch; excludes HTTP, tokenization, ingress, and client delay",
        "model": MODEL,
        "revision": REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "attention_backend": "reference_paged_gather",
        "kv_pool_bytes": runtime.kv_pool_bytes,
        "batch_size": args.batch_size,
        "prompt_tokens_per_request": list(map(len, prompt_ids)),
        "measured": measured,
        "top_device_self_time_ops": [
            {
                "name": item.key,
                "calls": item.count,
                "self_device_time_us": item.self_device_time_total,
                "self_cpu_time_us": item.self_cpu_time_total,
            }
            for item in ranked[:30]
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    profiler.export_chrome_trace(str(args.output_dir / "torch-trace.json"))
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "batch_size": args.batch_size,
        "prefill_step_wall_ms": measured["prefill_step_wall_ms"],
        "decode_step_wall_ms": measured["decode_step_wall_ms"],
        "top_device_op": result["top_device_self_time_ops"][0],
    }, indent=2))


if __name__ == "__main__":
    main()
