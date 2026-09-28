"""Locate first real-model attention difference without propagating it across layers.

For one teacher-forced decode step, compare each layer's FlashInfer output with
the gather oracle, then return the oracle output to preserve reference states.
This is a targeted correctness diagnostic, not a performance measurement.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from nanoserve.attention import AttentionBackend, FlashInferPagedAttention, ReferencePagedAttention
from nanoserve.runtime import ServingConfig, build_serving_runtime


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPT_LENGTHS = (1, 16, 17, 33)
ATOL = 0.03
RTOL = 0.03


class CompareDecode(AttentionBackend):
    name = "reference-output-with-flashinfer-observation"

    def __init__(self, query_heads: int) -> None:
        self.reference = ReferencePagedAttention()
        self.candidate = FlashInferPagedAttention(query_heads)
        self.records: list[dict] = []

    def prefill(self, query, cache, layer, metadata):
        return self.reference.prefill(query, cache, layer, metadata)

    def decode(self, query, cache, layer, metadata):
        expected = self.reference.decode(query, cache, layer, metadata)
        actual = self.candidate.decode(query, cache, layer, metadata)
        for row in range(metadata.batch_size):
            left = expected[row].float()
            right = actual[row].float()
            close = torch.isclose(left, right, atol=ATOL, rtol=RTOL)
            self.records.append({
                "layer": layer,
                "request_id": f"r{row}",
                "sequence_length": metadata.sequence_lengths[row],
                "max_absolute_error": float((left - right).abs().max().item()),
                "mean_absolute_error": float((left - right).abs().mean().item()),
                "mismatched_elements": int((~close).sum().item()),
                "elements": left.numel(),
            })
        # Do not feed candidate differences into later layers in this test.
        return expected


def run(model_dir: Path) -> dict:
    model_dir = model_dir.resolve()
    if model_dir.parent.name != "snapshots" or model_dir.name != REVISION:
        raise ValueError("model-dir must be the pinned Hugging Face snapshot")
    runtime = build_serving_runtime(ServingConfig(
        model_dir=model_dir, model_name=MODEL, device="cuda", dtype="bfloat16",
        kv_pool_mib=128, block_size=16, max_context_tokens=64,
        max_num_sequences=8, watermark=0.05,
    ))
    runner = runtime.worker.engine.model_runner
    manager = runner.cache_manager
    seed = tuple(runtime.codec.encode("The capital of France is"))
    if not seed:
        raise RuntimeError("tokenizer produced an empty seed prompt")
    prompt_ids = tuple(tuple((seed * (length // len(seed) + 1))[:length]) for length in PROMPT_LENGTHS)
    request_ids = tuple(f"r{index}" for index in range(len(prompt_ids)))
    for request_id, prompt in zip(request_ids, prompt_ids):
        manager.allocate(request_id, len(prompt) + 1)
    device = manager.cache.device
    with torch.inference_mode():
        prefill_logits = runner.forward(
            request_ids,
            [torch.tensor(prompt, dtype=torch.long, device=device) for prompt in prompt_ids],
        )
        next_tokens = [output[-1].argmax().reshape(1).to(torch.long) for output in prefill_logits]
        comparator = CompareDecode(runner.model.config.num_attention_heads)
        runner.attention_backend = comparator
        runner.forward(request_ids, next_tokens)
    for request_id in request_ids:
        manager.free(request_id)
    records = comparator.records
    return {
        "status": "passed" if all(item["mismatched_elements"] == 0 for item in records) else "failed",
        "model": MODEL,
        "revision": REVISION,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "prompt_lengths": list(PROMPT_LENGTHS),
        "layers": runner.model.config.num_hidden_layers,
        "atol": ATOL,
        "rtol": RTOL,
        "max_absolute_error": max(item["max_absolute_error"] for item in records),
        "mismatched_elements": sum(item["mismatched_elements"] for item in records),
        "first_mismatch": next((item for item in records if item["mismatched_elements"]), None),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    try:
        result = run(args.model_dir)
    except Exception as error:
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    result = {"schema_version": 1, "kind": "flashinfer-attention-layer-diagnostic", **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
