"""Compare real Qwen2.5 execution through contiguous and physical paged KV paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nanoserve.attention import ReferencePagedAttention
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
from nanoserve.model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM, load_safetensors

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPTS = [
    "The capital of France is",
    "Write one word for the color of grass:",
    "Two plus two equals",
    "A friendly greeting is",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("the CUDA device does not support BF16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    local = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt"],
    )
    config = Qwen2Config.from_json(os.path.join(local, "config.json"))
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    coverage = load_safetensors(model, local)
    tokenizer = AutoTokenizer.from_pretrained(local)
    prompts = [tokenizer(prompt, return_tensors="pt").input_ids[0].to("cuda") for prompt in PROMPTS]

    num_blocks = 32
    blocks = BlockManager(num_blocks, block_size=args.block_size, watermark=0)
    cache = PagedKVCache(
        KVCacheSpec(
            config.num_hidden_layers,
            num_blocks,
            args.block_size,
            config.num_key_value_heads,
            config.resolved_head_dim,
        ),
        dtype=dtype,
        device="cuda",
    )
    manager = PagedKVCacheManager(blocks, cache)
    runner = PagedQwen2Runner(model, manager, ReferencePagedAttention())
    request_ids = [f"prompt-{index}" for index in range(len(prompts))]
    for request_id, prompt in zip(request_ids, prompts):
        manager.allocate(request_id, int(prompt.numel()) + args.tokens)

    with torch.inference_mode():
        references = [model(prompt.unsqueeze(0)).logits[0].float() for prompt in prompts]
        paged = runner.forward(request_ids, prompts)

    results = []
    for text, reference, actual in zip(PROMPTS, references, paged):
        difference = (actual.float() - reference).abs()
        results.append(
            {
                "prompt": text,
                "prompt_tokens": actual.shape[0],
                "prefill_max_abs_logit_error": difference.max().item(),
                "prefill_mean_abs_logit_error": difference.mean().item(),
                "greedy_tokens_equal": True,
                "first_divergence": None,
            }
        )

    histories = [prompt.clone() for prompt in prompts]
    paged_logits = list(paged)
    generated = [[] for _ in prompts]
    for step in range(args.tokens):
        next_tokens = []
        for index, (history, current) in enumerate(zip(histories, paged_logits)):
            with torch.inference_mode():
                reference_logits = model(history.unsqueeze(0)).logits[0, -1].float()
            paged_last = current[-1].float()
            reference_token = int(reference_logits.argmax())
            paged_token = int(paged_last.argmax())
            if paged_token != reference_token and results[index]["first_divergence"] is None:
                paged_top = torch.topk(paged_last, 2).values
                reference_top = torch.topk(reference_logits, 2).values
                results[index]["greedy_tokens_equal"] = False
                results[index]["first_divergence"] = {
                    "generation_step": step,
                    "paged_token": paged_token,
                    "reference_token": reference_token,
                    "paged_top_two_margin": (paged_top[0] - paged_top[1]).item(),
                    "reference_top_two_margin": (reference_top[0] - reference_top[1]).item(),
                }
            next_tokens.append(torch.tensor([paged_token], device="cuda", dtype=torch.long))
            histories[index] = torch.cat((history, next_tokens[-1]))
            generated[index].append(paged_token)
        if step + 1 < args.tokens:
            paged_logits = list(runner.forward(request_ids, next_tokens))

    for index, result in enumerate(results):
        result["generated_text"] = tokenizer.decode(generated[index])
        result["status"] = "pass" if result["greedy_tokens_equal"] else "fail"

    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "block_size": args.block_size,
        "attention_backend": runner.attention_backend.name,
        "weight_coverage": {
            "loaded_tensors": coverage.loaded_tensors,
            "checkpoint_files": coverage.checkpoint_files,
            "tied_lm_head_from_embeddings": coverage.tied_lm_head_from_embeddings,
        },
        "cache_stats": manager.stats(),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if any(result["status"] != "pass" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
