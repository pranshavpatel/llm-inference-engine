"""Record real-model logit and greedy parity without using HF generate()."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoserve.model import Qwen2Config, Qwen2ForCausalLM, load_safetensors

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROMPTS = [
    "The capital of France is",
    "Write one word for the color of grass:",
    "Two plus two equals",
    "A friendly greeting is",
]


def resolve_dtype(name):
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    dtype = resolve_dtype(args.dtype)
    local = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt"],
    )
    config = Qwen2Config.from_json(os.path.join(local, "config.json"))
    custom = Qwen2ForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    coverage = load_safetensors(custom, local)
    reference = AutoModelForCausalLM.from_pretrained(
        local, dtype=dtype, attn_implementation="eager"
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(local)
    results = []
    for prompt in PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            reference_logits = reference(input_ids).logits.float()
            custom_logits = custom(input_ids).logits.float()
        difference = (custom_logits - reference_logits).abs()
        finite_logits = bool(
            torch.isfinite(custom_logits).all() and torch.isfinite(reference_logits).all()
        )
        cache = None
        incremental_pieces = []
        with torch.inference_mode():
            for position in range(input_ids.shape[1]):
                cached = custom(
                    input_ids[:, position : position + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = cached.past_key_values
                incremental_pieces.append(cached.logits.float())
        cache_difference = (torch.cat(incremental_pieces, dim=1) - custom_logits).abs()

        reference_tokens = input_ids
        with torch.inference_mode():
            for _ in range(args.tokens):
                token = reference(reference_tokens).logits[:, -1].argmax(-1, keepdim=True)
                reference_tokens = torch.cat((reference_tokens, token), dim=1)
            custom_tokens = custom.generate(input_ids, max_new_tokens=args.tokens)
        mismatches = (custom_tokens != reference_tokens).nonzero()
        divergence = None
        if mismatches.numel():
            position = int(mismatches[0, 1])
            prefix = custom_tokens[:, :position]
            with torch.inference_mode():
                ours = custom(prefix).logits[0, -1].float()
                theirs = reference(prefix).logits[0, -1].float()
            ours_top = torch.topk(ours, 2)
            theirs_top = torch.topk(theirs, 2)
            divergence = {
                "absolute_position": position,
                "custom_token": int(custom_tokens[0, position]),
                "reference_token": int(reference_tokens[0, position]),
                "custom_top_two_margin": (ours_top.values[0] - ours_top.values[1]).item(),
                "reference_top_two_margin": (theirs_top.values[0] - theirs_top.values[1]).item(),
            }
        results.append(
            {
                "prompt": prompt,
                "shape": list(custom_logits.shape),
                "teacher_logits_finite": finite_logits,
                "max_abs_logit_error": difference.max().item() if finite_logits else None,
                "mean_abs_logit_error": difference.mean().item() if finite_logits else None,
                "rms_logit_error": difference.square().mean().sqrt().item() if finite_logits else None,
                "cached_vs_full_max_abs_error": cache_difference.max().item() if finite_logits else None,
                "cached_vs_full_mean_abs_error": cache_difference.mean().item() if finite_logits else None,
                "greedy_tokens_equal": finite_logits and divergence is None,
                "first_divergence": divergence,
                "generated_text": tokenizer.decode(custom_tokens[0, input_ids.shape[1] :]),
                "status": "pass" if finite_logits and divergence is None else "fail",
            }
        )
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "weight_coverage": {
            "loaded_tensors": coverage.loaded_tensors,
            "checkpoint_files": coverage.checkpoint_files,
            "tied_lm_head_from_embeddings": coverage.tied_lm_head_from_embeddings,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
