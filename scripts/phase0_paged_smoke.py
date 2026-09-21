"""CUDA smoke test for a portable gather-based paged-attention fallback."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch
from torch.nn import functional as F


def attend(query, key, value, query_start):
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    query_positions = torch.arange(
        query_start, query_start + query.shape[2], device=query.device
    )
    key_positions = torch.arange(key.shape[2], device=query.device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def one_case(page_size, dtype, num_query_heads=12, num_kv_heads=2, head_dim=128):
    length = 33
    pages = math.ceil(length / page_size)
    generator = torch.Generator(device="cuda").manual_seed(20260921 + page_size)
    key = torch.randn((1, num_kv_heads, length, head_dim), device="cuda", dtype=dtype, generator=generator)
    value = torch.randn(key.shape, device="cuda", dtype=dtype, generator=generator)
    page_table = torch.randperm(pages + 3, device="cuda", generator=generator)[:pages]
    key_pool = torch.zeros((pages + 3, page_size, num_kv_heads, head_dim), device="cuda", dtype=dtype)
    value_pool = torch.zeros_like(key_pool)
    for logical, physical in enumerate(page_table.tolist()):
        start = logical * page_size
        stop = min(start + page_size, length)
        key_pool[physical, : stop - start].copy_(key[0, :, start:stop].transpose(0, 1))
        value_pool[physical, : stop - start].copy_(value[0, :, start:stop].transpose(0, 1))
    gathered_key = key_pool[page_table].reshape(-1, num_kv_heads, head_dim)[:length].transpose(0, 1).unsqueeze(0)
    gathered_value = value_pool[page_table].reshape(-1, num_kv_heads, head_dim)[:length].transpose(0, 1).unsqueeze(0)

    prefill_query = torch.randn(
        (1, num_query_heads, length, head_dim), device="cuda", dtype=dtype, generator=generator
    )
    decode_query = torch.randn(
        (1, num_query_heads, 1, head_dim), device="cuda", dtype=dtype, generator=generator
    )
    expected_prefill = attend(prefill_query, key, value, 0)
    actual_prefill = attend(prefill_query, gathered_key, gathered_value, 0)
    expected_decode = attend(decode_query, key, value, length - 1)
    actual_decode = attend(decode_query, gathered_key, gathered_value, length - 1)
    return {
        "page_size": page_size,
        "dtype": str(dtype).removeprefix("torch."),
        "query_heads": num_query_heads,
        "kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "prefill_max_abs_error": (actual_prefill - expected_prefill).abs().max().item(),
        "decode_max_abs_error": (actual_decode - expected_decode).abs().max().item(),
        "status": "pass",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for page_size in (1, 16, 32, 64):
            try:
                cases.append(one_case(page_size, dtype))
            except Exception as error:  # preserve the exact compatibility failure
                cases.append(
                    {
                        "page_size": page_size,
                        "dtype": str(dtype).removeprefix("torch."),
                        "status": "fail",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    report = {
        "schema_version": 1,
        "backend": "pytorch-sdpa-gather-reference",
        "optimized": False,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if any(case["status"] != "pass" for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
