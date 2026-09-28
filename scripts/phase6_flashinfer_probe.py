"""Check FlashInfer paged decode against nanoserve's BF16 gather oracle on CUDA.

This is a compatibility and correctness gate, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from nanoserve.attention import PagedBatchMetadata, ReferencePagedAttention
from nanoserve.attention.flashinfer import flashinfer_page_tensors
from nanoserve.memory import KVCacheSpec, PagedKVCache


ATOL = 0.03
RTOL = 0.03


def probe() -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    import flashinfer

    torch.manual_seed(20260928)
    lengths = (1, 16, 17, 33)
    page_size = 16
    tables = ((7, 9), (4, 8), (6, 1, 5), (3, 10, 0, 11))
    spec = KVCacheSpec(1, 12, page_size, 2, 128)
    cache = PagedKVCache(spec, dtype=torch.bfloat16, device="cuda")
    for length, table in zip(lengths, tables):
        key = torch.randn(2, length, 128, dtype=cache.dtype, device=cache.device)
        value = torch.randn_like(key)
        cache.write(0, table, 0, key, value)

    metadata = PagedBatchMetadata.from_lists(tables, lengths, (1,) * len(lengths))
    query = torch.randn(len(lengths), 12, 1, 128, dtype=cache.dtype, device=cache.device)
    expected = ReferencePagedAttention().decode(query, cache, 0, metadata)[:, :, 0, :]
    indptr, indices, last_page_len = flashinfer_page_tensors(metadata, cache)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=cache.device)
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        indptr, indices, last_page_len,
        num_qo_heads=12, num_kv_heads=2, head_dim=128, page_size=page_size,
        pos_encoding_mode="NONE", q_data_type=cache.dtype, kv_data_type=cache.dtype,
    )
    actual = wrapper.run(query[:, :, 0, :].contiguous(), (cache.keys[0], cache.values[0]))
    torch.cuda.synchronize()
    absolute_error = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual.float(), expected.float(), atol=ATOL, rtol=RTOL)
    return {
        "status": "passed" if bool(close.all()) else "failed",
        "torch": torch.__version__,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "page_size": page_size,
        "sequence_lengths": list(lengths),
        "query_heads": 12,
        "kv_heads": 2,
        "head_dim": 128,
        "atol": ATOL,
        "rtol": RTOL,
        "max_absolute_error": float(absolute_error.max().item()),
        "mismatched_elements": int((~close).sum().item()),
        "elements": actual.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"probe output already exists: {args.output}")
    try:
        result = probe()
    except Exception as error:
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "torch": torch.__version__,
        }
    result = {"schema_version": 1, "kind": "flashinfer-paged-decode-probe", **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
