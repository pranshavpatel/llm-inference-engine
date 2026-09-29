"""Opt-in FlashInfer paged decode with the gather backend as prefill oracle."""

from __future__ import annotations

import importlib.util
import platform

import torch

from nanoserve.attention.backend import AttentionBackend, BackendProbe, PagedBatchMetadata
from nanoserve.attention.reference import ReferencePagedAttention
from nanoserve.memory import PagedKVCache


def probe_flashinfer() -> BackendProbe:
    if platform.system() != "Linux":
        return BackendProbe(
            name="flashinfer",
            available=False,
            version=None,
            reason="FlashInfer publishes Linux-only runtime wheels",
        )
    if importlib.util.find_spec("flashinfer") is None:
        return BackendProbe(
            name="flashinfer",
            available=False,
            version=None,
            reason="flashinfer-python is not installed",
        )
    import flashinfer

    return BackendProbe(
        name="flashinfer",
        available=True,
        version=getattr(flashinfer, "__version__", "unknown"),
        reason="package import succeeded; kernel smoke test still required",
    )


def flashinfer_page_tensors(
    metadata: PagedBatchMetadata, cache: PagedKVCache
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate active page tables to FlashInfer's NHD paged-KV metadata.

    Reserved but unwritten tail pages must not be included in ``indices``.
    """
    metadata.validate(cache)
    indptr = [0]
    indices: list[int] = []
    last_page_len: list[int] = []
    page_size = cache.spec.block_size
    for table, length in zip(metadata.block_tables, metadata.sequence_lengths):
        active_pages = (length + page_size - 1) // page_size
        indices.extend(table[:active_pages])
        indptr.append(len(indices))
        last_page_len.append((length - 1) % page_size + 1)
    return tuple(
        torch.tensor(values, dtype=torch.int32, device=cache.device)
        for values in (indptr, indices, last_page_len)
    )


class FlashInferPagedAttention(AttentionBackend):
    """Use a planned NHD paged kernel for decode; retain reference prefill.

    A metadata object is shared across model layers in one runner forward, so
    the wrapper is planned once per decode step and reused for every layer.
    """

    name = "flashinfer-paged-decode-reference-prefill"

    def __init__(self, num_query_heads: int) -> None:
        if num_query_heads <= 0:
            raise ValueError("num_query_heads must be positive")
        self.num_query_heads = num_query_heads
        self.reference = ReferencePagedAttention()
        self._wrapper = None
        self._workspace = None
        self._planned_metadata: PagedBatchMetadata | None = None
        self._planned_cache: PagedKVCache | None = None

    def prefill(self, query, cache, layer, metadata):
        return self.reference.prefill(query, cache, layer, metadata)

    def decode(self, query, cache, layer, metadata):
        metadata.validate(cache)
        self.reference._validate_query(query, cache, metadata)
        if any(length != 1 for length in metadata.query_lengths):
            raise ValueError("decode requires exactly one query token per sequence")
        if query.shape[2] != 1 or query.shape[1] != self.num_query_heads:
            raise ValueError("decode query shape does not match the configured heads")
        if cache.device.type != "cuda" or cache.dtype != torch.bfloat16:
            raise ValueError("FlashInfer decode requires a CUDA BF16 KV cache")
        if cache.spec.block_size != 16 or cache.spec.head_dim != 128:
            raise ValueError("FlashInfer decode has been gated only for 16-token, 128-dim pages")
        if self.num_query_heads != 12 or cache.spec.num_key_value_heads != 2:
            raise ValueError("FlashInfer decode has been gated only for 12 query / 2 KV heads")

        if self._wrapper is None:
            import flashinfer

            # FlashInfer requires the workspace initialized to zero on first use.
            self._workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=cache.device)
            self._wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace, "NHD"
            )
        if self._planned_metadata is not metadata or self._planned_cache is not cache:
            indptr, indices, last_page_len = flashinfer_page_tensors(metadata, cache)
            self._wrapper.plan(
                indptr, indices, last_page_len,
                num_qo_heads=self.num_query_heads,
                num_kv_heads=cache.spec.num_key_value_heads,
                head_dim=cache.spec.head_dim,
                page_size=cache.spec.block_size,
                pos_encoding_mode="NONE",
                q_data_type=cache.dtype,
                kv_data_type=cache.dtype,
            )
            self._planned_metadata = metadata
            self._planned_cache = cache
        result = self._wrapper.run(
            query[:, :, 0, :].contiguous(),
            (cache.keys[layer], cache.values[layer]),
        )
        if result.shape != (metadata.batch_size, self.num_query_heads, cache.spec.head_dim):
            raise RuntimeError("FlashInfer decode returned an unexpected output shape")
        if result.dtype != cache.dtype or result.device != cache.device:
            raise RuntimeError("FlashInfer decode returned an unexpected dtype or device")
        return result.unsqueeze(2)
