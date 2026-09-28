"""Availability probe for the deferred optimized FlashInfer backend."""

from __future__ import annotations

import importlib.util
import platform

import torch

from nanoserve.attention.backend import BackendProbe, PagedBatchMetadata
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
