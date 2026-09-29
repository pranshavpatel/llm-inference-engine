"""Paged-attention contracts and correctness backends."""

from .backend import AttentionBackend, BackendProbe, PagedBatchMetadata
from .flashinfer import FlashInferPagedAttention, probe_flashinfer
from .reference import ReferencePagedAttention, contiguous_attention

__all__ = [
    "AttentionBackend",
    "BackendProbe",
    "FlashInferPagedAttention",
    "PagedBatchMetadata",
    "ReferencePagedAttention",
    "contiguous_attention",
    "probe_flashinfer",
]
