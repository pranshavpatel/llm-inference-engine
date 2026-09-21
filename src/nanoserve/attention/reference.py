"""Slow gather-based paged attention used as the correctness oracle."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from nanoserve.attention.backend import AttentionBackend, PagedBatchMetadata
from nanoserve.memory import PagedKVCache


def contiguous_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_start: int,
    softmax_scale: float | None = None,
) -> Tensor:
    """Explicit causal GQA for one sequence.

    Shapes are query ``[query_heads, query_tokens, head_dim]`` and K/V
    ``[kv_heads, sequence_tokens, head_dim]``.
    """
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("query, key, and value must be rank-three attention tensors")
    if query.shape[2] != key.shape[2]:
        raise ValueError("query and key head dimensions differ")
    if query.shape[0] % key.shape[0]:
        raise ValueError("query heads must be divisible by KV heads")
    if query_start < 0 or query_start + query.shape[1] > key.shape[1]:
        raise ValueError("query positions must lie within the KV sequence")
    groups = query.shape[0] // key.shape[0]
    expanded_key = key.repeat_interleave(groups, dim=0)
    expanded_value = value.repeat_interleave(groups, dim=0)
    scale = softmax_scale if softmax_scale is not None else query.shape[2] ** -0.5
    scores = torch.matmul(query, expanded_key.transpose(1, 2)) * scale
    query_positions = torch.arange(
        query_start, query_start + query.shape[1], device=query.device
    )
    key_positions = torch.arange(key.shape[1], device=query.device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
    probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probabilities, expanded_value)


class ReferencePagedAttention(AttentionBackend):
    name = "pytorch-gather-reference"

    def __init__(self, softmax_scale: float | None = None) -> None:
        if softmax_scale is not None and softmax_scale <= 0:
            raise ValueError("softmax_scale must be positive")
        self.softmax_scale = softmax_scale

    @staticmethod
    def _validate_query(query: Tensor, cache: PagedKVCache, metadata: PagedBatchMetadata) -> None:
        if query.ndim != 4:
            raise ValueError("query must have shape [batch, query_heads, query_tokens, head_dim]")
        if query.shape[0] != metadata.batch_size:
            raise ValueError("query batch does not match metadata")
        if query.shape[2] < max(metadata.query_lengths):
            raise ValueError("query tensor is shorter than a metadata query length")
        if query.shape[3] != cache.spec.head_dim:
            raise ValueError("query head dimension does not match the KV cache")
        if query.shape[1] % cache.spec.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if query.device != cache.device or query.dtype != cache.dtype:
            raise ValueError("query device and dtype must match the KV cache")

    def prefill(
        self,
        query: Tensor,
        cache: PagedKVCache,
        layer: int,
        metadata: PagedBatchMetadata,
    ) -> Tensor:
        metadata.validate(cache)
        self._validate_query(query, cache, metadata)
        output = torch.zeros_like(query)
        for batch_index, (table, sequence_length, query_length) in enumerate(
            zip(metadata.block_tables, metadata.sequence_lengths, metadata.query_lengths)
        ):
            key, value = cache.gather(layer, table, sequence_length)
            output[batch_index, :, :query_length] = contiguous_attention(
                query[batch_index, :, :query_length],
                key,
                value,
                query_start=sequence_length - query_length,
                softmax_scale=self.softmax_scale,
            )
        return output

    def decode(
        self,
        query: Tensor,
        cache: PagedKVCache,
        layer: int,
        metadata: PagedBatchMetadata,
    ) -> Tensor:
        if any(length != 1 for length in metadata.query_lengths):
            raise ValueError("decode requires exactly one query token per sequence")
        return self.prefill(query, cache, layer, metadata)
