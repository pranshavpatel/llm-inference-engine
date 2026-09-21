"""Backend-neutral metadata and paged-attention interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from torch import Tensor

from nanoserve.memory import PagedKVCache


@dataclass(frozen=True)
class BackendProbe:
    name: str
    available: bool
    version: str | None
    reason: str


@dataclass(frozen=True)
class PagedBatchMetadata:
    """Generic page tables for a static attention batch.

    Sequence lengths include the K/V entries for the current queries. Query
    tokens therefore occupy ``[sequence_length - query_length, sequence_length)``.
    """

    block_tables: tuple[tuple[int, ...], ...]
    sequence_lengths: tuple[int, ...]
    query_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        batch = len(self.block_tables)
        if batch == 0:
            raise ValueError("paged batch must contain at least one sequence")
        if len(self.sequence_lengths) != batch or len(self.query_lengths) != batch:
            raise ValueError("metadata fields must have identical batch length")
        for table, sequence_length, query_length in zip(
            self.block_tables, self.sequence_lengths, self.query_lengths
        ):
            if not table:
                raise ValueError("every sequence needs a nonempty block table")
            if sequence_length <= 0 or query_length <= 0:
                raise ValueError("sequence and query lengths must be positive")
            if query_length > sequence_length:
                raise ValueError("query length cannot exceed sequence length")

    @classmethod
    def from_lists(
        cls,
        block_tables: Sequence[Sequence[int]],
        sequence_lengths: Sequence[int],
        query_lengths: Sequence[int],
    ) -> "PagedBatchMetadata":
        return cls(
            tuple(tuple(table) for table in block_tables),
            tuple(sequence_lengths),
            tuple(query_lengths),
        )

    @property
    def batch_size(self) -> int:
        return len(self.block_tables)

    def validate(self, cache: PagedKVCache) -> None:
        for table, length in zip(self.block_tables, self.sequence_lengths):
            required = (length + cache.spec.block_size - 1) // cache.spec.block_size
            if len(table) < required:
                raise ValueError("block table does not cover its sequence")
            cache.validate_block_table(table)


class AttentionBackend(ABC):
    name: str

    @abstractmethod
    def prefill(
        self,
        query: Tensor,
        cache: PagedKVCache,
        layer: int,
        metadata: PagedBatchMetadata,
    ) -> Tensor:
        """Attend one or more query tokens for every sequence."""

    @abstractmethod
    def decode(
        self,
        query: Tensor,
        cache: PagedKVCache,
        layer: int,
        metadata: PagedBatchMetadata,
    ) -> Tensor:
        """Attend exactly one query token for every sequence."""
