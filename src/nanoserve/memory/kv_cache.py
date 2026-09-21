"""Physical paged KV tensors connected to the CPU block manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from nanoserve.config import positive_int
from nanoserve.memory.block_manager import BlockManager


@dataclass(frozen=True)
class KVCacheSpec:
    num_layers: int
    num_blocks: int
    block_size: int
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            positive_int(name, getattr(self, name))


@dataclass(frozen=True)
class KVAppendPlan:
    request_id: str
    start_position: int
    end_position: int
    block_table: tuple[int, ...]


def _page_ids(block_table: Sequence[int] | Tensor) -> tuple[int, ...]:
    if isinstance(block_table, Tensor):
        if block_table.ndim != 1:
            raise ValueError("block_table tensor must be one-dimensional")
        values = tuple(int(value) for value in block_table.detach().cpu().tolist())
    else:
        values = tuple(block_table)
    if not values:
        raise ValueError("block_table must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("block_table entries must be integers")
    if len(set(values)) != len(values):
        raise ValueError("a request block table cannot contain duplicate physical pages")
    return values


class PagedKVCache:
    """Preallocated per-layer K/V pages.

    Physical layout is ``[layer, page, offset, kv_head, head_dim]``. Public
    sequence tensors use ``[kv_head, token, head_dim]`` to match Qwen2 caches.
    """

    def __init__(
        self,
        spec: KVCacheSpec,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if not dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (
            spec.num_layers,
            spec.num_blocks,
            spec.block_size,
            spec.num_key_value_heads,
            spec.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)
        # Resolve aliases such as ``cuda`` to the concrete tensor device
        # (``cuda:0``) before validating caller tensors.
        self.device = self.keys.device

    def validate_block_table(self, block_table: Sequence[int] | Tensor) -> tuple[int, ...]:
        table = _page_ids(block_table)
        if any(page < 0 or page >= self.spec.num_blocks for page in table):
            raise IndexError("block_table contains a page outside the physical pool")
        return table

    def _validate_layer(self, layer: int) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("layer must be an integer")
        if not 0 <= layer < self.spec.num_layers:
            raise IndexError("layer is outside the KV cache")

    def _validate_values(self, key: Tensor, value: Tensor) -> int:
        expected_prefix = self.spec.num_key_value_heads
        if key.ndim != 3 or key.shape[0] != expected_prefix or key.shape[2] != self.spec.head_dim:
            raise ValueError("key must have shape [kv_heads, tokens, head_dim]")
        if value.shape != key.shape:
            raise ValueError("value must have the same shape as key")
        if key.shape[1] == 0:
            raise ValueError("a KV write must contain at least one token")
        if key.device != self.device or value.device != self.device:
            raise ValueError("key and value must already be on the cache device")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("key and value dtype must match the cache dtype")
        return key.shape[1]

    def validate_write(
        self,
        layer: int,
        block_table: Sequence[int] | Tensor,
        start_position: int,
        key: Tensor,
        value: Tensor,
    ) -> tuple[tuple[int, ...], int]:
        self._validate_layer(layer)
        table = self.validate_block_table(block_table)
        if isinstance(start_position, bool) or not isinstance(start_position, int) or start_position < 0:
            raise ValueError("start_position must be a nonnegative integer")
        count = self._validate_values(key, value)
        if start_position + count > len(table) * self.spec.block_size:
            raise IndexError("KV write exceeds the request block table")
        return table, count

    def write(
        self,
        layer: int,
        block_table: Sequence[int] | Tensor,
        start_position: int,
        key: Tensor,
        value: Tensor,
    ) -> None:
        table, count = self.validate_write(layer, block_table, start_position, key, value)
        positions = torch.arange(start_position, start_position + count, device=self.device)
        table_tensor = torch.tensor(table, dtype=torch.long, device=self.device)
        pages = table_tensor[positions // self.spec.block_size]
        offsets = positions % self.spec.block_size
        # Advanced indexing returns a temporary tensor, so ``.copy_()`` on the
        # indexed expression would not update the pool. Indexed assignment is
        # required for the scatter to reach physical storage.
        self.keys[layer, pages, offsets] = key.transpose(0, 1)
        self.values[layer, pages, offsets] = value.transpose(0, 1)

    def gather(
        self,
        layer: int,
        block_table: Sequence[int] | Tensor,
        length: int,
    ) -> tuple[Tensor, Tensor]:
        self._validate_layer(layer)
        table = self.validate_block_table(block_table)
        positive_int("length", length)
        if length > len(table) * self.spec.block_size:
            raise IndexError("gather length exceeds the request block table")
        positions = torch.arange(length, device=self.device)
        table_tensor = torch.tensor(table, dtype=torch.long, device=self.device)
        pages = table_tensor[positions // self.spec.block_size]
        offsets = positions % self.spec.block_size
        key = self.keys[layer, pages, offsets].transpose(0, 1).contiguous()
        value = self.values[layer, pages, offsets].transpose(0, 1).contiguous()
        return key, value

    def clear_pages(self, pages: Iterable[int]) -> None:
        page_ids = tuple(pages)
        if not page_ids:
            return
        if any(isinstance(page, bool) or not isinstance(page, int) for page in page_ids):
            raise ValueError("pages must contain integers")
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("pages must not contain duplicates")
        if any(page < 0 or page >= self.spec.num_blocks for page in page_ids):
            raise IndexError("page is outside the physical pool")
        index = torch.tensor(page_ids, dtype=torch.long, device=self.device)
        self.keys[:, index] = 0
        self.values[:, index] = 0

    def stats(self) -> dict:
        element_bytes = self.keys.element_size()
        bytes_per_block = (
            2
            * self.spec.num_layers
            * self.spec.block_size
            * self.spec.num_key_value_heads
            * self.spec.head_dim
            * element_bytes
        )
        return {
            "num_layers": self.spec.num_layers,
            "num_blocks": self.spec.num_blocks,
            "block_size": self.spec.block_size,
            "bytes_per_block": bytes_per_block,
            "physical_pool_bytes": bytes_per_block * self.spec.num_blocks,
            "dtype": str(self.dtype).removeprefix("torch."),
            "device": str(self.device),
        }


class PagedKVCacheManager:
    """Commit append-only KV ranges under BlockManager ownership."""

    def __init__(self, blocks: BlockManager, cache: PagedKVCache) -> None:
        if blocks.num_blocks != cache.spec.num_blocks:
            raise ValueError("block manager and physical cache block counts differ")
        if blocks.block_size != cache.spec.block_size:
            raise ValueError("block manager and physical cache block sizes differ")
        self.blocks = blocks
        self.cache = cache
        self._written: dict[str, int] = {}
        self._pending: dict[str, tuple[KVAppendPlan, set[int]]] = {}

    def allocate(self, request_id: str, tokens: int) -> tuple[int, ...]:
        table = self.blocks.allocate(request_id, tokens)
        self._written[request_id] = 0
        return table

    def reserve(self, request_id: str, tokens: int) -> tuple[int, ...]:
        return self.blocks.reserve(request_id, tokens)

    def begin_append(self, request_id: str, token_count: int) -> KVAppendPlan:
        if request_id not in self._written:
            raise KeyError(request_id)
        if request_id in self._pending:
            raise RuntimeError(f"request already has a pending KV append: {request_id}")
        positive_int("token_count", token_count)
        start = self._written[request_id]
        end = start + token_count
        if end > self.blocks.reserved_tokens(request_id):
            raise IndexError("KV append exceeds reserved request slots")
        plan = KVAppendPlan(request_id, start, end, self.blocks.block_table(request_id))
        self._pending[request_id] = (plan, set())
        return plan

    def write_layer(self, plan: KVAppendPlan, layer: int, key: Tensor, value: Tensor) -> None:
        pending = self._pending.get(plan.request_id)
        if pending is None or pending[0] != plan:
            raise RuntimeError("KV append plan is not active")
        written_layers = pending[1]
        if layer in written_layers:
            raise RuntimeError(f"layer {layer} was already written for this append")
        expected = plan.end_position - plan.start_position
        _, count = self.cache.validate_write(
            layer, plan.block_table, plan.start_position, key, value
        )
        if count != expected:
            raise ValueError("layer token count does not match the append plan")
        self.cache.write(layer, plan.block_table, plan.start_position, key, value)
        written_layers.add(layer)

    def commit_append(self, plan: KVAppendPlan) -> None:
        pending = self._pending.get(plan.request_id)
        if pending is None or pending[0] != plan:
            raise RuntimeError("KV append plan is not active")
        if len(pending[1]) != self.cache.spec.num_layers:
            raise RuntimeError("cannot commit until every layer has written K/V")
        if self._written[plan.request_id] != plan.start_position:
            raise RuntimeError("request KV position changed before commit")
        self._written[plan.request_id] = plan.end_position
        del self._pending[plan.request_id]

    def abort_append(self, plan: KVAppendPlan) -> None:
        pending = self._pending.get(plan.request_id)
        if pending is None or pending[0] != plan:
            raise RuntimeError("KV append plan is not active")
        del self._pending[plan.request_id]

    def append_is_pending(self, plan: KVAppendPlan) -> bool:
        pending = self._pending.get(plan.request_id)
        return pending is not None and pending[0] == plan

    def append(
        self,
        request_id: str,
        keys: Sequence[Tensor],
        values: Sequence[Tensor],
    ) -> None:
        if len(keys) != self.cache.spec.num_layers or len(values) != self.cache.spec.num_layers:
            raise ValueError("keys and values must contain one tensor per layer")
        counts = [key.shape[1] if key.ndim == 3 else 0 for key in keys]
        if len(set(counts)) != 1:
            raise ValueError("every layer must append the same token count")
        plan = self.begin_append(request_id, counts[0])
        try:
            for layer, (key, value) in enumerate(zip(keys, values)):
                self.write_layer(plan, layer, key, value)
            self.commit_append(plan)
        except Exception:
            if request_id in self._pending:
                self.abort_append(plan)
            raise

    def gather(self, request_id: str, layer: int) -> tuple[Tensor, Tensor]:
        if request_id not in self._written:
            raise KeyError(request_id)
        length = self._written[request_id]
        if length == 0:
            shape = (self.cache.spec.num_key_value_heads, 0, self.cache.spec.head_dim)
            empty = torch.empty(shape, dtype=self.cache.dtype, device=self.cache.device)
            return empty, empty.clone()
        return self.cache.gather(layer, self.blocks.block_table(request_id), length)

    def written_tokens(self, request_id: str) -> int:
        return self._written[request_id]

    def free(self, request_id: str) -> None:
        table = self.blocks.block_table(request_id)
        self._pending.pop(request_id, None)
        self.cache.clear_pages(table)
        self.blocks.free(request_id)
        del self._written[request_id]

    def stats(self) -> dict:
        return {
            **self.blocks.stats(),
            **self.cache.stats(),
            "completed_kv_tokens": sum(self._written.values()),
            "pending_appends": len(self._pending),
        }
