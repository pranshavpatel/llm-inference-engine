"""CPU page ownership and transactional growth. Physical tensors come later.

No sharing in v1: every allocated page has exactly one owner. Token counts here
mean reserved KV slots, not generated output length. A scheduler must reserve
slots before writing them and free them after completion or preemption.
"""
from collections import deque
from math import ceil, isfinite
from nanoserve.config import positive_int


class OutOfBlocks(RuntimeError):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16, watermark: float = 0.05):
        self.num_blocks = positive_int("num_blocks", num_blocks)
        self.block_size = positive_int("block_size", block_size)
        if not isfinite(watermark) or not 0 <= watermark < 1:
            raise ValueError("watermark must be finite and in [0, 1)")
        self.watermark_blocks = ceil(num_blocks * watermark)
        self._free = deque(range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._slots: dict[str, int] = {}

    def _required(self, tokens: int) -> int:
        positive_int("tokens", tokens)
        return (tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, tokens: int) -> bool:
        return self._required(tokens) <= len(self._free) - self.watermark_blocks

    def allocate(self, request_id: str, tokens: int) -> tuple[int, ...]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._tables:
            raise ValueError(f"request already owns pages: {request_id}")
        needed = self._required(tokens)
        if not self.can_allocate(tokens):
            raise OutOfBlocks("admission would consume the watermark reserve")
        self._tables[request_id] = [self._free.popleft() for _ in range(needed)]
        self._slots[request_id] = tokens
        return self.block_table(request_id)

    def can_reserve(self, request_id: str, tokens: int) -> bool:
        needed = self._required(tokens)
        current = self._slots[request_id]
        if tokens < current:
            raise ValueError("reservations cannot shrink; free the request instead")
        return needed - len(self._tables[request_id]) <= len(self._free)

    def reserve(self, request_id: str, tokens: int) -> tuple[int, ...]:
        """Grow to a total slot count; failed growth leaves all state intact."""
        if not self.can_reserve(request_id, tokens):
            raise OutOfBlocks("insufficient pages for growth")
        table = self._tables[request_id]
        needed = self._required(tokens) - len(table)
        table.extend(self._free.popleft() for _ in range(needed))
        self._slots[request_id] = tokens
        return tuple(table)

    def free(self, request_id: str) -> None:
        """Release once. Unknown IDs (including double frees) raise KeyError."""
        table = self._tables.pop(request_id)
        del self._slots[request_id]
        self._free.extend(table)

    def block_table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._tables[request_id])

    def slot(self, request_id: str, position: int) -> tuple[int, int]:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("position must be an integer")
        if not 0 <= position < self._slots[request_id]:
            raise IndexError("position is outside reserved KV slots")
        logical, offset = divmod(position, self.block_size)
        return self._tables[request_id][logical], offset

    def stats(self) -> dict:
        allocated = self.num_blocks - len(self._free)
        slots = allocated * self.block_size
        used = sum(self._slots.values())
        return {"total_blocks": self.num_blocks, "free_blocks": len(self._free),
                "allocated_blocks": allocated, "reserved_tokens": used,
                "allocated_slots": slots, "unused_tail_slots": slots - used,
                "internal_fragmentation": (slots - used) / slots if slots else 0.0,
                "active_requests": len(self._tables)}

    def check_invariants(self) -> None:
        """Expensive audit for tests/debugging, never required on the hot path."""
        owned = [page for table in self._tables.values() for page in table]
        pages = owned + list(self._free)
        assert len(pages) == self.num_blocks
        assert set(pages) == set(range(self.num_blocks))
        assert self._tables.keys() == self._slots.keys()
        for key, table in self._tables.items():
            assert len(table) == self._required(self._slots[key])
