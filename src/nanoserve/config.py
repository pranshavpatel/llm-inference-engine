"""Validated model geometry; no model downloads or tensor dependencies."""
from dataclasses import dataclass
from typing import Mapping, Any


def positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ModelGeometry:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            positive_int(name, getattr(self, name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModelGeometry":
        heads = positive_int("num_attention_heads", config["num_attention_heads"])
        hidden = positive_int("hidden_size", config["hidden_size"])
        head_dim = config.get("head_dim")
        if head_dim is None:
            if hidden % heads:
                raise ValueError("hidden_size must be divisible by attention heads")
            head_dim = hidden // heads
        return cls(config["num_hidden_layers"], heads,
                   config.get("num_key_value_heads", heads), hidden, head_dim)

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        positive_int("dtype_bytes", dtype_bytes)
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype_bytes

    def capacity(self, pool_bytes: int, block_size: int, dtype_bytes: int = 2) -> dict:
        positive_int("pool_bytes", pool_bytes)
        positive_int("block_size", block_size)
        per_token = self.kv_bytes_per_token(dtype_bytes)
        per_block = per_token * block_size
        blocks = pool_bytes // per_block
        return {"bytes_per_token": per_token, "bytes_per_block": per_block,
                "num_blocks": blocks, "token_capacity": blocks * block_size,
                "pool_bytes": blocks * per_block,
                "unusable_remainder_bytes": pool_bytes % per_block}
