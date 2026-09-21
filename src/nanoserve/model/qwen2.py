"""A small, readable Qwen2 causal language-model implementation.

This is the correctness path. It deliberately uses ordinary PyTorch operations
and a contiguous per-layer KV cache; it does not call Hugging Face model code or
``generate`` during engine execution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LayerKV = tuple[Tensor, Tensor]
KVCache = tuple[LayerKV, ...]


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads and self.head_dim is None:
            raise ValueError("hidden_size must be divisible by attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.resolved_head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("head_dim * num_attention_heads must equal hidden_size")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")

    @property
    def resolved_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Qwen2Config":
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in names})

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen2Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CausalLMOutput:
    logits: Tensor
    past_key_values: Optional[KVCache] = None


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(input_dtype)


def rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)

    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        cos, sin = self.cos_sin(position_ids, query.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_kv(hidden_states: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(groups, dim=1)


def build_attention_mask(
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return a broadcastable boolean mask where True entries may be attended."""
    query_positions = torch.arange(past_length, past_length + query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    allowed = allowed.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != key_length:
            raise ValueError("attention_mask must have shape [batch, total_key_length]")
        allowed = allowed & attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
    return allowed


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.resolved_head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[LayerKV] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[LayerKV]]:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.rotary_emb(query, key, position_ids)

        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape[:2] != (batch, self.num_key_value_heads):
                raise ValueError("cached key shape does not match the current batch/configuration")
            past_length = past_key.shape[2]
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        present = (key, value) if use_cache else None

        expanded_key = repeat_kv(key, self.num_key_value_groups)
        expanded_value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, expanded_key.transpose(2, 3)) * self.scaling
        allowed = build_attention_mask(
            query_length, key.shape[2], past_length, hidden_states.device, attention_mask
        )
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        output = torch.matmul(probabilities, expanded_value)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.o_proj(output), present


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor],
        past_key_value: Optional[LayerKV],
        use_cache: bool,
    ) -> tuple[Tensor, Optional[LayerKV]]:
        residual = hidden_states
        attention_output, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[KVCache]]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, nonzero_sequence]")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per layer")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")

        hidden_states = self.embed_tokens(input_ids)
        presents: list[LayerKV] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present = layer(
                hidden_states, position_ids, attention_mask, past, use_cache
            )
            if present is not None:
                presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(presents) if use_cache else None


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, present = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return CausalLMOutput(self.lm_head(hidden_states), present)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.shape[0] != 1:
            raise ValueError("the Phase 1 generator supports one request at a time")
        generated = input_ids
        if max_new_tokens == 0:
            return generated
        output = self(input_ids, use_cache=True)
        cache = output.past_key_values
        for step in range(max_new_tokens):
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if step + 1 < max_new_tokens:
                output = self(next_token, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
        return generated
