"""Static-batch Qwen2 execution against physical paged KV storage."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from nanoserve.attention import AttentionBackend, PagedBatchMetadata
from nanoserve.memory import KVAppendPlan, PagedKVCacheManager
from nanoserve.model.qwen2 import Qwen2ForCausalLM


class PagedQwen2Runner:
    """Correctness runner for append-only static batches.

    Requests must already own enough slots in ``cache_manager``. Each call may
    contain a different query length per request. All layer K/V writes are
    committed together only after logits have been produced successfully.
    """

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        cache_manager: PagedKVCacheManager,
        attention_backend: AttentionBackend,
    ) -> None:
        config = model.config
        spec = cache_manager.cache.spec
        if spec.num_layers != config.num_hidden_layers:
            raise ValueError("physical cache layer count does not match the model")
        if spec.num_key_value_heads != config.num_key_value_heads:
            raise ValueError("physical cache KV heads do not match the model")
        if spec.head_dim != config.resolved_head_dim:
            raise ValueError("physical cache head dimension does not match the model")
        parameter = next(model.parameters())
        if parameter.device != cache_manager.cache.device or parameter.dtype != cache_manager.cache.dtype:
            raise ValueError("model device and dtype must match the physical KV cache")
        if model.training:
            raise ValueError("paged reference execution requires model.eval()")
        self.model = model
        self.cache_manager = cache_manager
        self.attention_backend = attention_backend

    def _validate_inputs(self, request_ids: Sequence[str], token_batches: Sequence[Tensor]) -> None:
        if not request_ids or len(request_ids) != len(token_batches):
            raise ValueError("request_ids and token_batches need the same nonzero batch size")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("a static batch cannot contain a request twice")
        for request_id, tokens in zip(request_ids, token_batches):
            self.cache_manager.written_tokens(request_id)
            if tokens.ndim != 1 or tokens.numel() == 0:
                raise ValueError("each token batch must be a nonempty one-dimensional tensor")
            if tokens.dtype != torch.long:
                raise ValueError("token tensors must use torch.long")
            if tokens.device != self.cache_manager.cache.device:
                raise ValueError("token tensors must already be on the model device")

    @torch.inference_mode()
    def forward(
        self, request_ids: Sequence[str], token_batches: Sequence[Tensor]
    ) -> tuple[Tensor, ...]:
        self._validate_inputs(request_ids, token_batches)
        query_lengths = tuple(int(tokens.numel()) for tokens in token_batches)
        plans: list[KVAppendPlan] = []
        try:
            for request_id, length in zip(request_ids, query_lengths):
                plans.append(self.cache_manager.begin_append(request_id, length))

            batch = len(request_ids)
            maximum_query = max(query_lengths)
            device = self.cache_manager.cache.device
            padded_ids = torch.zeros((batch, maximum_query), dtype=torch.long, device=device)
            position_ids = torch.zeros_like(padded_ids)
            for row, (tokens, plan) in enumerate(zip(token_batches, plans)):
                length = query_lengths[row]
                padded_ids[row, :length] = tokens
                position_ids[row, :length] = torch.arange(
                    plan.start_position, plan.end_position, device=device
                )

            metadata = PagedBatchMetadata.from_lists(
                [plan.block_table for plan in plans],
                [plan.end_position for plan in plans],
                query_lengths,
            )
            hidden_states = self.model.model.embed_tokens(padded_ids)
            for layer_index, layer in enumerate(self.model.model.layers):
                residual = hidden_states
                normalized = layer.input_layernorm(hidden_states)
                query, key, value = layer.self_attn.project_qkv(normalized, position_ids)
                for row, plan in enumerate(plans):
                    length = query_lengths[row]
                    self.cache_manager.write_layer(
                        plan,
                        layer_index,
                        key[row, :, :length],
                        value[row, :, :length],
                    )
                if all(length == 1 for length in query_lengths):
                    attention_output = self.attention_backend.decode(
                        query, self.cache_manager.cache, layer_index, metadata
                    )
                else:
                    attention_output = self.attention_backend.prefill(
                        query, self.cache_manager.cache, layer_index, metadata
                    )
                attention_output = attention_output.transpose(1, 2).contiguous().view(
                    batch, maximum_query, -1
                )
                hidden_states = residual + layer.self_attn.o_proj(attention_output)
                residual = hidden_states
                hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

            logits = self.model.lm_head(self.model.model.norm(hidden_states))
            outputs = tuple(logits[row, :length] for row, length in enumerate(query_lengths))
            for plan in plans:
                self.cache_manager.commit_append(plan)
            return outputs
        except Exception:
            for plan in plans:
                if self.cache_manager.append_is_pending(plan):
                    self.cache_manager.abort_append(plan)
            raise
