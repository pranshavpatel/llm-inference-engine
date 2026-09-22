"""Construct a serving engine from a local Qwen2 safetensors checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from nanoserve.attention import ReferencePagedAttention
from nanoserve.engine import Engine
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
from nanoserve.model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM, load_safetensors
from nanoserve.scheduler import Scheduler, SchedulerConfig
from nanoserve.worker import HuggingFaceTokenCodec, InferenceWorker


@dataclass(frozen=True)
class ServingConfig:
    model_dir: Path
    model_name: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    kv_pool_mib: int = 2048
    block_size: int = 16
    max_context_tokens: int = 2048
    max_num_sequences: int = 16
    max_waiting_requests: int = 128
    command_capacity: int = 128
    watermark: float = 0.05


@dataclass(frozen=True)
class ServingRuntime:
    worker: InferenceWorker
    codec: HuggingFaceTokenCodec
    model_id: str
    num_blocks: int
    kv_pool_bytes: int
    checkpoint_files: tuple[str, ...]


def build_serving_runtime(config: ServingConfig) -> ServingRuntime:
    """Load weights and allocate a bounded KV pool before opening HTTP ingress."""
    from transformers import AutoTokenizer

    model_dir = Path(config.model_dir).resolve()
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    if config.model_name is not None and not config.model_name.strip():
        raise ValueError("model_name must be nonempty")
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.dtype]
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if dtype == torch.bfloat16:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("this CUDA device does not support bfloat16")
    elif device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU serving requires float32")
    elif device.type not in ("cuda", "cpu"):
        raise ValueError("device must be cpu or cuda")

    model_config = Qwen2Config.from_json(model_dir / "config.json")
    if config.max_context_tokens > model_config.max_position_embeddings:
        raise ValueError("max_context_tokens exceeds model max_position_embeddings")
    if config.kv_pool_mib <= 0:
        raise ValueError("kv_pool_mib must be positive")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if not math.isfinite(config.watermark) or not 0 <= config.watermark < 1:
        raise ValueError("watermark must be finite and in [0, 1)")

    bytes_per_block = (
        2
        * model_config.num_hidden_layers
        * config.block_size
        * model_config.num_key_value_heads
        * model_config.resolved_head_dim
        * torch.empty((), dtype=dtype).element_size()
    )
    num_blocks = (config.kv_pool_mib * 1024**2) // bytes_per_block
    if num_blocks < 1:
        raise ValueError("KV pool budget cannot hold one physical page")
    # The terminal sampled token is returned, not appended to physical KV.
    minimum_pages = max(1, (config.max_context_tokens - 1 + config.block_size - 1) // config.block_size)
    if num_blocks - math.ceil(num_blocks * config.watermark) < minimum_pages:
        raise ValueError("KV pool cannot admit a maximum-context request")

    scheduler_config = SchedulerConfig(
        max_num_sequences=config.max_num_sequences,
        max_batch_tokens=config.max_context_tokens,
        max_prefill_tokens=config.max_context_tokens,
        max_context_tokens=config.max_context_tokens,
        max_waiting_requests=config.max_waiting_requests,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = Qwen2ForCausalLM(model_config).to(device=device, dtype=dtype).eval()
    coverage = load_safetensors(model, model_dir)
    blocks = BlockManager(num_blocks, block_size=config.block_size, watermark=config.watermark)
    cache = PagedKVCache(
        KVCacheSpec(
            model_config.num_hidden_layers,
            num_blocks,
            config.block_size,
            model_config.num_key_value_heads,
            model_config.resolved_head_dim,
        ),
        dtype=dtype,
        device=device,
    )
    manager = PagedKVCacheManager(blocks, cache)
    runner = PagedQwen2Runner(model, manager, ReferencePagedAttention())
    scheduler = Scheduler(manager, scheduler_config)
    worker = InferenceWorker(
        Engine(scheduler, runner), command_capacity=config.command_capacity
    )
    return ServingRuntime(
        worker=worker,
        codec=HuggingFaceTokenCodec(tokenizer),
        model_id=config.model_name or model_dir.name,
        num_blocks=num_blocks,
        kv_pool_bytes=cache.stats()["physical_pool_bytes"],
        checkpoint_files=coverage.checkpoint_files,
    )
