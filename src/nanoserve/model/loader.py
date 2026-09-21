"""Direct, coverage-checked safetensors loading for the Qwen2 model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


class WeightCoverageError(RuntimeError):
    """Raised when checkpoint tensors do not exactly cover the model."""


@dataclass(frozen=True)
class LoadReport:
    loaded_tensors: int
    checkpoint_files: tuple[str, ...]
    tied_lm_head_from_embeddings: bool


def _checkpoint_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise WeightCoverageError("safetensors index has no weight_map")
        return {name: model_dir / filename for name, filename in weight_map.items()}
    if single_path.exists():
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as checkpoint:
            return {name: single_path for name in checkpoint.keys()}
    raise FileNotFoundError(f"no safetensors checkpoint found in {model_dir}")


def load_safetensors(model: "nn.Module", model_dir: str | Path) -> LoadReport:
    """Load weights without constructing a second full checkpoint state dict."""
    from safetensors import safe_open

    root = Path(model_dir)
    checkpoint = _checkpoint_map(root)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    state_names = set(model.state_dict())
    # Non-persistent buffers (notably RoPE inv_freq) are derived from config and
    # intentionally absent from both Hugging Face checkpoints and state_dict().
    destinations = {
        name: tensor for name, tensor in {**buffers, **parameters}.items() if name in state_names
    }
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))

    required = set(destinations)
    optional_missing = {"lm_head.weight"} if tied else set()
    missing = required - set(checkpoint) - optional_missing
    unexpected = set(checkpoint) - required
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected: " + ", ".join(sorted(unexpected)))
        raise WeightCoverageError("checkpoint coverage mismatch (" + "; ".join(details) + ")")

    loaded = 0
    by_file: dict[Path, list[str]] = {}
    for name, path in checkpoint.items():
        by_file.setdefault(path, []).append(name)
    with torch.no_grad():
        for path, names in by_file.items():
            if not path.exists():
                raise FileNotFoundError(f"checkpoint shard is missing: {path}")
            with safe_open(path, framework="pt", device="cpu") as shard:
                actual = set(shard.keys())
                indexed = set(names)
                if not indexed <= actual:
                    absent = ", ".join(sorted(indexed - actual))
                    raise WeightCoverageError(f"index entries absent from {path.name}: {absent}")
                for name in names:
                    tensor = shard.get_tensor(name)
                    destination = destinations[name]
                    if tensor.shape != destination.shape:
                        raise WeightCoverageError(
                            f"shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, "
                            f"model {tuple(destination.shape)}"
                        )
                    destination.copy_(tensor.to(device=destination.device, dtype=destination.dtype))
                    loaded += 1

    if tied and "lm_head.weight" not in checkpoint:
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise WeightCoverageError("configuration requests tied embeddings but parameters are not tied")
    return LoadReport(
        loaded_tensors=loaded,
        checkpoint_files=tuple(sorted(path.name for path in by_file)),
        tied_lm_head_from_embeddings=tied and "lm_head.weight" not in checkpoint,
    )
