import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .bench import make_trace
from .config import ModelGeometry


def scheduler_demo() -> dict:
    """Run a deterministic tiny-model scheduling smoke test on CPU."""
    import torch

    from .attention import ReferencePagedAttention
    from .engine import Engine
    from .memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
    from .model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM
    from .scheduler import Scheduler, SchedulerConfig

    torch.manual_seed(17)
    model_config = Qwen2Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(model_config).eval()
    blocks = BlockManager(3, block_size=2, watermark=0)
    cache = PagedKVCache(KVCacheSpec(1, 3, 2, 2, 4), dtype=torch.float32)
    manager = PagedKVCacheManager(blocks, cache)
    scheduler = Scheduler(
        manager,
        SchedulerConfig(
            max_num_sequences=2,
            max_batch_tokens=8,
            max_prefill_tokens=8,
            max_context_tokens=8,
        ),
    )
    engine = Engine(scheduler, PagedQwen2Runner(model, manager, ReferencePagedAttention()))
    engine.add_request("older", [1, 2], 3)
    engine.add_request("newer", [10, 11], 3)

    output_tokens = {"older": [], "newer": []}
    steps = 0
    started = time.perf_counter()
    while steps < 20:
        events = engine.step()
        steps += 1
        for event in events:
            if event.token_id is not None:
                output_tokens[event.request_id].append(event.token_id)
        states = scheduler.stats()["states"]
        if states["waiting"] == 0 and states["decoding"] == 0:
            break
        if not events:
            raise RuntimeError("scheduler demo stopped making progress")
    else:
        raise RuntimeError("scheduler demo exceeded its finite-workload step bound")

    elapsed_ms = (time.perf_counter() - started) * 1000
    snapshots = {request_id: scheduler.snapshot(request_id) for request_id in output_tokens}
    return {
        "kind": "tiny_random_model_correctness_demo",
        "steps": steps,
        "elapsed_ms": elapsed_ms,
        "output_tokens": output_tokens,
        "requests": {
            request_id: {
                "state": snapshot.state.value,
                "generated_tokens": len(snapshot.generated_token_ids),
                "preemptions": snapshot.preemptions,
                "recomputed_tokens": snapshot.recomputed_tokens,
                "finish_reason": snapshot.finish_reason.value,
            }
            for request_id, snapshot in snapshots.items()
        },
        "all_pages_released": manager.blocks.stats()["free_blocks"] == 3,
        "performance_claim": False,
    }


def doctor() -> dict:
    report = {"python": sys.version.split()[0], "platform": platform.platform(),
              "nvidia_smi": shutil.which("nvidia-smi"),
              "packages": {name: importlib.util.find_spec(name) is not None
                           for name in ("torch", "flashinfer", "safetensors", "transformers")},
              "gpu_execution_verified": False}
    if report["packages"]["torch"]:
        try:
            import torch

            report["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                               "cuda_available": torch.cuda.is_available()}
            if torch.cuda.is_available():
                probe = (torch.arange(8, device="cuda", dtype=torch.float32) ** 2).sum().item()
                report["torch"].update({"device": torch.cuda.get_device_name(0),
                                        "compute_capability": list(torch.cuda.get_device_capability(0)),
                                        "bf16_supported": torch.cuda.is_bf16_supported(),
                                        "probe_result": probe})
                report["gpu_execution_verified"] = probe == 140.0
        except (ImportError, RuntimeError, AssertionError) as exc:
            report["torch_error"] = f"{type(exc).__name__}: {exc}"
    if report["nvidia_smi"]:
        try:
            result = subprocess.run(
                [report["nvidia_smi"], "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            report["gpu_query"] = result.stdout.strip()
            report["gpu_query_returncode"] = result.returncode
            if result.returncode:
                report["gpu_query_error"] = result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["gpu_query_error"] = str(exc)
    report["next_step"] = "Run scripts/phase0_paged_smoke.py before selecting an optimized backend."
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="nanoserve")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect hardware/tool availability; does not install or download")
    sub.add_parser("scheduler-demo", help="Run a tiny CPU continuous-scheduling correctness demo")
    memory = sub.add_parser("memory", help="Compute KV capacity from model geometry")
    memory.add_argument("--config", type=Path, required=True)
    memory.add_argument("--pool-mib", type=int, default=4096)
    memory.add_argument("--block-size", type=int, default=16)
    memory.add_argument("--dtype-bytes", type=int, choices=(2, 4), default=2)
    trace = sub.add_parser("trace", help="Print a reproducible synthetic arrival plan")
    trace.add_argument("--count", type=int, default=100)
    trace.add_argument("--rate", type=float, default=2.0)
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--prompt-tokens", type=int, default=128)
    trace.add_argument("--output-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor()
        elif args.command == "scheduler-demo":
            report = scheduler_demo()
        elif args.command == "memory":
            geometry = ModelGeometry.from_config(json.loads(args.config.read_text()))
            report = geometry.capacity(args.pool_mib * 1024**2, args.block_size, args.dtype_bytes)
        else:
            report = make_trace(args.count, args.rate, args.seed, args.prompt_tokens, args.output_tokens)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0
