import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .bench import make_trace
from .config import ModelGeometry


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
        elif args.command == "memory":
            geometry = ModelGeometry.from_config(json.loads(args.config.read_text()))
            report = geometry.capacity(args.pool_mib * 1024**2, args.block_size, args.dtype_bytes)
        else:
            report = make_trace(args.count, args.rate, args.seed, args.prompt_tokens, args.output_tokens)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0
