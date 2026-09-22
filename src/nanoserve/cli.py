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


def serve_demo(host: str, port: int) -> int:
    """Serve a tiny random model for local HTTP contract testing."""
    import torch

    from .attention import ReferencePagedAttention
    from .engine import Engine
    from .memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
    from .model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM
    from .scheduler import Scheduler, SchedulerConfig
    from .server import make_server
    from .worker import ByteTokenCodec, InferenceWorker

    if not 0 <= port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    torch.manual_seed(17)
    model_config = Qwen2Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(model_config).eval()
    blocks = BlockManager(32, block_size=16, watermark=0.05)
    cache = PagedKVCache(KVCacheSpec(1, 32, 16, 2, 4), dtype=torch.float32)
    manager = PagedKVCacheManager(blocks, cache)
    scheduler = Scheduler(
        manager,
        SchedulerConfig(
            max_num_sequences=8,
            max_batch_tokens=256,
            max_prefill_tokens=256,
            max_context_tokens=256,
            max_waiting_requests=64,
        ),
    )
    worker = InferenceWorker(
        Engine(scheduler, PagedQwen2Runner(model, manager, ReferencePagedAttention()))
    )
    server = make_server(
        worker,
        ByteTokenCodec(),
        model="nanoserve-tiny-random",
        host=host,
        port=port,
    )
    worker.start()
    bound_host, bound_port = server.server_address
    print(
        json.dumps(
            {
                "status": "ready",
                "address": f"http://{bound_host}:{bound_port}",
                "model": "nanoserve-tiny-random",
                "correctness_demo": True,
                "performance_claim": False,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.stop()
    return 0


def serve_checkpoint(args) -> int:
    """Serve a local checkpoint through the same worker and HTTP contract."""
    from .runtime import ServingConfig, build_serving_runtime
    from .server import make_server

    if not 0 <= args.port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    runtime = build_serving_runtime(
        ServingConfig(
            model_dir=args.model_dir,
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            kv_pool_mib=args.kv_pool_mib,
            block_size=args.block_size,
            max_context_tokens=args.max_context_tokens,
            max_num_sequences=args.max_num_sequences,
            max_waiting_requests=args.max_waiting_requests,
            command_capacity=args.command_capacity,
            watermark=args.watermark,
        )
    )
    server = make_server(
        runtime.worker,
        runtime.codec,
        model=runtime.model_id,
        host=args.host,
        port=args.port,
    )
    runtime.worker.start()
    bound_host, bound_port = server.server_address
    print(
        json.dumps(
            {
                "status": "ready",
                "address": f"http://{bound_host}:{bound_port}",
                "model": runtime.model_id,
                "kv_blocks": runtime.num_blocks,
                "kv_pool_bytes": runtime.kv_pool_bytes,
                "checkpoint_files": runtime.checkpoint_files,
                "attention_backend": "reference_paged_gather",
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.worker.stop()
    return 0


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
    serve = sub.add_parser("serve-demo", help="Serve a tiny random model for HTTP contract testing")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    checkpoint = sub.add_parser("serve", help="Serve a local Qwen2 safetensors checkpoint")
    checkpoint.add_argument("--model-dir", type=Path, required=True)
    checkpoint.add_argument("--model-name")
    checkpoint.add_argument("--host", default="127.0.0.1")
    checkpoint.add_argument("--port", type=int, default=8000)
    checkpoint.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    checkpoint.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    checkpoint.add_argument("--kv-pool-mib", type=int, default=2048)
    checkpoint.add_argument("--block-size", type=int, default=16)
    checkpoint.add_argument("--max-context-tokens", type=int, default=2048)
    checkpoint.add_argument("--max-num-sequences", type=int, default=16)
    checkpoint.add_argument("--max-waiting-requests", type=int, default=128)
    checkpoint.add_argument("--command-capacity", type=int, default=128)
    checkpoint.add_argument("--watermark", type=float, default=0.05)
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
    requests = sub.add_parser("trace-requests", help="Save a checksummed completion workload")
    requests.add_argument("--count", type=int, default=20)
    requests.add_argument("--rate", type=float, default=2.0)
    requests.add_argument("--seed", type=int, default=0)
    requests.add_argument("--model", required=True)
    requests.add_argument("--revision", required=True)
    requests.add_argument("--prompt", action="append", dest="prompts", required=True)
    requests.add_argument("--max-tokens", type=int, default=16)
    requests.add_argument("--ignore-eos", action="store_true", help="Require max-tokens generated tokens")
    requests.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay", help="Replay a saved completion trace")
    replay.add_argument("--trace", type=Path, required=True)
    replay.add_argument("--engine", choices=("nanoserve", "vllm", "hf"), required=True)
    replay.add_argument("--endpoint")
    replay.add_argument("--model-dir", type=Path)
    replay.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    replay.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    replay.add_argument("--timeout-s", type=float, default=60)
    replay.add_argument("--max-workers", type=int, default=32)
    replay.add_argument("--bounded-drain-s", type=float, help="Stop a fixed-window HTTP trace after this drain interval")
    replay.add_argument("--output", type=Path, required=True)
    analysis = sub.add_parser("analyze-replay", help="Export honest full-cohort metrics from one saved replay")
    analysis.add_argument("--trace", type=Path, required=True)
    analysis.add_argument("--replay", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    sweep_report = sub.add_parser("report-sweep", help="Regenerate paired pilot CSV and SVG diagnostics")
    sweep_report.add_argument("--plan", type=Path, required=True, help="Path to sweep-plan.json")
    sweep_report.add_argument("--replays", action="append", required=True, metavar="ENGINE=DIR")
    sweep_report.add_argument("--output-dir", type=Path, required=True)
    sweep = sub.add_parser("plan-sweep", help="Save fixed-window Poisson traces for paired pilot runs")
    sweep.add_argument("--rates", required=True, help="Comma-separated target requests per second")
    sweep.add_argument("--repetitions", type=int, default=3)
    sweep.add_argument("--duration-s", type=float, required=True)
    sweep.add_argument("--seed", type=int, default=0)
    sweep.add_argument("--model", required=True)
    sweep.add_argument("--revision", required=True)
    sweep.add_argument("--prompt", action="append", dest="prompts", required=True)
    sweep.add_argument("--max-tokens", type=int, default=64)
    sweep.add_argument("--ignore-eos", action="store_true", help="Require max-tokens generated tokens")
    sweep.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor()
        elif args.command == "scheduler-demo":
            report = scheduler_demo()
        elif args.command == "serve-demo":
            return serve_demo(args.host, args.port)
        elif args.command == "serve":
            return serve_checkpoint(args)
        elif args.command == "memory":
            geometry = ModelGeometry.from_config(json.loads(args.config.read_text()))
            report = geometry.capacity(args.pool_mib * 1024**2, args.block_size, args.dtype_bytes)
        elif args.command == "trace":
            report = make_trace(args.count, args.rate, args.seed, args.prompt_tokens, args.output_tokens)
        elif args.command == "trace-requests":
            from .replay import make_completion_trace

            report = make_completion_trace(
                count=args.count,
                rate=args.rate,
                seed=args.seed,
                model=args.model,
                revision=args.revision,
                prompts=args.prompts,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        elif args.command == "analyze-replay":
            from .experiment import write_analysis

            report = write_analysis(args.trace, args.replay, args.output_dir)
        elif args.command == "report-sweep":
            from .experiment import write_sweep_report

            replay_dirs = {}
            for item in args.replays:
                if "=" not in item:
                    raise ValueError("--replays must use ENGINE=DIR")
                name, directory = item.split("=", 1)
                if not name or not directory or name in replay_dirs:
                    raise ValueError("--replays engine names must be unique and nonempty")
                replay_dirs[name] = Path(directory)
            report = write_sweep_report(args.plan, replay_dirs, args.output_dir)
            report = {key: value for key, value in report.items() if key not in ("runs", "by_rate")}
        elif args.command == "plan-sweep":
            from .experiment import write_sweep_plan

            report = write_sweep_plan(
                args.output_dir,
                rates=[float(item.strip()) for item in args.rates.split(",")],
                repetitions=args.repetitions,
                duration_s=args.duration_s,
                base_seed=args.seed,
                model=args.model,
                revision=args.revision,
                prompts=args.prompts,
                max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
            )
        else:
            from .replay import (
                HTTPCompletionsAdapter,
                HuggingFaceAdapter,
                replay_completion_trace,
                validate_completion_trace,
            )

            workload = json.loads(args.trace.read_text(encoding="utf-8"))
            validate_completion_trace(workload)
            if args.engine in ("nanoserve", "vllm"):
                if not args.endpoint:
                    raise ValueError("--endpoint is required for HTTP replay")
                adapter = HTTPCompletionsAdapter(
                    args.endpoint,
                    workload["model"],
                    timeout_s=args.timeout_s,
                    name=f"{args.engine}-http",
                )
            else:
                if args.bounded_drain_s is not None:
                    raise ValueError("--bounded-drain-s is supported only for HTTP engines")
                if args.model_dir is None:
                    raise ValueError("--model-dir is required for Hugging Face replay")
                if (
                    args.model_dir.parent.name == "snapshots"
                    and args.model_dir.name != workload["revision"]
                ):
                    raise ValueError("checkpoint snapshot does not match trace revision")
                adapter = HuggingFaceAdapter(
                    args.model_dir, device=args.device, dtype=args.dtype
                )
            if args.bounded_drain_s is None:
                report = replay_completion_trace(
                    workload, adapter, max_workers=args.max_workers
                )
            else:
                from .replay import replay_bounded_http_trace

                report = replay_bounded_http_trace(
                    workload, adapter, drain_s=args.bounded_drain_s,
                    max_workers=args.max_workers,
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report = {key: value for key, value in report.items() if key != "records"}
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0
