"""Collect Phase 6 fixed-window NanoServe scores and resource samples on one GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

from phase6_vm_compare import (
    MODEL, PORT, REVISION, archive, capture, require_free_port,
    stop_server, wait_ready,
)
from nanoserve.phase6_score import score_fixed_window


def run(*args: str) -> None:
    visible = [f"<prompt {len(value)} chars>" if index and args[index - 1] == "--prompt" else value
               for index, value in enumerate(args)]
    print("+", sys.executable, "-m nanoserve", " ".join(visible), flush=True)
    subprocess.run([sys.executable, "-m", "nanoserve", *args], check=True)


def _prompt_near(tokenizer, target: int) -> tuple[str, int]:
    unit = "This is an intentionally synthetic inference benchmark request about a small city. "
    prompt = unit
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target:
        prompt += unit
    count = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, count


def _sample_metrics(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        at = time.time()
        sample: dict = {"at_utc_epoch_s": at}
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as response:
                sample["metrics"] = json.load(response)
        except Exception as error:
            sample["metrics_error"] = f"{type(error).__name__}: {error}"
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 text=True, capture_output=True, timeout=2, check=True)
            sample["gpu_memory_used_mib"] = int(gpu.stdout.splitlines()[0].strip())
        except Exception as error:
            sample["gpu_memory_error"] = f"{type(error).__name__}: {error}"
        samples.append(sample)
        stop.wait(1.0)


def _replay(trace: Path, output: Path, *, config_name: str) -> None:
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_metrics, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        run("replay", "--trace", str(trace), "--engine", "nanoserve",
            "--endpoint", f"http://127.0.0.1:{PORT}/v1/completions",
            "--bounded-drain-s", "30", "--max-workers", "256",
            "--token-window-endpoint", f"http://127.0.0.1:{PORT}/metrics/token-window",
            "--output", str(output))
    finally:
        stop.set()
        sampler.join(timeout=3)
        output.with_suffix(".metrics.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    trace_data = json.loads(trace.read_text(encoding="utf-8"))
    replay_data = json.loads(output.read_text(encoding="utf-8"))
    score = score_fixed_window(trace_data, replay_data)
    score["configuration"] = config_name
    output.with_suffix(".score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    if not score["client_send_lag_gate_passed"]:
        print(f"Warning: client send-lag gate failed: {config_name} {trace.name}", flush=True)
    if score["emitted_output_tokens_per_s"] is None:
        print(f"Warning: missing server token-window count: {config_name} {trace.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    if not model_dir.is_dir() or REVISION not in model_dir.parts:
        parser.error(f"--model-dir must be a snapshot directory containing {REVISION}")
    output = args.output_dir.resolve()
    if output.exists() or output.with_suffix(".zip").exists():
        parser.error("output directory and sibling archive must not already exist")
    output.mkdir(parents=True)

    status = 0
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        short = [_prompt_near(tokenizer, target) for target in (40, 120, 240)]
        long = [_prompt_near(tokenizer, target) for target in (1024, 2048, 3072)]
        if any(count + 64 > 512 for _, count in short) or any(count + 32 > 4096 for _, count in long):
            raise RuntimeError("generated prompts exceed configured context")
        capture(["git", "rev-parse", "HEAD"], output / "engine-commit.txt")
        capture(["git", "status", "--short", "--untracked-files=no"], output / "git-tracked-status.txt")
        capture(["nvidia-smi"], output / "nvidia-smi-before.txt")
        capture([sys.executable, "-c", "import torch, flashinfer; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('flashinfer', flashinfer.__version__); print('gpu', torch.cuda.get_device_name(0))"], output / "packages.txt")
        capture(["uv", "pip", "freeze"], output / "pip-freeze.txt")
        (output / "run-config.json").write_text(json.dumps({
            "model": MODEL, "revision": REVISION, "model_dir": str(model_dir),
            "slo": {"client_ttft_s": 1.0, "server_reported_tpot_s": 0.1},
            "client_send_lag_p99_gate_s": 0.05, "repetitions": 3,
            "offered_interval_s": 60, "bounded_drain_s": 30,
            "short": {"prompt_token_counts": [count for _, count in short], "output_tokens": 64,
                      "rates_rps": [0.5, 1, 2, 3, 4, 6], "context": 512, "kv_pool_mib": 256},
            "long": {"prompt_token_counts": [count for _, count in long], "output_tokens": 32,
                     "rates_rps": [0.1, 0.2, 0.4, 0.8, 1.6], "context": 4096, "kv_pool_mib": 256},
            "policy": "synthetic prompts; ignore EOS; fresh server for each configuration; 8-request warmup",
            "ablation": "reference attention, fixed 256MiB KV pool; block size 16 vs 32 and max sequences 1 vs 16 at 2 rps",
        }, indent=2) + "\n", encoding="utf-8")
        for name, rates, prompts, output_tokens in (
            ("short", "0.5,1,2,3,4,6", short, 64),
            ("long", "0.1,0.2,0.4,0.8,1.6", long, 32),
        ):
            run("plan-sweep", "--rates", rates, "--repetitions", "3", "--duration-s", "60",
                "--seed", "1643", "--model", MODEL, "--revision", REVISION,
                *(item for prompt, _ in prompts for item in ("--prompt", prompt)),
                "--max-tokens", str(output_tokens), "--ignore-eos",
                "--output-dir", str(output / f"{name}-plan"))
        configs = [
            ("reference-b16-n16-short", "reference", 16, 16, "short", None),
            ("flashinfer-b16-n16-short", "flashinfer", 16, 16, "short", None),
            ("reference-b16-n1-ablation", "reference", 16, 1, "short", 2),
            ("reference-b32-n1-ablation", "reference", 32, 1, "short", 2),
            ("reference-b32-n16-ablation", "reference", 32, 16, "short", 2),
            ("reference-b16-n16-long", "reference", 16, 16, "long", None),
            ("flashinfer-b16-n16-long", "flashinfer", 16, 16, "long", None),
        ]
        for name, backend, block_size, max_sequences, workload, only_rate_index in configs:
            require_free_port()
            config_dir = output / name
            config_dir.mkdir()
            context = 512 if workload == "short" else 4096
            command = [sys.executable, "-m", "nanoserve", "serve",
                       "--model-dir", str(model_dir), "--model-name", MODEL,
                       "--device", "cuda", "--dtype", "bfloat16", "--kv-pool-mib", "256",
                       "--block-size", str(block_size), "--max-context-tokens", str(context),
                       "--max-num-sequences", str(max_sequences), "--attention-backend", backend,
                       "--host", "127.0.0.1", "--port", str(PORT)]
            print(f"Starting {name}", flush=True)
            with (output / f"{name}-server.log").open("wb") as log:
                server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                try:
                    wait_ready(server)
                    warmup = output / f"{name}-warmup.json"
                    run("trace-requests", "--count", "8", "--rate", "2", "--seed", "1644",
                        "--model", MODEL, "--revision", REVISION,
                        "--prompt", (short if workload == "short" else long)[0][0],
                        "--max-tokens", "64" if workload == "short" else "32",
                        "--ignore-eos", "--output", str(config_dir / "warmup-trace.json"))
                    run("replay", "--trace", str(config_dir / "warmup-trace.json"),
                        "--engine", "nanoserve", "--endpoint", f"http://127.0.0.1:{PORT}/v1/completions",
                        "--output", str(warmup))
                    warmup_data = json.loads(warmup.read_text(encoding="utf-8"))
                    if warmup_data["summary"] != {"requests": 8, "completed": 8, "failed": 0, "missing_usage": 0}:
                        raise RuntimeError(f"{name} warmup did not complete cleanly")
                    for trace in sorted((output / f"{workload}-plan").glob("trace-rate-*.json")):
                        if only_rate_index is not None and f"rate-{only_rate_index:02d}-" not in trace.name:
                            continue
                        print(f"{name}: {trace.name}", flush=True)
                        _replay(trace, config_dir / trace.name, config_name=name)
                    with urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as response:
                        (output / f"{name}-final-metrics.json").write_bytes(response.read())
                finally:
                    stop_server(server)
            capture(["nvidia-smi"], output / f"nvidia-smi-{name}-after.txt")
        print("Scored collection complete", flush=True)
    except Exception as error:
        status = 1
        (output / "run-error.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        print(f"Run stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        capture(["nvidia-smi"], output / "nvidia-smi-final.txt")
        try:
            bundle = archive(output)
            print(f"Send this archive: {bundle}", flush=True)
        except Exception as error:
            print(f"Archive failed; preserve {output}: {error}", file=sys.stderr, flush=True)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
