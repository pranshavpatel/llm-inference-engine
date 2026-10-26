"""Replay the saved Phase 6 traces against a pinned, feature-matched vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from phase6_vm_compare import MODEL, REVISION, archive, capture
from nanoserve.phase6_score import score_fixed_window


PORT = 8002


def ready(server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM server exited before readiness ({server.returncode})")
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM did not become ready in 240 seconds")


def stop(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=25)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


def run_replay(trace: Path, output: Path, *, bounded: bool) -> None:
    command = [sys.executable, "-m", "nanoserve", "replay", "--trace", str(trace),
               "--engine", "vllm", "--endpoint", f"http://127.0.0.1:{PORT}/v1/completions",
               "--max-workers", "256", "--output", str(output)]
    if bounded:
        command.extend(("--bounded-drain-s", "30"))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-dir", type=Path, required=True,
                        help="Unarchived phase6-vm-scored directory, including short-plan and long-plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    nano = args.nano_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.with_suffix(".zip").exists():
        parser.error("output directory and sibling archive must not already exist")
    for workload in ("short", "long"):
        manifest = nano / f"{workload}-plan" / "sweep-plan.json"
        if not manifest.is_file():
            parser.error(f"missing {manifest}")
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        if plan.get("model") != MODEL or plan.get("revision") != REVISION:
            parser.error(f"unexpected model/revision in {manifest}")
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            parser.error(f"port {PORT} is occupied")
    except (ConnectionRefusedError, TimeoutError):
        pass

    output.mkdir(parents=True)
    status = 0
    try:
        capture(["git", "rev-parse", "HEAD"], output / "engine-commit.txt")
        capture(["git", "status", "--short", "--untracked-files=no"], output / "git-tracked-status.txt")
        capture(["nvidia-smi"], output / "nvidia-smi-before.txt")
        capture([sys.executable, "-c", "import torch,vllm,transformers; print('torch',torch.__version__); print('vllm',vllm.__version__); print('transformers',transformers.__version__); print('cuda',torch.version.cuda)"], output / "packages.txt")
        capture(["uv", "pip", "freeze"], output / "pip-freeze.txt")
        for workload in ("short", "long"):
            shutil.copytree(nano / f"{workload}-plan", output / f"{workload}-plan")
        for workload, context in (("short", 512), ("long", 4096)):
            folder = output / workload
            folder.mkdir()
            command = ["vllm", "serve", MODEL, "--revision", REVISION,
                       "--tokenizer-revision", REVISION, "--generation-config", "vllm",
                       "--dtype", "bfloat16", "--max-model-len", str(context),
                       "--max-num-seqs", "16", "--max-num-batched-tokens", str(context),
                       "--kv-cache-memory-bytes", "268435456",
                       "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
                       "--enable-per-request-metrics", "--host", "127.0.0.1", "--port", str(PORT)]
            (output / f"{workload}-serve-command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            print(f"Starting vLLM {workload} server", flush=True)
            with (output / f"{workload}-server.log").open("wb") as log:
                server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                try:
                    ready(server)
                    warmup_trace = nano / f"reference-b16-n16-{workload}" / "warmup-trace.json"
                    shutil.copy2(warmup_trace, folder / "warmup-trace.json")
                    run_replay(folder / "warmup-trace.json", folder / "warmup.json", bounded=False)
                    warmup = json.loads((folder / "warmup.json").read_text(encoding="utf-8"))
                    if (warmup["summary"]["completed"] != 8 or warmup["summary"]["failed"] != 0
                            or warmup["summary"]["missing_usage"] != 0):
                        raise RuntimeError(f"vLLM {workload} warmup failed")
                    for trace in sorted((output / f"{workload}-plan").glob("trace-rate-*.json")):
                        print(f"vLLM {workload}: {trace.name}", flush=True)
                        replay_path = folder / trace.name
                        run_replay(trace, replay_path, bounded=True)
                        score = score_fixed_window(
                            json.loads(trace.read_text(encoding="utf-8")),
                            json.loads(replay_path.read_text(encoding="utf-8")),
                        )
                        score["configuration"] = f"vllm-{workload}"
                        replay_path.with_suffix(".score.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
                        if not score["client_send_lag_gate_passed"] or score["missing_window_tpot"]:
                            print(f"Warning: data-quality gate failed: {trace.name}", flush=True)
                finally:
                    stop(server)
            capture(["nvidia-smi"], output / f"nvidia-smi-{workload}-after.txt")
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
