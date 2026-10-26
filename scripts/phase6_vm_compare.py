"""Run a paired, same-host reference/FlashInfer serving comparison on Linux."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import zipfile


REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RATES = "1,2,4,6,10,16"
PORT = 8001


def run(*args: str, output: Path | None = None) -> None:
    command = [sys.executable, "-m", "nanoserve", *args]
    print("+", " ".join(command), flush=True)
    if output is None:
        subprocess.run(command, check=True)
    else:
        with output.open("w", encoding="utf-8") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def capture(command: list[str], output: Path) -> None:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        output.write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            output.with_name(output.name + ".error.txt").write_text(
                f"command={command!r}\nexit_code={result.returncode}\n", encoding="utf-8")
    except OSError as exc:
        output.with_name(output.name + ".error.txt").write_text(
            f"command={command!r}\n{type(exc).__name__}: {exc}\n", encoding="utf-8")


def wait_ready(server: subprocess.Popen[bytes], seconds: float = 240) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited before readiness (exit {server.returncode})")
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/ready", timeout=1) as response:
                if json.load(response).get("status") == "ready":
                    return
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("server did not become ready within 240 seconds")


def require_free_port() -> None:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            raise RuntimeError(f"port {PORT} is occupied; stop the existing server before this run")
    except (ConnectionRefusedError, TimeoutError):
        pass


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


def save_endpoint(path: Path) -> None:
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as response:
            path.write_bytes(response.read())
    except (URLError, TimeoutError, OSError) as exc:
        path.write_text(f"metrics unavailable: {exc}\n", encoding="utf-8")


def archive(output_dir: Path) -> Path:
    target = output_dir.with_suffix(".zip")
    if target.exists():
        raise FileExistsError(f"archive already exists: {target}")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=str(path.relative_to(output_dir.parent)))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Absolute pinned Hugging Face snapshot directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; a sibling .zip is created")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    if not model_dir.is_dir() or REVISION not in model_dir.parts:
        parser.error(f"--model-dir must be a snapshot path containing {REVISION}")
    output = args.output_dir.resolve()
    if output.with_suffix(".zip").exists():
        parser.error(f"archive already exists: {output.with_suffix('.zip')}")
    output.mkdir(parents=True, exist_ok=False)
    for backend in ("reference", "flashinfer"):
        (output / backend).mkdir()

    exit_code = 0
    try:
        capture(["git", "rev-parse", "HEAD"], output / "engine-commit.txt")
        capture(["git", "status", "--short", "--untracked-files=no"], output / "git-tracked-status.txt")
        capture(["nvidia-smi"], output / "nvidia-smi-before.txt")
        capture([sys.executable, "-c", "import torch, flashinfer; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('flashinfer', flashinfer.__version__); print('gpu', torch.cuda.get_device_name(0))"], output / "packages.txt")
        capture(["uv", "pip", "freeze"], output / "pip-freeze.txt")
        (output / "run-config.json").write_text(json.dumps({
            "model": MODEL, "revision": REVISION, "model_dir": str(model_dir),
            "backends": ["reference", "flashinfer"], "rates_rps": [1, 2, 4, 6, 10, 16],
            "repetitions": 3, "offered_interval_s": 60, "bounded_drain_s": 20,
            "max_workers": 256, "max_tokens": 16, "ignore_eos": True,
            "dtype": "bfloat16", "kv_pool_mib": 128, "block_size": 16,
            "max_context_tokens": 64, "max_num_sequences": 16,
            "client_send_lag_p99_gate_s": 0.05,
            "notes": "Exploratory paired comparison; p99 needs sufficient per-run samples. Full-run throughput includes drain.",
        }, indent=2) + "\n", encoding="utf-8")

        run("plan-sweep", "--rates", RATES, "--repetitions", "3",
            "--duration-s", "60", "--seed", "1641", "--model", MODEL,
            "--revision", REVISION, "--prompt", "The capital of France is",
            "--prompt", "Two plus two equals", "--max-tokens", "16",
            "--ignore-eos", "--output-dir", str(output / "plan"))
        run("trace-requests", "--count", "8", "--rate", "4", "--seed", "1642",
            "--model", MODEL, "--revision", REVISION,
            "--prompt", "The capital of France is", "--prompt", "Two plus two equals",
            "--max-tokens", "16", "--ignore-eos", "--output", str(output / "warmup-trace.json"))

        for backend in ("reference", "flashinfer"):
            require_free_port()
            log = output / f"{backend}-server.log"
            command = [sys.executable, "-m", "nanoserve", "serve",
                       "--model-dir", str(model_dir), "--model-name", MODEL,
                       "--device", "cuda", "--dtype", "bfloat16", "--kv-pool-mib", "128",
                       "--block-size", "16", "--max-context-tokens", "64",
                       "--max-num-sequences", "16", "--attention-backend", backend,
                       "--host", "127.0.0.1", "--port", str(PORT)]
            print(f"Starting {backend} server; log: {log}", flush=True)
            capture(["nvidia-smi"], output / f"nvidia-smi-{backend}-before.txt")
            with log.open("wb") as handle:
                server = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                          env=os.environ.copy())
                try:
                    wait_ready(server)
                    run("replay", "--trace", str(output / "warmup-trace.json"),
                        "--engine", "nanoserve", "--endpoint", f"http://127.0.0.1:{PORT}/v1/completions",
                        "--max-workers", "256", "--output", str(output / f"{backend}-warmup.json"))
                    warmup = json.loads((output / f"{backend}-warmup.json").read_text(encoding="utf-8"))
                    if (warmup["summary"]["completed"] != 8 or warmup["summary"]["failed"] != 0 or
                            warmup["summary"]["missing_usage"] != 0 or
                            any(record.get("finish_reason") != "length" or
                                (record.get("usage") or {}).get("completion_tokens") != 16
                                for record in warmup["records"])):
                        raise RuntimeError(f"{backend} warmup did not complete cleanly")
                    for trace in sorted((output / "plan").glob("trace-rate-*.json")):
                        print(f"{backend}: {trace.name}", flush=True)
                        run("replay", "--trace", str(trace), "--engine", "nanoserve",
                            "--endpoint", f"http://127.0.0.1:{PORT}/v1/completions",
                            "--bounded-drain-s", "20", "--max-workers", "256",
                            "--output", str(output / backend / trace.name))
                    save_endpoint(output / f"{backend}-final-metrics.json")
                finally:
                    stop_server(server)
            capture(["nvidia-smi"], output / f"nvidia-smi-{backend}-after.txt")

        run("report-sweep", "--plan", str(output / "plan" / "sweep-plan.json"),
            "--replays", f"reference={output / 'reference'}",
            "--replays", f"flashinfer={output / 'flashinfer'}",
            "--output-dir", str(output / "report"), output=output / "report-command.txt")
        print("Paired sweep complete.", flush=True)
    except Exception as exc:
        exit_code = 1
        (output / "run-error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"Run stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        capture(["nvidia-smi"], output / "nvidia-smi-final.txt")
        try:
            bundle = archive(output)
            print(f"Send this archive: {bundle}", flush=True)
        except Exception as exc:
            print(f"Archive failed; preserve output directory {output}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
