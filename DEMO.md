# Three-minute project walkthrough

This is a reproducible presentation outline, not a recorded GPU benchmark.
The local server below uses a tiny randomly initialized Qwen2 model to show
the HTTP, scheduler, and KV ownership behavior without downloading weights.
Its text and speed are not meaningful model-quality or performance results.

## 0:00–0:45 — What is implemented

Show [README.md](README.md) and [DESIGN.md](DESIGN.md): safetensors-loaded
Qwen2 reference execution, allocator-owned physical KV pages, continuous
decode-first scheduling with recompute preemption, and a narrow streaming
`/v1/completions` server. Reference paged gather is the default; FlashInfer
paged decode is an experimental Linux opt-in.

## 0:45–1:50 — Live serving and memory state

In terminal A on Windows, from the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve serve-demo --host 127.0.0.1 --port 8000
```

In terminals B and C, start these streaming requests close together:

```powershell
curl.exe -N -H "Content-Type: application/json" -d '{"model":"nanoserve-tiny-random","prompt":"hello","max_tokens":240,"temperature":0,"stream":true}' http://127.0.0.1:8000/v1/completions
```

While they run, inspect from terminal D:

```powershell
$m = Invoke-RestMethod http://127.0.0.1:8000/metrics
$m.states
$m.cache | Select-Object active_requests,allocated_blocks,free_blocks,unused_tail_slots
$m.worker | Select-Object submitted,completed,cancelled,generated_tokens
```

Interrupt terminal C with Ctrl+C before its stream completes, then recheck
metrics. The tiny CPU model can finish all 240 tokens before a human
interrupts it; only call this a live cancellation if the `cancelled` counter
actually rises and active pages are released. Otherwise, show the
deterministic disconnect test instead:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q tests/test_server.py::ServerTests::test_stream_disconnect_requests_cancellation
```

This test holds a streaming worker open, closes the client socket, and checks
that the server requests cancellation. Stop terminal A with Ctrl+C afterward.

## 1:50–3:00 — Evidence and limitation

Show [the saved Phase 6 scored run](environment/phase6-vm-scored/README.md)
and its [throughput/TTFT](environment/phase6-vm-scored/report/throughput-vs-ttft.svg)
and [goodput](environment/phase6-vm-scored/report/goodput-vs-rate.svg)
plots. On the same L40S, with synthetic 40/131/248-token prompts, 64 fixed
output tokens, and 2 offered requests/s, the median of three 60-second runs
emitted 75.4 tokens/s on reference gather versus 136.6 on opt-in FlashInfer.
SLO goodput under predeclared TTFT <= 1 s and TPOT <= 100 ms was 0.00
versus 2.12 requests/s; the reference path was overloaded. At higher rates,
FlashInfer also developed queueing and failures—do not imply universal or
sustainable performance. The optimized path's longer greedy outputs differ
from reference and it uses an extra 128 MiB planning workspace. End with
the remaining limit: these prompts are synthetic, p99 samples are small,
and a same-session vLLM baseline is pending.

## Evidence-backed portfolio wording

- Built a single-GPU Qwen2 inference engine with allocator-owned paged KV,
  continuous batching, recompute preemption, and a streaming completions API;
  validated core model and replay behavior against saved reference runs.
- Added an experimental FlashInfer paged-decode path and a checksummed,
  fixed-window L40S benchmark. At 2 offered requests/s on 64-token synthetic
  completions, measured 136.6 emitted tokens/s and 2.12 SLO-qualified
  requests/s in the median run versus 75.4 tokens/s and 0.00 qualified
  requests/s on an overloaded reference path. This is workload-specific,
  not an exact-output or universal-speedup claim.
