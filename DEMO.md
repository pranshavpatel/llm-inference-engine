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

Show [the saved Phase 6 comparison](environment/phase6-vm-compare/README.md)
and its [latency/failure diagnostics](environment/phase6-vm-compare/report/).
On the same L40S at 4 offered requests/s, median-of-run p50 client TTFT was
168.0 ms with reference gather and 72.9 ms with opt-in FlashInfer. At 6,
reference queue delay grew, while FlashInfer completed 1,049/1,064 requests
within the 60-second offered windows. At 16, both paths overloaded. Explain
that this is a short-prompt diagnostic, not a sustainable-throughput claim;
longer greedy outputs differ between backends and FlashInfer uses an extra
128 MiB planning workspace. End with what a future scored benchmark needs:
mixed lengths, predeclared SLO goodput, enough tail samples, and explicit
paging/batching ablations.

## Evidence-backed portfolio wording

- Built a single-GPU Qwen2 inference engine with allocator-owned paged KV,
  continuous batching, recompute preemption, and a streaming completions API;
  validated core model and replay behavior against saved reference runs.
- Added an experimental FlashInfer paged-decode path. On an L40S under a
  pinned, fixed-16-token workload at 4 offered requests/s, median-of-run p50
  client TTFT was 72.9 ms versus 168.0 ms for the reference gather path.
  Longer greedy output text differed, so this is a workload-specific
  diagnostic, not an exact-output or universal-speedup claim.
