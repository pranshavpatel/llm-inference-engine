# nanoserve

`nanoserve` is an educational single-GPU LLM inference engine. It now has a custom Qwen2 reference implementation, allocator-owned physical KV pages, gather-based paged attention, and a Phase 3 continuous scheduler that drives the paged model runner.

This is still a correctness milestone. The paged backend deliberately gathers K/V before ordinary PyTorch attention. An optimized paged kernel and controlled performance benchmarks are not implemented, and there are no performance claims.

## Reproducible setup

Python 3.10 or newer is required. The allocator and planning utilities retain a dependency-free base install:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_core.py" -v
```

The validated GPU environment uses the official PyTorch CUDA 13.0 wheel on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/cuda-cu130.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu"
.\.venv\Scripts\python.exe scripts/phase0_paged_smoke.py
$env:NANOSERVE_RUN_GPU_PAGING = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_paged.py -q -m gpu
```

The exact validated versions are PyTorch `2.13.0+cu130`, Transformers `4.57.6`, safetensors `0.6.2`, and pytest `8.4.2`. `requirements/model.txt` is the platform-neutral model stack; `requirements/flashinfer-linux.txt` records the proposed optimized-backend pin.

FlashInfer `0.6.18.post1` publishes Linux-only wheels. This Windows host has neither WSL nor Docker, so FlashInfer was not installed and no optimized-backend compatibility is claimed. The Phase 0 fallback smoke test gathers noncontiguous physical pages and executes PyTorch SDPA. It is a correctness check, not an optimized backend.

## Continuous scheduling

`Scheduler` owns the request state machine and uses decode-first reservation followed by FCFS admission. It enforces sequence, token, context, waiting-queue, and physical-pool bounds. When decode growth cannot fit, it preempts the newest active request, releases its pages, and later reconstructs KV from the retained prompt and generated-token history. Recomputed tokens are never emitted again or counted as new output.

`Engine.step()` executes decode and prefill groups separately against the shared `PagedQwen2Runner`, greedily samples one token per selected request, and commits results only after physical KV positions match the plan. Completion, EOS, cancellation, and execution failure release pages. Only one synchronous step may be in flight.

Run the deterministic tiny-model demonstration, which intentionally forces one preemption in a three-page pool:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve scheduler-demo
```

The JSON report includes output tokens, steps, preemptions, recomputed tokens, and final page release. Its elapsed time is diagnostic only and is explicitly not a performance claim.

## Completion serving

Phase 4 now includes a dependency-free HTTP correctness path. `InferenceWorker` is the only owner of the synchronous engine: request threads communicate through a bounded command queue and receive token events through per-request queues. Cancellation waits for an already-running model step and then releases request state before the next step.

The implemented API subset is:

- `POST /v1/completions` with `model`, string `prompt`, positive `max_tokens`, `stream`, `n=1`, greedy `temperature=0`, and optional boolean `ignore_eos` for fixed-output experiments.
- JSON completions with exact prompt/completion token accounting, or SSE chunks terminated by `data: [DONE]`.
- `GET /health`, `GET /ready`, and JSON `GET /metrics`. The worker metrics include a cumulative `generated_tokens` counter: one increment per emitted model token, excluding recomputed history and terminal events without a token. Take differences between boundary samples for an internal output-token count; the counter alone does not define a benchmark measurement window.
- `400` for unsupported inputs, `429` for bounded-queue overload, `503` when the worker is unavailable, and disconnect cancellation for streaming responses.

Run a local contract demo backed by a deterministic, randomly initialized tiny Qwen2 model:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve serve-demo --host 127.0.0.1 --port 8000
```

From another shell:

```powershell
$body = @{ model = "nanoserve-tiny-random"; prompt = "hello"; max_tokens = 4; stream = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/completions -ContentType application/json -Body $body
```

The demo model and byte codec test transport and lifecycle behavior only; their text is not meaningful.

To serve a locally available Qwen2 safetensors checkpoint with its tokenizer, use the `serve` command. It loads only local files, checks tensor coverage, and derives physical page count from the KV budget:

```powershell
$env:PYTHONPATH = "src"
$snapshot = ".hf-cache\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
.\.venv\Scripts\python.exe -m nanoserve serve --model-dir $snapshot --model-name Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype bfloat16 --kv-pool-mib 128 --max-context-tokens 64
```

The `serve` command defaults to a loopback bind, BF16 CUDA, a 2 GiB KV pool, and a 2,048-token context. The smaller settings above are for a short correctness smoke. On this host, the pinned real checkpoint started with 292 KV pages in a 128 MiB pool and returned ` Paris.` for a two-token completion to `The capital of France is`, with five prompt tokens and two completion tokens counted. This is one functional check, not a throughput measurement. The server currently has no authentication, TLS, chat endpoint, sampling, or multi-process deployment support.

## Saved trace replay

`trace-requests` writes a checksummed workload with fixed arrival offsets and prompt text. `replay` sends those requests at their scheduled times and retains one record per request, including failures, actual send lag, first content, completion, exact usage when the endpoint supplies it, and timestamped content chunks. HTTP chunks are not assumed to equal model tokens.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve trace-requests --count 20 --rate 2 --seed 7 --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --prompt "The capital of France is" --prompt "Two plus two equals" --max-tokens 16 --output trace.json
.\.venv\Scripts\python.exe -m nanoserve replay --trace trace.json --engine nanoserve --endpoint http://127.0.0.1:8000/v1/completions --output nanoserve-run.json
.\.venv\Scripts\python.exe -m nanoserve replay --trace trace.json --engine hf --model-dir $snapshot --output hf-run.json
```

On a separate supported vLLM host, run its OpenAI-compatible completions server with the same pinned model, then use `--engine vllm --endpoint http://HOST:PORT/v1/completions`. The HTTP adapter requests a final usage chunk through `stream_options.include_usage`, which [vLLM's completion protocol supports](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/). Check the saved `missing_usage` count before using token metrics. The trace revision is checked against a Hugging Face snapshot directory name when available; a remote HTTP server's loaded revision still must be verified in its launch configuration. The local Hugging Face adapter serializes greedy requests on one loaded model; it is an initial functional baseline, not a tuned static-batch comparison.

The committed two-request debug trace was replayed against the pinned Hugging Face checkpoint, nanoserve, and vLLM 0.30.0. All three completed 2/2 requests with matching text (` four,` and ` Paris.`), `length` finish reasons, two completion tokens each, matching prompt-token usage, and no missing usage. The vLLM run used an Ubuntu 24 NVIDIA L40S VM with `VLLM_USE_FLASHINFER_SAMPLER=0` because the VM lacked `nvcc`; its launch pinned both model and tokenizer revisions to the trace's commit. See `environment/phase4-vllm-smoke.json` and `environment/README.md` for the saved evidence and environment details. These short smoke records are functional evidence only; the Windows and Linux runs used different GPUs and are not benchmark results.

## Phase 5 experiment groundwork

`plan-sweep` writes reproducible, independently seeded Poisson arrival traces for fixed offered-load intervals. Each trace is saved once and can be replayed unchanged against all engines. Use `--ignore-eos` for the fixed-output benchmark policy: EOS token IDs still count toward `max_tokens`, and a completed HTTP replay is marked failed unless usage reports exactly that many generated tokens with a `length` finish. Without the flag, normal EOS behavior remains available for correctness checks. vLLM 0.30 supports the `ignore_eos` completion parameter as a [server extension](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/). Use a new output directory for each plan:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m nanoserve plan-sweep --rates 1,2,4 --repetitions 3 --duration-s 30 --seed 42 --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --prompt "The capital of France is" --prompt "Two plus two equals" --max-tokens 16 --ignore-eos --output-dir phase5-pilot-plan
```

`analyze-replay` validates a saved record against its checksummed trace, then exports a manifest, per-request CSV, timestamped chunk-event JSONL, and aggregate JSON. For a small format check using the committed Phase 4 records:

```powershell
.\.venv\Scripts\python.exe -m nanoserve analyze-replay --trace environment/phase4-debug-trace.json --replay environment/phase4-vllm-smoke.json --output-dir phase4-vllm-analysis
```

For a fixed-window trace, HTTP replay can now stop after an explicit drain interval. Work still in flight becomes `timed_out`; work queued behind the client worker limit becomes `not_sent`. The command interrupts active sockets and refuses to save a report if its client threads cannot stop. For example, with nanoserve already serving the matching checkpoint:

```powershell
.\.venv\Scripts\python.exe -m nanoserve replay --trace phase5-pilot-plan/trace-rate-00-rep-00.json --engine nanoserve --endpoint http://127.0.0.1:8000/v1/completions --bounded-drain-s 10 --max-workers 32 --output phase5-pilot-nanoserve.json
```

The aggregate reports full-run completed output tokens per second, client time-to-first-content, end-to-end latency, send lag, and inter-content-chunk gaps with sample counts. Rejections, drain timeouts, unsent work, and missing usage remain visible. It intentionally does **not** call chunk gaps token ITL/TPOT or report SLO goodput: HTTP chunks need not equal model tokens. Full-run throughput includes startup and drain, not a steady measurement window. The installed-vLLM fixed-output policy pilot passed, but one longer generated text diverged; see [the saved pilot evidence and next correctness check](environment/phase5-vm-pilot/README.md). Scored same-GPU sweeps, profiles, and plots remain before any comparative performance claim.

For the first Ubuntu L40S same-GPU policy check, follow [the Phase 5 VM pilot runbook](environment/PHASE5_VM_PILOT.md). It runs vLLM and nanoserve sequentially, saves both full replay files, and records the environment and server startup configuration.

After a larger paired pilot, save each engine's replay files under its own directory using the trace filenames from `sweep-plan.json` (for example `phase5-runs/nanoserve/trace-rate-00-rep-00.json`). The report command validates every trace/replay pairing and regenerates run-level CSV, per-rate CSV, and two labeled SVG diagnostics:

```powershell
.\.venv\Scripts\python.exe -m nanoserve report-sweep --plan phase5-pilot-plan/sweep-plan.json --replays nanoserve=phase5-runs/nanoserve --replays vllm=phase5-runs/vllm --output-dir phase5-pilot-report
```

The report displays the median and range of **per-run** p99 TTFT values; it does not pool requests and relabel that value as a pooled p99. Hollow plot points denote at least one failed repetition. Zero observed failures alone does not establish stable throughput or a sustainable frontier.

## Physical paged reference

`PagedKVCache` preallocates K/V tensors with layout `[layer, physical_page, offset, kv_head, head_dim]`. `PagedKVCacheManager` connects those tensors to `BlockManager`, tracks completed KV tokens separately from reserved slots, clears released pages, and commits an append only after every transformer layer has written the same token range.

`ReferencePagedAttention` gathers request-local K/V using generic block tables and implements causal GQA in ordinary PyTorch. `PagedQwen2Runner` uses that backend for variable-length static batches and incremental decode. Page tables may include capacity reserved for future tokens; sequence lengths expose only committed entries.

Run real-model contiguous-versus-paged diagnostics with:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HOME = "$PWD\.hf-cache"
.\.venv\Scripts\python.exe scripts/paged_model_parity.py --dtype float32 --tokens 8
.\.venv\Scripts\python.exe scripts/paged_model_parity.py --dtype bfloat16 --tokens 8
```

## Reference model

`src/nanoserve/model/` implements:

- Qwen2 configuration validation, RMSNorm, RoPE with cache offsets, grouped-query attention, biased Q/K/V projections, and SwiGLU blocks.
- Tied and untied language-model heads.
- Full forward execution and single-request greedy cached decoding without calling `AutoModelForCausalLM.generate()`.
- Direct single-file or sharded safetensors loading with missing, unexpected, and shape coverage checks.

The development checkpoint is `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Model files and the local `.hf-cache/` are ignored by Git.

Run the opt-in real-model test with:

```powershell
$env:PYTHONPATH = "src"
$env:HF_HOME = "$PWD\.hf-cache"
$env:NANOSERVE_RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_qwen2.py -q -m gpu
```

Generate the detailed precision report without using the Hugging Face generation helper:

```powershell
.\.venv\Scripts\python.exe scripts/model_parity.py --dtype bfloat16 --tokens 8
.\.venv\Scripts\python.exe scripts/model_parity.py --dtype float32 --tokens 8
```

## Recorded correctness evidence

On the RTX 6000 Ada environment in `environment/phase0-manifest.json`:

- FP32 teacher-forced logits were bit-for-bit identical to Hugging Face eager attention on four fixed prompts. Cached versus full custom logits had maximum absolute error `1.1610984802246094e-4`, and all 32 greedy tokens matched.
- BF16 produced finite logits and all 32 greedy tokens matched. Across the fixed prompts, custom-versus-HF teacher-forced maximum absolute error was at most `0.59375` and mean absolute error at most `0.05159274488687515`. Cached-versus-full maximum absolute error was at most `1.5625`; use FP32 as the strict numerical oracle.
- FP16 real-model execution produced non-finite logits in both the custom and Hugging Face eager paths on this stack, so FP16 is rejected and BF16 is the selected inference dtype.
- The gather-based paged smoke passed page sizes 1, 16, 32, and 64 for FP16 and BF16 at 12 query heads, 2 KV heads, and head dimension 128. The gathered and contiguous SDPA inputs produced identical outputs in these cases.
- The physical page pool and reference backend passed all eight CUDA page-size/dtype combinations. CPU coverage includes fragmented page tables, B−1/B/B+1 boundaries, variable-length static batches, transactional failures, clearing, and repeated mixed-length reuse.
- The real-model FP32 paged static-batch path matched all 32 greedy tokens. Its largest prefill logit error versus individual contiguous forwards was `1.329183578491211e-4` and largest mean error was `1.3605588719656225e-5`.
- The BF16 paged static-batch path matched 31/32 greedy tokens. The one divergence occurred at a contiguous-reference top-two margin of exactly `0.0`; the paged margin was `0.125`. This near-tie is preserved in the evidence rather than hidden by weakening a tolerance.
- The Phase 3 CPU suite exercises staggered continuous admission, decode-first execution, simultaneous progress, EOS, cancellation, queue and context bounds, transactional runner failures, forced recompute preemption, and a real tiny-Qwen scheduler integration.
- The Phase 4 suite verifies single-thread engine ownership, concurrent submissions, bounded ingress, cancellation after in-flight work, worker failure propagation, JSON completions, SSE framing, tokenizer byte boundaries, validation, overload responses, health/readiness, metrics, startup from a saved tiny checkpoint, trace checksums, failed-request retention, and streamed usage parsing. The Phase 5 CPU additions validate fixed-window sweep planning, bounded HTTP drain with queued/in-flight accounting, ignore-EOS fixed token counts, saved-record consistency, sweep provenance, honest metric labels, and export files. The full non-GPU suite passes `102` tests; `9` GPU tests remain opt-in.

These are correctness observations, not latency or throughput measurements. Raw reports are committed under `environment/`.

## Foundation commands

```powershell
$env:PYTHONPATH = "src"
python -m nanoserve memory --config configs/qwen2.5-7b.geometry.json
python -m nanoserve trace --count 100 --rate 2 --seed 42
```

`BlockManager` remains the CPU ownership authority. `PagedKVCacheManager` now maps its immutable page tables to physical tensors, distinguishes reserved from completed KV tokens, and zeroes pages before returning them to the allocator.

The deferred Phase 2 optimization gate is to validate FlashInfer on a Linux CUDA host and compare its kernels against both contiguous and gather-based paged references. Phase 4 has complete functional replay records for all three engines; controlled performance claims still require a comparable, isolated target GPU environment.
