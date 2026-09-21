# nanoserve

`nanoserve` is an educational single-GPU LLM inference engine. It now has a custom Qwen2 reference implementation, allocator-owned physical KV pages, gather-based paged attention, and a Phase 3 continuous scheduler that drives the paged model runner.

This is still a correctness milestone. The paged backend deliberately gathers K/V before ordinary PyTorch attention. An optimized paged kernel, production-checkpoint server startup, and controlled performance benchmarks are not implemented, and there are no performance claims.

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

- `POST /v1/completions` with `model`, string `prompt`, positive `max_tokens`, `stream`, `n=1`, and greedy `temperature=0`.
- JSON completions with exact prompt/completion token accounting, or SSE chunks terminated by `data: [DONE]`.
- `GET /health`, `GET /ready`, and JSON `GET /metrics`.
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

The demo model and byte codec test transport and lifecycle behavior only; their text is not meaningful. The reusable `HuggingFaceTokenCodec` adapts a loaded tokenizer, but loading a production checkpoint into the server CLI remains a separate integration step. The server currently has no authentication, TLS, chat endpoint, sampling, or multi-process deployment support.

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
- The initial Phase 4 suite verifies single-thread engine ownership, concurrent submissions, bounded ingress, cancellation after in-flight work, worker failure propagation, JSON completions, SSE framing, validation, overload responses, health/readiness, and metrics. The full non-GPU suite passes `67` tests; `9` GPU tests remain opt-in.

These are correctness observations, not latency or throughput measurements. Raw reports are committed under `environment/`.

## Foundation commands

```powershell
$env:PYTHONPATH = "src"
python -m nanoserve memory --config configs/qwen2.5-7b.geometry.json
python -m nanoserve trace --count 100 --rate 2 --seed 42
```

`BlockManager` remains the CPU ownership authority. `PagedKVCacheManager` now maps its immutable page tables to physical tensors, distinguishes reserved from completed KV tokens, and zeroes pages before returning them to the allocator.

The deferred Phase 2 optimization gate is to validate FlashInfer on a Linux CUDA host and compare its kernels against both contiguous and gather-based paged references. The next Phase 4 increment is production-checkpoint startup plus identical saved-trace adapters for nanoserve, Hugging Face, and vLLM; controlled performance claims still require the target Linux GPU environment.
