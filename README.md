# nanoserve

`nanoserve` is an educational single-GPU LLM inference engine. It now has a custom Qwen2 reference implementation plus a Phase 2 correctness path that writes static batches into allocator-owned physical KV pages and executes gather-based paged attention. The model-memory calculator, environment doctor, and seeded arrival plans remain intact.

This is still a correctness milestone. The paged backend deliberately gathers K/V before ordinary PyTorch attention; an optimized paged kernel, continuous batching, scheduling, HTTP serving, and benchmarks are not implemented, and there are no performance claims.

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

These are correctness observations, not latency or throughput measurements. Raw reports are committed under `environment/`.

## Foundation commands

```powershell
$env:PYTHONPATH = "src"
python -m nanoserve memory --config configs/qwen2.5-7b.geometry.json
python -m nanoserve trace --count 100 --rate 2 --seed 42
```

`BlockManager` remains the CPU ownership authority. `PagedKVCacheManager` now maps its immutable page tables to physical tensors, distinguishes reserved from completed KV tokens, and zeroes pages before returning them to the allocator.

The remaining Phase 2 gate is to validate FlashInfer on a Linux CUDA host and compare its optimized prefill/decode kernels against both contiguous and gather-based paged references. After that, the next milestone is continuous scheduling and recompute preemption.
