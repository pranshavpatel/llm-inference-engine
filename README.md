# nanoserve

`nanoserve` is an educational single-GPU LLM inference engine. Phase 1 now has a custom Qwen2 reference implementation with direct safetensors loading, full-sequence execution, greedy generation, and a contiguous per-layer KV cache. The earlier CPU page allocator, model-memory calculator, environment doctor, and seeded arrival plans remain intact.

This is still a correctness milestone. Physical paged KV tensors, an optimized paged-attention kernel, batching, scheduling, HTTP serving, and benchmarks are not implemented, and there are no performance claims.

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
```

The exact validated versions are PyTorch `2.13.0+cu130`, Transformers `4.57.6`, safetensors `0.6.2`, and pytest `8.4.2`. `requirements/model.txt` is the platform-neutral model stack; `requirements/flashinfer-linux.txt` records the proposed optimized-backend pin.

FlashInfer `0.6.18.post1` publishes Linux-only wheels. This Windows host has neither WSL nor Docker, so FlashInfer was not installed and no optimized-backend compatibility is claimed. The Phase 0 fallback smoke test gathers noncontiguous physical pages and executes PyTorch SDPA. It is a correctness check, not an optimized backend.

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

These are correctness observations, not latency or throughput measurements. Raw reports are committed under `environment/`.

## Foundation commands

```powershell
$env:PYTHONPATH = "src"
python -m nanoserve memory --config configs/qwen2.5-7b.geometry.json
python -m nanoserve trace --count 100 --rate 2 --seed 42
```

`BlockManager` still owns only CPU page metadata. It reserves KV slots transactionally, leaves a configurable admission watermark, returns immutable page-table snapshots, and exposes internal fragmentation without allocating GPU tensors.

The next milestone is to connect the block manager to a physical GPU KV pool, add a slow paged-attention oracle, validate FlashInfer on a Linux CUDA host, and compare the optimized path against both contiguous and paged references.
