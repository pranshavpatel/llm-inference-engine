# Correctness evidence

- `phase0-manifest.json` records the machine, driver, CUDA runtime, pinned packages, container availability, and backend decision.
- `paged-attention-smoke.json` is produced by `scripts/phase0_paged_smoke.py` and records every tested page-size/dtype case.
- `model-parity-float32.json` is the strict teacher-forced and cached/full numerical oracle.
- `model-parity-bfloat16.json` records the selected inference dtype and fixed-prompt greedy results.
- `model-parity-float16.json` preserves the rejected configuration and its non-finite-logit failure.
- `paged-model-parity-float32.json` records the real-model physical-paged FP32 architecture gate.
- `paged-model-parity-bfloat16.json` records the selected-dtype static-batch near-tie divergence with top-two margins.
- `phase4-debug-trace.json` is a checksummed two-request open-loop completion workload with pinned model revision.
- `phase4-hf-smoke.json`, `phase4-nanoserve-smoke.json`, and `phase4-vllm-smoke.json` are complete per-request functional replay records for that trace. All three completed 2/2 requests with the same outputs (` four,` and ` Paris.`), finish reasons (`length`), and exact token usage. `phase4-vllm-environment.json` records the user-reported VM and launch configuration associated with the vLLM replay.

The Hugging Face and nanoserve records were collected on a shared Windows RTX 6000 Ada display GPU. The vLLM record was collected separately on an Ubuntu 24 NVIDIA L40S VM (driver 580.159.03, vLLM 0.30.0), with BF16 weights, a 64-token maximum context, and `--gpu-memory-utilization 0.3`. The VM had no `nvcc`, so vLLM used `VLLM_USE_FLASHINFER_SAMPLER=0` to select its native sampling fallback; attention used FlashAttention 2. These timestamps are diagnostic only, not controlled latency or throughput benchmarks. Same-GPU comparison remains for the performance phase.
