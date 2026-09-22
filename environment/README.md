# Correctness evidence

- `phase0-manifest.json` records the machine, driver, CUDA runtime, pinned packages, container availability, and backend decision.
- `paged-attention-smoke.json` is produced by `scripts/phase0_paged_smoke.py` and records every tested page-size/dtype case.
- `model-parity-float32.json` is the strict teacher-forced and cached/full numerical oracle.
- `model-parity-bfloat16.json` records the selected inference dtype and fixed-prompt greedy results.
- `model-parity-float16.json` preserves the rejected configuration and its non-finite-logit failure.
- `paged-model-parity-float32.json` records the real-model physical-paged FP32 architecture gate.
- `paged-model-parity-bfloat16.json` records the selected-dtype static-batch near-tie divergence with top-two margins.
- `phase4-debug-trace.json` is a checksummed two-request open-loop completion workload with pinned model revision.
- `phase4-hf-smoke.json` and `phase4-nanoserve-smoke.json` are complete per-request functional replay records for that trace. Both returned the same two short outputs and exact usage.

The Phase 4 replay records include diagnostic timestamps, but were collected on a shared Windows display GPU with separate engine runs. They are not controlled latency or throughput benchmarks.
