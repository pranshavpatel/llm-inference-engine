# Phase 0 evidence

- `phase0-manifest.json` records the machine, driver, CUDA runtime, pinned packages, container availability, and backend decision.
- `paged-attention-smoke.json` is produced by `scripts/phase0_paged_smoke.py` and records every tested page-size/dtype case.
- `model-parity-float32.json` is the strict teacher-forced and cached/full numerical oracle.
- `model-parity-bfloat16.json` records the selected inference dtype and fixed-prompt greedy results.
- `model-parity-float16.json` preserves the rejected configuration and its non-finite-logit failure.

These files contain correctness evidence only. They do not contain timing measurements or benchmark claims.
