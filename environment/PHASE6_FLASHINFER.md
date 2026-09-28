# Phase 6: FlashInfer decode compatibility gate

The L40S reference-paged profile in `phase6-paged-profile/README.md` identifies
per-request/layer gather and small-op dispatch as the next backend target. Before
altering the serving path, run one isolated BF16 FlashInfer paged-decode check
against the existing gather oracle. It covers 16-token pages, GQA (12 query /
2 KV heads), noncontiguous physical pages, and full/partial final pages.

On the Ubuntu L40S VM, with the existing repository and `.venv-vllm-managed`
environment, run:

```bash
cd ~/llm-inference-engine
git fetch origin codex/phase-6-optimized-attention
git switch -c codex/phase-6-optimized-attention --track origin/codex/phase-6-optimized-attention
source .venv-vllm-managed/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
uv pip install -r requirements/flashinfer-linux.txt
uv pip check
python scripts/phase6_flashinfer_probe.py --output phase6-flashinfer-probe.json
uv pip freeze > phase6-flashinfer-packages.txt
```

Send `phase6-flashinfer-probe.json` and `phase6-flashinfer-packages.txt`. The
probe saves its exception in JSON and exits nonzero if installation, kernel
loading, or numerical parity fails. The declared BF16 gate is `atol=rtol=0.03`;
do not widen it merely to make the check pass. No speedup is claimed from the
probe, and the reference backend stays the default until the optimized path
passes end-to-end parity and a same-host measurement.

The first VM attempt imported FlashInfer but reached JIT compilation during
`wrapper.plan()` and failed because `nvcc` is unavailable. This is an
environment/kernel-loading failure, not a numerical mismatch. Before trying
to install a system CUDA toolkit, use the matching prebuilt packages:

```bash
uv pip install 'flashinfer-cubin==0.6.18.post1' \
  --index-url https://flashinfer.ai/whl
uv pip install 'flashinfer-jit-cache==0.6.18.post1+cu130' \
  --index-url https://flashinfer.ai/whl/cu130
uv pip check
flashinfer show-config > phase6-flashinfer-config.txt
python scripts/phase6_flashinfer_probe.py \
  --output phase6-flashinfer-probe-aot.json
```

The VM currently reports PyTorch `2.13.0+cu132`, whereas FlashInfer publishes
the matching JIT-cache wheel for CUDA 13.0, not 13.2. The AOT smoke result must
therefore decide compatibility; do not claim it works based on wheel install
alone. If it still requests `nvcc` or reports a CUDA-library mismatch, keep
the resulting JSON and config rather than repeatedly changing dependencies.

The returned AOT probe **passed**; see `phase6-flashinfer-probe/`. The next
gate is model-level parity for the opt-in decode adapter, followed by a
same-host, same-protocol measurement before any speed claim.

The adapter is now selectable with `nanoserve serve --attention-backend
flashinfer`; `reference` remains the default. The opt-in path uses FlashInfer
for one-token decode batches and the gather oracle for prefill or mixed-query
batches. It is restricted to the probed BF16 12/2-head, 128-dim, 16-token-page
geometry. Run the model-level check on the same VM after fetching the latest
branch:

```bash
git pull --ff-only origin codex/phase-6-optimized-attention
python scripts/phase6_model_backend_parity.py \
  --model-dir "$(<phase5-vm-sweep-knee/model-snapshot-path.txt)" \
  --output phase6-model-backend-parity.json
```

Return `phase6-model-backend-parity.json` whether it passes or fails. The script
uses one loaded model and separate KV pools, compares four page-boundary
prompt lengths through prefill and eight teacher-forced decode steps, and
records per-request full-logit errors and top-token mismatches. It does not
time the backends. Do not run headline performance comparisons until this
gate and a server-level correctness replay pass.

FlashInfer's package is Linux-only and its documented paged-decode wrapper
accepts separate NHD K/V tensors with int32 `indptr`, `indices`, and
`last_page_len` metadata. The repository pins `flashinfer-python==0.6.18.post1`
for this initial gate; the pinned package may require additional kernel
downloads on first use. Do not switch package versions without recording the
new version and repeating the gate.
