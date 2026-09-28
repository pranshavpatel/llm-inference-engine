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

The returned model-level run **failed**: prefill logits matched exactly, but
decode produced 1,547,252 out-of-tolerance logits over the four requests and
eight steps, with two top-token changes and a maximum absolute logit difference
of 12.03125. The raw result is `phase6-flashinfer-model-parity.json` (source
SHA-256 `83cc2cb3d3bb15f9c745999f566a5c32c0dd9ae1a1e0a8e0130a9b0fa68d9a71`).
The synthetic attention probe is therefore insufficient to approve the
optimized serving path. Do not relax the fixed model-logit tolerance to turn
this into a pass.

One bounded diagnostic isolates the first decode step by comparing each
layer's attention output on the **same** reference hidden states and KV cache;
it returns the reference output to later layers so differences cannot
propagate. On the L40S VM:

```bash
git pull --ff-only origin codex/phase-6-optimized-attention
python scripts/phase6_attention_diagnose.py \
  --model-dir "$(<phase5-vm-sweep-knee/model-snapshot-path.txt)" \
  --output phase6-attention-diagnostic.json
```

Return `phase6-attention-diagnostic.json` even if the diagnostic exits nonzero.

That diagnostic found the gap concentrated in the first two layers while
keeping all subsequent hidden states on the reference path: layer 0 had 669
out-of-tolerance attention elements (maximum absolute error 1.7890625), layer
1 had 219, and only 20 combined occurred across layers 2–27. The raw result
is `phase6-attention-diagnostic.json`, source SHA-256
`abee2f7d53a459f6698b2889f90c125af9a2c050e6381856fc0faa43b112bfbc`.
This is not evidence of a page-table fault by itself: the synthetic page
probe passed, and BF16 versus fused-kernel arithmetic can differ. One final
targeted check compares both paths with a float32 attention oracle on layers
0–1, using identical real-model Q/K/V. Run the updated script with a new
output filename:

```bash
git pull --ff-only origin codex/phase-6-optimized-attention
python scripts/phase6_attention_diagnose.py \
  --model-dir "$(<phase5-vm-sweep-knee/model-snapshot-path.txt)" \
  --output phase6-attention-oracle.json
```

If FlashInfer matches the float32 oracle and the BF16 gather path does not,
the model-logit difference is a numerical-reference issue rather than a
page-mapping bug. If neither matches, keep the optimized path gated and do
not add more benchmark work to it without a specific correctness fix.

The returned float32-oracle check supports the numerical-reference
explanation. In all eight request/layer-0-or-1 comparisons, FlashInfer had
**zero** elements outside the original attention tolerance against the
float32 oracle; its largest absolute error was `0.00390625`. The BF16 gather
path had 888 out-of-tolerance elements against the same oracle, with a
maximum absolute error of `1.7890625`. The raw artifact is
`phase6-attention-oracle.json`, source SHA-256
`d95bf126cc5526d100c2067fb46ef8f0b8ce602ebb4e34f260246bf9545af896`.
This validates the targeted decode kernel against higher-precision attention
where the original gap was largest; it does not make the earlier full-model
BF16-reference parity run pass, establish exact greedy text, or prove a
serving speedup. The optimized backend stays opt-in.

Next run one short functional HTTP replay on the same VM. Keep the reference
service stopped and use two terminals. Terminal A:

```bash
cd ~/llm-inference-engine
git pull --ff-only origin codex/phase-6-optimized-attention
source .venv-vllm-managed/bin/activate
python -m nanoserve serve \
  --model-dir "$(<phase5-vm-sweep-knee/model-snapshot-path.txt)" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype bfloat16 --kv-pool-mib 128 \
  --max-context-tokens 64 --attention-backend flashinfer \
  --host 127.0.0.1 --port 8001 2>&1 | tee phase6-flashinfer-server.log
```

After the server prints ready, terminal B:

```bash
cd ~/llm-inference-engine
source .venv-vllm-managed/bin/activate
python -m nanoserve replay \
  --trace environment/phase4-debug-trace.json \
  --engine nanoserve \
  --endpoint http://127.0.0.1:8001/v1/completions \
  --output phase6-flashinfer-server-smoke.json
```

Return the replay JSON and server log. The gate is complete requests, no
server errors, and matching token-usage contract. This is a functional smoke,
not a throughput comparison.

FlashInfer's package is Linux-only and its documented paged-decode wrapper
accepts separate NHD K/V tensors with int32 `indptr`, `indices`, and
`last_page_len` metadata. The repository pins `flashinfer-python==0.6.18.post1`
for this initial gate; the pinned package may require additional kernel
downloads on first use. Do not switch package versions without recording the
new version and repeating the gate.
