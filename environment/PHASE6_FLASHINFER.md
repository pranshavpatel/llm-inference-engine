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

FlashInfer's package is Linux-only and its documented paged-decode wrapper
accepts separate NHD K/V tensors with int32 `indptr`, `indices`, and
`last_page_len` metadata. The repository pins `flashinfer-python==0.6.18.post1`
for this initial gate; the pinned package may require additional kernel
downloads on first use. Do not switch package versions without recording the
new version and repeating the gate.
