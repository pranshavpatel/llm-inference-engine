# Phase 6 scored L40S collection

This is the remaining fixed-window NanoServe measurement on a fresh Ubuntu L40S
VM. It uses synthetic prompt lengths, not a production trace. The frozen
project SLOs are client TTFT <= 1 second and server-reported TPOT <= 100 ms.
Only requests completed before the 60-second offered-load cutoff qualify for
goodput. A 30-second drain retains failures and unfinished work but adds no
goodput. NanoServe's server-side monotonic token-event counter measures output
tokens actually emitted in the 60-second interval; HTTP chunks are not used
as token events. The p99 client send-lag gate is 50 ms.

## Fresh VM preparation (no root access needed)

First check `nvidia-smi`; do not run a scored sweep if NVML reports a
driver/library mismatch. Install `uv` into your home directory if absent,
then clone the repository and use the checked Phase 6 branch:

```bash
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
git clone https://github.com/pranshavpatel/llm-inference-engine.git
cd llm-inference-engine
git switch codex/phase-6-optimized-attention
uv python install 3.12
uv venv --python 3.12 --managed-python --seed .venv-phase6
source .venv-phase6/bin/activate
uv pip install 'torch==2.13.0' -r requirements/model.txt -r requirements/flashinfer-linux.txt -e .
flashinfer install-cubin-wheel
flashinfer install-jit-cache-wheel
uv pip check
python scripts/phase6_flashinfer_probe.py --output phase6-fresh-probe-managed.json
```

The managed Python is required on restricted Ubuntu images where system
`Python.h` is absent. `--seed` installs pip because the pinned FlashInfer
wheel helper invokes `python -m pip`. Do not install vLLM for this NanoServe
collection: the current vLLM 0.30 package metadata requires Transformers 5,
which conflicts with this project's pinned Transformers 4.57.6 model stack.
The run is valid only if the probe status is `passed`. The previous working
L40S stack had torch 2.13.0+cu132, FlashInfer 0.6.18.post1, matching cubin
0.6.18.post1, and JIT cache 0.6.18.post1+cu130, without `nvcc`. Capture any
install/probe error and stop rather than substituting a different kernel
stack without a compatibility check. This collection does not start vLLM.

Download the exact Qwen snapshot and run the one-command collection:

```bash
MODEL_DIR="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))')"
OUT="phase6-vm-scored-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/phase6_vm_scored.py --model-dir "$MODEL_DIR" --output-dir "$OUT"
```

The script archives all collected data at `$OUT.zip`, including partial data
after an ordinary run error. Send the entire zip, plus
`phase6-fresh-probe-managed.json`.
The run executes reference and FlashInfer on identical short mixed and long
prefill plans, then a 2x2 block-size (16/32) x batching-limit (1/16) ablation
on reference attention at fixed 256 MiB KV memory. The 2x2 does **not** claim
an unimplemented no-paging mode. Each configuration has a fresh server and an
eight-request warmup. Three independently seeded repetitions are paired across
configurations. Resource samples include cache fragmentation/free blocks,
queue state, preemptions/recomputation, and GPU memory used. A missing
token-window query or failed send-lag gate is flagged in each score.

The short rates are 0.5/1/2/3/4/6 requests/s, 64 fixed output tokens; long
rates are 0.1/0.2/0.4/0.8/1.6 requests/s, 32 fixed output tokens. Both offer
for 60 seconds and drain for at most 30. Low-rate repetitions have too few
completions for strong p99 claims. A high-load point with rising backlog or
failures is overload, not sustainable throughput. The run may take roughly
2 hours plus fresh environment/model setup; allow up to 4 hours and export
the archive before the VM expires.

Install commands follow the [uv environment documentation](https://docs.astral.sh/uv/pip/environments/),
[FlashInfer kernel-wheel documentation](https://docs.flashinfer.ai/cli.html),
and [Hugging Face snapshot API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
