# Phase 6 same-session vLLM baseline

Run this only after the NanoServe scored archive is safely exported. It
replays the *same saved short and long Poisson plans* against vLLM 0.30.0 on
the same L40S and checkpoint. It is a feature-matched baseline: BF16,
256 MiB KV pool, 16 maximum sequences, pinned model/tokenizer revision,
prefix caching and chunked prefill disabled, with vLLM's graph/optimized
backend left enabled. The vLLM context/batch-token cap is 512 for short
and 4096 for long, matching the NanoServe run. Per-request metrics are
enabled for true-token TPOT SLO scoring; this mode can have CPU overhead.

In the still-running VM, start from the repository root. The NanoServe
collection output directory must still exist. This uses a separate
uv-managed Python environment with headers and pip; do not modify the
working `.venv-phase6` environment.

```bash
cd ~/llm-inference-engine
git pull --ff-only origin codex/phase-6-optimized-attention
uv venv --python 3.12 --managed-python --seed .venv-vllm-phase6
source .venv-vllm-phase6/bin/activate
uv pip install 'vllm==0.30.0' -e .
uv pip check
vllm --version
OUT="phase6-vm-vllm-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/phase6_vm_vllm.py \
  --nano-dir phase6-vm-scored-20260929T173830Z \
  --output-dir "$OUT"
```

Send the resulting `$OUT.zip`, including if the runner reports an error;
ordinary run errors still produce a partial archive. Stop if `uv pip check`
or `vllm --version` fails. The runner saves the exact traces, server launch
settings, environment, warmup, every replay and SLO score. It uses the same
predeclared TTFT <= 1 s and TPOT <= 100 ms with 60-second offered window
and 30-second bounded drain. vLLM does not implement NanoServe's
`/metrics/token-window` endpoint, so this artifact does **not** claim exact
within-window emitted-token throughput for vLLM; the paired SLO-goodput and
client latency are comparable. No new prompt distribution or SLO is chosen.

See the pinned [vLLM 0.30 serving flags](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/)
and [per-request metrics](https://docs.vllm.ai/en/v0.30.0/features/per_request_metrics/).
