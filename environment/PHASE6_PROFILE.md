# Focused paged-runner profile on the L40S

The Phase 5 knee pilot shows nanoserve queueing sharply above the clean
4 requests/s point. Before changing attention or scheduling, capture one
operator-level diagnostic on the same model and GPU. This is **not** an HTTP
throughput run and its instrumented wall times are not benchmark scores.

Stop any running vLLM or nanoserve server. On the Ubuntu VM after this branch
is available:

```bash
cd ~/llm-inference-engine
git pull --ff-only origin codex/phase-5-bounded-replay
source .venv-vllm-managed/bin/activate
nvidia-smi
python scripts/profile_paged_decode.py \
  --model-dir "$(<phase5-vm-sweep-knee/model-snapshot-path.txt)" \
  --output-dir phase6-paged-profile
tar -czf phase6-paged-profile.tar.gz phase6-paged-profile
```

The script warms one eight-request batch, then profiles an identical batch
with one prefill and 15 decode steps. It uses the pinned local Qwen snapshot,
BF16, a 64-token context, the reference paged-gather backend, and the same
128 MiB KV budget as the pilot. Return the tarball, particularly `summary.json`
and `torch-trace.json`. If the profiler fails because CUDA tracing is
unavailable, return the exact error rather than substituting CPU timings.

Use the operator trace to decide whether gather/copy, attention matmul and
softmax, model projections/MLP, or Python scheduling dominates the step. A
server-level score or cross-engine SLO comparison needs a separate, declared
protocol after profiling; do not compare this profile's wall times with vLLM
HTTP latency.

The returned profile has been summarized in `phase6-paged-profile/README.md`.
The next bounded step is the FlashInfer compatibility gate in
`PHASE6_FLASHINFER.md`; the raw 464 MB trace is intentionally not committed.
