# Phase 6 same-L40S backend comparison

This run compares **nanoserve reference paged gather** with **nanoserve
FlashInfer paged decode** on the same Ubuntu L40S VM. It is not a vLLM
comparison. The opt-in FlashInfer path passed a short functional replay, but
the full-model BF16-gather logit comparison did not; preserve that limitation
when interpreting either backend's text or performance.

Stop any earlier server on port 8001. In the VM repository, with the pinned
snapshot path previously saved by the Phase 5 knee run:

```bash
cd ~/llm-inference-engine
git status --short
git switch codex/phase-6-optimized-attention
git pull --ff-only origin codex/phase-6-optimized-attention
source .venv-vllm-managed/bin/activate
uv pip install -e .
uv pip check
MODEL_DIR="$(<phase5-vm-sweep-knee/model-snapshot-path.txt)"
OUT="phase6-vm-compare-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/phase6_vm_compare.py --model-dir "$MODEL_DIR" --output-dir "$OUT"
```

If the snapshot-path file is missing, pass the absolute pinned Hugging Face
snapshot directory manually via `--model-dir`; its path must contain revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Do not delete or reset any
VM files to make `git pull` work.

The script creates a new directory and a sibling `.zip`. Send the **entire
`.zip`**, including if the script reports an error; partial output is useful.
The script starts and stops only its own server processes. Its sequence is:

1. Record commit, Git status, GPU state, package versions, and run settings.
2. Generate one checksummed sweep plan and fixed-output warmup trace.
3. Run the reference server, then the FlashInfer server, with the same BF16
   model snapshot, 16-token pages, 128 MiB KV pool, 64-token context, 16
   active sequences, and endpoint protocol. Require each warmup to complete.
4. Replay the same 1/2/4/6/10/16 requests/s Poisson traces, three independent
   60-second repetitions per rate, 20-second bounded drain, 256 parked
   client workers, 16 generated tokens, and ignore-EOS policy.
5. Save every raw replay, server log, final metrics, and a regenerable paired
   sweep report. Archive all output even after a failure.

Before reading server latency or capacity, require p99 client send lag at
each point to be at most 50 ms. Report every failed, timed-out, not-sent, and
missing-usage request; do not silently discard overloaded points. A request
that finishes only during drain is not proof that offered load was sustainable.
The existing `report-sweep` p99 values are medians of three per-run p99s;
low-rate runs have far fewer than 1,000 completions each, so they are not
strong headline tail-latency estimates. Full-run output-token throughput
includes drain and is not steady-state throughput. We will inspect the raw
records for stable-window behavior and draw only claims supported by the run.

The one-hour-plus VM session is the last *planned* GPU data collection for
the core release if it completes and passes data-quality checks. Local
analysis, documentation, and PR review can then proceed without the VM.
