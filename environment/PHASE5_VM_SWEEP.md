# Phase 5 paired load pilot (Ubuntu L40S)

Run this after the fixed-output pilot. It is the first **load-harness pilot**,
not a headline throughput claim. Run one server at a time on the same L40S.
The saved plan is identical for both. Keep the server logs: they establish the
actual settings. These runs use BF16, a 64-token context limit, 16 active
sequences, and approximately 128 MiB of KV cache for each engine. vLLM's
prefix caching and chunked prefill are disabled for this feature-matched pilot;
its compiled/CUDA-graph and FlashAttention paths remain enabled. nanoserve
still uses its reference paged-gather backend, so disclose that limitation.

## Prepare once

Stop any running vLLM or nanoserve server, then in `~/llm-inference-engine`:

```bash
git status --short
git pull --ff-only
source .venv-vllm/bin/activate
PYTHONPATH=src python -m nanoserve plan-sweep \
  --rates 0.5,1,2 --repetitions 3 --duration-s 30 --seed 42 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --prompt "The capital of France is" --prompt "Two plus two equals" \
  --max-tokens 16 --ignore-eos --output-dir phase5-vm-sweep/plan
mkdir -p phase5-vm-sweep/vllm phase5-vm-sweep/nanoserve
```

If `git status` lists tracked changes, preserve them before pulling; do not
reset them. Use a new output directory if `phase5-vm-sweep` already exists.
Do not regenerate the plan between engines.

## vLLM

Start this server in one terminal and wait for readiness:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --tokenizer-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --generation-config vllm --dtype bfloat16 \
  --max-model-len 64 --max-num-seqs 16 --max-num-batched-tokens 64 \
  --kv-cache-memory-bytes 134217728 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee phase5-vm-sweep/vllm-server.log
```

In a second terminal:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
PYTHONPATH=src python -m nanoserve replay \
  --trace phase5-vm-pilot/fixed-trace.json --engine vllm \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --output phase5-vm-sweep/vllm-warmup.json
for trace in phase5-vm-sweep/plan/trace-rate-*.json; do
  PYTHONPATH=src python -m nanoserve replay \
    --trace "$trace" --engine vllm \
    --endpoint http://127.0.0.1:8000/v1/completions \
    --bounded-drain-s 20 --max-workers 32 \
    --output "phase5-vm-sweep/vllm/${trace##*/}" || break
done
```

Stop vLLM with Ctrl+C and wait for its process to exit before starting
nanoserve. If startup, warmup, or any replay fails, keep the partial files and
log and stop rather than silently skipping a trace.

## nanoserve

Use the absolute local snapshot path found during the fixed-output pilot as
`PHASE5_MODEL_DIR`. Start nanoserve in one terminal:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
PHASE5_MODEL_DIR=/absolute/path/to/the/pinned/snapshot
PYTHONPATH=src python -m nanoserve serve \
  --model-dir "$PHASE5_MODEL_DIR" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype bfloat16 --kv-pool-mib 128 \
  --max-context-tokens 64 --max-num-sequences 16 \
  --host 127.0.0.1 --port 8001 \
  2>&1 | tee phase5-vm-sweep/nanoserve-server.log
```

In a second terminal, after readiness:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
PYTHONPATH=src python -m nanoserve replay \
  --trace phase5-vm-pilot/fixed-trace.json --engine nanoserve \
  --endpoint http://127.0.0.1:8001/v1/completions \
  --output phase5-vm-sweep/nanoserve-warmup.json
for trace in phase5-vm-sweep/plan/trace-rate-*.json; do
  PYTHONPATH=src python -m nanoserve replay \
    --trace "$trace" --engine nanoserve \
    --endpoint http://127.0.0.1:8001/v1/completions \
    --bounded-drain-s 20 --max-workers 32 \
    --output "phase5-vm-sweep/nanoserve/${trace##*/}" || break
done
PYTHONPATH=src python -m nanoserve report-sweep \
  --plan phase5-vm-sweep/plan/sweep-plan.json \
  --replays nanoserve=phase5-vm-sweep/nanoserve \
  --replays vllm=phase5-vm-sweep/vllm \
  --output-dir phase5-vm-sweep/report
```

Stop nanoserve with Ctrl+C. Return the entire `phase5-vm-sweep/` directory as
an archive, including plan, all replay JSON files, report, warmup records, and
both server logs. If the report fails, return the files anyway. We will check
failures, send lag, latency samples, capacity, and run provenance before using
any numbers. Full-run throughput includes the drain and is not steady-state
throughput; this pilot does not establish SLO goodput.

## After the first pilot

The saved 0.5/1/2 requests/s pilot completed without overload. To locate the
capacity knee, repeat the same procedure with a **new** directory named
`phase5-vm-sweep-high`: use `--rates 4,8,16` in `plan-sweep` and replace only
the `phase5-vm-sweep/` output-directory prefix in commands above with
`phase5-vm-sweep-high/`. Keep the model, server flags, warmup, fixed 30-second
offered interval, 20-second bounded drain, and three paired repetitions
unchanged. Do not replace `phase5-vm-pilot/` in the warmup trace path. Return
the complete high-rate directory as an archive, including logs and any partial
outputs if a run fails. These are still pilot measurements, not a scored
sustainable-throughput frontier.
