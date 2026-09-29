# Phase 5 same-GPU policy pilot (Ubuntu L40S)

This is a correctness and harness pilot, not a throughput comparison. Run both engines on the **same** L40S, one at a time. Keep each server bound to `127.0.0.1`. Do not merge or publish benchmark claims from this two-request trace.

## Prepare one trace

In the Ubuntu repository, preserve any local work before switching branches:

```bash
cd ~/llm-inference-engine
git status --short
git fetch origin codex/phase-5-bounded-replay
git switch codex/phase-5-bounded-replay
git pull --ff-only
source .venv-vllm/bin/activate
mkdir -p phase5-vm-pilot
git rev-parse HEAD | tee phase5-vm-pilot/engine-commit.txt
vllm --version | tee phase5-vm-pilot/vllm-version.txt
nvidia-smi | tee phase5-vm-pilot/nvidia-smi.txt
python -m nanoserve trace-requests \
  --count 2 --rate 100 --seed 7 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --prompt "The capital of France is" \
  --prompt "Two plus two equals" \
  --max-tokens 8 --ignore-eos \
  --output phase5-vm-pilot/fixed-trace.json
```

If `git switch` reports local changes, stop and preserve them; do not reset or delete them. The model snapshot can be located from the existing Hugging Face cache without a new download:

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", local_files_only=True))'
```

## vLLM first

In one terminal, start vLLM 0.30.0 with the same pinned model and tokenizer. `--generation-config vllm` prevents model-repository generation defaults from silently changing the request policy; see the [vLLM 0.30 serving guide](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/).

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --tokenizer-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --generation-config vllm --dtype bfloat16 \
  --max-model-len 64 --gpu-memory-utilization 0.3 \
  --host 127.0.0.1 --port 8000 2>&1 | tee phase5-vm-pilot/vllm-server.log
```

In a second terminal, after the server prints that it is ready:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
python -m nanoserve replay \
  --trace phase5-vm-pilot/fixed-trace.json \
  --engine vllm --endpoint http://127.0.0.1:8000/v1/completions \
  --output phase5-vm-pilot/vllm-fixed.json
```

The expected gate is `completed: 2`, `failed: 0`, `missing_usage: 0`, with **eight** completion tokens and `length` finish reason in each record. Output text itself may include hidden/special tokens and need not match the Phase 4 two-token smoke. If this gate fails, preserve the full JSON and server error output; do not proceed to a scored sweep.

Stop vLLM with Ctrl+C and wait for its process to exit before starting nanoserve. Running both together would invalidate a same-GPU comparison.

## nanoserve on the same GPU

Use the absolute snapshot path printed by the cache command above as `PHASE5_MODEL_DIR` in the terminal that starts nanoserve:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
PHASE5_MODEL_DIR=/absolute/path/printed/by/snapshot_download
python -m nanoserve serve \
  --model-dir "$PHASE5_MODEL_DIR" \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype bfloat16 \
  --kv-pool-mib 128 --max-context-tokens 64 \
  --host 127.0.0.1 --port 8001 2>&1 | tee phase5-vm-pilot/nanoserve-server.log
```

In a second terminal, run the *same* trace:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
python -m nanoserve replay \
  --trace phase5-vm-pilot/fixed-trace.json \
  --engine nanoserve --endpoint http://127.0.0.1:8001/v1/completions \
  --output phase5-vm-pilot/nanoserve-fixed.json
```

Apply the same 2/2, eight-token, `length`-finish gate. Stop nanoserve with Ctrl+C when done. Please return the `phase5-vm-pilot/` files, especially `fixed-trace.json`, both replay JSON files, the three environment text files, and the startup/configuration lines in both server logs. The full JSON records matter more than terminal summary snippets.
