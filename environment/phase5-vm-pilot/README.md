# Phase 5 fixed-output policy pilot (Ubuntu L40S)

These are the raw files supplied from the Ubuntu VM, plus regenerable
`analyze-replay` exports. The initial ZIP had SHA-256
`c7f01421f4b05e9b7a9ca099ad377bc2989c240aae79457c497aa35adf30b7dc`.
The engine checkout was `629d010411dd5668966dbcb1e5021f341d26ea54`, vLLM was
`0.30.0`, and all three runs used the same two-request trace, model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16 dtype, and L40S GPU.
The later HF eager run used torch `2.13.0+cu132` and transformers `5.17.0`.

All three runs passed the **fixed-output contract**: 2/2 completed, zero failures,
zero missing usage, eight completion tokens per request, and `length` finish
reasons. The saved trace checksum is
`6b23eae8b7abd3d2b9339fccfcf56d3481eff4604095305213847219f25969f9`.

The texts are not fully identical. HF eager and nanoserve match exactly for
both requests. For `r000000` (`Two plus two equals`), all three produce
` four, but` in the first three chunks; vLLM next produces ` what`, while HF
and nanoserve produce another ` but`. For `r000001`, all three produce
` Paris. The capital of France is also`. This is evidence against a nanoserve
versus HF correctness mismatch on these two requests, but does not establish
the cause of vLLM's different greedy choice. Do not claim numerical parity
with vLLM until the choice is characterized.
An earlier BF16 paged-versus-eager check on a different GPU also found a
near-tie on this prompt (`../paged-model-parity-bfloat16.json`); it does not
establish the cause of the L40S divergence.
The earlier Windows BF16 Hugging Face comparison generated ` four, but but`
(`../model-parity-bfloat16.json`), matching the L40S nanoserve prefix, but
different hardware and execution paths make that indirect evidence only.

The runs were **not** a performance comparison: there were only two requests,
no warmup or bounded measurement interval, and the resource configurations
were not matched. The vLLM log reports 8.81 GiB of KV cache, prefix caching,
chunked prefill, compilation, and CUDA graphs; nanoserve reports a 128 MiB KV
pool and reference paged-gather attention. The analysis files intentionally
label full-run throughput and inter-content-chunk gaps as diagnostics, not
steady-state throughput or token-level TPOT.
HF replay serialized inference through its eager adapter: the second request
was sent roughly 1.03 seconds after its intended arrival. Its timing must
not be compared with the two HTTP server runs.

## HF diagnostic already completed

The following command produced `hf-fixed.json`; it was a correctness
adjudication, not a timed baseline. The absolute snapshot directory came
from the pilot runbook as `PHASE5_MODEL_DIR`:

```bash
cd ~/llm-inference-engine
source .venv-vllm/bin/activate
PHASE5_MODEL_DIR=/absolute/path/to/the/pinned/snapshot
python -m nanoserve replay \
  --trace phase5-vm-pilot/fixed-trace.json \
  --engine hf --model-dir "$PHASE5_MODEL_DIR" \
  --device cuda --dtype bfloat16 --max-workers 1 \
  --output phase5-vm-pilot/hf-fixed.json
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)' \
  | tee phase5-vm-pilot/hf-env.txt
```

The remaining numerical check is the top-token logit margin at the shared
prefix where vLLM and HF/nanoserve diverged. A close margin would support a
precision-sensitive greedy tie; a wide margin would require deeper vLLM
configuration or implementation investigation.

After pulling the updated Phase 5 branch on the VM, with both servers stopped,
run the local-only BF16 eager probe against the same pinned snapshot:

```bash
cd ~/llm-inference-engine
git status --short
git pull --ff-only
source .venv-vllm/bin/activate
PHASE5_MODEL_DIR=/absolute/path/to/the/pinned/snapshot
PYTHONPATH=src python scripts/phase5_logit_probe.py \
  --model-dir "$PHASE5_MODEL_DIR" \
  --trace phase5-vm-pilot/fixed-trace.json \
  --hf-replay phase5-vm-pilot/hf-fixed.json \
  --request-id r000000 \
  --output phase5-vm-pilot/hf-logit-probe-r000000.json
```

If `git status` shows tracked local changes, preserve them before pulling; do
not reset them to make the command work.

Send back the JSON even if the command reports that its generated text does
not match the saved HF replay. The file contains only top-five logits per
generation step, not the full model output tensor.
