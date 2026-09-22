# Phase 5 fixed-output policy pilot (Ubuntu L40S)

These are the raw files supplied from the Ubuntu VM, plus regenerable
`analyze-replay` exports. The received ZIP had SHA-256
`c7f01421f4b05e9b7a9ca099ad377bc2989c240aae79457c497aa35adf30b7dc`.
The engine checkout was `629d010411dd5668966dbcb1e5021f341d26ea54`, vLLM was
`0.30.0`, and both runs used the same two-request trace, model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16 dtype, and L40S GPU.

Both runs passed the **fixed-output contract**: 2/2 completed, zero failures,
zero missing usage, eight completion tokens per request, and `length` finish
reasons. The saved trace checksum is
`6b23eae8b7abd3d2b9339fccfcf56d3481eff4604095305213847219f25969f9`.

The texts are not fully identical. For `r000000` (`Two plus two equals`), both
produce ` four, but` in the first three chunks; vLLM next produces ` what`,
where nanoserve next produces another ` but`. For `r000001`, both produce
` Paris. The capital of France is also`. The cause of the first divergence is
not established. Do not score a comparative benchmark or claim numerical
parity from this pilot until the divergent request is investigated.
An earlier BF16 paged-versus-eager check on a different GPU also found a
near-tie on this prompt (`../paged-model-parity-bfloat16.json`); it does not
establish the cause of the L40S divergence.
The earlier Windows BF16 Hugging Face comparison generated ` four, but but`
(`../model-parity-bfloat16.json`), matching the L40S nanoserve prefix, but
different hardware and execution paths make that indirect evidence only.

The run was **not** a performance comparison: there were only two requests,
no warmup or bounded measurement interval, and the resource configurations
were not matched. The vLLM log reports 8.81 GiB of KV cache, prefix caching,
chunked prefill, compilation, and CUDA graphs; nanoserve reports a 128 MiB KV
pool and reference paged-gather attention. The analysis files intentionally
label full-run throughput and inter-content-chunk gaps as diagnostics, not
steady-state throughput or token-level TPOT.

## Next diagnostic on the VM

After both servers have stopped, replay the saved trace against the pinned
Hugging Face eager model in BF16. This is a correctness adjudication, not a
timed baseline. Use the absolute snapshot directory located in the pilot
runbook as `PHASE5_MODEL_DIR`:

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

Return `hf-fixed.json` and `hf-env.txt`. If the HF run errors, preserve the
error output instead of changing model, precision, or token policy.
