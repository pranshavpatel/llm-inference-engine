# nanoserve

An LLM inference engine project focused on paged KV memory management and continuous batching. **Early implementation:** CPU page allocation, memory accounting, environment inspection, and seeded arrival plans work. Model execution, GPU attention, scheduling, and serving are not implemented yet. There are no throughput claims.

## Run the foundation

Python 3.10 or newer is required. The current foundation has no third-party runtime dependencies. From the repository root:

```sh
PYTHONPATH=src python3 -m nanoserve doctor
PYTHONPATH=src python3 -m nanoserve memory --config configs/qwen2.5-7b.geometry.json
PYTHONPATH=src python3 -m nanoserve trace --count 100 --rate 2 --seed 42 > trace.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Use a Python 3.10+ executable explicitly if `python3` points to an older installation. Alternatively install in your own virtual environment with `python -m pip install -e .` to use the `nanoserve` command.

`doctor` inspects availability without installing packages or downloading models. Run it on the library GPU machine and save its JSON output. An available package or a successful NVIDIA query does **not** validate CUDA execution or the attention backend. Those need a later GPU smoke test.

`memory` accepts a Hugging Face-style model config. The included geometry file is a small source-attributed fixture, not a revision-pinned model download. `--pool-mib` is KV pool capacity only: it excludes model weights, activations, and backend workspaces. For the included geometry, 16-bit K/V uses **56 KiB per token**, or **896 KiB per 16-token page** across all layers.

`trace` creates a synthetic fixed-length arrival plan with exponential inter-arrival times and a checksum. It contains lengths, not prompt text or token IDs, and is not yet an executable serving benchmark. A workload materializer and asynchronous replay client will follow.

## Page allocator

`BlockManager` owns page IDs and reserves KV slots before writes. Admission preserves a configurable watermark; existing requests may consume it while growing. Failed allocation leaves ownership unchanged. Returned page tables are immutable snapshots. Freeing an unknown request raises an error to expose lifecycle bugs.

The allocator counts **reserved** tokens, which can include slots scheduled for a write. Later execution metrics must separately count completed KV writes. It owns CPU metadata only; it does not allocate GPU tensors or share prefix pages.

See [DESIGN.md](DESIGN.md) for invariants and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap. Next: verify the library CUDA environment, pin the model/backend stack, and implement the contiguous-cache model reference before optimized paged execution.
