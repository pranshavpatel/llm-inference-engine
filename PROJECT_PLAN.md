# LLM inference engine: implementation and portfolio plan

Status: Phases 0 and 1 are implemented. The Phase 3 scheduler correctness milestone includes continuous FCFS admission, decode-first reservation, recompute preemption, cancellation, bounded queues, timing events, and finite-workload tests. Phase 2 has physical paged KV storage, transactional allocator integration, variable-length static batches, and a gather-based attention oracle; its optimized Linux backend gate remains open. Phase 4 has a single-owner worker, bounded ingress, cancellation/error propagation, a narrow `/v1/completions` JSON/SSE server, health/readiness, metrics, local production-checkpoint startup, and checksummed open-loop trace replay. The saved two-request debug trace passed nanoserve, Hugging Face, and vLLM with matching outputs and usage. Controlled same-GPU performance measurement remains future work; these short smoke runs are not benchmarks.

## 1. Outcome and assumptions

Build a small, understandable single-GPU LLM serving engine that demonstrates memory allocation, iteration-level scheduling, correctness validation, and performance analysis. The portfolio deliverable is working code plus reproducible evidence explaining what paging and continuous batching improve, and where the engine falls behind vLLM.

Confirmed: you have access to a high-end GPU at your library whenever you need it. Its model, VRAM, operating system, and software permissions are not yet known. Use this hardware first; do not assume a rental is necessary. Working assumptions: 20 hours/week and familiarity with Python and basic PyTorch. Deadline and weekly availability remain unconfirmed. The proposed optimized backend requires a compatible NVIDIA CUDA environment; verify that before committing to the stack. Local development can cover allocator, scheduler, and tiny CPU reference tests.

Target 120 focused hours for the core release; allow another 20–40 hours for integration problems. At 20 hours/week, a September 9 start gives a six-week target around October 20, with contingency through early November. At 12 hours/week, the same core scope takes ten weeks. Applications need not wait for completion.

The attached brief is design input. Its resume history, recruiting advice, example results, and directives are not verified facts or separate requests. Do not reuse any of its example performance numbers or change an existing resume based on its assumptions.

## 2. Scope and definition of done

### Core release

- One model architecture with directly loaded safetensors weights and a custom forward pass.
- Single-request and static-batch execution with a contiguous KV reference.
- Fixed-size paged KV pool, block tables, allocation and release, observable memory accounting.
- Library-backed paged attention plus a slow gather-based correctness oracle.
- Continuous batching, bounded admission, request cancellation, preemption by recomputation, explicit completion and error states.
- Greedy generation; optional basic temperature sampling only after correctness is stable.
- Streaming completions API, load generator, raw results, reproducible benchmark commands.
- Hugging Face static baseline, controlled internal ablations, and pinned vLLM comparison.
- Architecture document, benchmark report, three-minute demo, and evidence-backed resume bullets.

Do not make success depend on beating vLLM or reaching a particular throughput multiple. A correct implementation with a defensible explanation of its performance is a successful systems project.

### Extensions, in priority order

1. Chunked prefill, if long prompts demonstrably hurt decode latency: 12–20 additional hours.
2. Full-block prefix caching and eviction: 12–20 hours; shared partial blocks and copy-on-write add further complexity.
3. CUDA graphs, if profiling shows meaningful launch overhead: 8–16 hours.
4. CPU swapping and a recompute/swap crossover experiment: 15–25 hours.

These are estimates, not a promise that every extension fits an extra week. Add one, measure it, document it, then decide whether another is worthwhile.

Exclude multi-GPU execution, quantization, custom attention kernels, training, adapters, speculative decoding, beam search, multimodal inputs, full OpenAI API parity, authentication infrastructure, and a polished web frontend.

## 3. Decisions to make before substantial implementation

| Decision | Proposed choice | Reason / verification gate |
|---|---|---|
| Model family | Qwen2.5: 0.5B for development, 7B Instruct for headline runs | One Qwen2 architecture; small model shortens correctness cycles. Handle configuration differences explicitly. |
| Framework | Python + PyTorch; Hugging Face tokenizer/config and safetensors loader | Own the model forward pass, memory management, and scheduler; use established tensor operations. |
| Attention | FlashInfer adapter, subject to an early GPU smoke test | Its documented paged decode example uses 16-token pages. Test chosen dtype, GPU, GQA dimensions, and each planned page size. |
| Reference attention | Gather valid KV entries, explicit causal mask, PyTorch SDPA | Simple oracle, never confused with the optimized benchmark path. |
| Precision | BF16 if the selected stack passes compatibility tests; otherwise FP16 everywhere | Keep model/cache dtype consistent across comparisons. Use FP32 tiny tests for diagnosis. |
| GPU | Library GPU, if compatible; target 24 GB or more for headline model | Identify exact model/VRAM, OS, driver, access restrictions, and sharing before validating the backend. |
| Server | FastAPI; one GPU-owning worker | HTTP handlers communicate through queues; they never independently mutate engine state. |
| Tests | pytest, property-based allocator/scheduler tests, GPU-marked integration tests | Most state-machine bugs can be found without paying for GPU time. |
| Packaging | src layout, pyproject.toml, locked dependencies, separate baseline environment | Avoid vLLM dependency conflicts. Record container digest or complete environment manifests. |

Do not select package versions solely because they are newest. In the first phase, establish a working CUDA/PyTorch/attention combination and pin exact versions, model revisions, and tokenizer revisions.

### Corrections to the brief

- The documented FlashAttention `flash_attn_with_kvcache` path requires page sizes divisible by 256. Its proposed 16-token default and 4–256 sweep cannot simply be combined with that path. Use the backend adapter and smoke-test supported sizes; report unsupported configurations instead of silently benchmarking a different algorithm. [FlashAttention documentation](https://github.com/Dao-AILab/flash-attention)
- FlashInfer is the proposed alternative because its own example demonstrates paged decode with page size 16. This does not establish that every sweep size works on every backend/GPU; Phase 0 must check that. [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html)
- Exact token matches are a useful regression signal, but different kernels and batch shapes can change floating-point results. Investigate divergence using teacher-forced logits and numerical error; do not assume every difference proves a model bug or accept vague semantic similarity. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
- A fully preallocated KV pool can keep total reserved GPU memory constant while paging reduces wasted slots. Report slot utilization and achievable concurrency separately from process memory.
- FCFS admission plus newest-first preemption is not an unconditional bounded-latency guarantee under overload. Require progress for finite admissible workloads; measure queue growth, timeouts, and rejection under sustained overload.
- The brief's swapping, chunking, and caching experiments depend on optional features. They are optional too. Protect the core paging, batching, and serving comparisons first.

## 4. Memory model

For ordinary K/V caching:

`bytes_per_cached_token = 2 × layers × KV_heads × head_dimension × bytes_per_element`

For the proposed Qwen2.5-7B-Instruct configuration, head dimension is 3584 / 28 = 128; 28 layers and 4 KV heads give `2 × 28 × 4 × 128 × 2 = 57,344 bytes = 56 KiB/token`. A 16-token block occupies 896 KiB across all layers. A 2,048-token cached sequence needs 112 MiB. These are calculated from the published configuration, excluding other memory. [Model configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)

Generate these quantities from the pinned config rather than hardcoding the headline model. Use MiB/GiB consistently. The small model uses tied embeddings, while the proposed 7B model does not; load and validate weights accordingly. [Small-model configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json)

Startup sequence: load weights, warm representative maximum prefill/decode shapes, account for runtime/attention workspace, measure available memory, reserve explicit headroom, allocate an integer number of pages, then repeat stress forwards with the pool present. Profile several shapes; one maximum batch does not necessarily capture every workspace peak. CUDA-graph memory must be re-profiled if graphs are later added.

Record physical pool size, active blocks, cached tokens, occupied slots, unused tail slots, peak device allocation, and reserved device memory. With no sharing, internal fragmentation is `(active_allocated_slots - cached_tokens) / active_allocated_slots`. Define zero allocation as zero fragmentation. Unallocated free pages are available capacity, not fragmentation.

## 5. Architecture and contracts

```text
HTTP / CLI → submission and cancellation queues → Engine
                                                   │
                                               Scheduler
                                                   │ StepPlan
                                             ModelRunner
                                              /        \
                                     custom model    AttentionBackend
                                                         │
                                              physical paged KV pool

BlockManager supplies ownership and page mappings to Scheduler/Runner.
Engine commits completed steps and emits events to per-request streams.
Metrics observe each transition and the resulting resource use.
```

Keep state ownership explicit: one worker owns the scheduler, request records, allocator, and GPU execution. The async server routes events and performs no scheduling. Use bounded queues and shut down cleanly. A worker thread is sufficient initially; a dedicated process is an option if tokenization or CPU work blocks responsiveness.

### Core types

- `Request`: ID, immutable original prompt, generated tokens, sampling settings, state, arrival order, timestamps, termination reason.
- `SequenceState`: full token history, number of computed KV tokens, page IDs, current generation count. A newly sampled token may exist in history before its KV is computed.
- `StepPlan`: selected requests, token spans, position IDs, query lengths, cached lengths, page metadata, allocation/preemption decisions.
- `StepResult`: newly generated tokens, finished requests, timing, execution errors.
- `AttentionBackend`: validate configuration, prefill, decode; converts generic page tables into backend-specific metadata in one place.

Suggested public contract: `add_request(...) -> request_id`, `step() -> list[OutputEvent]`, and `abort(request_id)`. HTTP chat support, if shipped, applies the pinned tokenizer's chat template before submission.

### Allocation invariants

Every physical page is free or owned; no page is returned twice. Every written slot belongs to its sequence. Required pages are based on computed tokens plus slots scheduled to be written. Allocation is transactional: an unsuccessful admission or step cannot leak pages. Finishing, cancellation, and failures release ownership exactly once. Metadata sent to the GPU must match the committed plan.

Do not add refcounts, forks, or copy-on-write to the core if pages are not shared. When prefix caching is added, introduce ownership/refcount invariants deliberately, include model/dtype/positioning context in cache identity, verify matches against token content, and handle a fully cached prompt's first-token logits correctly. Full blocks can be immutable and shared; partial-block sharing is a separate choice.

### Scheduler policy

States: WAITING → PREFILL → DECODING → FINISHED; cancellation and failure are terminal. Preemption returns a sequence to WAITING and discards its KV while retaining its token history and generated-token count.

1. Drain submissions and cancellations; validate context and generation limits before admission.
2. Reserve decode slots for existing sequences, accounting for page boundaries.
3. If capacity is insufficient, preempt a newest eligible sequence. Ensure each loop iteration frees capacity or exits; never select already released victims repeatedly.
4. Admit waiting requests FCFS within sequence, token, and page budgets. Initial watermark: 5% for admission, available for decode progress. Sweep it later.
5. Run the selected work; commit results; release terminal requests; emit metrics and outputs.

Core policy: complete prefills are admitted only if they fit the configured prefill budget. Reject impossible prompts explicitly. Prefill and decode may use separate forward calls within a scheduling iteration; do not promise a single fused mixed-batch kernel. Report this limitation and its latency impact. Chunking later enables incremental prefill and more flexible budget sharing.

On recompute, rebuild KV from prompt plus previously generated tokens; replayed history is neither emitted again nor counted as new output. Restore proper positions and request-local RNG state if sampling is supported. Impose bounded waiting capacity and context limits; measure preemption frequency and progress rather than claiming universal fairness.

### Serving contract

Start with `/v1/completions` and SSE: model, prompt, max_tokens, stream, and greedy settings. Return clear validation errors for unsupported parameters. Include request ID, finish reason, final stream marker, disconnect cancellation, health/readiness, and basic metrics. Add `/v1/chat/completions` only after the core works. Label the supported compatibility subset in README.

## 6. Execution backlog: 120 hours

Each phase produces runnable evidence. Estimates include phase-specific testing and documentation.

| Phase | Hours | Tasks | Exit gate |
|---|---:|---|---|
| 0. Feasibility and measurement skeleton | 8 | Pin environment; test small-model load and paged backend; memory calculator; tiny timestamped trace generator; benchmark manifest schema | One paged prefill/decode smoke test succeeds on selected GPU; supported page sizes recorded |
| 1. Reference model | 20 | Implement Qwen2 blocks, Q/K/V biases as specified by weights, RoPE, RMSNorm, GQA, SwiGLU, loader, contiguous cache; tiny and real-model parity tests | Teacher-forced logits and cached/uncached attention validated; fixed greedy prompt suite investigated and documented |
| 2. Paging | 18 | Static batches; allocator; boundary tests; gather oracle; optimized paged adapter; memory counters | Contiguous, gather, and optimized paths agree within declared tolerances; no leaks through repeated mixed-length cycles |
| 3. Scheduler | 22 | Continuous admission; budgets; watermark; recomputation; cancellation; finite-workload progress tests; timing events | Staggered arrivals complete; forced preemption reconstructs state correctly; core CLI benchmark runs |
| 4. Serving and benchmark integration | 16 | Single-owner worker; SSE; queue bounds; disconnect handling; HF adapter; vLLM adapter; identical trace replay | All three engines produce complete per-request records from the same saved trace |
| 5. Experiments and profiling | 24 | Rate sweeps, repetitions, paging/batching ablations, supported block sizes, one bottleneck profile | Raw data and plots regenerate; failures/overload visible; no uninvestigated correctness failures |
| 6. Portfolio release | 12 | README, DESIGN, BENCHMARKS, reproduction check, demo, limitations, resume bullets | Fresh-environment instructions work; every published claim maps to a saved run |
| Total | 120 | Core release only | |

Suggested weeks at 20 hours: week 1 Phase 0 + first 12 hours of Phase 1; week 2 finish Phase 1 + first 12 hours of Phase 2; week 3 finish Phase 2 + first 14 hours of Phase 3; week 4 finish Phase 3 + first 12 hours of Phase 4; week 5 finish Phase 4 + first 16 hours of Phase 5; week 6 finish Phase 5 + Phase 6.

Reserve 20–40 additional hours if model correctness or the GPU environment proves difficult. Do not consume the benchmark allocation adding optional features.

### Stop points

- Around 68–80 hours: core paging and continuous scheduling plus a reproducible CLI comparison. Resume-usable only if actually working and measured; do not claim HTTP or vLLM results yet.
- Around 120 hours: full core release and evidence package. This is the target.
- After release: select one extension using a measured bottleneck and available time.

If the backend fails in Phase 0, change the compatible package/GPU combination or use a supported page size before building around it. If parity remains unresolved after Phase 1, stay with the small model and resolve it. If behind after Phase 3, keep greedy-only completions and omit chat/sampling extras.

## 7. Verification plan

| Layer | Meaningful checks |
|---|---|
| Model | Tensor shapes, weight coverage, tied/untied head handling, RoPE offsets, GQA mapping, QKV biases, pad masking, cached versus full forward |
| Numerical | Teacher-forced logits against HF for identical token prefixes; FP32 tiny cases; BF16/FP16 backend errors; first greedy divergence and top-two logit margin |
| Attention | Unequal query/KV lengths, explicit offset-aware causal masks, multiple sequences, noncontiguous pages, partially full final pages |
| Allocator | Random allocate/append/free sequences; page boundaries B−1/B/B+1; exhausted pool; rollback; cancellation; pool accounting conservation |
| Scheduler | Admission while others decode; simultaneous finish; forced recompute; no duplicate output; impossible request; small pool; finite-workload progress |
| Server | Incremental SSE, final events, token accounting, disconnect cleanup, bounded queues, worker error propagation |
| Harness | Known timestamp fixtures; deterministic trace replay; missing/failed requests retained; stream chunks not assumed to equal model tokens |

Select tolerances before accepting optimized-path changes, using small high-precision references and recorded dtype/backend behavior. Do not continually relax tolerances to make failures pass. Aim for 10 fixed prompts × 100 greedy tokens; distinguish true state/mask bugs from documented near-tie numerical divergence. Exact same-path deterministic resume tests remain especially valuable.

Run fast state tests in CPU CI on each change; run GPU correctness checks at model/backend milestones and before benchmark releases. Profile only after correctness passes. Do not put benchmark speed assertions in general CI.

## 8. Benchmark protocol

### Baselines and attribution

1. HF `generate()` static batches: useful familiar reference; document batch formation, padding, EOS behavior, and queueing policy. Tune a small disclosed batch-size grid.
2. Own engine, 2×2 matrix: contiguous/static, paged/static, contiguous/continuous, paged/continuous. Hold model, dtype, workload, and resource limits fixed. This separates the effects of batching and allocation. If attention implementations differ, disclose the confound and include an oracle-backed small control experiment.
3. vLLM: same GPU, model/tokenizer revision, dtype, maximum context, input trace, output-length policy, and comparable memory budget. Record launch flags and actual available KV capacity. Equal memory-utilization fractions alone do not guarantee equal KV capacity.

Use a controlled feature-matched comparison with prefix caching disabled and optional features disclosed, plus a production-default vLLM run if budget permits. Report CUDA graphs and chunking flags instead of silently weakening the baseline. Avoid claiming equal latency from arbitrary nearby points; compare within a declared latency threshold and label interpolation.

### Workloads

- Debug trace: a few fixed requests, page-boundary lengths, deterministic outputs.
- Short mixed requests: proposed 32–256 prompt tokens and 32–128 output tokens.
- Long-prefill mix: proposed 1,024–4,096 prompt tokens with short outputs; cap to the configured context/budget.
- Trace-derived length distribution: pin a public dataset revision and tokenizer; save sampled lengths, seed, filtering, and trace checksum. Inspect redistribution rights before including raw conversation text; a redistributable fixture and reconstruction script can replace committed text.
- Memory pressure: long enough sequences and concurrency to trigger allocation limits and recomputation.

Synthetic lengths are diagnostic workloads, not evidence of production realism. Prefix-sharing workloads belong to the caching extension.

Generate and save Poisson inter-arrivals in advance. Replay identical arrival offsets and requests for each engine. The client must not wait for one request to finish before issuing the next. Measure intended versus actual send times to detect a saturated load generator. Official vLLM serving benchmarks support request-rate driven arrivals and configurable SLO metrics; use them to cross-check the custom harness. [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve.html)

Start with 5–7 rates selected by a pilot, covering low load through overload. Run at least three independently seeded repetitions, paired across engines. For headline p99 points, target at least 1,000 measured completions per repetition; show sample counts and uncertainty and avoid strong tail claims if budget only permits fewer. Fix warmup and cache-reset policy. Set a fixed offered-load interval, then a bounded drain; retain unfinished requests and report rejection/timeout rates. A rising queue is an unstable overload point, not sustainable throughput.

### Metric definitions

- Client TTFT: first nonempty content event minus actual request send time; separately record scheduled-arrival latency and server queue/prefill time when instrumented.
- ITL: elapsed time between actual token emissions in internal instrumentation. HTTP chunks may contain multiple tokens; name external measurements inter-chunk latency unless true token timing is available.
- TPOT: `(last-token time − first-token time) / (output_tokens − 1)` for outputs longer than one token. This is not a tail-ITL metric.
- End-to-end latency: completion minus actual send time.
- Throughput: output tokens emitted during the declared steady measurement interval divided by that interval; disclose warmup and drain exclusion. Report full-cohort completion metrics separately.
- Goodput: requests completed within predeclared per-request TTFT and TPOT SLOs per measurement second, with an explicit cohort/window policy. Proposed starting SLOs: TTFT ≤ 1 second and TPOT ≤ 100 ms; label them project targets and freeze after the pilot before scored runs.
- Resource behavior: allocated/used KV slots, time-weighted fragmentation, free pages, preemptions, recomputed tokens, peak GPU memory, and queue depth.

For controlled performance runs, request fixed output lengths with a consistent ignore-EOS policy supported by each adapter; label that choice. Test normal EOS generation separately. Sampled tokens and recomputed historical tokens must never inflate output throughput.

### Required figures

1. Output throughput versus p99 TTFT, with rates labeled; distinguish stable and overloaded points. Only nondominated stable points form the sustainable frontier.
2. Goodput versus offered request rate, with timeout/rejection fraction.
3. Paging/batching 2×2 comparison at fixed resources.
4. Supported block size versus internal fragmentation and throughput.
5. One profile explaining the dominant gap to vLLM, including prefill/decode time and CPU scheduling overhead.

An optional watermark sweep of 0%, 5%, and 10% is cheaper than adding swapping and produces a useful scheduler result. Extension experiments require their corresponding implementation first.

Save environment manifest, engine commit, configuration, trace checksum, intended arrivals, per-request CSV, token/chunk event JSONL, aggregate JSON, and figures under a run ID. Record medians and spread across repetitions; do not average percentile values and call the result a pooled percentile.

## 9. Repository plan

```text
PROJECT_PLAN.md
README.md
DESIGN.md
BENCHMARKS.md
pyproject.toml
src/nanoserve/
  config.py, types.py, engine.py, scheduler.py, runner.py, sampling.py
  model/qwen2.py, model/loader.py
  memory/block_manager.py, memory/kv_cache.py
  attention/reference.py, attention/flashinfer.py
  server.py, metrics.py
configs/                    # development, headline, benchmark settings
scripts/                    # environment check, memory math, reproduction, plots
tests/unit/                 # CPU state tests and tiny models
tests/gpu/                  # parity, paging, recompute integration
tests/integration/          # API and harness
bench/                      # trace generation, clients, adapters, sweeps
results/                    # manifests, compact raw results, figures
.github/workflows/          # CPU checks; GPU checks when hardware exists
```

Proposed commands still to implement: `nanoserve generate` and a single benchmark reproduction entrypoint. `nanoserve doctor` and `nanoserve serve` are available. Document expected runtime, downloads, hardware, and output locations. Keep model weights and large profiler binaries outside Git; provide checksums and retrieval instructions for released artifacts.

## 10. Budget and operational plan

Use the library GPU as the primary environment. At the first visit, record its model and VRAM (for NVIDIA, `nvidia-smi`), operating system, driver, Python environment, ability to install packages or use containers, available disk, download access, persistent storage, remote access, session limits, and whether other users share the device. Do not assume that physical availability means exclusive compute access. Save environment details and export results before ending each session.

If it is a compatible NVIDIA GPU with less VRAM, use the 0.5B model for the full pipeline and select a fitting headline model in the same family after measuring memory. If it is non-NVIDIA or software installation is restricted, reassess the attention backend; a rental is a fallback only if the library environment cannot support the project. For comparable benchmark runs, obtain permitted exclusive sessions or record contention and avoid attributing shared-device interference to the engine.

Use the first short GPU session to validate compilation and compatibility. Develop CPU components locally, then batch GPU sessions around parity, backend integration, and benchmarks. A proposed allocation is 10% environment validation, 35% integration, 40% experiments, and 15% reruns.

No rental budget is assumed or authorized. If the library GPU is unsuitable, first establish an acceptable fallback budget. Before choosing a provider, calculate `GPU hours × current hourly rate + storage + transfers`, verify actual availability, and preserve benchmark/rerun funds. Stop idle instances and keep a spending log. Do not keep a public demo GPU running continuously for a resume project; a recording and reproducible launch instructions suffice.

GPU provisioning is outside this planning task. If no GPU budget is available, finish the CPU scheduler/allocator and tiny model, but describe it as a correctness prototype until GPU measurements exist.

## 11. Release, resume, and interview evidence

README opening: what was implemented, a real headline result once available, the architecture figure, main benchmark plot, and an explicit statement that attention kernels come from a library. Follow with a quickstart, hardware requirements, testing instructions, limitations, and benchmark reproduction.

BENCHMARKS.md must explain where the engine loses, which optimization might address it, and the profile supporting that hypothesis. Negative results and unsupported configurations are useful evidence.

Three-minute demo: start the service, submit overlapping requests, show streaming output and changing active sequence/page counts, complete a cancellation, and show a saved throughput/latency comparison. Do not spend core hours on a dashboard.

Resume templates, to fill only after measurement:

- Implemented a single-GPU LLM inference engine in PyTorch with paged KV memory, continuous batching, and recompute preemption; validated model outputs and cache behavior against reference implementations.
- Sustained [X] output tokens/s on [GPU/model] under [latency bound], reaching [Y]% of vLLM throughput under the stated comparison protocol; published reproducible load traces and benchmark results.
- Reduced unused KV allocation by [X]% and increased admitted concurrency from [A] to [B] at fixed memory through [measured paging/block-size change].

Use two strong bullets if the third does not add independent evidence. Maintain a claim-to-run table linking each number to its source result. Do not describe PyTorch/library-backed execution as custom CUDA kernels.

Prepare concise explanations of request-to-first-token flow, page ownership, prefill/decode differences, out-of-memory admission, recomputation costs, watermark behavior, numerical parity, open-loop overload, SLO goodput, and the principal vLLM gap. Connect to prior internship work only using verified experience and actual results.

## 12. First work session

1. Identify the library GPU and environment restrictions; confirm weekly time and target deadline; update the assumptions above.
2. Initialize packaging and CPU tests; record architecture/model decision.
3. Run a small GPU environment/backend spike before committing to cache layout.
4. Generate the model memory table from a pinned config.
5. Create the trace/manifest schema and one tiny benchmark fixture.
6. Begin the model reference path only after the compatibility gate passes.

The original first implementation milestone was Phase 0. Current progress is summarized at the top of this document; no GPU rental or performance result is claimed.
