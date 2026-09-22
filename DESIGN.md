# Current implementation contracts

## Qwen2 reference execution

The Phase 1 model is a correctness implementation for Qwen2.5. It owns the transformer forward pass and never calls Hugging Face model execution or `generate()` in engine code. Hugging Face is used only by tests as an external oracle.

Each decoder block performs pre-normalized grouped-query self-attention followed by a pre-normalized SwiGLU MLP. Q, K, and V projections have biases; the attention output and all MLP projections do not. RMSNorm accumulates variance in FP32. RoPE uses absolute position IDs, including the past-cache offset during incremental decoding. KV heads are repeated in contiguous query-head groups.

The attention mask is explicitly offset-aware: a query written after `past_length` may attend through its own absolute sequence position but not future positions. A supplied two-dimensional padding mask covers the complete key length. Softmax is evaluated in FP32 and converted back to the query dtype.

The contiguous cache is an immutable tuple of `(key, value)` pairs, one per layer. Each tensor has shape `[batch, kv_heads, cached_tokens, head_dim]`. A cached call concatenates new K/V tensors and returns a new tuple. This intentionally favors clarity over allocation efficiency. Phase 1 generation accepts exactly one request and uses greedy `argmax` sampling.

## Configuration and weight loading

`Qwen2Config` accepts Hugging Face JSON fields needed by the implementation and ignores unrelated metadata. It validates positive dimensions, query/KV head divisibility, and the relationship between hidden size, query heads, and head dimension.

The loader reads safetensors directly, including Hugging Face shard indexes, and copies one shard at a time rather than materializing a second complete state dictionary. Before copying, it requires exact model/checkpoint name coverage and validates every tensor shape. Derived non-persistent RoPE frequencies are not checkpoint weights. For tied embeddings, an omitted `lm_head.weight` is valid only when the language-model head and input embedding are actually the same parameter. Untied configurations require both tensors.

## Numerical gates

Tiny deterministic CPU tests use FP32 and declare `rtol=1e-5, atol=2e-6` for cached/full and Hugging Face teacher-forced comparisons. Greedy tokens must match exactly.

Real-model FP32 teacher-forced logits are the strict architecture gate. With pinned Qwen2.5-1.5B weights, the four-prompt suite recorded zero error against Hugging Face eager attention. Different query shapes select different matrix-multiplication paths, so real-model cached/full FP32 comparison uses the measured envelope `max_abs <= 1.17e-4`; the largest recorded mean absolute error was `8.50e-6`.

BF16 is the selected deployment dtype because this GPU reports native support, all inspected logits are finite, and 32/32 fixed greedy tokens match the FP32-validated Hugging Face loop. BF16 reductions are shape- and layout-sensitive: the recorded teacher-forced custom/reference maximum absolute error is `0.59375`, while cached/full maximum absolute error is `1.5625`. These bounds are recorded observations, not a license to increase tolerances automatically. Any greedy divergence must record the first divergent position and both top-two margins. FP16 is rejected on this stack because both implementations produced non-finite real-model logits.

## Phase 0 backend decision

The preferred optimized backend remains FlashInfer `0.6.18.post1`, but its published wheels require Linux. The current host is native Windows without WSL or Docker. Building the architecture around an unexercised FlashInfer API would therefore violate the compatibility gate.

The Phase 0 fallback is a standalone PyTorch SDPA smoke test that scatters contiguous KV tensors into noncontiguous physical pages, gathers them through a page table, and checks prefill and decode outputs against the original tensors. Page sizes 1, 16, 32, and 64 passed in FP16 and BF16 for the Qwen2.5-1.5B attention geometry. It remains a feasibility record and has no performance claim.

## Physical paged KV

The physical layout is `[layer, physical_page, offset, kv_head, head_dim]`, with separate preallocated key and value tensors. Sequence-facing tensors use `[kv_head, token, head_dim]`. Page-table scatter and gather validate page ownership ranges, device, dtype, shape, and capacity before mutation. Advanced-indexed writes use indexed assignment because `.copy_()` on a PyTorch advanced-indexing result would modify a temporary tensor.

`PagedKVCacheManager` composes the physical pool with `BlockManager`. Allocation reserves slots but begins with zero completed KV tokens. An append creates a plan with fixed start/end positions; every layer must write exactly that range before commit advances the completed-token counter. Failure aborts the plan without exposing partial pages through a committed length. Released pages are zeroed before ownership returns to the allocator.

Generic `PagedBatchMetadata` contains block tables, committed sequence lengths, and query lengths. Block tables may be longer than the committed sequence requires because the allocator can reserve future decode capacity. Backends may read only the committed sequence length.

`ReferencePagedAttention` is the slow correctness oracle. It gathers each sequence, repeats KV heads for GQA, applies an offset-aware causal mask, evaluates softmax in FP32, and returns padded static-batch output. `PagedQwen2Runner` performs Qwen2 projection and RoPE, writes each layer through append plans, and supports variable-length static prefill and incremental decode. It is not a scheduler and does not dynamically admit work during a step.

## Page ownership

A physical page is either free or belongs to exactly one request. There is no prefix sharing. Page tables map logical pages to physical IDs; position `p` maps through `table[p // block_size]` at offset `p % block_size`.

`allocate(id, tokens)` reserves a positive total KV slot count for a new request. `reserve(id, tokens)` grows an existing reservation, including growth within the current last page. Requests cannot shrink in place. `free(id)` releases all ownership once. Allocation and growth check capacity before mutating state, so expected resource-exhaustion errors are transactional.

Admission leaves `ceil(num_blocks * watermark)` free pages; growth may consume all remaining free pages. A very small pool may therefore admit nothing at a nonzero watermark. Watermarks reduce admission capacity; they do not guarantee freedom from preemption or starvation.

`check_invariants()` verifies the complete partition of physical page IDs, balanced request metadata, and the expected page count per reservation. It is a test/debug audit, not a production scheduling operation. The allocator is not thread safe; the planned engine has one state owner.

## Continuous scheduler

Request state transitions are `WAITING -> PREFILL -> DECODING -> FINISHED`, with `CANCELLED` and `FAILED` as terminal alternatives. A sampled token is appended to request history immediately but gains KV only if the request is selected in a later decode step. The final sampled token therefore never needs a cache slot.

Each scheduling iteration first computes page demand for every existing decode request. If growth would exceed free pages, the newest active request is preempted until the remaining decodes fit. Preemption discards physical KV, retains token history, and returns the request to the FCFS waiting queue. The preempted request is excluded from admission during the same iteration so releasing it always enables forward progress rather than immediate churn.

Decode reservations happen before admission. Waiting requests are admitted in arrival order only when sequence, total-token, prefill-token, watermark, and physical capacity permit a complete prefill. Chunked prefill is not implemented. Decode and prefill work use separate runner calls within an iteration; this keeps the current correctness path simple but adds launch and scheduling overhead.

The scheduler uses a plan/execute/commit boundary. Only one plan may be in flight. A commit first validates every sampled token and every physical KV end position, then updates request histories and emits events. A runner exception fails all selected work and releases its ownership. Cancellation is immediate between synchronous steps; cancellation of work already in flight is rejected until that step resolves.

Preemption reconstruction runs the complete prompt plus already generated history at positions starting from zero. Its sampled result is the next new output; historical tokens are never re-emitted. Counters distinguish preemption count from recomputed KV tokens. Request snapshots and output events expose monotonic timestamps for queue, first-schedule, emission, and completion measurement.

## Worker and HTTP ownership

`InferenceWorker` runs the synchronous engine on one dedicated background thread. HTTP threads never access scheduler, allocator, model, or GPU state directly. Submissions and cancellation use a bounded command queue; each accepted request owns an event queue consumed by exactly one response path. The scheduler remains the second admission bound for waiting requests.

A cancellation received during model execution cannot interrupt that tensor operation. It is processed immediately after the step commits, so a token may have been produced internally before cancellation becomes terminal. The request event stream then receives one cancellation event and its pages are released. Stopping the worker applies the same cleanup to every active handle.

Runner exceptions already transition selected scheduler requests to `FAILED`. The worker inspects those snapshots, forwards terminal error events to the corresponding consumers, and continues serving unaffected queued requests. An exception that does not produce any failed scheduler state is treated as a fatal worker invariant failure.

The standard-library HTTP server intentionally implements a narrow completions contract. Non-streaming handlers collect request-local events; streaming handlers translate each event into one SSE record and cancel on a broken connection. EOS maps to the OpenAI-style `stop` finish reason while length remains `length`. Health is process liveness, readiness requires a running inference worker, and metrics expose scheduler, cache, and worker counters as JSON rather than claiming Prometheus compatibility.

`serve-demo` uses a seeded random tiny model and a byte codec so transport tests require no checkpoint download. It is not a language-quality or performance demonstration. `serve` constructs the same worker/service objects from a local Qwen2 config, tokenizer, and coverage-checked safetensors checkpoint. It chooses the physical page count from the configured KV byte budget and rejects a pool that cannot admit a maximum-context request. Startup opens HTTP ingress only after model and KV allocation succeed.

The HTTP server decodes the entire generated token prefix at each output event. This matters for byte-level tokenizers, where individual token pieces can be invalid UTF-8. Streaming withholds an incomplete replacement character until later tokens complete it; non-streaming decodes the full output once. EOS is excluded from decoded text and maps to the `stop` finish reason.

## Open-loop trace replay

Completion traces contain prompt text, maximum output tokens, intended arrival offsets, model ID, revision, and a checksum. The replay client schedules submissions against one monotonic start time, independent of request completion. The request record keeps intended and actual submission times, first nonempty content, completion, finish reason, stream chunks, output text, usage when returned, and failures. Chunk count is not treated as token count. A final usage-only SSE record is requested from HTTP endpoints through `stream_options.include_usage`; nanoserve emits it after terminal success.

The HTTP adapter targets either nanoserve or a vLLM OpenAI-compatible completions endpoint. The local Hugging Face adapter loads the same saved checkpoint and serializes greedy requests with a cached model forward. Its queueing and batch formation differ from continuous nanoserve scheduling and must be disclosed in any performance comparison. The two-request debug trace validates record completeness and matching short outputs between nanoserve and Hugging Face, but does not establish a throughput or latency result.

## Accounting and arrival plans

Reserved slots track space promised for KV writes. Completed KV tokens advance only after every layer finishes an append transaction. Tail fragmentation uses reserved tokens and excludes wholly free pages. Pool counters report physical bytes, bytes per block, completed tokens, active blocks, and pending append count; they are accounting values, not performance measurements.

The trace builder uses an isolated seeded random generator. Arrival offsets are cumulative exponential draws starting after time zero. The checksum covers canonical JSON of the payload excluding the checksum itself. The trace is synthetic and contains lengths, not prompt text or token IDs.

## Deferred work

An optimized Linux backend, chat completions, an actual vLLM trace replay, and controlled benchmarks remain unimplemented. No throughput, latency, concurrency, or memory-saving result is claimed. FlashInfer agreement on the exact target geometry remains a deferred Phase 2 optimization gate; the validated gather backend is the current scheduler and HTTP correctness path.

Real-model FP32 static-batch paged execution matched all 32 greedy tokens and stayed within `1.33e-4` maximum prefill logit error versus individual contiguous forwards. BF16 matched 31/32 tokens; the first divergence had an exactly tied contiguous top-two score and a `0.125` paged margin. FP32 remains the architecture oracle, and the BF16 divergence is a recorded numerical limitation.
