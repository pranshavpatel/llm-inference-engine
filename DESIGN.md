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

The Phase 0 fallback is a standalone PyTorch SDPA smoke test that scatters contiguous KV tensors into noncontiguous physical pages, gathers them through a page table, and checks prefill and decode outputs against the original tensors. Page sizes 1, 16, 32, and 64 passed in FP16 and BF16 for the Qwen2.5-1.5B attention geometry. This establishes layout feasibility only: the fallback is not wired into the model, is not optimized, and has no performance claim.

## Page ownership

A physical page is either free or belongs to exactly one request. There is no prefix sharing. Page tables map logical pages to physical IDs; position `p` maps through `table[p // block_size]` at offset `p % block_size`.

`allocate(id, tokens)` reserves a positive total KV slot count for a new request. `reserve(id, tokens)` grows an existing reservation, including growth within the current last page. Requests cannot shrink in place. `free(id)` releases all ownership once. Allocation and growth check capacity before mutating state, so expected resource-exhaustion errors are transactional.

Admission leaves `ceil(num_blocks * watermark)` free pages; growth may consume all remaining free pages. A very small pool may therefore admit nothing at a nonzero watermark. Watermarks reduce admission capacity; they do not guarantee freedom from preemption or starvation.

`check_invariants()` verifies the complete partition of physical page IDs, balanced request metadata, and the expected page count per reservation. It is a test/debug audit, not a production scheduling operation. The allocator is not thread safe; the planned engine has one state owner.

## Accounting and arrival plans

Reserved slots track space promised for KV writes. They do not imply that a model has performed those writes. Tail fragmentation uses reserved tokens and excludes wholly free pages. Physical tensor allocation and completed-token accounting remain Phase 2 work.

The trace builder uses an isolated seeded random generator. Arrival offsets are cumulative exponential draws starting after time zero. The checksum covers canonical JSON of the payload excluding the checksum itself. The trace is synthetic and contains lengths, not prompt text or token IDs.

## Deferred work

Physical GPU KV pages, a reusable paged-attention interface, an optimized Linux backend, static or continuous batching, scheduling, serving, and benchmarks remain unimplemented. No throughput, latency, concurrency, or memory-saving result is claimed. The next gate is agreement among contiguous attention, a gather-based paged oracle, and FlashInfer on the exact target geometry before connecting scheduling or serving.
