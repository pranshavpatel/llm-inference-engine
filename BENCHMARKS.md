# Benchmark evidence and limits

There is **no scored sustainable-throughput frontier or SLO-goodput claim yet**.
The saved L40S runs below include a controlled, same-host reference-versus-
FlashInfer diagnostic. They show where the reference engine loses under the
fixed short-prompt workload, but do not establish a universal speedup.

## Paired reference-backend pilots

The [low-rate sweep](environment/phase5-vm-sweep/README.md) used the same L40S,
pinned Qwen2.5-1.5B-Instruct checkpoint, BF16, 64-token context, 16 active
sequences, 4,672 KV-token slots, fixed 16-token outputs, and paired 30-second
arrival traces for nanoserve and vLLM 0.30.0. All 348 requests completed for
each engine with usage present. Median-of-three-run p99 client TTFT for
nanoserve at 0.5/1/2 requests/s was 111/121/173 ms, versus 17/18/17 ms for
vLLM. Each run had only 17–66 requests, so the tail estimates are weak.

The [4/8/16 requests/s follow-up](environment/phase5-vm-sweep-high/README.md)
found a clean 4 requests/s point, but its 32-worker replay client fell behind
at 8 and 16 requests/s. Those higher points do not measure server capacity.
The [5/6/7 requests/s knee pilot](environment/phase5-vm-sweep-knee/README.md)
used 256 client workers and passed its send-lag gate. All 1,673 requests
eventually completed across the nine paired traces, but nanoserve completed
only 73/465, 155/559, and 245/649 by each 30-second offered-load boundary.
Its median-of-three-run p99 client TTFT rose to 1.90/7.36/12.43 seconds;
vLLM remained near 18 ms. Even 5 requests/s was variable, so it is not an
established stable-capacity point. The full-run throughput includes drain and
must not be used as a steady-state score.

These pilots compare a gather-based reference attention path against vLLM's
optimized FlashAttention 2, compiled, and CUDA-graph execution. Prefix
caching and chunked prefill were disabled in vLLM, but the remaining backend
differences are material. The result is evidence of a large observed gap in
these runs, not an isolated causal estimate for any one optimization.

## Profile and optimized-backend gate

The [reference-paged L40S profile](environment/phase6-paged-profile/README.md)
ran eight requests for 16 tokens in one process. It recorded 3,584 attention
softmax calls (8 requests × 16 steps × 28 layers), consistent with per-request,
per-layer gather and small-op dispatch. Its instrumented step timings are not
HTTP latency scores. This profile motivated testing a batched FlashInfer
paged-decode kernel.

The [FlashInfer kernel probe](environment/phase6-flashinfer-probe/README.md)
passed on the target page size and head geometry. A full-model comparison to
BF16 gather did **not** pass; two of 32 top-token decisions differed. The
[targeted float32 attention oracle](environment/PHASE6_FLASHINFER.md) supported
the optimized kernel on the two worst-differing layers, but does not overturn
the full-model result. The [two-request HTTP smoke](environment/phase6-flashinfer-server-smoke/README.md)
then passed with matching text, finish reasons, and token usage. The optimized
decode path remains opt-in and reference prefill remains in place.

## Paired Phase 6 backend diagnostic

The [same-L40S run](environment/phase6-vm-compare/README.md) paired reference
and FlashInfer nanoserve on 36 replays from one checksummed 1/2/4/6/10/16
requests/s plan, with three 60-second repetitions per rate. All client
send-lag gates passed, and all completed requests had 16 output tokens and
usage. At 4 requests/s, median-of-run p50 client TTFT was 168.0 ms for
reference versus 72.9 ms for FlashInfer; 683/711 versus 708/711 requests
finished within the 60-second offered windows. At 6, the reference path
accumulated a queue (826/1,064 finished by 60 seconds), whereas FlashInfer
finished 1,049/1,064 by then with no failures. Both paths overloaded at 16.
The data supports an observed latency/backlog improvement on this pinned
short-prompt workload, not a scored capacity frontier. The long greedy text
differed for all 4,468 requests completed by both paths, so the optimized
backend is not an exact-output replacement for BF16 gather.
The two servers had equal 128 MiB KV pools, but FlashInfer allocated an
additional 128 MiB planning workspace; total GPU memory was not matched.

For a future headline comparison, use longer and mixed-length workloads,
predeclare the goodput/SLO rule, collect enough completions per repetition
for tail estimates, and separate paging/batching ablations. A broader vLLM
comparison must also state its feature and kernel differences. Until those
runs exist, do not put a sustained-throughput or universal speedup number in
the README or resume.
