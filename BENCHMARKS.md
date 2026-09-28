# Benchmark evidence and limits

There is **no scored release benchmark or FlashInfer speedup claim yet**. The
saved L40S runs below are exploratory load and operator diagnostics. They show
where the reference engine loses and motivate the next measurement, but do not
establish a sustainable-throughput frontier or an SLO goodput comparison.

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
decode path remains opt-in and reference prefill remains in place. No
FlashInfer-versus-reference latency or throughput comparison has been run.

## Remaining release measurement

Use a later, isolated same-L40S session to compare reference nanoserve and
opt-in FlashInfer nanoserve under identical pinned model, trace, KV budget,
output policy, and client protocol. Capture warmup, environment, raw replays,
server logs, client send lag, error/usage checks, and a predeclared latency or
goodput acceptance rule. Only then report an optimized-path speed comparison.
A broader vLLM comparison must separately state its feature and kernel
differences. Until those runs exist, no headline throughput or speedup number
belongs in the README or resume.
