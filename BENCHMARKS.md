# Benchmark evidence and limits

There is now a scored fixed-window SLO-goodput sweep on a pinned L40S and
synthetic short/long workloads. It is **not** a production-workload result,
universal speedup, or strong p99 sustainable-throughput frontier. The older
Phase 5 and Phase 6 runs below remain diagnostics. The
[predeclared scored protocol](environment/PHASE6_VM_SCORED.md) froze its SLOs,
measurement window, quality gates, and resource samples before collection.

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

| Published observation | Saved evidence |
| --- | --- |
| 4 requests/s median-of-run p50 client TTFT: 168.0 versus 72.9 ms | [`phase6-vm-compare/summary.json`](environment/phase6-vm-compare/summary.json), `by_rate` entries for 4 requests/s; original trace and per-request replay files in its `source.zip` |
| 6 requests/s completions by the 60-second boundary: 826 versus 1,049 of 1,064 | The same summary's 6 requests/s `completed_in_offered_window` entries |
| 4,468/4,468 paired completed 16-token strings differ | The same summary's `paired_both_completed` and `paired_different_output_text` fields, regenerated from `source.zip` |
| Targeted worst-layer float32 attention oracle: zero FlashInfer elements outside the declared tolerance | [`phase6-attention-oracle.json`](environment/phase6-attention-oracle.json) and [method notes](environment/PHASE6_FLASHINFER.md) |

## Fixed-window Phase 6 score

The [fresh L40S scored archive](environment/phase6-vm-scored/README.md)
contains 75 independently auditable replays: reference and FlashInfer on
identical short (40/131/248 input, 64 fixed output tokens) and long-prefill
(1,028/2,055/3,082 input, 32 fixed output tokens) synthetic prompt plans,
plus a 2x2 supported block-size/batching-limit ablation. Each rate had three
paired 60-second offered windows and a 30-second bounded drain. All scores
recomputed from the source zip; every client send-lag gate passed, with no
missing usage, TPOT, or within-window emitted-token count. The reference
and FlashInfer servers had the same 256 MiB KV pool; the FlashInfer planning
workspace is an additional allocation.

The project SLOs were TTFT <= 1 s and TPOT <= 100 ms. At 2 offered
requests/s on the short workload, the median run emitted 75.4 tokens/s on
reference and 136.6 on FlashInfer. Median SLO goodput was 0.00 versus
2.12 requests/s. The reference path failed 92 of 387 requests across
repetitions and built a queue; FlashInfer failed none. This is evidence of
a higher observed capacity under the pinned workload, not a 1.81x
low-load speedup. FlashInfer's short-workload goodput peaked at 2.60
requests/s at 3 offered requests/s, but the median per-run p99 client TTFT
was 1.77 s. At 4 requests/s its sampled queue grew and p99 reached 11.62 s;
at 6, 210/1,095 offered requests failed. Those points are not sustainable.

The long-prefill 1.6 requests/s point gave median SLO goodput of 0.083
reference versus 0.917 FlashInfer requests/s, but both had p99 TTFT far
above 1 s (17.40 and 3.55 s). Low-rate runs had small per-repetition counts,
so the published p99 plots are descriptive, not strong tail estimates. The
ablation at 2 requests/s showed that raising the batch cap from 1 to 16
lifted reference emitted throughput from about 32 to 75–76 tokens/s;
doubling block size from 16 to 32 at batch cap 16 raised time-weighted
fragmentation from 4.41% to 8.69% with little throughput change. All four
ablation points were overloaded, and the factors were block size and batching
limit—not paging-on versus paging-off.

| Claim or figure | Saved run |
| --- | --- |
| All short/long per-rate counts, SLO goodput, emitted-token throughput, sample counts, and resource behavior | [Scored summary](environment/phase6-vm-scored/report/summary.json), [per-run CSV](environment/phase6-vm-scored/report/runs.csv), and [source zip](environment/phase6-vm-scored/source.zip) |
| Output throughput versus p99 TTFT | [Short-workload plot](environment/phase6-vm-scored/report/throughput-vs-ttft.svg) |
| Goodput versus offered rate and failures | [Goodput plot](environment/phase6-vm-scored/report/goodput-vs-rate.svg) and [failure plot](environment/phase6-vm-scored/report/failure-vs-rate.svg) |
| Supported 2x2 and block-size tradeoff | [Ablation figure](environment/phase6-vm-scored/report/ablation-2x2.svg) and [block-size figure](environment/phase6-vm-scored/report/block-size.svg) |
| Reference-path per-request attention dispatch bottleneck | [Paged runner profile](environment/phase6-paged-profile/README.md) |

The measured gap to vLLM still requires a feature-matched same-session
baseline on these exact traces; the earlier Phase 5 comparison used a
different workload and VM software stack. A 1,000-completion-per-repetition
headline p99 and a public trace-derived workload also remain future work.
Do not present these synthetic-workload numbers as production performance
or exact numerical parity between attention backends.
