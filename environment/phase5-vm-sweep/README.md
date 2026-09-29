# Phase 5 paired L40S load pilot

The raw files came from the Ubuntu L40S VM archive with SHA-256
`ddc0dd06c4c4f74e8f18546acea34c0a8d80f316748caa400cbe61fc5f3c354e`.
The saved `report-sweep` validates all nine plan traces and both engines'
replays. This is a load-harness pilot, not a scored capacity result.

The plan pins Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the vLLM log confirms its
loaded model/tokenizer revisions. The nanoserve readiness log records the
model name and checkpoint filename, but not the snapshot path, so that
revision relies on the VM runbook setup. Both engines used BF16, a 64-token
context limit, 16 active sequences, and 4,672 available KV-token slots.
vLLM had prefix caching and chunked prefill disabled, with FlashAttention and
compiled/CUDA-graph execution enabled. nanoserve used its reference paged-
gather attention backend. The nine paired 30-second traces used 16-token
ignore-EOS outputs, rates 0.5/1/2 requests/s, and three seeds per rate.
Both warmups completed 2/2 requests.

Each engine completed all 348 requests; there were zero failed, timed-out,
not-sent, or missing-usage records. Every completed record reported 16 output
tokens and `length` finish. The largest per-run p99 client send lag was under
1.8 ms. The reported median-of-run p99 client TTFT values were:

| Offered requests/s | nanoserve | vLLM |
| ---: | ---: | ---: |
| 0.5 | 111 ms | 17 ms |
| 1 | 121 ms | 18 ms |
| 2 | 173 ms | 17 ms |

These p99 estimates have only 17–66 requests per run, so they are pilot
observations, not strong tail-latency claims. Full-run delivered output rate
tracked the offered load for both engines. That rate includes the drain and
is not steady-state throughput. vLLM did not report server TPOT in this run,
so a cross-engine TPOT/SLO goodput comparison is unavailable. No saturation
point was observed by 2 requests/s. The next necessary measurement is a
paired higher-rate sweep under the same settings, followed by a declared
latency/failure acceptance rule before identifying sustainable capacity.

The [VM runbook](../PHASE5_VM_SWEEP.md) records the commands. The raw plan,
replays, warmups, server logs, and generated report are retained here for
reproduction. The existing two-request greedy-text difference remains a
documented correctness limitation, not an explanation for these timing gaps.
