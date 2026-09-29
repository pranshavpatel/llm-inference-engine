# Phase 5 high-rate L40S pilot

The original VM files are retained byte-for-byte in `source.zip`; its SHA-256 is
`9829849de5c75810ca4259df34cfcbd589bb48b3973b9f8f7f1d0202d798e6b7`.
The plan and all 18 replay files inside it passed `report-sweep` validation;
the small generated `report/` is also unpacked here for browsing. All
rates used the same model/launch policy as the lower-rate pilot: BF16,
64-token context, 16 active sequences, 4,672 available KV-token slots, and
16-token ignore-EOS outputs. Each engine completed its 2/2 warmup.

| Offered rate | nanoserve | vLLM | Interpretation |
| ---: | ---: | ---: | --- |
| 4 requests/s | 362/362 complete; median run p99 client TTFT 349 ms | 362/362; 18 ms | Client send lag <0.5 ms; usable pilot point. |
| 8 requests/s | 705/734 complete; median run p99 TTFT 3.53 s | 734/734; 18 ms | nanoserve's p99 send lag reached 11.7–18.7 s; not a clean server-capacity point. |
| 16 requests/s | 720/1478 complete; median run p99 TTFT 3.62 s | 1478/1478; 19 ms | nanoserve had 96 timed-out and 662 not-sent requests, with 30.6–34.3 s p99 send lag; heavily client-limited. |

The bounded replay driver used only 32 HTTP workers. Under slow nanoserve
responses, those workers remained occupied and queued arrivals were sent
late or not at all. The 8 and 16 requests/s nanoserve rows therefore mix
server backlog with load-generator backpressure and must **not** be used as
server-only sustainable-capacity measurements. vLLM had no failures and
sub-millisecond p99 send lag in all nine high-rate runs, so it also did not
reach a capacity limit here. The higher vLLM delivered-token rate reflects
the higher offered rates, not a measured maximum.

The narrow next experiment is to bracket the nanoserve knee between 4 and 8
requests/s with enough client concurrency to keep intended and actual send
times close. The driver must report send lag and unfinished work separately;
if its send-lag gate fails, discard that rate as a server-capacity point.
No current pilot establishes a full sustainable-throughput frontier or
cross-engine SLO goodput.
