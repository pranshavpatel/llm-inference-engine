# Current implementation contracts

## Page ownership

A physical page is either free or belongs to exactly one request. There is no prefix sharing in this version. Page tables map logical pages to physical IDs; position `p` maps through `table[p // block_size]` at offset `p % block_size`.

`allocate(id, tokens)` reserves a positive total KV slot count for a new request. `reserve(id, tokens)` grows an existing reservation, including growth within the current last page. Requests cannot shrink in place. `free(id)` releases all ownership once. Allocation and growth check capacity before mutating state, so expected resource-exhaustion errors are transactional.

Admission leaves `ceil(num_blocks * watermark)` free pages; growth may consume all remaining free pages. A very small pool may therefore admit nothing at a nonzero watermark: use zero explicitly for tiny tests. Watermarks reduce admission capacity; they do not guarantee freedom from preemption or starvation.

`check_invariants()` verifies the complete partition of physical page IDs, balanced request metadata, and the expected page count per reservation. It is an expensive debug/test audit, not a production scheduling operation. This allocator is not thread safe; the planned engine has one state owner.

## Accounting

Reserved slots track space promised for KV writes. They do not imply that a model has performed those writes. Tail fragmentation uses reserved tokens and excludes wholly free pages. Physical tensor allocation, completed-token accounting, and GPU memory measurements will be added with the model runner.

## Arrival plans

The trace builder uses an isolated seeded random generator. Arrival offsets are cumulative exponential draws starting after time zero. The checksum covers canonical JSON of the payload excluding the checksum itself. Identical parameters reproduce the plan in the same recorded Python environment. The current trace is explicitly synthetic and contains no real conversation data.

## Deferred gates

Hardware/backend execution has not been validated. No kernel package versions are pinned until a compatible library GPU environment is established. No model parity, scheduler throughput, or API compatibility is claimed by this foundation.
