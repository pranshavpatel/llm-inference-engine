"""Decode-first FCFS scheduler with bounded recompute preemption."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from nanoserve.config import positive_int
from nanoserve.memory import PagedKVCacheManager
from nanoserve.types import (
    FinishReason,
    OutputEvent,
    Request,
    RequestSnapshot,
    RequestState,
    StepPlan,
    WorkItem,
)


class QueueFull(RuntimeError):
    pass


class StepInFlight(RuntimeError):
    pass


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_sequences: int = 16
    max_batch_tokens: int = 2048
    max_prefill_tokens: int = 2048
    max_context_tokens: int = 2048
    max_waiting_requests: int = 128

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            positive_int(name, getattr(self, name))
        if self.max_num_sequences > self.max_batch_tokens:
            raise ValueError("max_batch_tokens must fit one decode token per active sequence")
        if self.max_prefill_tokens > self.max_batch_tokens:
            raise ValueError("max_prefill_tokens cannot exceed max_batch_tokens")
        if self.max_context_tokens > self.max_prefill_tokens:
            raise ValueError(
                "max_prefill_tokens must cover max_context_tokens for recompute preemption"
            )


class Scheduler:
    def __init__(
        self,
        cache_manager: PagedKVCacheManager,
        config: SchedulerConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_manager = cache_manager
        self.config = config
        self._clock = clock
        self._requests: dict[str, Request] = {}
        self._arrival = 0
        self._step = 0
        self._in_flight: Optional[StepPlan] = None

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: Optional[int] = None,
    ) -> str:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        prompt = tuple(prompt_token_ids)
        if not prompt or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prompt):
            raise ValueError("prompt_token_ids must contain nonnegative integers")
        positive_int("max_new_tokens", max_new_tokens)
        if len(prompt) + max_new_tokens > self.config.max_context_tokens:
            raise ValueError("request exceeds max_context_tokens")
        block_size = self.cache_manager.blocks.block_size
        prompt_pages = math.ceil(len(prompt) / block_size)
        admission_pages = (
            self.cache_manager.blocks.num_blocks
            - self.cache_manager.blocks.watermark_blocks
        )
        if prompt_pages > admission_pages:
            raise ValueError("prompt cannot fit the physical KV admission capacity")
        # The last sampled token is emitted but never needs a KV slot because
        # the request becomes terminal immediately afterward.
        maximum_kv_tokens = len(prompt) + max_new_tokens - 1
        maximum_pages = math.ceil(maximum_kv_tokens / block_size)
        if maximum_pages > self.cache_manager.blocks.num_blocks:
            raise ValueError("request cannot fit the physical KV pool")
        waiting = sum(request.state == RequestState.WAITING for request in self._requests.values())
        if waiting >= self.config.max_waiting_requests:
            raise QueueFull("waiting request capacity is full")
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            arrival_order=self._arrival,
            submitted_at=self._clock(),
        )
        self._arrival += 1
        self._requests[request_id] = request
        return request_id

    def snapshot(self, request_id: str) -> RequestSnapshot:
        request = self._requests[request_id]
        return RequestSnapshot(
            request_id=request.request_id,
            state=request.state,
            prompt_tokens=len(request.prompt_token_ids),
            generated_token_ids=tuple(request.generated_token_ids),
            computed_tokens=request.computed_tokens,
            preemptions=request.preemptions,
            recomputed_tokens=request.recomputed_tokens,
            finish_reason=request.finish_reason,
            error=request.error,
            submitted_at=request.submitted_at,
            first_scheduled_at=request.first_scheduled_at,
            finished_at=request.finished_at,
        )

    def _active(self) -> list[Request]:
        return [request for request in self._requests.values() if request.state == RequestState.DECODING]

    def _waiting(self) -> list[Request]:
        return sorted(
            (request for request in self._requests.values() if request.state == RequestState.WAITING),
            key=lambda request: request.arrival_order,
        )

    def _extra_pages_for_decode(self, request: Request) -> int:
        required = math.ceil(len(request.history) / self.cache_manager.blocks.block_size)
        return required - len(self.cache_manager.blocks.block_table(request.request_id))

    def _preempt(self, request: Request) -> None:
        previously_computed = request.computed_tokens
        self.cache_manager.free(request.request_id)
        request.state = RequestState.WAITING
        request.computed_tokens = 0
        request.preemptions += 1
        request.pending_recompute_tokens += previously_computed

    def schedule(self) -> StepPlan:
        if self._in_flight is not None:
            raise StepInFlight("commit or fail the current step before scheduling another")
        active = sorted(self._active(), key=lambda request: request.arrival_order)
        free_pages = self.cache_manager.blocks.stats()["free_blocks"]
        page_demand = sum(self._extra_pages_for_decode(request) for request in active)
        preempted: list[str] = []
        while page_demand > free_pages:
            if not active:
                raise RuntimeError("decode page demand cannot be satisfied")
            victim = active.pop()
            demand = self._extra_pages_for_decode(victim)
            freed = len(self.cache_manager.blocks.block_table(victim.request_id))
            self._preempt(victim)
            preempted.append(victim.request_id)
            page_demand -= demand
            free_pages += freed

        decodes: list[WorkItem] = []
        for request in active:
            pending = request.history[request.computed_tokens :]
            if len(pending) != 1:
                raise RuntimeError("decoding requests must have exactly one uncomputed token")
            self.cache_manager.reserve(request.request_id, len(request.history))
            decodes.append(
                WorkItem(request.request_id, pending, request.computed_tokens, is_prefill=False)
            )

        batch_tokens = len(decodes)
        prefill_tokens = 0
        active_count = len(active)
        prefills: list[WorkItem] = []
        excluded = set(preempted)
        for request in self._waiting():
            if request.request_id in excluded:
                continue
            history = request.history
            if active_count >= self.config.max_num_sequences:
                break
            if prefill_tokens + len(history) > self.config.max_prefill_tokens:
                break
            if batch_tokens + len(history) > self.config.max_batch_tokens:
                break
            if not self.cache_manager.blocks.can_allocate(len(history)):
                break
            self.cache_manager.allocate(request.request_id, len(history))
            request.state = RequestState.PREFILL
            request.first_scheduled_at = request.first_scheduled_at or self._clock()
            recomputed = min(request.pending_recompute_tokens, len(history))
            prefills.append(
                WorkItem(request.request_id, history, 0, is_prefill=True, recomputed_tokens=recomputed)
            )
            prefill_tokens += len(history)
            batch_tokens += len(history)
            active_count += 1

        plan = StepPlan(self._step, tuple(prefills), tuple(decodes), tuple(preempted))
        self._step += 1
        if not plan.is_empty:
            self._in_flight = plan
        return plan

    def _require_plan(self, plan: StepPlan) -> None:
        if self._in_flight != plan:
            raise StepInFlight("step plan is not the current in-flight plan")

    def commit(self, plan: StepPlan, sampled_tokens: Mapping[str, int]) -> tuple[OutputEvent, ...]:
        self._require_plan(plan)
        expected = {work.request_id for work in plan.work}
        if set(sampled_tokens) != expected:
            raise ValueError("sampled_tokens must exactly cover the step work")
        # Validate the whole result before mutating request state. If a runner
        # returns malformed output, ``fail`` can still release the untouched
        # plan without partially emitting tokens from earlier work items.
        for work in plan.work:
            token = sampled_tokens[work.request_id]
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("sampled token IDs must be nonnegative integers")
            computed_end = work.start_position + len(work.token_ids)
            if self.cache_manager.written_tokens(work.request_id) != computed_end:
                raise RuntimeError("physical KV commit does not match scheduled work")

        events: list[OutputEvent] = []
        for work in plan.work:
            request = self._requests[work.request_id]
            computed_end = work.start_position + len(work.token_ids)
            request.computed_tokens = computed_end
            if work.recomputed_tokens:
                request.recomputed_tokens += work.recomputed_tokens
                request.pending_recompute_tokens -= work.recomputed_tokens
            token = sampled_tokens[request.request_id]
            request.generated_token_ids.append(token)
            reason = None
            if request.eos_token_id is not None and token == request.eos_token_id:
                reason = FinishReason.EOS
            elif len(request.generated_token_ids) >= request.max_new_tokens:
                reason = FinishReason.LENGTH
            if reason is not None:
                request.state = RequestState.FINISHED
                request.finish_reason = reason
                request.finished_at = self._clock()
                self.cache_manager.free(request.request_id)
            else:
                request.state = RequestState.DECODING
            events.append(
                OutputEvent(
                    request.request_id,
                    token,
                    reason is not None,
                    reason,
                    emitted_at=self._clock(),
                )
            )
        self._in_flight = None
        return tuple(events)

    def fail(self, plan: StepPlan, error: BaseException) -> tuple[OutputEvent, ...]:
        self._require_plan(plan)
        message = f"{type(error).__name__}: {error}"
        events = []
        for work in plan.work:
            request = self._requests[work.request_id]
            self.cache_manager.free(request.request_id)
            request.state = RequestState.FAILED
            request.finish_reason = FinishReason.ERROR
            request.finished_at = self._clock()
            request.error = message
            events.append(
                OutputEvent(
                    request.request_id,
                    None,
                    True,
                    FinishReason.ERROR,
                    error=message,
                    emitted_at=self._clock(),
                )
            )
        self._in_flight = None
        return tuple(events)

    def abort(self, request_id: str) -> Optional[OutputEvent]:
        request = self._requests[request_id]
        if request.is_terminal:
            return None
        if self._in_flight is not None and any(
            work.request_id == request_id for work in self._in_flight.work
        ):
            raise StepInFlight("cannot abort a request while its GPU step is in flight")
        if request.state == RequestState.DECODING:
            self.cache_manager.free(request_id)
        request.state = RequestState.CANCELLED
        request.finish_reason = FinishReason.CANCELLED
        request.finished_at = self._clock()
        return OutputEvent(
            request_id,
            None,
            True,
            FinishReason.CANCELLED,
            emitted_at=self._clock(),
        )

    def stats(self) -> dict:
        state_counts = {
            state.value: sum(request.state == state for request in self._requests.values())
            for state in RequestState
        }
        return {
            "requests": len(self._requests),
            "states": state_counts,
            "preemptions": sum(request.preemptions for request in self._requests.values()),
            "recomputed_tokens": sum(
                request.recomputed_tokens for request in self._requests.values()
            ),
            "step_in_flight": self._in_flight is not None,
            "cache": self.cache_manager.stats(),
        }
