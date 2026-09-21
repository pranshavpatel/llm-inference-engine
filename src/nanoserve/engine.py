"""Synchronous correctness engine connecting scheduler and paged model runner."""

from __future__ import annotations

from typing import Sequence

import torch

from nanoserve.scheduler import Scheduler
from nanoserve.types import OutputEvent, StepPlan, WorkItem


class Engine:
    def __init__(self, scheduler: Scheduler, model_runner) -> None:
        if scheduler.cache_manager is not model_runner.cache_manager:
            raise ValueError("scheduler and runner must share one KV cache manager")
        self.scheduler = scheduler
        self.model_runner = model_runner

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
    ) -> str:
        return self.scheduler.add_request(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            eos_token_id=eos_token_id,
        )

    def abort(self, request_id: str) -> OutputEvent | None:
        return self.scheduler.abort(request_id)

    def _execute_group(self, work: tuple[WorkItem, ...]) -> dict[str, int]:
        if not work:
            return {}
        device = self.scheduler.cache_manager.cache.device
        token_batches = [torch.tensor(item.token_ids, dtype=torch.long, device=device) for item in work]
        logits = self.model_runner.forward([item.request_id for item in work], token_batches)
        if len(logits) != len(work):
            raise RuntimeError("runner returned the wrong number of logit batches")
        if any(output.ndim != 2 or output.shape[0] != len(item.token_ids) for item, output in zip(work, logits)):
            raise RuntimeError("runner logits do not match the scheduled token spans")
        return {
            item.request_id: int(output[-1].argmax(dim=-1))
            for item, output in zip(work, logits)
        }

    def step(self) -> tuple[OutputEvent, ...]:
        plan: StepPlan = self.scheduler.schedule()
        if plan.is_empty:
            return ()
        try:
            sampled = self._execute_group(plan.decodes)
            sampled.update(self._execute_group(plan.prefills))
            return self.scheduler.commit(plan, sampled)
        except Exception as error:
            self.scheduler.fail(plan, error)
            raise
