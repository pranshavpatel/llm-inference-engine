"""Core request, scheduling, and output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FinishReason(str, Enum):
    LENGTH = "length"
    EOS = "eos"
    CANCELLED = "cancelled"
    ERROR = "error"


TERMINAL_STATES = {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED}


@dataclass
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_order: int
    submitted_at: float
    eos_token_id: Optional[int] = None
    state: RequestState = RequestState.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    computed_tokens: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    pending_recompute_tokens: int = 0
    first_scheduled_at: Optional[float] = None
    finished_at: Optional[float] = None
    finish_reason: Optional[FinishReason] = None
    error: Optional[str] = None

    @property
    def history(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.generated_token_ids)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    token_ids: tuple[int, ...]
    start_position: int
    is_prefill: bool
    recomputed_tokens: int = 0


@dataclass(frozen=True)
class StepPlan:
    step_id: int
    prefills: tuple[WorkItem, ...]
    decodes: tuple[WorkItem, ...]
    preempted_request_ids: tuple[str, ...] = ()

    @property
    def work(self) -> tuple[WorkItem, ...]:
        return self.prefills + self.decodes

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


@dataclass(frozen=True)
class OutputEvent:
    request_id: str
    token_id: Optional[int]
    finished: bool
    finish_reason: Optional[FinishReason] = None
    error: Optional[str] = None
    emitted_at: Optional[float] = None


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    state: RequestState
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    computed_tokens: int
    preemptions: int
    recomputed_tokens: int
    finish_reason: Optional[FinishReason]
    error: Optional[str]
    submitted_at: float
    first_scheduled_at: Optional[float]
    finished_at: Optional[float]
