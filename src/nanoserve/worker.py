"""Single-owner background worker for synchronous inference engines."""

from __future__ import annotations

import copy
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from nanoserve.scheduler import QueueFull
from nanoserve.types import FinishReason, OutputEvent, RequestState


class WorkerClosed(RuntimeError):
    pass


class WorkerQueueFull(QueueFull):
    pass


class TokenCodec(Protocol):
    eos_token_id: Optional[int]

    def encode(self, text: str) -> Sequence[int]: ...

    def decode_tokens(self, token_ids: Sequence[int]) -> str: ...


class HuggingFaceTokenCodec:
    """Small adapter around a Hugging Face tokenizer instance."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, text: str) -> Sequence[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class ByteTokenCodec:
    """Dependency-free byte codec for the random-model server demo."""

    eos_token_id = None

    def encode(self, text: str) -> Sequence[int]:
        return tuple(text.encode("utf-8"))

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 255
            for token_id in token_ids
        ):
            raise ValueError("byte tokens must be integers in [0, 255]")
        return bytes(token_ids).decode("latin-1")


@dataclass
class RequestHandle:
    request_id: str
    prompt_tokens: int
    max_tokens: int
    created_at: float
    _events: queue.Queue[OutputEvent] = field(default_factory=queue.Queue, repr=False)

    def next_event(self, timeout: Optional[float] = None) -> OutputEvent:
        return self._events.get(timeout=timeout)

    def iter_events(self, timeout: Optional[float] = None):
        while True:
            event = self.next_event(timeout)
            yield event
            if event.finished:
                return


@dataclass(frozen=True)
class _Submit:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: Optional[int]
    handle: RequestHandle
    reply: queue.Queue


@dataclass(frozen=True)
class _Cancel:
    request_id: str
    reply: queue.Queue


@dataclass(frozen=True)
class _Stop:
    reply: queue.Queue


class InferenceWorker:
    """Own all engine state and GPU execution on one background thread."""

    def __init__(
        self,
        engine,
        *,
        command_capacity: int = 128,
        idle_wait_seconds: float = 0.001,
        clock=time.time,
    ) -> None:
        if (
            isinstance(command_capacity, bool)
            or not isinstance(command_capacity, int)
            or command_capacity <= 0
        ):
            raise ValueError("command_capacity must be a positive integer")
        if idle_wait_seconds < 0:
            raise ValueError("idle_wait_seconds must be nonnegative")
        self.engine = engine
        self.command_capacity = command_capacity
        self.idle_wait_seconds = idle_wait_seconds
        self._clock = clock
        self._commands: queue.Queue = queue.Queue(maxsize=command_capacity)
        self._handles: dict[str, RequestHandle] = {}
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._running = False
        self._fatal_error: Optional[str] = None
        self._counters = {
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
            "rejected": 0,
        }
        self._stats_cache: dict = {}

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._running and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            if self._thread is not None:
                raise WorkerClosed("a stopped worker cannot be restarted")
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="nanoserve-inference-worker",
                daemon=True,
            )
            self._thread.start()

    def _put_command(self, command, *, admission: bool, deadline: float | None = None) -> None:
        if not self.is_running:
            raise WorkerClosed("inference worker is not running")
        if admission:
            try:
                self._commands.put_nowait(command)
            except queue.Full as error:
                with self._state_lock:
                    self._counters["rejected"] += 1
                raise WorkerQueueFull("worker command queue is full") from error
            return
        while self.is_running:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for worker command capacity")
            try:
                wait = 0.1 if deadline is None else min(0.1, deadline - time.monotonic())
                self._commands.put(command, timeout=max(0, wait))
                return
            except queue.Full:
                continue
        raise WorkerClosed("inference worker stopped before accepting the command")

    def _wait_reply(self, reply: queue.Queue, *, deadline: float | None = None):
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for inference worker reply")
            try:
                wait = 0.1 if deadline is None else min(0.1, deadline - time.monotonic())
                result = reply.get(timeout=max(0, wait))
            except queue.Empty:
                if not self.is_running:
                    raise WorkerClosed("inference worker stopped before replying")
                continue
            if isinstance(result, BaseException):
                raise result
            return result

    def submit(
        self,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        *,
        eos_token_id: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> RequestHandle:
        identifier = request_id or f"cmpl-{uuid.uuid4().hex}"
        prompt = tuple(prompt_token_ids)
        handle = RequestHandle(identifier, len(prompt), max_new_tokens, self._clock())
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._put_command(
            _Submit(identifier, prompt, max_new_tokens, eos_token_id, handle, reply),
            admission=True,
        )
        return self._wait_reply(reply)

    def cancel(self, request_id: str) -> Optional[OutputEvent]:
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._put_command(_Cancel(request_id, reply), admission=False)
        return self._wait_reply(reply)

    def stats(self) -> dict:
        with self._state_lock:
            result = copy.deepcopy(self._stats_cache)
            result["worker"] = {
                **self._counters,
                "running": self._running and self._thread is not None and self._thread.is_alive(),
                "active_handles": len(self._handles),
                "command_queue_depth": self._commands.qsize(),
                "command_queue_capacity": self.command_capacity,
                "fatal_error": self._fatal_error,
            }
            return result

    def stop(self, timeout: float = 5) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if not self.is_running:
            return
        deadline = time.monotonic() + timeout
        reply: queue.Queue = queue.Queue(maxsize=1)
        self._put_command(_Stop(reply), admission=False, deadline=deadline)
        self._wait_reply(reply, deadline=deadline)
        thread = self._thread
        if thread is not None:
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise TimeoutError("inference worker did not stop")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()

    def _refresh_stats(self) -> None:
        stats = self.engine.scheduler.stats()
        with self._state_lock:
            self._stats_cache = stats

    def _deliver(self, event: OutputEvent) -> None:
        handle = self._handles.get(event.request_id)
        if handle is None:
            return
        handle._events.put(event)
        if event.finished:
            del self._handles[event.request_id]
            with self._state_lock:
                if event.finish_reason in (FinishReason.LENGTH, FinishReason.EOS):
                    self._counters["completed"] += 1
                elif event.finish_reason == FinishReason.CANCELLED:
                    self._counters["cancelled"] += 1
                else:
                    self._counters["failed"] += 1

    def _submit(self, command: _Submit) -> None:
        try:
            self.engine.add_request(
                command.request_id,
                command.prompt_token_ids,
                command.max_new_tokens,
                eos_token_id=command.eos_token_id,
            )
            self._handles[command.request_id] = command.handle
            with self._state_lock:
                self._counters["submitted"] += 1
            command.reply.put(command.handle)
        except BaseException as error:
            with self._state_lock:
                self._counters["rejected"] += 1
            command.reply.put(error)

    def _cancel(self, command: _Cancel) -> None:
        try:
            if command.request_id not in self._handles:
                command.reply.put(None)
                return
            event = self.engine.abort(command.request_id)
            if event is not None:
                self._deliver(event)
            command.reply.put(event)
        except BaseException as error:
            command.reply.put(error)

    def _cancel_all(self) -> None:
        for request_id in tuple(self._handles):
            try:
                event = self.engine.abort(request_id)
            except BaseException as error:
                event = OutputEvent(
                    request_id,
                    None,
                    True,
                    FinishReason.ERROR,
                    error=f"{type(error).__name__}: {error}",
                    emitted_at=self._clock(),
                )
            if event is not None:
                self._deliver(event)

    def _deliver_failed_snapshots(self, error: BaseException) -> bool:
        delivered = False
        for request_id in tuple(self._handles):
            snapshot = self.engine.scheduler.snapshot(request_id)
            if snapshot.state != RequestState.FAILED:
                continue
            self._deliver(
                OutputEvent(
                    request_id,
                    None,
                    True,
                    FinishReason.ERROR,
                    error=snapshot.error or f"{type(error).__name__}: {error}",
                    emitted_at=self._clock(),
                )
            )
            delivered = True
        return delivered

    def _run(self) -> None:
        stopping = False
        try:
            self._refresh_stats()
            while not stopping:
                timeout = 0 if self._handles else 0.05
                try:
                    command = self._commands.get(timeout=timeout)
                except queue.Empty:
                    command = None

                processed = 0
                if isinstance(command, _Submit):
                    self._submit(command)
                    processed = 1
                elif isinstance(command, _Cancel):
                    self._cancel(command)
                    processed = 1
                elif isinstance(command, _Stop):
                    self._cancel_all()
                    command.reply.put(True)
                    stopping = True
                    processed = 1

                # Bound ingress work so a continuously refilled queue cannot starve decode.
                while not stopping and processed < self.command_capacity:
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    processed += 1
                    if isinstance(command, _Submit):
                        self._submit(command)
                    elif isinstance(command, _Cancel):
                        self._cancel(command)
                    elif isinstance(command, _Stop):
                        self._cancel_all()
                        command.reply.put(True)
                        stopping = True

                if not stopping and self._handles:
                    try:
                        events = self.engine.step()
                    except BaseException as error:
                        if not self._deliver_failed_snapshots(error):
                            raise
                    else:
                        for event in events:
                            self._deliver(event)
                        if not events and self.idle_wait_seconds:
                            time.sleep(self.idle_wait_seconds)
                self._refresh_stats()
        except BaseException as error:
            with self._state_lock:
                self._fatal_error = f"{type(error).__name__}: {error}"
            self._cancel_all()
        finally:
            with self._state_lock:
                self._running = False
            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    break
                command.reply.put(WorkerClosed("inference worker stopped"))
