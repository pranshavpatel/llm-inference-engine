import queue
import threading
import time
import unittest

import torch

from nanoserve.engine import Engine
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
from nanoserve.scheduler import Scheduler, SchedulerConfig
from nanoserve.types import FinishReason, OutputEvent, RequestState
from nanoserve.worker import InferenceWorker, RequestHandle, WorkerClosed, WorkerQueueFull, _Submit


def make_manager():
    blocks = BlockManager(16, block_size=2, watermark=0)
    cache = PagedKVCache(KVCacheSpec(1, 16, 2, 1, 2), dtype=torch.float32)
    return PagedKVCacheManager(blocks, cache)


def make_scheduler(manager):
    return Scheduler(
        manager,
        SchedulerConfig(
            max_num_sequences=4,
            max_batch_tokens=16,
            max_prefill_tokens=16,
            max_context_tokens=16,
            max_waiting_requests=8,
        ),
    )


class RecordingRunner:
    def __init__(self, cache_manager, *, fail=False, entered=None, release=None):
        self.cache_manager = cache_manager
        self.fail = fail
        self.entered = entered
        self.release = release
        self.thread_ids = []

    def forward(self, request_ids, token_batches):
        self.thread_ids.append(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(2)
        if self.fail:
            raise RuntimeError("injected worker failure")
        outputs = []
        spec = self.cache_manager.cache.spec
        for request_id, tokens in zip(request_ids, token_batches):
            count = int(tokens.numel())
            shape = (spec.num_key_value_heads, count, spec.head_dim)
            keys = [
                torch.zeros(
                    shape,
                    dtype=self.cache_manager.cache.dtype,
                    device=self.cache_manager.cache.device,
                )
                for _ in range(spec.num_layers)
            ]
            self.cache_manager.append(request_id, keys, [key.clone() for key in keys])
            logits = torch.zeros((count, 64), dtype=torch.float32)
            for row, token in enumerate(tokens.tolist()):
                logits[row, (token + 1) % 64] = 1
            outputs.append(logits)
        return tuple(outputs)


class WorkerTests(unittest.TestCase):
    def test_worker_completes_overlapping_requests_on_one_owner_thread(self):
        manager = make_manager()
        runner = RecordingRunner(manager)
        engine = Engine(make_scheduler(manager), runner)
        main_thread = threading.get_ident()

        with InferenceWorker(engine) as worker:
            first = worker.submit([1, 2], 3, request_id="first")
            second = worker.submit([10], 2, request_id="second")
            first_events = list(first.iter_events(timeout=2))
            second_events = list(second.iter_events(timeout=2))
            stats = worker.stats()

        self.assertEqual([event.token_id for event in first_events], [3, 4, 5])
        self.assertEqual([event.token_id for event in second_events], [11, 12])
        self.assertEqual(len(set(runner.thread_ids)), 1)
        self.assertNotEqual(runner.thread_ids[0], main_thread)
        self.assertEqual(stats["worker"]["completed"], 2)
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)

    def test_cancel_waits_for_current_step_then_releases_request(self):
        manager = make_manager()
        entered = threading.Event()
        release = threading.Event()
        runner = RecordingRunner(manager, entered=entered, release=release)
        scheduler = make_scheduler(manager)
        worker = InferenceWorker(Engine(scheduler, runner))
        worker.start()
        handle = worker.submit([1], 8, request_id="cancel-me")
        self.assertTrue(entered.wait(2))

        result = []
        cancel_thread = threading.Thread(target=lambda: result.append(worker.cancel("cancel-me")))
        cancel_thread.start()
        time.sleep(0.02)
        self.assertTrue(cancel_thread.is_alive())
        release.set()
        cancel_thread.join(2)

        events = list(handle.iter_events(timeout=2))
        worker.stop()
        self.assertEqual(events[-1].finish_reason, FinishReason.CANCELLED)
        self.assertEqual(scheduler.snapshot("cancel-me").state, RequestState.CANCELLED)
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)
        self.assertEqual(result[0].finish_reason, FinishReason.CANCELLED)

    def test_runner_failure_reaches_request_stream_and_worker_survives(self):
        manager = make_manager()
        runner = RecordingRunner(manager, fail=True)
        scheduler = make_scheduler(manager)

        with InferenceWorker(Engine(scheduler, runner)) as worker:
            handle = worker.submit([1, 2], 2, request_id="broken")
            event = handle.next_event(timeout=2)
            stats = worker.stats()

        self.assertTrue(event.finished)
        self.assertEqual(event.finish_reason, FinishReason.ERROR)
        self.assertIn("injected worker failure", event.error)
        self.assertTrue(stats["worker"]["running"])
        self.assertEqual(stats["worker"]["failed"], 1)

    def test_worker_rejects_calls_outside_lifecycle(self):
        manager = make_manager()
        worker = InferenceWorker(Engine(make_scheduler(manager), RecordingRunner(manager)))
        with self.assertRaises(WorkerClosed):
            worker.submit([1], 1)
        worker.start()
        worker.stop()
        with self.assertRaises(WorkerClosed):
            worker.submit([1], 1)
        with self.assertRaises(WorkerClosed):
            worker.start()

    def test_command_ingress_is_bounded_under_concurrent_submitters(self):
        entered = threading.Event()
        release = threading.Event()

        class StatsOnlyScheduler:
            def stats(self):
                return {"states": {}}

        class BlockingEngine:
            def __init__(self):
                self.scheduler = StatsOnlyScheduler()
                self.requests = []
                self.first = True

            def add_request(self, request_id, prompt, max_tokens, *, eos_token_id=None):
                if self.first:
                    self.first = False
                    entered.set()
                    release.wait(2)
                self.requests.append(request_id)

            def step(self):
                requests, self.requests = self.requests, []
                return tuple(
                    OutputEvent(request_id, 1, True, FinishReason.LENGTH)
                    for request_id in requests
                )

            def abort(self, request_id):
                return OutputEvent(request_id, None, True, FinishReason.CANCELLED)

        worker = InferenceWorker(BlockingEngine(), command_capacity=1)
        worker.start()
        handles = []
        first = threading.Thread(
            target=lambda: handles.append(worker.submit([1], 1, request_id="first"))
        )
        second = threading.Thread(
            target=lambda: handles.append(worker.submit([2], 1, request_id="second"))
        )
        first.start()
        self.assertTrue(entered.wait(2))
        second.start()
        deadline = time.time() + 2
        while worker.stats()["worker"]["command_queue_depth"] != 1:
            if time.time() >= deadline:
                self.fail("second submission did not reach the bounded command queue")
            time.sleep(0.001)

        with self.assertRaises(WorkerQueueFull):
            worker.submit([3], 1, request_id="rejected")

        release.set()
        first.join(2)
        second.join(2)
        for handle in handles:
            self.assertTrue(handle.next_event(timeout=2).finished)
        self.assertEqual(worker.stats()["worker"]["rejected"], 1)
        worker.stop()

    def test_continuously_refilled_ingress_does_not_starve_engine_steps(self):
        class StatsOnlyScheduler:
            def stats(self):
                return {"states": {}}

        class SelfFeedingEngine:
            def __init__(self):
                self.scheduler = StatsOnlyScheduler()
                self.worker = None
                self.admitted = 0
                self.active = []
                self.first_step_after = None

            def add_request(self, request_id, prompt, max_tokens, *, eos_token_id=None):
                self.admitted += 1
                self.active.append(request_id)
                if self.admitted < 12:
                    next_id = f"queued-{self.admitted}"
                    handle = RequestHandle(next_id, 1, 1, time.time())
                    self.worker._commands.put_nowait(
                        _Submit(next_id, (1,), 1, None, handle, queue.Queue(maxsize=1))
                    )

            def step(self):
                if self.first_step_after is None:
                    self.first_step_after = self.admitted
                active, self.active = self.active, []
                return tuple(OutputEvent(request_id, 1, True, FinishReason.LENGTH) for request_id in active)

            def abort(self, request_id):
                return OutputEvent(request_id, None, True, FinishReason.CANCELLED)

        engine = SelfFeedingEngine()
        worker = InferenceWorker(engine, command_capacity=3)
        engine.worker = worker
        with worker:
            handle = worker.submit([1], 1, request_id="first")
            self.assertTrue(handle.next_event(timeout=2).finished)
        self.assertLessEqual(engine.first_step_after, 3)

    def test_stop_timeout_includes_wait_for_worker_reply(self):
        manager = make_manager()
        entered = threading.Event()
        release = threading.Event()
        runner = RecordingRunner(manager, entered=entered, release=release)
        worker = InferenceWorker(Engine(make_scheduler(manager), runner))
        worker.start()
        worker.submit([1], 8, request_id="slow")
        self.assertTrue(entered.wait(2))
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                worker.stop(timeout=0.05)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            worker._thread.join(2)
        self.assertFalse(worker.is_running)


if __name__ == "__main__":
    unittest.main()
