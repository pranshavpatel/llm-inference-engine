import unittest

import torch

from nanoserve.attention import ReferencePagedAttention
from nanoserve.cli import scheduler_demo
from nanoserve.engine import Engine
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
from nanoserve.model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM
from nanoserve.scheduler import QueueFull, Scheduler, SchedulerConfig, StepInFlight
from nanoserve.types import FinishReason, RequestState


def make_manager(*, blocks=12, block_size=2, layers=1, heads=1, dim=2, watermark=0):
    block_manager = BlockManager(blocks, block_size=block_size, watermark=watermark)
    cache = PagedKVCache(
        KVCacheSpec(layers, blocks, block_size, heads, dim),
        dtype=torch.float32,
    )
    return PagedKVCacheManager(block_manager, cache)


def make_scheduler(manager, *, sequences=4, context=16, waiting=16):
    return Scheduler(
        manager,
        SchedulerConfig(
            max_num_sequences=sequences,
            max_batch_tokens=context,
            max_prefill_tokens=context,
            max_context_tokens=context,
            max_waiting_requests=waiting,
        ),
    )


class FakeRunner:
    """Deterministic runner that exercises real physical KV transactions."""

    def __init__(self, cache_manager, *, vocab_size=64):
        self.cache_manager = cache_manager
        self.vocab_size = vocab_size

    def forward(self, request_ids, token_batches):
        outputs = []
        spec = self.cache_manager.cache.spec
        for request_id, tokens in zip(request_ids, token_batches):
            length = int(tokens.numel())
            shape = (spec.num_key_value_heads, length, spec.head_dim)
            keys = [
                torch.zeros(shape, dtype=self.cache_manager.cache.dtype)
                for _ in range(spec.num_layers)
            ]
            values = [key.clone() for key in keys]
            self.cache_manager.append(request_id, keys, values)
            logits = torch.zeros((length, self.vocab_size), dtype=torch.float32)
            for row, token in enumerate(tokens.tolist()):
                logits[row, (token + 1) % self.vocab_size] = 1
            outputs.append(logits)
        return tuple(outputs)


def run_until_idle(engine, scheduler, *, limit=30):
    events = []
    for _ in range(limit):
        step_events = engine.step()
        events.extend(step_events)
        states = scheduler.stats()["states"]
        if states["waiting"] == 0 and states["decoding"] == 0:
            return events
        if not step_events:
            raise AssertionError("scheduler made no progress with unfinished requests")
    raise AssertionError("finite workload did not complete")


class SchedulerTests(unittest.TestCase):
    def test_request_lifecycle_emits_each_token_once_and_releases_pages(self):
        manager = make_manager()
        scheduler = make_scheduler(manager)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("a", [3, 4], 3)

        events = run_until_idle(engine, scheduler)

        self.assertEqual([event.token_id for event in events], [5, 6, 7])
        self.assertEqual(sum(event.finished for event in events), 1)
        snapshot = scheduler.snapshot("a")
        self.assertEqual(snapshot.state, RequestState.FINISHED)
        self.assertEqual(snapshot.generated_token_ids, (5, 6, 7))
        self.assertEqual(snapshot.finish_reason, FinishReason.LENGTH)
        self.assertEqual(manager.blocks.stats()["free_blocks"], 12)
        manager.blocks.check_invariants()

    def test_eos_finishes_early_and_releases_pages(self):
        manager = make_manager()
        scheduler = make_scheduler(manager)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("eos", [8], 5, eos_token_id=9)

        events = engine.step()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].token_id, 9)
        self.assertEqual(events[0].finish_reason, FinishReason.EOS)
        self.assertEqual(scheduler.snapshot("eos").state, RequestState.FINISHED)
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)

    def test_no_eos_stop_emits_fixed_token_count_including_eos_id(self):
        manager = make_manager()
        scheduler = make_scheduler(manager)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("fixed", [8], 3, eos_token_id=None)

        events = run_until_idle(engine, scheduler)

        self.assertEqual([event.token_id for event in events], [9, 10, 11])
        self.assertEqual(events[-1].finish_reason, FinishReason.LENGTH)
        self.assertEqual(scheduler.snapshot("fixed").generated_token_ids, (9, 10, 11))

    def test_staggered_arrivals_complete_while_existing_request_decodes(self):
        manager = make_manager()
        scheduler = make_scheduler(manager, sequences=2)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("first", [1, 2], 4)
        self.assertEqual([event.token_id for event in engine.step()], [3])
        engine.add_request("second", [10, 11, 12], 2)

        events = run_until_idle(engine, scheduler)

        self.assertEqual(
            [event.token_id for event in events if event.request_id == "first"],
            [4, 5, 6],
        )
        self.assertEqual(
            [event.token_id for event in events if event.request_id == "second"],
            [13, 14],
        )
        self.assertEqual(scheduler.snapshot("first").state, RequestState.FINISHED)
        self.assertEqual(scheduler.snapshot("second").state, RequestState.FINISHED)

    def test_forced_preemption_recomputes_without_duplicate_output(self):
        manager = make_manager(blocks=3, block_size=2)
        scheduler = make_scheduler(manager, sequences=2, context=8)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("older", [1, 2], 3)
        engine.add_request("newer", [10, 11], 3)

        events = run_until_idle(engine, scheduler)

        older = scheduler.snapshot("older")
        newer = scheduler.snapshot("newer")
        self.assertEqual(older.generated_token_ids, (3, 4, 5))
        self.assertEqual(newer.generated_token_ids, (12, 13, 14))
        self.assertEqual(newer.preemptions, 1)
        self.assertEqual(newer.recomputed_tokens, 2)
        self.assertEqual(len(events), 6)
        self.assertEqual(manager.blocks.stats()["free_blocks"], 3)
        manager.blocks.check_invariants()

    def test_abort_waiting_and_decoding_requests(self):
        manager = make_manager()
        scheduler = make_scheduler(manager, sequences=1)
        engine = Engine(scheduler, FakeRunner(manager))
        engine.add_request("active", [1], 3)
        engine.add_request("waiting", [2], 3)
        engine.step()

        waiting_event = engine.abort("waiting")
        active_event = engine.abort("active")

        self.assertEqual(waiting_event.finish_reason, FinishReason.CANCELLED)
        self.assertEqual(active_event.finish_reason, FinishReason.CANCELLED)
        self.assertIsNone(engine.abort("active"))
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)

    def test_in_flight_step_must_be_resolved_before_other_mutations(self):
        manager = make_manager()
        scheduler = make_scheduler(manager)
        scheduler.add_request("a", [1], 2)
        plan = scheduler.schedule()

        with self.assertRaises(StepInFlight):
            scheduler.schedule()
        with self.assertRaises(StepInFlight):
            scheduler.abort("a")

        events = scheduler.fail(plan, RuntimeError("cancel test step"))
        self.assertEqual(events[0].finish_reason, FinishReason.ERROR)
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)

    def test_bad_commit_is_transactional_then_failure_cleans_up(self):
        manager = make_manager()
        scheduler = make_scheduler(manager)
        scheduler.add_request("a", [1], 2)
        scheduler.add_request("b", [2], 2)
        plan = scheduler.schedule()
        FakeRunner(manager).forward(
            [item.request_id for item in plan.prefills],
            [torch.tensor(item.token_ids) for item in plan.prefills],
        )

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            scheduler.commit(plan, {"a": 2, "b": True})

        self.assertEqual(scheduler.snapshot("a").generated_token_ids, ())
        self.assertEqual(scheduler.snapshot("b").generated_token_ids, ())
        scheduler.fail(plan, RuntimeError("invalid sampler output"))
        self.assertEqual(manager.blocks.stats()["active_requests"], 0)

    def test_runner_failure_marks_selected_requests_failed_without_leaks(self):
        class FailingRunner:
            def __init__(self, cache_manager):
                self.cache_manager = cache_manager

            def forward(self, request_ids, token_batches):
                raise RuntimeError("injected runner failure")

        manager = make_manager()
        scheduler = make_scheduler(manager)
        engine = Engine(scheduler, FailingRunner(manager))
        engine.add_request("a", [1, 2], 2)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            engine.step()

        snapshot = scheduler.snapshot("a")
        self.assertEqual(snapshot.state, RequestState.FAILED)
        self.assertEqual(snapshot.finish_reason, FinishReason.ERROR)
        self.assertIn("injected runner failure", snapshot.error)
        self.assertEqual(manager.blocks.stats()["free_blocks"], 12)

    def test_queue_context_and_physical_capacity_are_bounded(self):
        manager = make_manager(blocks=2, block_size=2, watermark=0.5)
        scheduler = make_scheduler(manager, context=8, waiting=1)
        scheduler.add_request("a", [1, 2], 1)
        with self.assertRaises(QueueFull):
            scheduler.add_request("b", [1], 1)

        other_manager = make_manager(blocks=2, block_size=2)
        other = make_scheduler(other_manager, context=8)
        with self.assertRaisesRegex(ValueError, "context"):
            other.add_request("context", [1, 2, 3], 6)
        with self.assertRaisesRegex(ValueError, "physical KV pool"):
            other.add_request("pool", [1, 2], 5)


class TinyModelEngineIntegrationTests(unittest.TestCase):
    def test_scheduler_drives_real_paged_qwen_runner(self):
        torch.manual_seed(17)
        config = Qwen2Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
        model = Qwen2ForCausalLM(config).eval()
        manager = make_manager(blocks=12, block_size=2, layers=1, heads=2, dim=4)
        runner = PagedQwen2Runner(model, manager, ReferencePagedAttention())
        scheduler = make_scheduler(manager, sequences=2, context=12)
        engine = Engine(scheduler, runner)
        engine.add_request("a", [1, 2, 3], 2)
        engine.add_request("b", [4, 5], 2)

        events = run_until_idle(engine, scheduler)

        self.assertEqual(len(events), 4)
        self.assertEqual(len(scheduler.snapshot("a").generated_token_ids), 2)
        self.assertEqual(len(scheduler.snapshot("b").generated_token_ids), 2)
        self.assertEqual(manager.blocks.stats()["free_blocks"], 12)
        manager.blocks.check_invariants()

    def test_cli_scheduler_demo_completes_forced_preemption_workload(self):
        report = scheduler_demo()

        self.assertEqual(report["steps"], 5)
        self.assertEqual(report["requests"]["newer"]["preemptions"], 1)
        self.assertEqual(report["requests"]["newer"]["recomputed_tokens"], 2)
        self.assertTrue(report["all_pages_released"])
        self.assertFalse(report["performance_claim"])


if __name__ == "__main__":
    unittest.main()
