import http.client
import json
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nanoserve.model import Qwen2Config, Qwen2ForCausalLM
from nanoserve.runtime import ServingConfig, build_serving_runtime
from nanoserve.server import make_server


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp.name)
        model_config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )
        (self.model_dir / "config.json").write_text(json.dumps(asdict(model_config)))
        torch.manual_seed(3)
        model = Qwen2ForCausalLM(model_config).eval()
        save_file(model.state_dict(), str(self.model_dir / "model.safetensors"))
        vocab = {"[UNK]": 0, "[EOS]": 1, "hello": 2}
        vocab.update({f"token{i}": i for i in range(3, 32)})
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            eos_token="[EOS]",
        )
        wrapped.save_pretrained(str(self.model_dir))

    def tearDown(self):
        self.temp.cleanup()

    def serving_config(self, **overrides):
        values = dict(
            model_dir=self.model_dir,
            model_name="tiny-checkpoint",
            device="cpu",
            dtype="float32",
            kv_pool_mib=1,
            block_size=2,
            max_context_tokens=8,
            max_num_sequences=2,
            max_waiting_requests=4,
        )
        values.update(overrides)
        return ServingConfig(**values)

    def test_local_checkpoint_serves_completion_and_releases_pages(self):
        runtime = build_serving_runtime(self.serving_config())
        server = make_server(runtime.worker, runtime.codec, model=runtime.model_id, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        runtime.worker.start()
        thread.start()
        try:
            host, port = server.server_address
            connection = http.client.HTTPConnection(host, port, timeout=5)
            body = json.dumps(
                {"model": "tiny-checkpoint", "prompt": "hello", "max_tokens": 2}
            )
            connection.request(
                "POST",
                "/v1/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            result = json.loads(response.read())
            connection.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(result["usage"]["prompt_tokens"], 1)
            self.assertLessEqual(result["usage"]["completion_tokens"], 2)
            self.assertIn(result["choices"][0]["finish_reason"], ("stop", "length"))
            deadline = time.monotonic() + 2
            while runtime.worker.stats()["cache"]["active_requests"] != 0:
                if time.monotonic() >= deadline:
                    self.fail("worker did not publish the released KV pages")
                time.sleep(0.01)
            self.assertEqual(runtime.checkpoint_files, ("model.safetensors",))
        finally:
            server.shutdown()
            server.server_close()
            runtime.worker.stop()
            thread.join(2)

    def test_configuration_rejects_context_above_model_capacity(self):
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            build_serving_runtime(self.serving_config(max_context_tokens=17))

    def test_configuration_rejects_unsupported_cpu_dtype(self):
        with self.assertRaisesRegex(ValueError, "CPU serving"):
            build_serving_runtime(self.serving_config(dtype="bfloat16"))


if __name__ == "__main__":
    unittest.main()
