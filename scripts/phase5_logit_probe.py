"""Inspect eager BF16 greedy-token margins for a saved Phase 5 pilot request.

This is a local-only correctness diagnostic, not a timing or throughput run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nanoserve.experiment import analyze_replay


def inspect_request(trace: dict, replay: dict, request_id: str) -> tuple[dict, dict]:
    """Select a matching successful request without trusting file order."""
    if (
        replay.get("schema_version") != 1
        or replay.get("kind") != "completion-replay"
        or replay.get("adapter") != "huggingface-eager"
        or replay.get("trace_sha256") != trace["sha256"]
        or replay.get("model") != trace["model"]
        or replay.get("revision") != trace["revision"]
    ):
        raise ValueError("HF replay does not match the checksummed trace")
    analyze_replay(trace, replay)
    requests = [item for item in trace["requests"] if item["request_id"] == request_id]
    records = [item for item in replay.get("records", []) if item.get("request_id") == request_id]
    if len(requests) != 1 or len(records) != 1 or records[0].get("status") != "completed":
        raise ValueError("request ID must identify one completed HF replay record")
    return requests[0], records[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--hf-replay", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    replay_bytes = args.hf_replay.read_bytes()
    replay = json.loads(replay_bytes)
    request, reference = inspect_request(trace, replay, args.request_id)
    resolved_model_dir = args.model_dir.resolve()
    if resolved_model_dir.parent.name != "snapshots" or resolved_model_dir.name != trace["revision"]:
        raise ValueError("model-dir must be the pinned Hugging Face snapshot from the trace")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    model_dir = str(resolved_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda").eval()
    prompt_tokens = tokenizer.encode(request["prompt"], add_special_tokens=False)
    if len(prompt_tokens) != reference["usage"]["prompt_tokens"]:
        raise ValueError("local tokenizer prompt count disagrees with HF replay")

    generated: list[int] = []
    steps = []
    with torch.inference_mode():
        inputs = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
        output = model(inputs, use_cache=True)
        for step in range(request["max_tokens"]):
            logits = output.logits[0, -1].float()
            values, ids = torch.topk(logits, 5)
            chosen = int(logits.argmax())
            candidates = [
                {
                    "token_id": int(token_id),
                    "text": tokenizer.decode([int(token_id)], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False),
                    "logit": float(value),
                }
                for value, token_id in zip(values.tolist(), ids.tolist())
            ]
            steps.append({
                "generation_step": step,
                "chosen_token_id": chosen,
                "chosen_text": tokenizer.decode([chosen], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),
                "top_two_logit_margin": float(values[0] - values[1]),
                "top_five": candidates,
            })
            generated.append(chosen)
            if step + 1 < request["max_tokens"]:
                next_input = torch.tensor([[chosen]], dtype=torch.long, device="cuda")
                output = model(next_input, past_key_values=output.past_key_values, use_cache=True)

    generated_text = tokenizer.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    result = {
        "schema_version": 1,
        "kind": "phase5-hf-logit-probe",
        "performance_claim": False,
        "model": trace["model"],
        "revision": trace["revision"],
        "trace_sha256": trace["sha256"],
        "hf_replay_file_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "request_id": request["request_id"],
        "prompt": request["prompt"],
        "dtype": "bfloat16",
        "attention_backend": "huggingface-eager",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "generated_text": generated_text,
        "matches_hf_replay": generated_text == reference["output_text"],
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(json.dumps({key: result[key] for key in ("request_id", "generated_text", "matches_hf_replay")}, indent=2))
    if not result["matches_hf_replay"]:
        raise SystemExit("probe generation disagrees with saved HF replay; preserve the JSON for investigation")


if __name__ == "__main__":
    main()
