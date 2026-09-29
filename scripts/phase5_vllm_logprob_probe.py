"""Capture vLLM top-token probabilities for one saved Phase 5 pilot request.

This is a correctness diagnostic, not a timing or throughput run. The server
must use the same pinned model and launch configuration as the pilot replay.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
from urllib.parse import urlsplit

from nanoserve.experiment import analyze_replay
from nanoserve.replay import validate_completion_trace


def prepare_probe(trace: dict, replay: dict, request_id: str) -> tuple[dict, dict]:
    """Verify provenance and reproduce the replay's generation parameters."""
    validate_completion_trace(trace)
    if (
        replay.get("schema_version") != 1
        or replay.get("kind") != "completion-replay"
        or replay.get("adapter") != "vllm-http"
        or replay.get("trace_sha256") != trace["sha256"]
        or replay.get("model") != trace["model"]
        or replay.get("revision") != trace["revision"]
    ):
        raise ValueError("vLLM replay does not match the checksummed trace")
    analyze_replay(trace, replay)
    requests = [item for item in trace["requests"] if item["request_id"] == request_id]
    records = [item for item in replay.get("records", []) if item.get("request_id") == request_id]
    if len(requests) != 1 or len(records) != 1 or records[0].get("status") != "completed":
        raise ValueError("request ID must identify one completed vLLM replay record")
    request = requests[0]
    payload = {
        "model": trace["model"],
        "prompt": request["prompt"],
        "max_tokens": request["max_tokens"],
        "temperature": 0,
        "stream": False,
        "logprobs": 10,
    }
    if request.get("ignore_eos", False):
        payload["ignore_eos"] = True
    return payload, records[0]


def query_vllm(endpoint: str, payload: dict) -> tuple[int, dict]:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.path != "/v1/completions"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("endpoint must be a local HTTP /v1/completions URL")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    try:
        connection.request(
            "POST", "/v1/completions", body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        raw = response.read()
        try:
            return response.status, json.loads(raw)
        except json.JSONDecodeError:
            return response.status, {"non_json_response": raw.decode("utf-8", errors="replace")}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--vllm-replay", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    replay_bytes = args.vllm_replay.read_bytes()
    payload, reference = prepare_probe(trace, json.loads(replay_bytes), args.request_id)
    status, response = query_vllm(args.endpoint, payload)
    choices = response.get("choices", []) if isinstance(response, dict) else []
    choice = choices[0] if status == 200 and len(choices) == 1 else {}
    result = {
        "schema_version": 1,
        "kind": "phase5-vllm-logprob-probe",
        "performance_claim": False,
        "trace_sha256": trace["sha256"],
        "vllm_replay_file_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "request_id": args.request_id,
        "replay_output_text": reference["output_text"],
        "request_payload": payload,
        "http_status": status,
        "response": response,
        "matches_vllm_replay": choice.get("text") == reference["output_text"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(json.dumps({key: result[key] for key in ("request_id", "http_status", "matches_vllm_replay")}, indent=2))
    if status != 200 or not choice.get("logprobs"):
        raise SystemExit("vLLM did not return completion logprobs; preserve the JSON for investigation")


if __name__ == "__main__":
    main()
