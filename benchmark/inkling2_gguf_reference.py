"""Capture llama.cpp's greedy reference for an Inkling2 GGUF.

Queries a running ``llama-server`` (loopback only) for each prompt: the token
ids of the raw prompt (``/tokenize``), then a greedy continuation with the
top-k probabilities of every step (``/completion``, ``top_k=1``,
``n_probs``). The JSON it writes is the parity oracle for TinyTitan's native
Inkling2 GGUF engine: the same prompt ids must produce the same greedy ids,
with each step's top-k in the same order.

Usage (start the server on the model first, e.g.
``llama-server -m <gguf> -ngl 99 -fa on -c 4096 -np 1 --host 127.0.0.1 --port 18090``)::

    python3 benchmark/inkling2_gguf_reference.py --port 18090 \
        --output tests/TinyTitan/Inkling2GGUF/Inkling2GGUFReferences/q4kxl-reference.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Water boils at a temperature of",
    "1, 2, 3, 4, 5,",
    "The three primary colors are",
    "In 1969, the first person to walk on the Moon was",
]


def post(port: int, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 - loopback only
        return json.loads(response.read())


def capture(port: int, prompt: str, steps: int, top: int) -> dict:
    ids = post(port, "/tokenize", {"content": prompt, "add_special": False})["tokens"]
    result = post(
        port,
        "/completion",
        {
            "prompt": ids,
            "n_predict": steps,
            "top_k": 1,
            "temperature": 0.0,
            "n_probs": top,
            "cache_prompt": False,
            "post_sampling_probs": False,
        },
    )
    generated = []
    for step in result["completion_probabilities"]:
        generated.append(
            {
                "id": step["id"],
                "logprob": step["logprob"],
                "top": [[c["id"], c["logprob"]] for c in step["top_logprobs"]],
            }
        )
    return {"prompt": prompt, "prompt_ids": ids, "text": result["content"], "steps": generated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    props = json.loads(
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/props", timeout=60).read()  # noqa: S310
    )
    cases = [capture(args.port, p, args.steps, args.top) for p in PROMPTS]
    model = props.get("model_path", "")
    payload = {"model": model.rsplit("/", 1)[-1], "steps": args.steps, "cases": cases}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    for case in cases:
        print(repr(case["prompt"]), "->", repr(case["text"]))


if __name__ == "__main__":
    main()
