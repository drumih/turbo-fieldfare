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

``--long`` captures the long-context reference instead: needle-retrieval prompts
of thousands of numbered records, ending in a question about an early record,
so the answer depends on global attention reaching tens of thousands of tokens
back. The prompt text is not stored (it is hundreds of KB); the case stores the
generator's inputs, the id count and an FNV-1a hash of the ids, and the Swift
test regenerates the text with the same generator
(``Inkling2GGUFLongContextTests.longPrompt``). The server needs a context at
least as long as the longest case (``-c 65536``); ``--max-records 2100`` keeps
to the ~27K case, for a model whose llama.cpp run does not fit 64K. (The Q4
file does not fit even ``-c 28672`` on a 24 GB Mac, so only Q2 has a long
reference.) It needs the Unsloth llama.cpp build, which
knows the ``inkling2`` architecture; a 56K-token prefill takes about 20 minutes.
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


# The long-context cases: (records, queried record). About 14 tokens a record.
LONG_CASES = [(2100, 3), (4400, 5)]

LONG_WORDS = [
    "amber",
    "basil",
    "cedar",
    "delta",
    "ember",
    "fable",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kestrel",
    "lantern",
    "meadow",
    "nectar",
    "onyx",
    "pepper",
    "quartz",
    "raven",
    "saffron",
    "thistle",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "anchor",
    "birch",
    "cobalt",
    "dune",
    "falcon",
    "glacier",
]


def long_prompt(records: int, query: int) -> str:
    """The needle prompt; `Inkling2GGUFLongContextTests.longPrompt` must match it."""
    lines = ["Below is a list of records. Remember them.\n\n"]
    for index in range(1, records + 1):
        word = LONG_WORDS[(index * 13 + 7) % len(LONG_WORDS)]
        lines.append(f"Record {index:05d}: the code word is {word}.\n")
    lines.append(
        f"\nQuestion: what is the code word of record {query:05d}?\n"
        f"Answer: the code word of record {query:05d} is"
    )
    return "".join(lines)


def fnv1a(ids: list[int]) -> str:
    """FNV-1a 64 over the ids as little-endian uint32, as hex."""
    value = 0xCBF29CE484222325
    for token in ids:
        for byte in int(token).to_bytes(4, "little"):
            value ^= byte
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def post(port: int, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310 - loopback only
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
    parser.add_argument("--long", action="store_true", help="capture the long-context cases")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="with --long: only the cases of at most this many records (a smaller -c)",
    )
    args = parser.parse_args()
    props = json.loads(
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/props", timeout=60).read()  # noqa: S310
    )
    if args.long:
        cases = []
        for records, query in LONG_CASES:
            if args.max_records is not None and records > args.max_records:
                continue
            case = capture(args.port, long_prompt(records, query), args.steps, args.top)
            ids = case.pop("prompt_ids")
            case.pop("prompt")
            case.update(
                records=records,
                query=query,
                prompt_id_count=len(ids),
                prompt_id_fnv1a=fnv1a(ids),
                answer=LONG_WORDS[(query * 13 + 7) % len(LONG_WORDS)],
            )
            cases.append(case)
            print(records, "records,", len(ids), "tokens ->", repr(case["text"]))
    else:
        cases = [capture(args.port, p, args.steps, args.top) for p in PROMPTS]
    model = props.get("model_path", "")
    payload = {"model": model.rsplit("/", 1)[-1], "steps": args.steps, "cases": cases}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    for case in cases:
        if "prompt" in case:
            print(repr(case["prompt"]), "->", repr(case["text"]))


if __name__ == "__main__":
    main()
