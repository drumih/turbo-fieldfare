#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval through the OpenAI-compatible server.

Admission alone cannot tell a working long context from a broken one: a prompt
the server accepts and prefills returns HTTP 200 whether or not the model can
still attend across it. This plants one fact at a fractional depth of a filler
prompt, asks for it back with a real multi-token completion, and exits non-zero
on any miss. Sliding-window degradation is position dependent, so the depth is
swept.

usage:
  Scripts/context_retrieval.py --max-context 131072 [--depths 0.1,0.5,0.9]
      [--server http://127.0.0.1:8080] [--model gemma-4-26b-a4b-it]
      [--reserve 4096] [--words N] [--out results.jsonl]

The server must already be running with `--max-context` at or above the value
given here. Each depth is a fresh prompt, so each one pays a full prefill.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

SENTENCES = [
    "The harbour town kept its ledgers in a stone office beside the customs house.",
    "Every clerk knew the tide tables better than the calendar.",
    "Cargo moved by barge at dawn, by cart at noon, and by memory at night.",
    "The old pilots argued about channels that had silted up decades ago.",
    "A visiting surveyor once counted forty-one warehouses and was told he had missed some.",
    "Rain came in from the west most afternoons and left the quays slick until dusk.",
    "The lighthouse keeper logged every passing hull by its lantern colour.",
    "Rope was sold by the fathom and argued over by the inch.",
    "The ferry ran on the hour except when the harbourmaster said otherwise.",
    "Nobody remembered who had named the eastern mole, and nobody asked.",
]

NEEDLE = "The passcode for the archive room is {secret}."
QUESTION = (
    "\n\nWhat is the passcode for the archive room? "
    "Answer with the passcode only."
)


def build_prompt(words, depth, secret, rng):
    """Filler of about `words` words with the needle at fractional `depth`."""
    body = []
    count = 0
    needle_at = int(words * depth)
    placed = False
    while count < words:
        if not placed and count >= needle_at:
            body.append(NEEDLE.format(secret=secret))
            placed = True
        sentence = rng.choice(SENTENCES)
        body.append(sentence)
        count += len(sentence.split())
    if not placed:
        body.append(NEEDLE.format(secret=secret))
    return " ".join(body) + QUESTION


def ask(server, model, prompt, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 32,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--reserve", type=int, default=4096,
                        help="tokens left free below --max-context (default 4096)")
    parser.add_argument("--words", type=int, default=None,
                        help="filler words; default fills the context at ~1.2 tokens per word")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--out", default=None, help="append one JSON line per depth")
    args = parser.parse_args(argv)

    depths = [float(value) for value in args.depths.split(",") if value]
    if not depths or any(not 0 <= depth <= 1 for depth in depths):
        parser.error("--depths must be fractions in [0, 1]")
    words = args.words or int((args.max_context - args.reserve) / 1.2)
    if words <= 0:
        parser.error("--max-context minus --reserve leaves no room for filler")

    rng = random.Random(args.seed)
    misses = 0
    for depth in depths:
        secret = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        prompt = build_prompt(words, depth, secret, rng)
        try:
            body, seconds = ask(args.server, args.model, prompt, args.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            print(f"depth={depth} HTTP {error.code}: {detail}", file=sys.stderr)
            return 2
        answer = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage", {})
        hit = secret in answer
        misses += 0 if hit else 1
        record = {
            "max_context": args.max_context,
            "depth": depth,
            "words": words,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "seconds": round(seconds, 3),
            "hit": hit,
            "answer": answer.strip()[:120],
        }
        print(json.dumps(record))
        if args.out:
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
    print(f"{len(depths) - misses}/{len(depths)} recalled at {args.max_context}")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
