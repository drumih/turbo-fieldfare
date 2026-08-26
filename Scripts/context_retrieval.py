#!/usr/bin/env python3
"""Measure whether a long context is actually *used*, not merely admitted.

``Scripts/context_ladder.py`` answers an admission question: it sends repeated
filler and a one-token completion, so a level passes when the server accepts
the prompt, prefills it, and stays within memory. That cannot distinguish a
healthy long context from one the model can no longer attend across — both
return HTTP 200.

The distinction matters for this architecture specifically. Gemma 4 26B-A4B
sets ``slidingWindow: 1024`` and marks only layers 5, 11, 17, 23, and 29 as
full attention, so 25 of 30 layers see a 1024-token window and all long-range
signal rests on 5 layers. Extending the context cap does not by itself extend
what the model can retrieve.

This harness runs needle-in-a-haystack. A distinctive fact is planted at a
fractional depth inside filler, the model is asked to repeat it, and the answer
is checked for the needle. Depth is swept because sliding-window degradation is
position dependent: a needle near the end of the prompt stays inside the local
window of every layer, while one near the middle must survive on the full
attention layers alone.

Interpreting results
--------------------
Recall at a depth is evidence the full-attention path carries information from
that position. A miss is a genuine finding, not a harness failure: report it
with the depth and prompt-token count rather than retrying until it passes.
Filler is repetitive by design, which is the easy case for retrieval; real
agent history is harder, so treat recall here as an upper bound.

Examples
--------
Validate the probe where retrieval is known to work::

    python3 Scripts/context_retrieval.py --model scratch/gemma4.gturbo \\
      --max-context 16384 --filler-words 14000

Sweep the ladder rungs::

    python3 Scripts/context_retrieval.py --model scratch/gemma4.gturbo \\
      --max-context 262144 --filler-words 250000 --depths 0.1,0.5,0.9
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "scratch" / "gemma4.gturbo"
DEFAULT_OUT = ROOT / "benchmark-results" / "context-retrieval"
# A short, distinctive, non-word token sequence: unlikely to be produced by
# chance, and cheap to check for verbatim.
NEEDLE_CODE = "74391-QM"
NEEDLE = f"The Aurora Vault access code is {NEEDLE_CODE}."
QUESTION = "What is the Aurora Vault access code? Answer with only the code."


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def wait_ready(proc: subprocess.Popen[str], logs: list[str], ready: threading.Event) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        logs.append(line.rstrip())
        if "TurboFieldfareServer ready" in line:
            ready.set()


def start_server(options: argparse.Namespace) -> tuple[subprocess.Popen[str], list[str]]:
    command = [
        str(options.server),
        "--model", str(options.model),
        "--port", str(options.port),
        "--max-context", str(options.max_context),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs: list[str] = []
    ready = threading.Event()
    threading.Thread(target=wait_ready, args=(proc, logs, ready), daemon=True).start()
    if not ready.wait(options.ready_timeout):
        proc.terminate()
        raise RuntimeError(f"server did not become ready: {logs[-10:]}")
    return proc, logs


def build_prompt(filler_words: int, depth: float) -> str:
    filler = ["word"] * filler_words
    at = max(0, min(len(filler), int(len(filler) * depth)))
    return " ".join(filler[:at] + [NEEDLE] + filler[at:]) + f"\n\n{QUESTION}"


def probe(options: argparse.Namespace, depth: float, logs: list[str]) -> dict:
    payload = json.dumps({
        "model": options.model_id,
        "messages": [{"role": "user", "content": build_prompt(options.filler_words, depth)}],
        "temperature": 0,
        "max_completion_tokens": options.max_completion_tokens,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{options.port}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    started = time.time()
    record: dict[str, object] = {
        "max_context": options.max_context,
        "depth": depth,
        "filler_words": options.filler_words,
        "needle": NEEDLE_CODE,
        "source_commit": options.commit,
    }
    try:
        with urllib.request.urlopen(request, timeout=options.request_timeout) as response:
            body = json.loads(response.read())
        answer = (body["choices"][0]["message"].get("content") or "").strip()
        usage = body.get("usage", {})
        record.update({
            "status": "completed",
            "elapsed_seconds": round(time.time() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "answer": answer[:200],
            "found": NEEDLE_CODE in answer,
        })
    except Exception as exc:
        record.update({
            "status": "error",
            "elapsed_seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "found": False,
        })
    line = next((x for x in reversed(logs) if " completed in " in x), None)
    if line:
        match = re.search(r"pp=([0-9.]+)s pp_tok_s=([0-9.]+)", line)
        if match:
            record["pp_seconds"] = float(match.group(1))
            record["pp_tokens_per_second"] = float(match.group(2))
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-haystack retrieval probe for long contexts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server", type=Path,
                        default=ROOT / ".build/release/TurboFieldfareServer")
    parser.add_argument("--model-id", default="gemma-4-26b-a4b-it")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-context", type=int, required=True,
                        help="Server context cap for this run.")
    parser.add_argument("--filler-words", type=int, required=True,
                        help="Filler words around the needle. Keep the resulting "
                             "prompt inside the cap with room for the answer.")
    parser.add_argument("--depths", default="0.1,0.5,0.9",
                        help="Comma-separated needle depths in [0,1].")
    parser.add_argument("--max-completion-tokens", type=int, default=24)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--ready-timeout", type=float, default=180)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    options = parser.parse_args(argv)
    try:
        options.depths = [float(part) for part in options.depths.split(",") if part.strip()]
    except ValueError:
        parser.error("--depths must be a comma-separated list of numbers")
    if any(d < 0 or d > 1 for d in options.depths):
        parser.error("--depths values must fall in [0, 1]")
    return options


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if not options.model.is_dir():
        print(f"missing model directory: {options.model}", file=sys.stderr)
        return 2
    if not options.server.is_file():
        print(f"missing server binary: {options.server}\n"
              "build it with: swift build -c release --product TurboFieldfareServer",
              file=sys.stderr)
        return 2
    options.commit = git_commit()
    options.out.mkdir(parents=True, exist_ok=True)

    proc, logs = start_server(options)
    records: list[dict] = []
    try:
        for depth in options.depths:
            record = probe(options, depth, logs)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)

    label = f"{options.max_context // 1024}K"
    (options.out / f"{label}.json").write_text(json.dumps(records, indent=2) + "\n")
    found = sum(1 for r in records if r.get("found"))
    print(f"RECALL {found}/{len(records)} at {label}", flush=True)
    # A miss is a real result worth surfacing in the exit code, so a sweep can
    # be scripted without parsing stdout.
    return 0 if found == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
