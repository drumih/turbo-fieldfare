#!/usr/bin/env python3
"""ANE-vs-GPU prefill A/B across the installed models.

One variable: `TINYTITAN_PREFILL_ANE`. Arms alternate `off, on, on, off` after a
discarded warm-up per arm, so thermal drift and page-cache state land on both
arms rather than on whichever ran second.

The metric is prefill seconds from the CLI's own footer, on a prompt long enough
to fill more than one 4,096-token chunk — the only workload the sidecar serves
(a short prompt is one partial chunk and deliberately stays on the GPU, and a
non-4,096 prefill chunk never matches the sidecar at all).

The script refuses to report a speedup for an arm that did not use the ANE: the
runtime prints `ane-prefill fallback ... using the GPU path` when a chunk is
ineligible, and such an arm is recorded as fallen back rather than as an ANE
number. A model with no sidecar is reported `no sidecar` — with the switch
explicitly `on`, the runtime fails the load rather than pretending.

  python3 benchmark/ane_prefill_ab_matrix.py \
      --models qwen3.5_2B_4Bit qwen3.5_4B_4Bit --label v5.6 --record
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CLI = ROOT / ".build/release/TinyTitanCLI"
RESULTS = ROOT / "benchmark/ane-prefill"
EXTRA_ARGS: list[str] = []

PREFILL_CHUNK = 4096
FALLBACK_MARKER = "ane-prefill fallback"
NO_SIDECAR_MARKER = "is missing; run "
# One token: prefill is the metric, and a longer continuation only adds decode
# time. It does cost the digest its diagnostic value — with a single greedy
# token both arms usually emit the same one — so treat the digests as a
# determinism check, not as evidence the arms differ.
MAX_NEW = 1
# Measured for this text: 23,000 characters is 4,333 tokens. Used only to warn
# when a prompt cannot reach one full chunk, i.e. when the ANE cannot engage.
CHARACTERS_PER_TOKEN = 5.31

FOOTER = re.compile(
    r"\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s new=(\d+)tok decode=([\d.]+)s "
    r"tok/s=([\d.]+)\]"
)

# A fixed, self-contained body. Deriving it from repo files would make the
# prompt length move whenever those files are edited, and prompt length is the
# quadratic term this benchmark is about.
#
# It is deliberately *just* over one chunk (~4,300 tokens, measured at ~5.3
# characters per token for this text): the ANE only serves a full 4,096-token
# chunk, so a short prompt would measure two GPU arms and call one of them
# "ANE". Shorter is faster; below ~21,750 characters it is meaningless.
PARAGRAPH = (
    "Swift and C++ differ in memory management, dispatch, compilation and type "
    "safety, and a fair comparison names each axis before it judges either "
    "language. "
)
PROMPT_CHARACTERS = 23_000


def prompt(characters: int = PROMPT_CHARACTERS) -> str:
    repeats = characters // len(PARAGRAPH) + 1
    return (PARAGRAPH * repeats)[:characters]


def parse_footer(stderr: str) -> dict | None:
    match = FOOTER.search(stderr)
    if not match:
        return None
    return {
        "finish": match.group(1),
        "prefill_tokens": int(match.group(2)),
        "prefill_seconds": float(match.group(3)),
        "decode_tokens": int(match.group(4)),
        "decode_seconds": float(match.group(5)),
        "decode_tokens_per_second": float(match.group(6)),
    }


def run_arm(model: str, ane: bool, characters: int, chunk: int, timeout: int = 3600) -> dict:
    env = os.environ.copy()
    env["TINYTITAN_PREFILL_ANE"] = "on" if ane else "off"
    command = [
        str(CLI),
        "--model",
        str(MODELS_DIR / model),
        "--prompt",
        prompt(characters),
        "--max-new",
        str(MAX_NEW),
        "--temperature",
        "0",
        "--prefill-chunk",
        str(chunk),
        *EXTRA_ARGS,
    ]
    proc = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=ROOT, timeout=timeout, check=False
    )
    arm: dict = {"ane": ane, "exit": proc.returncode}
    if proc.returncode != 0:
        err = proc.stderr
        if NO_SIDECAR_MARKER in err:
            arm["error"] = "no sidecar"
        else:
            # The last few lines, not just one: a load failure prints its
            # reason above whatever the process says on the way out.
            tail = [line.strip() for line in err.strip().splitlines() if line.strip()]
            arm["error"] = " / ".join(tail[-3:])[:400] or f"exit {proc.returncode}"
        return arm
    footer = parse_footer(proc.stderr)
    if footer is None:
        arm["error"] = "no [stop=...] footer"
        return arm
    arm.update(footer)
    arm["used_ane"] = FALLBACK_MARKER not in proc.stderr
    if not arm["used_ane"]:
        arm["fallback_reason"] = next(
            (line.strip() for line in proc.stderr.splitlines() if FALLBACK_MARKER in line),
            FALLBACK_MARKER,
        )
    arm["response_sha256"] = hashlib.sha256(proc.stdout.strip().encode()).hexdigest()
    arm["response_head"] = proc.stdout.strip()[:120]
    return arm


def measure(model: str, repeats: int, characters: int, chunk: int) -> dict:
    """Warm both arms, then alternate `repeats` measured runs per arm.

    The warm-ups are discarded because the first ANE run pays Core ML's compile
    (~68 s against an 86 s prefill on AgentWorld 4-bit). One OFF run is measured
    before the ANE arm is attempted, so a model the ANE cannot serve — no
    sidecar, or one whose geometry the runtime refuses — still reports its GPU
    prefill time instead of only the refusal. That is the Qwen 3.8 case.
    """
    arms: dict[str, list[dict]] = {"off": [], "on": []}
    warm_off = run_arm(model, False, characters, chunk)
    if "error" in warm_off:
        return {"model": model, "arms": arms, "error": f"off warm-up: {warm_off['error']}"}
    arms["off"].append(run_arm(model, False, characters, chunk))
    if "error" in arms["off"][-1]:
        return {"model": model, "arms": arms, "error": f"off arm: {arms['off'][-1]['error']}"}
    warm_on = run_arm(model, True, characters, chunk)
    if "error" in warm_on:
        return {"model": model, "arms": arms, "ane_unavailable": warm_on["error"]}
    arms["on"].append(run_arm(model, True, characters, chunk))
    if "error" in arms["on"][-1]:
        return {"model": model, "arms": arms, "error": f"on arm: {arms['on'][-1]['error']}"}
    for _ in range(max(0, repeats - 1)):
        for ane in (False, True):
            run = run_arm(model, ane, characters, chunk)
            if "error" in run:
                return {
                    "model": model,
                    "arms": arms,
                    "error": f"{'on' if ane else 'off'} arm: {run['error']}",
                }
            arms["on" if ane else "off"].append(run)
    return {"model": model, "arms": arms}


def summarize(record: dict, chunk: int = PREFILL_CHUNK) -> dict:
    out: dict = {"model": record["model"]}
    if "error" in record:
        out["error"] = record["error"]
        return out
    if "ane_unavailable" in record:
        out["ane_unavailable"] = record["ane_unavailable"]
    for name, runs in record["arms"].items():
        good = [r for r in runs if "error" not in r]
        if not good:
            out[name] = {"error": runs[0].get("error", "no runs") if runs else "no runs"}
            continue
        out[name] = {
            "prefill_seconds_median": statistics.median(r["prefill_seconds"] for r in good),
            "prefill_tokens": good[0]["prefill_tokens"],
            "used_ane": all(r.get("used_ane", False) for r in good),
            "runs": len(good),
            "digests": sorted({r["response_sha256"] for r in good}),
            "head": good[0].get("response_head", ""),
        }
    off = out.get("off", {}).get("prefill_seconds_median")
    on = out.get("on", {}).get("prefill_seconds_median")
    # The ANE serves a full chunk or a continuation of one, so a prompt that
    # does not reach 4,096 tokens measures two GPU arms. Say so rather than
    # report "no speedup", which reads like a finding about the ANE.
    tokens = out.get("off", {}).get("prefill_tokens")
    if tokens is not None and tokens < chunk:
        out["prompt_too_short"] = (
            f"{tokens} prompt tokens is under one {chunk}-token chunk; "
            f"the ANE cannot engage, so these are two GPU arms"
        )
    if (
        off is not None
        and on is not None
        and out.get("on", {}).get("used_ane")
        and "prompt_too_short" not in out
    ):
        if on:
            out["speedup"] = off / on
            out["saved_seconds"] = off - on
        else:
            out["no_speedup"] = "the ANE arm's prefill median is 0 s, so no speedup is computable"
    return out


def format_row(r: dict) -> str:
    if "error" in r:
        return f"{r['model']:<44} {'-':>9} {'-':>9} {'-':>8} {'-':>9}  {r['error']}"
    off = r["off"].get("prefill_seconds_median")
    on = r["on"].get("prefill_seconds_median")
    note = ""
    if "ane_unavailable" in r:
        note = f"ANE unavailable: {r['ane_unavailable'][:70]}"
    elif "prompt_too_short" in r:
        note = r["prompt_too_short"]
    elif r["on"].get("used_ane") is False:
        note = "ANE arm fell back to the GPU"
    elif not r["off"].get("used_ane", True):
        note = "OFF arm reported an ANE fallback (unexpected)"
    elif "no_speedup" in r:
        note = r["no_speedup"]
    speedup = f"{r['speedup']:.3f}" if "speedup" in r else "-"
    return (
        f"{r['model']:<44} "
        f"{f'{off:.2f}' if off is not None else '-':>9} "
        f"{f'{on:.2f}' if on is not None else '-':>9} "
        f"{speedup:>8} "
        f"{str(r.get('on', {}).get('used_ane')):>9}  {note}"
    )


def new_record(repeats: int, characters: int, max_new: int, chunk: int = PREFILL_CHUNK) -> dict:
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "prompt_characters": characters,
        "prefill_chunk": chunk,
        "max_new_tokens": max_new,
        "repeats_per_arm": repeats,
        "results": [],
    }


def stored_models(record: dict) -> set[str]:
    """Models already measured into this record, in the order they were stored.

    A model whose run *failed* is not counted: a refusal is worth re-attempting
    after whatever caused it is fixed, and treating it as done would silently
    keep the failure.
    """
    return {r["model"] for r in record.get("results", []) if "error" not in r}


def store_result(record: dict, result: dict) -> dict:
    """Replace this model's row (or append it), keeping the others."""
    kept = [r for r in record.get("results", []) if r["model"] != result["model"]]
    kept.append(result)
    record["results"] = kept
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True, help="install directory names under models/"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="measured runs per arm (default 2). Each arm also "
        "gets one discarded warm-up, which the first ANE "
        "run needs for Core ML's compile",
    )
    parser.add_argument(
        "--prompt-characters",
        type=int,
        default=PROMPT_CHARACTERS,
        help=f"prompt size (default {PROMPT_CHARACTERS}, about "
        f"4,300 tokens). The prompt must reach one full "
        f"--prefill-chunk or the ANE cannot engage at all",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=PREFILL_CHUNK,
        help=f"the runtime's prefill chunk (default "
        f"{PREFILL_CHUNK}). It must equal the sidecar's "
        f"chunk, so it also picks which sidecar directory "
        f"is loaded — a model can carry several widths "
        f"(ane_prefill-1024 beside ane_prefill)",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--ram-budget",
        default=None,
        help="passed through to the CLI as --ram-budget (e.g. 10G), so a "
        "budgeted install is measured at the budget it runs at",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="write benchmark/ane-prefill/<label>.json, after "
        "every model so a held run keeps what it measured",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="with --record, skip models already measured into "
        "the record (this is how a held run resumes)",
    )
    args = parser.parse_args()

    global EXTRA_ARGS
    EXTRA_ARGS = ["--ram-budget", args.ram_budget] if args.ram_budget else []

    chunk_floor = int(args.prefill_chunk * CHARACTERS_PER_TOKEN)
    if args.prompt_characters < chunk_floor:
        print(
            f"warning: {args.prompt_characters} characters is under one full "
            f"{args.prefill_chunk}-token chunk (~{chunk_floor} characters), so "
            f"the ANE arm would measure the GPU path",
            file=sys.stderr,
        )

    path = None
    record = new_record(args.repeats, args.prompt_characters, MAX_NEW, args.prefill_chunk)
    if args.record:
        RESULTS.mkdir(parents=True, exist_ok=True)
        label = args.label or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = RESULTS / f"{label}.json"
        if path.exists() and args.skip_done:
            try:
                record = json.loads(path.read_text())
                record.setdefault("results", [])
            except ValueError:
                record = new_record(
                    args.repeats, args.prompt_characters, MAX_NEW, args.prefill_chunk
                )
    done = stored_models(record)
    if args.skip_done and done:
        print(f"resuming {path.name}: {len(done)} model(s) already measured", flush=True)

    header = f"{'model':<44} {'off s':>9} {'on s':>9} {'speedup':>8} {'ANE used':>9}  note"
    print(header, flush=True)

    # Each model is printed *and stored* as it finishes: a full matrix is hours
    # of prefill, and a run that is held or interrupted at hour two must not
    # discard the rows it already earned.
    for name in args.models:
        if args.skip_done and name in done:
            print(f"{name:<44} skipped (already in {path.name})", flush=True)
            continue
        result = summarize(
            measure(name, args.repeats, args.prompt_characters, args.prefill_chunk),
            args.prefill_chunk,
        )
        print(format_row(result), flush=True)
        if path is not None:
            store_result(record, result)
            record["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            )
            # Whole-file rewrite after each model: the file is small and a
            # partial row is worse than none.
            path.write_text(json.dumps(record, indent=2) + "\n")

    if path is not None:
        print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
