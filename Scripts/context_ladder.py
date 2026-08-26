#!/usr/bin/env python3
"""Measure controlled long-context server timing and memory behavior.

Each ladder level starts a fresh release ``TurboFieldfareServer`` on the
loopback interface, waits for its ready message, and submits one
OpenAI-compatible chat completion with ``max_completion_tokens=1``. The prompt
is deliberately reproducible rather than semantic: ``"word "`` is repeated
until the selected context cap is reached while reserving
``UNIFORM_RESERVE_TOKENS`` and the measured chat-template overhead. Server
reported prompt usage is authoritative; the packing estimate is not treated as
the token count.

While the request runs, the harness samples macOS memory pressure, swap usage,
process footprint, RSS/VSZ, and Metal allocation every ten seconds. It also
captures the server's completion log, including prompt/decode durations and
throughput. A level is considered complete only when the HTTP request succeeds,
the server reports the expected prompt and one-token completion, and the
request is not rejected, timed out, cancelled, or terminated by OOM. Resource
pressure is recorded as measurement data rather than a hidden pass/fail rule.

WHAT THIS DOES NOT MEASURE
--------------------------
A one-token completion on repeated filler proves *admission*: that the server
accepts the prompt, prefills it, and stays within memory. It cannot show that
the model still attends across that span. Gemma 4 runs sliding-window attention
on 25 of its 30 layers, so long-range signal rests on 5 full-attention layers
and a degraded long context would still return HTTP 200 here. Use
``Scripts/context_retrieval.py`` for that question; the two are complementary
and neither substitutes for the other.

Rungs are only comparable when produced by the same binary. Record the commit
for every run and re-measure the whole ladder after any change to the prefill
or attention path.

Examples
--------
Full default ladder::

    python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo

One rung, merged into the existing aggregate::

    python3 Scripts/context_ladder.py --model scratch/gemma4.gturbo --levels 64
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
DEFAULT_PORT = 8080
DEFAULT_MODEL_ID = "gemma-4-26b-a4b-it"
UNIFORM_RESERVE_TOKENS = 8_192
CHAT_TEMPLATE_OVERHEAD_TOKENS = 13
# Label -> context cap. 64 is the baseline rung; it is part of the ladder so a
# baseline can be produced by the same binary and protocol as the rungs it is
# compared against.
LEVEL_CONTEXTS = {
    64: 65_536,
    96: 98_304,
    128: 131_072,
    192: 196_608,
    256: 262_144,
}
DEFAULT_LEVELS = [64, 96, 128, 192, 256]
DEFAULT_OUT = ROOT / "benchmark-results" / "context-ladder"


def command_text(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def git_commit() -> str | None:
    try:
        return command_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()
    except Exception:
        return None


def memory_sample(pid: int, phase: str) -> dict:
    sample: dict[str, object] = {"time": time.time(), "phase": phase}
    try:
        text = command_text(["memory_pressure", "-Q"])
        match = re.search(r"free percentage: (\d+)%", text)
        if match:
            sample["memory_free_pct"] = int(match.group(1))
    except Exception as exc:
        sample["memory_error"] = str(exc)
    try:
        text = command_text(["sysctl", "vm.swapusage"])
        match = re.search(r"used = ([0-9.]+)([MG])", text)
        if match:
            value = float(match.group(1))
            sample["swap_used_mb"] = value * (1024 if match.group(2) == "G" else 1)
    except Exception as exc:
        sample["swap_error"] = str(exc)
    try:
        text = command_text(["ps", "-o", "rss=,vsz=,%mem=,etime=", "-p", str(pid)])
        parts = text.split()
        if len(parts) >= 4:
            sample.update({"rss_kb": int(parts[0]), "vsz_kb": int(parts[1]), "percent_mem": float(parts[2]), "elapsed": parts[3]})
    except Exception as exc:
        sample["ps_error"] = str(exc)
    try:
        text = command_text(["footprint", "-p", str(pid)])
        match = re.search(r"Footprint:\s+([0-9.]+) MB", text)
        if match:
            sample["footprint_mb"] = float(match.group(1))
        match = re.search(r"(\d+(?:\.\d+)?) MB\s+0 B\s+0 B\s+\d+\s+IOAccelerator \(graphics\)", text)
        if match:
            sample["ioaccelerator_mb"] = float(match.group(1))
        match = re.search(r"phys_footprint_peak:\s+([0-9.]+) MB", text)
        if match:
            sample["footprint_peak_mb"] = float(match.group(1))
    except Exception as exc:
        sample["footprint_error"] = str(exc)
    return sample


def wait_ready(proc: subprocess.Popen[str], logs: list[str], ready: threading.Event) -> None:
    if proc.stdout is None:
        return
    for line in proc.stdout:
        line = line.rstrip()
        logs.append(line)
        if "TurboFieldfareServer ready" in line:
            ready.set()


def run_level(label: int, max_context: int, options: argparse.Namespace) -> dict:
    words = max_context - options.reserve - CHAT_TEMPLATE_OVERHEAD_TOKENS
    command = [
        str(options.server),
        "--model", str(options.model),
        "--port", str(options.port),
        "--max-context", str(max_context),
    ]
    started = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs: list[str] = []
    ready = threading.Event()
    threading.Thread(target=wait_ready, args=(proc, logs, ready), daemon=True).start()
    if not ready.wait(options.ready_timeout):
        proc.terminate()
        raise RuntimeError(f"server did not become ready for {label}K: {logs[-10:]}")

    samples: list[dict] = []
    stop_sampling = threading.Event()

    def sample_loop() -> None:
        while not stop_sampling.is_set():
            samples.append(memory_sample(proc.pid, "request" if len(samples) else "ready"))
            stop_sampling.wait(options.sample_interval)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    text = "word " * words
    payload = json.dumps({
        "model": options.model_id,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_completion_tokens": 1,
    }).encode()
    request_start = time.time()
    result: dict[str, object] = {
        "label_k": label,
        "max_context": max_context,
        "target_words": words,
        "uniform_reserve_tokens": options.reserve,
        "expected_prompt_tokens": words + CHAT_TEMPLATE_OVERHEAD_TOKENS,
        "server_pid": proc.pid,
        "server_command": " ".join(command),
        "source_commit": options.commit,
        "client_timeout_seconds": options.request_timeout,
        "ready_seconds": round(request_start - started, 3),
        "samples": samples,
    }
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{options.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=options.request_timeout) as response:
            body = json.loads(response.read())
        result["client_status"] = "completed"
        result["client_elapsed_seconds"] = round(time.time() - request_start, 3)
        result["usage"] = body.get("usage")
    except Exception as exc:
        result["client_status"] = "error"
        result["client_error"] = f"{type(exc).__name__}: {exc}"
        result["client_elapsed_seconds"] = round(time.time() - request_start, 3)
    finally:
        stop_sampling.set()
        sampler.join(timeout=2)
        # Give the log reader a moment to receive the completion line.
        time.sleep(1)
        completion_line = next((line for line in reversed(logs) if " completed in " in line), None)
        cancel_line = next((line for line in reversed(logs) if " cancelled by client " in line), None)
        result["completion_log"] = completion_line
        result["cancel_log"] = cancel_line
        result["samples"] = samples
        for key in ("footprint_mb", "ioaccelerator_mb", "rss_kb", "vsz_kb", "percent_mem"):
            values = [sample[key] for sample in samples if key in sample]
            if values:
                result[f"peak_{key}"] = max(values)
        free_values = [sample["memory_free_pct"] for sample in samples if "memory_free_pct" in sample]
        swap_values = [sample["swap_used_mb"] for sample in samples if "swap_used_mb" in sample]
        if free_values:
            result["minimum_memory_free_pct"] = min(free_values)
        if swap_values:
            result["peak_swap_used_mb"] = max(swap_values)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=15)
    match = completion_line and re.search(
        r"completed in ([0-9.]+)s prompt=(\d+) cached=(\d+) completion=(\d+) "
        r"pp=([0-9.]+)s pp_tok_s=([0-9.]+) tg=([0-9.]+)s tg_tok_s=([0-9.]+) finish=(\w+)",
        completion_line,
    )
    if match:
        result.update({
            "server_total_seconds": float(match.group(1)),
            "prompt_tokens": int(match.group(2)),
            "cached_tokens": int(match.group(3)),
            "completion_tokens": int(match.group(4)),
            "pp_seconds": float(match.group(5)),
            "pp_tokens_per_second": float(match.group(6)),
            "tg_seconds": float(match.group(7)),
            "tg_tokens_per_second": float(match.group(8)),
            "finish_reason": match.group(9),
        })
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TurboFieldfare long-context ladder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Completed .gturbo model directory.")
    parser.add_argument("--server", type=Path,
                        default=ROOT / ".build/release/TurboFieldfareServer",
                        help="Release server binary.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="API model identifier sent in the request body.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Loopback port for the harness-owned server.")
    parser.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS),
                        help="Comma-separated rung labels in K, e.g. '64' or '64,256'.")
    parser.add_argument("--reserve", type=int, default=UNIFORM_RESERVE_TOKENS,
                        help="Tokens reserved inside every cap. Keep uniform across rungs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Directory for per-rung and aggregate artifacts.")
    parser.add_argument("--request-timeout", type=float, default=3600,
                        help="Client time bound for one request, in seconds.")
    parser.add_argument("--ready-timeout", type=float, default=120,
                        help="Seconds to wait for the server ready line.")
    parser.add_argument("--sample-interval", type=float, default=10,
                        help="Seconds between resource samples.")
    options = parser.parse_args(argv)
    try:
        options.levels = [int(part) for part in options.levels.split(",") if part.strip()]
    except ValueError:
        parser.error("--levels must be a comma-separated list of integers")
    unknown = [x for x in options.levels if x not in LEVEL_CONTEXTS]
    if unknown:
        parser.error(f"unknown level(s) {unknown}; choose from {sorted(LEVEL_CONTEXTS)}")
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

    # Merge into any existing aggregate so a single rung can be re-measured
    # without repeating the hours-long rungs beside it.
    aggregate_path = options.out / "results.json"
    aggregate: dict[int, dict] = {}
    if aggregate_path.is_file():
        try:
            for entry in json.loads(aggregate_path.read_text()):
                aggregate[entry["label_k"]] = entry
        except Exception as exc:
            print(f"ignoring unreadable aggregate {aggregate_path}: {exc}", file=sys.stderr)

    for label in options.levels:
        max_context = LEVEL_CONTEXTS[label]
        words = max_context - options.reserve - CHAT_TEMPLATE_OVERHEAD_TOKENS
        print(f"START {label}K context={max_context} words={words} reserve={options.reserve}", flush=True)
        result = run_level(label, max_context, options)
        aggregate[label] = result
        print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "completion_log", "cancel_log"}}, sort_keys=True), flush=True)
        (options.out / f"{label}K.json").write_text(json.dumps(result, indent=2) + "\n")

    ordered = [aggregate[k] for k in sorted(aggregate)]
    aggregate_path.write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"WROTE {aggregate_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
