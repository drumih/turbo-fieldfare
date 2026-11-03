#!/usr/bin/env python3
"""Record TinyTitan's internal speeds for a release, and compare them.

One command measures the engine's own numbers on a fixed prompt and writes them
to `benchmark/internal-speeds/<label>.json`, so the next release can be diffed
against this one and a speed regression is caught before it ships:

  GPU   fused QKV GEMV, routed MoE and GDN in-projection bandwidth (GB/s)
  CPU   int8 affine GEMV bandwidth (GB/s) at every thread width
  model prefill tok/s, decode tok/s, TTFT, total seconds, and the effective
        decode bandwidth (weight bytes / decode seconds)
  ANE   prefill tok/s when the model ships an ANE sidecar; recorded as not
        applicable with the reason otherwise (the sidecar exporter is
        qwen36-only, so a dense install cannot have one)
  text  the response, its SHA-256, and a small quality proxy: length, keyword
        coverage and trigram repetition

Usage:
  tools/internal-speeds.py --record [--label v5.6]
      [--model models/qwen3.5_4B_4Bit] [--max-new 256]
      [--baseline benchmark/internal-speeds/<previous>.json]
  tools/internal-speeds.py --compare <baseline.json> <candidate.json>
      [--threshold 10]

`--record --baseline <previous>` measures, writes the new record, prints the
comparison, and exits non-zero when a performance metric regressed by more than
the threshold. That is the release gate; see docs/release-process.md.

Records are per (model, prompt): with no `--baseline`, the comparison picks the
newest previous record **for the same model and prompt**, because a 125B MoE's
decode rate is not a baseline for the 4B's. Different models are recorded side
by side — the ANE prefill number only exists for a qwen36 install with a
sidecar — and are never compared against one another.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / ".build/release/TinyTitanBench"
CLI = ROOT / ".build/release/TinyTitanCLI"
RESULTS = ROOT / "benchmark/internal-speeds"

DEFAULT_MODEL = "models/qwen3.5_4B_4Bit"
DEFAULT_PROMPT = "difference swift vs c++ in detail"
DEFAULT_MAX_NEW = 256
DEFAULT_THRESHOLD = 10.0

# The ANE prefill sidecar is a fixed 4,096-token program, and the runtime only
# routes a chunk to it when the configured prefill chunk matches and the chunk
# is full (a short prompt is padded at ~2 s of ANE work against under a second
# on the GPU, so it falls back on purpose). Measuring ANE prefill therefore
# needs a prompt that fills a chunk and an explicit 4,096 chunk; the fixed
# short prompt's ANE row would otherwise report the GPU fallback under an ANE
# label.
ANE_PREFILL_CHUNK = 4096
ANE_PROMPT_CHARACTERS = 32_000
ANE_PROMPT_SENTENCE = (
    "Swift and C++ differ in memory management, dispatch, compilation and type "
    "safety, and a fair comparison names each axis before it judges either "
    "language. "
)
ANE_FALLBACK_MARKER = "ane-prefill fallback"

FOOTER = re.compile(
    r"\[stop=(\S+) prefill=(\d+)tok/([\d.]+)s "
    r"new=(\d+)tok decode=([\d.]+)s tok/s=([\d.]+)\]"
)
ACHIEVED = re.compile(r"achieved=([\d.]+) GB/s")
BYTES_PER_LAUNCH = re.compile(r"bytes/launch=(\d+)")
CPU_ROW = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

# The prompt asks for a comparison, so a healthy answer names the languages and
# the axes they differ on. Coverage is a proxy, not a grade: it moves when the
# model stops answering the question, and it is recorded next to the text so a
# human can check the one release where it drops.
QUALITY_KEYWORDS = (
    "swift",
    "c++",
    "performance",
    "memory",
    "compile",
    "type",
    "arc",
    "manual",
    "safety",
    "runtime",
    "speed",
    "garbage",
)

# metric path -> "higher" (a drop is a regression) or "lower" (a rise is one)
PERF_METRICS = {
    "gpu.qkv_gemv_gbps": "higher",
    "gpu.routed_moe_gbps": "higher",
    "gpu.gdn_inproj_gbps": "higher",
    "cpu.best_gbps": "higher",
    "generation.prefill_tokens_per_second": "higher",
    "generation.decode_tokens_per_second": "higher",
    "generation.effective_decode_gbps": "higher",
    "generation.ttft_seconds": "lower",
    "generation.decode_seconds": "lower",
    "generation.total_seconds": "lower",
    "ane.prefill_tokens_per_second": "higher",
    "ane.effective_prefill_gbps": "higher",
}


def run(cmd: list[str], env: dict | None = None, timeout: int = 3600) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=ROOT, check=False
    )
    return proc.returncode, proc.stdout, proc.stderr


def git(*args: str) -> str:
    code, out, _ = run(["git", *args], timeout=30)
    return out.strip() if code == 0 else ""


def model_family(model: str) -> str | None:
    """The manifest's architecture family, or None when it cannot be read."""
    try:
        manifest = json.loads((ROOT / model / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    arch = manifest.get("arch")
    return arch.get("family") if isinstance(arch, dict) else None


def model_total_bytes(model: str) -> int:
    """Every byte the manifest declares, falling back to the resident weights.

    `model_weights.bin` is the whole model only for a dense install. A MoE keeps
    the routed experts in `packed_experts/` — AgentWorld 4-bit declares 1.92 GB
    of 20.08 GB — and those are the bytes a prefill chunk actually sweeps.
    """
    try:
        manifest = json.loads((ROOT / model / "manifest.json").read_text())
        files = manifest.get("files")
        if isinstance(files, dict) and files:
            total = sum(
                int(entry.get("size", 0)) for entry in files.values() if isinstance(entry, dict)
            )
            if total > 0:
                return total
    except (OSError, ValueError, TypeError):
        pass
    weights = ROOT / model / "model_weights.bin"
    return weights.stat().st_size if weights.exists() else 0


# Families the exporter can build a sidecar for, because the Core ML graph
# computes their attention block. `qwen38flash` is one of them — its QSA indexer
# selects keys rather than changing the arithmetic, and the runtime folds that
# selection into the additive mask the graph already takes — but *installing* a
# sidecar for it is not recommended, which `missing_ane_reason` says below. The
# one-layer MTP draft is not: the runtime verifies it rather than prefilling it
# on the ANE.
ANE_EXPORTABLE_FAMILIES = ("qwen36", "qwen3_5_dense", "qwen38flash", "qwen4flash")


def missing_ane_reason(model: str, family: str | None) -> str:
    """Why no ANE number was recorded, in terms of what was observed."""
    if family in ("qwen38flash", "qwen4flash"):
        return (
            f"the ANE prefill path does not pay for a {family} model: "
            f"measured 0.72x against the GPU at 4,333 tokens on the 3.8 "
            f"preview, because the GPU already attends to only the indexer's "
            f"~2,051 selected keys while the ANE graph is dense over the "
            f"context and its per-variant Core ML load is 7-14 s. A sidecar "
            f"is exportable only to re-measure"
        )
    if family in ("qwen38flash_mtp", "qwen4flash_mtp"):
        return (
            f"ANE prefill does not serve a {family} model: the one-layer "
            f"MTP draft is verified rather than prefilled on the ANE"
        )
    reason = (
        f"no ANE prefill sidecar at {model}/ane_prefill; export one "
        f"with tools/export_ane_prefill.py --model {model}"
    )
    if family and family not in ANE_EXPORTABLE_FAMILIES:
        reason += f" — the exporter has no graph for the {family} family"
    return reason


def ane_prompt() -> str:
    """A deterministic prompt long enough to fill an ANE chunk."""
    repeats = ANE_PROMPT_CHARACTERS // len(ANE_PROMPT_SENTENCE) + 1
    return (ANE_PROMPT_SENTENCE * repeats)[:ANE_PROMPT_CHARACTERS]


def ane_usage(stderr: str) -> tuple[bool, str | None]:
    """Did the ANE path run? Returns (used, fallback line when it did not).

    The runtime prints one fallback line per request when a chunk is not
    eligible, so its absence is the evidence that the ANE served the prefill.
    """
    for line in stderr.splitlines():
        if ANE_FALLBACK_MARKER in line:
            return False, line.strip()
    return True, None


def ane_cache_is_warm(model: str) -> bool:
    """Has Core ML already compiled this model's sidecar for the ANE?

    The runtime caches the compiled programs beside the sidecar, so a cold cache
    means the first run also pays the compile and its rate is not comparable.
    """
    return bool(sorted((ROOT / model / "ane_prefill").glob("compiled-v*")))


def measure_kernel(kernel: str, iterations: int) -> dict:
    code, out, err = run([str(BENCH), kernel, str(iterations)], timeout=600)
    log = out + err
    achieved = ACHIEVED.search(log)
    if code != 0 or not achieved:
        return {"error": f"{kernel}: exit {code}", "output": log[-400:]}
    record: dict = {"gbps": float(achieved.group(1))}
    launch = BYTES_PER_LAUNCH.search(log)
    if launch:
        record["bytes_per_launch"] = int(launch.group(1))
    return record


def measure_cpu_gemv(iterations: int) -> dict:
    code, out, err = run([str(BENCH), "cpugemv", str(iterations)], timeout=600)
    if code != 0:
        return {"error": f"cpugemv: exit {code}", "output": (out + err)[-400:]}
    rows = []
    for line in out.splitlines():
        match = CPU_ROW.match(line)
        if match:
            rows.append(
                {
                    "threads": int(match.group(1)),
                    "ms_per_pass": float(match.group(2)),
                    "gbps": float(match.group(3)),
                    "tok_per_second_2b_8bit": float(match.group(4)),
                }
            )
    if not rows:
        return {"error": "cpugemv: no table rows parsed", "output": out[-400:]}
    best = max(rows, key=lambda row: row["gbps"])
    return {"best_gbps": best["gbps"], "best_threads": best["threads"], "rows": rows}


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


def measure_generation(
    model: str,
    prompt: str,
    max_new: int,
    ane: bool = False,
    prefill_chunk: int | None = None,
    ram_budget: str | None = None,
) -> dict:
    import os

    env = os.environ.copy()
    if ane:
        env["TINYTITAN_PREFILL_ANE"] = "on"
    command = [
        str(CLI),
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-new",
        str(max_new),
        "--temperature",
        "0",
    ]
    if prefill_chunk is not None:
        command += ["--prefill-chunk", str(prefill_chunk)]
    if ram_budget is not None:
        command += ["--ram-budget", ram_budget]
    code, out, err = run(command, env=env, timeout=3600)
    if code != 0:
        return {"error": f"cli exit {code}", "stderr": err[-800:]}
    measured = parse_footer(err)
    if measured is None:
        return {"error": "no [stop=...] footer", "stderr": err[-800:]}
    text = out.strip()
    prefill_tokens = measured["prefill_tokens"]
    decode_seconds = measured["decode_seconds"]
    weights = ROOT / model / "model_weights.bin"
    weight_bytes = weights.stat().st_size if weights.exists() else 0
    prefill_seconds = measured["prefill_seconds"]
    measured.update(
        {
            "prefill_tokens_per_second": prefill_tokens / prefill_seconds
            if prefill_seconds
            else 0.0,
            # The first generated token is sampled at the end of prefill, so the
            # prompt's processing time is the time to first token.
            "ttft_seconds": prefill_seconds,
            "total_seconds": prefill_seconds + decode_seconds,
            "weight_bytes": weight_bytes,
            # Every token re-reads the weight set, so the bandwidth is the weight
            # bytes times the tokens decoded, over the decode wall time.
            "effective_decode_gbps": weight_bytes * measured["decode_tokens"] / decode_seconds / 1e9
            if decode_seconds
            else 0.0,
            "response": text,
            "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "response_characters": len(text),
            "response_tokens": measured["decode_tokens"],
        }
    )
    if ane:
        # A prefill chunk sweeps the whole weight set -- 4,096 tokens at top-8
        # touch every routed expert -- so the traffic is the model's declared
        # bytes per chunk, not the resident weights. It is an end-to-end prefill
        # bandwidth (the ANE attends, the experts still run on the GPU and
        # stream from disk), not an ANE-only figure, which one wall time cannot
        # separate.
        chunks = max(1, -(-prefill_tokens // ANE_PREFILL_CHUNK))
        total_bytes = model_total_bytes(model)
        measured["chunks"] = chunks
        measured["model_total_bytes"] = total_bytes
        measured["effective_prefill_gbps"] = (
            total_bytes * chunks / prefill_seconds / 1e9 if prefill_seconds else 0.0
        )
        # A fallback line means the number is the GPU's, under an ANE label.
        # Say so rather than record it as an ANE result.
        measured["used_ane"], fallback = ane_usage(err)
        if fallback:
            measured["fallback_reason"] = fallback
    return measured


def quality(measured: dict) -> dict:
    text = measured.get("response", "")
    lowered = text.lower()
    present = [word for word in QUALITY_KEYWORDS if word in lowered]
    words = re.findall(r"[a-z0-9+]+", lowered)
    trigrams = [tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2))]
    unique = len(set(trigrams))
    repetition = 1.0 - (unique / len(trigrams)) if trigrams else 0.0
    return {
        "keyword_coverage": len(present) / len(QUALITY_KEYWORDS),
        "keywords_present": present,
        "trigram_repetition": repetition,
        "response_sha256": measured.get("response_sha256"),
        "characters": measured.get("response_characters", 0),
    }


def environment() -> dict:
    _, swift, _ = run(["swift", "--version"], timeout=60)
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_describe": git("describe", "--tags", "--always", "--dirty"),
        "macos": platform.mac_ver()[0],
        "swift": swift.strip(),
        "python": platform.python_version(),
        "physical_memory_bytes": None,
    }


def measure(
    model: str, prompt: str, max_new: int, iterations: int, ram_budget: str | None = None
) -> dict:
    print(f"measuring kernels ({iterations} iterations each)...", flush=True)
    gpu = {
        "qkv_gemv": measure_kernel("baseline", iterations),
        "routed_moe": measure_kernel("moe", iterations),
        "gdn_inproj": measure_kernel("gdn_inproj", iterations),
    }
    gpu["qkv_gemv_gbps"] = gpu["qkv_gemv"].get("gbps")
    gpu["routed_moe_gbps"] = gpu["routed_moe"].get("gbps")
    gpu["gdn_inproj_gbps"] = gpu["gdn_inproj"].get("gbps")

    print("measuring CPU GEMV...", flush=True)
    cpu = measure_cpu_gemv(iterations)

    print(f"measuring generation on {model} ...", flush=True)
    generation = measure_generation(model, prompt, max_new, ram_budget=ram_budget)

    model_dir = ROOT / model
    sidecar = model_dir / "ane_prefill/ane_prefill.json"
    family = model_family(model)
    if sidecar.exists():
        print("measuring ANE prefill (sidecar present)...", flush=True)
        warm = ane_cache_is_warm(model)
        if not warm:
            # Core ML compiles the sidecar for the ANE on first use. On
            # AgentWorld 4-bit that added ~68 s to a 86 s prefill, so a cold
            # measurement reads as a ~44% regression against a warm one and
            # would block a release that changed nothing. Warm it first and say
            # the run was warmed.
            print(
                "  the ANE compile cache is cold; warming it first so the number is comparable",
                file=sys.stderr,
                flush=True,
            )
            measure_generation(
                model,
                ane_prompt(),
                1,
                ane=True,
                prefill_chunk=ANE_PREFILL_CHUNK,
                ram_budget=ram_budget,
            )
        ane = measure_generation(
            model, ane_prompt(), 1, ane=True, prefill_chunk=ANE_PREFILL_CHUNK, ram_budget=ram_budget
        )
        ane["cache_was_warm"] = warm
        ane["applicable"] = "error" not in ane and ane.get("used_ane", False)
        if "error" not in ane and not ane.get("used_ane", False):
            ane["reason"] = "the runtime did not route the chunk to the ANE: " + ane.get(
                "fallback_reason", ""
            )
    else:
        ane = {"applicable": False, "reason": missing_ane_reason(model, family)}

    env = environment()
    try:
        env["physical_memory_bytes"] = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.strip()
        )
    except Exception:
        pass

    return {
        "schema": 1,
        "environment": env,
        "model": {
            "path": model,
            "family": family,
            "prompt": prompt,
            "max_new_tokens": max_new,
            "ram_budget": ram_budget,
        },
        "gpu": gpu,
        "cpu": cpu,
        "generation": generation,
        "ane": ane,
        "quality": quality(generation),
    }


def get_path(record: dict, dotted: str):
    node = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def compare(baseline: dict, candidate: dict, threshold: float) -> bool:
    """Print the diff and return True when nothing regressed.

    A metric that is not a number on both sides is skipped rather than failed,
    because a qwen36-only ANE row is absent from every other model's record. That
    is why the count is printed: the same rule makes a candidate whose probes all
    wrote `{"error": …}` compare as zero rows and no regression, which is a record
    that measured nothing, not a build that held its speeds.
    """
    print(f"{'metric':<44} {'baseline':>12} {'now':>12} {'delta':>9}  status")
    ok = True
    compared = 0
    for metric, direction in PERF_METRICS.items():
        before = get_path(baseline, metric)
        after = get_path(candidate, metric)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        if before == 0:
            continue
        compared += 1
        delta = (after - before) / before * 100.0
        if direction == "higher":
            regressed = delta < -threshold
        else:
            regressed = delta > threshold
        status = "regressed" if regressed else "ok"
        if regressed:
            ok = False
        print(f"{metric:<44} {before:>12.3f} {after:>12.3f} {delta:>8.1f}%  {status}")

    print(f"{compared} metric(s) compared")
    base_quality = baseline.get("quality", {})
    now_quality = candidate.get("quality", {})
    base_cov = base_quality.get("keyword_coverage")
    now_cov = now_quality.get("keyword_coverage")
    if isinstance(base_cov, (int, float)) and isinstance(now_cov, (int, float)):
        print(
            f"{'quality.keyword_coverage':<44} {base_cov:>12.3f} "
            f"{now_cov:>12.3f} {(now_cov - base_cov) * 100:>8.1f}%  "
            f"{'ok' if now_cov >= base_cov - 0.01 else 'dropped'}"
        )
        if now_cov < base_cov - 0.01:
            ok = False
    if base_quality.get("response_sha256") != now_quality.get("response_sha256"):
        print(
            "note: the greedy response changed; review the recorded text "
            "before treating the numbers as comparable"
        )
    if compared == 0:
        print(
            "FAIL: no metric was comparable — a candidate whose probes all failed "
            "is not a candidate that regressed in nothing"
        )
        return False
    return ok


def default_label(describe: str, model: str) -> str:
    """The record's file name: checkout state, plus the model when it is not the
    mandatory one.

    Two models recorded at the same commit must not resolve to one file — the
    4B's record is the release gate's baseline and must not be overwritten by an
    extra model's.
    """
    if model == DEFAULT_MODEL:
        return describe
    return f"{describe}-{pathlib.Path(model).name}"


def newest_baseline(out: pathlib.Path, model: str, prompt: str) -> str | None:
    """The newest previous record for the same model and prompt.

    Records are separate files per model (only a qwen36 install can have an ANE
    sidecar), so the newest file on disk is not necessarily a comparable one.
    """
    for path in sorted(RESULTS.glob("*.json"), reverse=True):
        if path.resolve() == out.resolve():
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        recorded = record.get("model") or {}
        if recorded.get("path") == model and recorded.get("prompt") == prompt:
            return str(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--baseline")
    parser.add_argument("--label")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--ram-budget",
        help="resident-memory target passed to the CLI (e.g. 12G). It is "
        "recorded in the model block, so a row says which ceiling produced it",
    )
    args = parser.parse_args()

    if args.compare:
        baseline = json.load(open(args.compare[0], encoding="utf-8"))
        candidate = json.load(open(args.compare[1], encoding="utf-8"))
        print(f"baseline {args.compare[0]} -> candidate {args.compare[1]}")
        return 0 if compare(baseline, candidate, args.threshold) else 1

    if not args.record:
        parser.error("pass --record or --compare")

    record = measure(args.model, args.prompt, args.max_new, args.iterations, args.ram_budget)
    RESULTS.mkdir(parents=True, exist_ok=True)
    label = (
        args.label
        or default_label(record["environment"]["git_describe"], args.model)
        or "unlabeled"
    )
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")

    generation = record["generation"]
    if "error" in generation:
        print(f"generation FAILED: {generation['error']}", file=sys.stderr)
        return 1
    print(
        f"  prefill {generation['prefill_tokens_per_second']:.1f} tok/s, "
        f"decode {generation['decode_tokens_per_second']:.1f} tok/s, "
        f"ttft {generation['ttft_seconds']:.2f}s, "
        f"effective decode {generation['effective_decode_gbps']:.1f} GB/s"
    )
    print(
        f"  gpu qkv {record['gpu']['qkv_gemv_gbps']} GB/s, "
        f"moe {record['gpu']['routed_moe_gbps']} GB/s, "
        f"gdn {record['gpu']['gdn_inproj_gbps']} GB/s; "
        f"cpu best {record['cpu'].get('best_gbps')} GB/s"
    )
    print(
        f"  quality coverage {record['quality']['keyword_coverage']:.2f}, "
        f"trigram repetition {record['quality']['trigram_repetition']:.2f}"
    )

    baseline_path = args.baseline
    if baseline_path is None:
        # Default to the newest previous record *of the same run shape* so a
        # release always compares against something without being told which
        # file, and never against a different model's numbers.
        baseline_path = newest_baseline(out, args.model, args.prompt)
        if baseline_path:
            print(
                f"comparing against the newest previous record for this "
                f"model and prompt: {baseline_path}"
            )
        else:
            print(
                f"no previous record for {args.model} on this prompt; this "
                f"is the first, so there is nothing to compare",
                file=sys.stderr,
            )
    if baseline_path:
        baseline = json.load(open(baseline_path, encoding="utf-8"))
        print()
        return 0 if compare(baseline, record, args.threshold) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
