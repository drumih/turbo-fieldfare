#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass". Phase 2 of the production audit needs this before RealForwardRunner
# is decomposed.
#
#   tools/golden-baseline.sh [target ...]     # default: ornith-8
#   tools/golden-baseline.sh --check [...]    # compare against the stored file
#   tools/golden-baseline.sh --server [...]   # CLI and server must agree
#
# Targets: ornith-4, ornith-8, ornith-bf16dense-4, ornith122-4, ornith122-8,
# qwen36-4, qwen36-8, qwen38-4, qwen38-8, agentworld-4, agentworld-8,
# katcoder-4, katcoder-8, k2-4, k2-8, and the dense qwen35-{2b,4b,9b}-{4,8}.
# The katcoder and qwen35 targets are declared whether
# or not their install is present: a target is inert until the directory is
# there, and release.sh demands the baseline as soon as it is.
# Bare 4 and 8 still mean
# Ornith, because the stored baselines are named for it.
#
# Qwen3.8-Flash-Next is the broader gate: it is the only family exercising
# hyper-connections, the PLE block and the QSA indexer, and at 8-bit it is the
# only one exercising the affine GEMV path through them. Ornith reaches none
# of that, so an Ornith-only baseline can pass while those kernels are wrong.
#
# Determinism comes from greedy decoding: --temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels — exactly what a runtime refactor must not change.
#
# --server exists because the stored baselines only ever drove TinyTitanCLI, and a
# server binary 15 hours older than the runtime it links went unnoticed until
# it refused a promoted 8-bit model by hand. The two modes catch different
# things and neither substitutes for the other:
#
#   the stored file   the runtime changed        (CLI vs its own past output)
#   --server          the two front ends diverged (CLI vs server, same build)
#
# So --server stores nothing. It renders one message list through both paths at
# the same settings and requires the text to match, which needs no re-capture
# after a deliberate numerics change and cannot go stale. Comparing raw
# --prompt output against the server would be meaningless: the server always
# applies the chat template, so both sides are driven from --messages-file.
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M3 is not a reference for an M4. Re-capture after a deliberate numerics
# change; a diff at any other time is a regression.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="$ROOT/.build/release/TinyTitanCLI"
# K2-Horizon has no CLI path: its baseline drives the GPU probe instead, which
# is the same graph the server serves. One bench binary, one baseline format.
BENCH="$ROOT/.build/release/TinyTitanBench"
# shellcheck source=model-guard.sh
. "$SCRIPT_DIR/model-guard.sh"
SERVER="$ROOT/.build/release/TinyTitanServer"
LAUNCHER="$ROOT/tools/server_launcher.sh"

# The launcher resolves a model key or the catalog id the server advertises
# (`<modelID>_<bits>-Bit`), not a directory, so read the id the install itself
# declares. python3 is already a dependency of this script.
catalog_id_for() {
  python3 - "$1" <<'PY'
import json, pathlib, sys
manifest = pathlib.Path(sys.argv[1]) / "manifest.json"
data = json.loads(manifest.read_text())
bits = data["quant"]["routedExpert"]["weightBits"]
print(f'{data["modelID"]}_{bits}-Bit')
PY
}
OUT_DIR="${OUT_DIR:-$ROOT/benchmark/golden}"

# The expert routing profile is per-machine history: it seeds the cache's LFU
# score, which changes the order experts land in their slots and therefore the
# last-bit summation order. A stored golden is one exact text per (machine,
# build, model), so this gate runs with the profile off; a profile-enabled run
# differing from it is cache-state, not a regression. See
# docs/expert-profile-warm-start-2026-10-20.md.
export TINYTITAN_EXPERT_PROFILE=0

PROMPT="${PROMPT:-Explain what a mutex is and when you would use one.}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"
# Loading a 220 GB install off SSD is minutes, not seconds, and the readiness
# line is block-buffered when stdout is a file — so poll the port, not the log.
PORT="${PORT:-8757}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

# Those four numbers are read far from here, and only one of them is checked
# anywhere before it costs a model load: MAX_NEW and SEED go to the CLI's own
# parser (sources/TinyTitanCLI/Args.swift:254) and to the python body builder
# below, PORT to the launcher, and READY_TIMEOUT to nothing but the
# `[ "$waited" -lt "$READY_TIMEOUT" ]` that decides whether the server is polled
# at all. A word is the expensive shape. `test` errors on a non-integer and takes
# the false branch -- `[ 0 -lt 1800s ]` prints "integer expression expected" and
# is false -- so the poll never runs, `ready` stays 0, and the gate reports
# "server: FAILED — not ready after 1800s": the operator's own typo returned as a
# model that loaded too slowly, over a run that never waited for anything. Refuse
# at the read, in the exit-2 style the checks below this file already use.
#
# Refused, and deliberately not:
#   - anything with a non-digit in it (`-5` too: the CLI refuses it after the
#     load, the body builder's int() accepts it, so the two readers disagree);
#   - zero for MAX_NEW and READY_TIMEOUT, both of which mean "poll/generate for
#     no time at all" and report a timeout or an empty run instead;
#   - a port the port space does not hold, and a leading-zero port, because
#     tools/server_launcher.sh:1233 range-checks with `(( ))`, which reads octal
#     and leaves the verdict unevaluated on `08757` (measured: two "value too
#     great for base" lines, then the launch proceeds unchecked).
#   - a leading zero on the other three: `test` and int() both read `0900` as 900,
#     so refusing it would be a rule no reader of this value has.
#   - an out-of-range but numeric size: that is the CLI's named error, and this
#     script does not restate the engine's bounds.
for name in MAX_NEW SEED PORT READY_TIMEOUT; do
  case "${!name}" in
    *[!0-9]*)
      echo "unknown $name: ${!name} (a number)" >&2
      exit 2
      ;;
  esac
done
if [ "$MAX_NEW" -eq 0 ]; then
  echo "unknown MAX_NEW: 0 (at least one token)" >&2
  exit 2
fi
if [ "$READY_TIMEOUT" -eq 0 ]; then
  echo "unknown READY_TIMEOUT: 0 (seconds to poll for readiness)" >&2
  exit 2
fi
case "$PORT" in
  0*)
    echo "unknown PORT: $PORT (a port 1-65535, no leading zero)" >&2
    exit 2
    ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "unknown PORT: $PORT (a port 1-65535)" >&2
  exit 2
fi

# Pin the ANE prefill switch OFF, and export it so the CLI, the server, and
# anything the server spawns all inherit it.
#
# `RuntimePrefillANE.environmentValue` returns `.on` when the variable is unset,
# so without this line the gate follows that default. It must not, because the
# ANE sidecar is deliberately NOT byte-identical to the GPU path -- fp16 with a
# different reduction order, ~1% mean per-layer deviation -- which is the one
# property this script exists to rely on.
#
# The runner reasons that stored baselines still hold because the ANE path only
# engages at >= one full 4,096-token chunk and the baseline prompt is shorter.
# That is true of the current PROMPT and MAX_NEW, but both are overridable above,
# and a longer PROMPT is the natural thing to pass when testing long context.
# Worse, whether a sidecar exists at all is a property of the machine, not of
# the (machine, build, model) triple the SCOPE note names -- so the reference
# would silently stop meaning one thing. A gate whose meaning depends on an
# unrelated local file is not a gate.
#
# This measures the GPU path, deliberately not following the default: the ANE
# path is a qualified speed/quality trade rather than the reference arithmetic,
# and a regression in the GPU path is what this has to catch.
export TINYTITAN_PREFILL_ANE=off

mode=capture
server_mode=0
targets=()
for arg in "$@"; do
  case "$arg" in
    --check)  mode=check ;;
    --server) server_mode=1 ;;
    *)        targets+=("$arg") ;;
  esac
done
[ ${#targets[@]} -eq 0 ] && targets=(ornith-8)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI — run: swift build -c release" >&2
  exit 2
fi
if [ "$server_mode" = 1 ] && [ ! -x "$SERVER" ]; then
  echo "missing $SERVER — run: swift build -c release --product TinyTitanServer" >&2
  exit 2
fi

# AGENTS.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
#
# The refusal names what it matched, because the pattern is deliberately broad:
# `swiftpm-testing-helper` belongs to *any* SwiftPM package on the machine, so a
# run in an unrelated checkout trips it. That is the safe direction -- a foreign
# test helper may itself be driving the GPU and this script cannot tell -- but
# "a model process is already running" sent the operator looking for the wrong
# thing when it was another project's tests. Do not narrow the pattern to fix
# that: reporting the matches is what lets a human judge.
#
# The guard answers three things (tools/model-guard.sh), and this used to act on
# one: `if busy=$(pgrep ... 2>/dev/null)` is false on pgrep's error status, so a
# pgrep that could not read the process table reported no model process and the
# capture ran beside the operator's server. Not knowing is a refusal too.
model_guard_status=0
busy="$(model_guard_matches 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm')" \
  || model_guard_status=$?
case "$model_guard_status" in
  0)
    echo "refusing to start: these processes match the model-process guard" >&2
    echo "$busy" | sed 's/^/  /' >&2
    echo "stop them yourself, or re-run when they are gone. This script never terminates a process it did not start." >&2
    exit 3
    ;;
  1) : ;;
  *)
    echo "refusing to start: the model-process guard could not answer, so nothing is known about what is running." >&2
    echo "See the model-guard message above. A capture taken beside a live model process measures nothing." >&2
    exit 3
    ;;
esac

# Checked, because this script does not run under `set -e` and a silent failure
# here turns every later capture into a write to a path that does not exist.
# `benchmark/golden` being a regular file rather than a directory is not
# hypothetical -- it is what a botched checkout or a stray redirect leaves.
mkdir -p "$OUT_DIR" "$ROOT/.build" || {
  echo "cannot create $OUT_DIR — is a plain file in the way?" >&2
  exit 2
}
status=0

# Only ever holds a PID this script started, which is what makes killing it in
# the trap compatible with the rule above.
srv_pid=""
cleanup() {
  if [ -n "$srv_pid" ] && kill -0 "$srv_pid" 2>/dev/null; then
    kill "$srv_pid" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$srv_pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$srv_pid" 2>/dev/null
  fi
  srv_pid=""
}
trap cleanup EXIT INT TERM

# Render the same message list through the server and through the CLI, and
# require the generated text to match.
server_check() {
  local model="$1" t="$2"
  local work msg cli_out srv_out srv_log body
  work="$(mktemp -d "$ROOT/.build/golden-server.XXXXXX")"
  msg="$work/messages.json"
  cli_out="$work/cli.txt"; srv_out="$work/server.txt"; srv_log="$work/server.log"

  python3 - "$msg" "$PROMPT" <<'PY'
import json, sys
json.dump([{"role": "user", "content": sys.argv[2]}], open(sys.argv[1], "w"))
PY

  "$CLI" --model "$model" --messages-file "$msg" --max-new "$MAX_NEW" \
         --temperature 0 --seed "$SEED" --quiet > "$cli_out" 2>"$work/cli.err"
  if [ $? -ne 0 ]; then
    echo "  server: FAILED — CLI leg exited nonzero"
    sed 's/^/    /' "$work/cli.err" | head -10
    rm -rf "$work"; return 1
  fi

  # Prompt caching is off so a rerun cannot answer from a cached prefix, which
  # would prove the cache works rather than that the paths agree.
  #
  # Started through the user-facing launcher, like every other harness, so the
  # gate's server is configured the way a user's server is. The flags the
  # launcher adds -- native 262,144 context, no YaRN, KV 8-bit, no MTP -- are
  # the server's own defaults, which is all this invocation relied on before,
  # and the launcher installs its cleanup trap with the server, so the pid
  # captured below really does own it.
  local catalog_id
  if ! catalog_id="$(catalog_id_for "$model")"; then
    echo "  server: FAILED — cannot read the catalog id from ${model#$ROOT/}"
    rm -rf "$work"; return 1
  fi
  "$LAUNCHER" --client server --model "$catalog_id" --port "$PORT" \
      --prompt-cache off --thinking off \
      > "$srv_log" 2>&1 &
  srv_pid=$!

  local ready=0 waited=0
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "  server: FAILED — exited during load"
      sed 's/^/    /' "$srv_log" | tail -5
      rm -rf "$work"; srv_pid=""; return 1
    fi
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
            "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = "200" ]; then
      ready=1; break
    fi
    sleep 5; waited=$(( waited + 5 ))
  done
  if [ "$ready" != 1 ]; then
    echo "  server: FAILED — not ready after ${READY_TIMEOUT}s"
    cleanup; rm -rf "$work"; return 1
  fi

  local model_id
  model_id="$(curl -s --max-time 30 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"

  body="$(python3 - "$model_id" "$PROMPT" "$MAX_NEW" "$SEED" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1],
                  "messages": [{"role": "user", "content": sys.argv[2]}],
                  "max_tokens": int(sys.argv[3]),
                  "seed": int(sys.argv[4]),
                  "temperature": 0}))
PY
)"
  curl -s --max-time 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$body" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"], end="")' \
    > "$srv_out" 2>"$work/srv.err"
  if [ ! -s "$srv_out" ]; then
    echo "  server: FAILED — no completion returned"
    sed 's/^/    /' "$work/srv.err" | head -10
    cleanup; rm -rf "$work"; return 1
  fi
  cleanup

  # The CLI writes the generated text to stdout; the server returns it inside a
  # JSON string. Whether a trailing newline survives that is a property of the
  # transport, not of the tokens, so normalise trailing whitespace per line and
  # trailing blank lines on both sides. Everything else still has to match
  # exactly -- this must fail when the two front ends really do disagree.
  norm() {
    python3 -c 'import sys; print("\n".join(l.rstrip() for l in sys.stdin.read().split("\n")).rstrip())' \
      < "$1"
  }
  if diff -q <(norm "$cli_out") <(norm "$srv_out") >/dev/null; then
    echo "  server: ok — CLI and server agree ($(wc -c < "$srv_out" | tr -d ' ') bytes)"
    rm -rf "$work"; return 0
  fi
  echo "  server: MISMATCH — CLI and server disagree on the same messages:"
  diff <(norm "$cli_out") <(norm "$srv_out") | head -30 | sed 's/^/    /'
  rm -rf "$work"; return 1
}

for t in "${targets[@]+"${targets[@]}"}"; do
  kind=cli
  case "$t" in
    4|ornith-4) dir="ornith-1.5_35B_A3B_4Bit"; file="ornith-1.5-35b-a3b-4bit.txt"; q=4 ;;
    8|ornith-8) dir="ornith-1.5_35B_A3B_8Bit"; file="ornith-1.5-35b-a3b-8bit.txt"; q=8 ;;
    # The bf16-dense / 4-bit-expert build is a different install, not a width:
    # its attention, GDN and shared-expert weights are the checkpoint's own
    # bf16, so its output is not expected to match ornith-4. It needs its own
    # baseline for the same reason the other targets do -- a runtime change
    # must not move this graph either -- and it is declared here before the
    # install exists so the first release that ships it cannot skip it.
    ornith-bf16dense-4) dir="ornith-1.5_35B_A3B-bf16dense_4Bit"
                        file="ornith-1.5-35b-a3b-bf16dense-4bit.txt"; q=4 ;;
    # Ornith 1.5 122B: the wider member of the same family, so it exercises the
    # same graph at hidden 3072 and 48 layers. Declared before its install is on
    # every machine, like the other targets: inert where the directory is
    # missing, demanded by release.sh the moment it is there.
    ornith122-4) dir="ornith-1.5_122B_4Bit"; file="ornith-1.5-122b-4bit.txt"; q=4 ;;
    ornith122-8) dir="ornith-1.5_122B_8Bit"; file="ornith-1.5-122b-8bit.txt"; q=8 ;;
    qwen38-4)   dir="qwen3.8-flash-next_125B_A6B_4Bit"
                file="qwen3.8-flash-next-125b-a6b-4bit.txt"; q=4 ;;
    qwen38-8)   dir="qwen3.8-flash-next_125B_A6B_8Bit"
                file="qwen3.8-flash-next-125b-a6b-8bit.txt"; q=8 ;;
    qwen36-4) dir="qwen3.6_35B_A3B_4Bit"; file="qwen3.6-35b-a3b-4bit.txt"; q=4 ;;
    qwen36-8) dir="qwen3.6_35B_A3B_8Bit"; file="qwen3.6-35b-a3b-8bit.txt"; q=8 ;;
    agentworld-4) dir="qwen-agentworld_35B_A3B_4Bit"; file="qwen-agentworld-35b-a3b-4bit.txt"; q=4 ;;
    agentworld-8) dir="qwen-agentworld_35B_A3B_8Bit"; file="qwen-agentworld-35b-a3b-8bit.txt"; q=8 ;;
    katcoder-4) dir="kat-coder-v2.5_35B_A3B_4Bit"; file="kat-coder-v2.5-35b-a3b-4bit.txt"; q=4 ;;
    katcoder-8) dir="kat-coder-v2.5_35B_A3B_8Bit"; file="kat-coder-v2.5-35b-a3b-8bit.txt"; q=8 ;;
    # The dense Qwen 3.5 models, which had no baseline at all. They are the one
    # shape that runs on either engine, so this drives the default engine (GPU);
    # the CPU-versus-GPU equivalence is a separate opt-in test.
    qwen35-2b-4) dir="qwen3.5_2B_4Bit"; file="qwen3.5-2b-4bit.txt"; q=4 ;;
    qwen35-2b-8) dir="qwen3.5_2B_8Bit"; file="qwen3.5-2b-8bit.txt"; q=8 ;;
    qwen35-4b-4) dir="qwen3.5_4B_4Bit"; file="qwen3.5-4b-4bit.txt"; q=4 ;;
    qwen35-4b-8) dir="qwen3.5_4B_8Bit"; file="qwen3.5-4b-8bit.txt"; q=8 ;;
    qwen35-9b-4) dir="qwen3.5_9B_4Bit"; file="qwen3.5-9b-4bit.txt"; q=4 ;;
    qwen35-9b-8) dir="qwen3.5_9B_8Bit"; file="qwen3.5-9b-8bit.txt"; q=8 ;;
    # K2-Horizon installs, driven by the probe: `TinyTitanBench k2gpu`.
    k2-4) dir="k2-horizon-4bit.ssdai"; file="k2-horizon-4bit.txt"; q=4; kind=k2 ;;
    k2-8) dir="k2-horizon-8bit.ssdai"; file="k2-horizon-8bit.txt"; q=8; kind=k2 ;;
    *) echo "unknown target: $t (ornith-4, ornith-8, ornith-bf16dense-4, ornith122-4, ornith122-8, qwen36-4, qwen36-8, qwen38-4, qwen38-8, agentworld-4, agentworld-8, katcoder-4, katcoder-8, qwen35-2b-4, qwen35-2b-8, qwen35-4b-4, qwen35-4b-8, qwen35-9b-4, qwen35-9b-8, k2-4, k2-8)" >&2
       status=1; continue ;;
  esac
  model="$ROOT/models/$dir"
  if [ ! -f "$model/verified-install.json" ]; then
    echo "missing baseline model $t: no verified install at ${model#$ROOT/}" >&2
    status=1
    continue
  fi
  file="$OUT_DIR/$file"
  work="$(mktemp "$ROOT/.build/golden-baseline.XXXXXX")"

  echo "== $t =="

  if [ "$kind" = "k2" ]; then
    # The bench owns capture and comparison for this family
    # (`TINYTITAN_K2_GOLDEN`), so the body format is the token ids it prints
    # rather than CLI text. --server has no K2 twin: there is no CLI leg.
    if [ "$server_mode" = 1 ]; then
      echo "  server: not applicable — K2 has no CLI leg to compare against"
      status=1
    elif [ ! -x "$BENCH" ]; then
      echo "  FAILED: missing $BENCH — run: swift build -c release"
      status=1
    elif [ "$mode" = check ] && [ ! -f "$file" ]; then
      echo "  no baseline at ${file#$ROOT/}; run without --check first"
      status=1
    else
      if [ "$mode" = capture ]; then
        TINYTITAN_K2_GOLDEN="$file" TINYTITAN_K2_GOLDEN_UPDATE=1 \
          "$BENCH" k2gpu "$model" "$PROMPT" "$MAX_NEW" \
          > "$work" 2>"$work.err"
      else
        TINYTITAN_K2_GOLDEN="$file" \
          "$BENCH" k2gpu "$model" "$PROMPT" "$MAX_NEW" \
          > "$work" 2>"$work.err"
      fi
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "  FAILED (exit $rc)"
        sed 's/^/  /' "$work.err" | head -10
        status=1
      else
        grep -E '^(tokens|output|golden): ' "$work" | sed 's/^/  /'
      fi
    fi
    rm -f "$work" "$work.err"
    continue
  fi

  if [ "$server_mode" = 1 ]; then
    server_check "$model" "$t" || status=1
    rm -f "$work" "$work.err"
    continue
  fi

  # --quiet keeps the timing footer out of the compared text; only the
  # generated tokens are the contract. Timings vary run to run by design.
  "$CLI" --model "$model" --prompt "$PROMPT" --max-new "$MAX_NEW" \
         --temperature 0 --seed "$SEED" --quiet > "$work" 2>"$work.err"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED (exit $rc)"; sed 's/^/  /' "$work.err" | head -20
    rm -f "$work" "$work.err"; status=1; continue
  fi

  if [ "$mode" = capture ]; then
    # An empty generation is refused, not stored. `--check` compares only the
    # body (everything after `---`), so an empty body certifies an empty run:
    # the gate would print "identical to baseline" forever while having
    # captured nothing. A CLI that exits 0 and prints nothing is a failure of
    # the run, not a result.
    if [ ! -s "$work" ]; then
      echo "  REFUSED: exit 0 with no output — an empty body would pass every"
      echo "  future --check without asserting anything. Nothing was stored."
      rm -f "$work" "$work.err"; status=1; continue
    fi
    # Assemble into a temp file, then copy it into place with a *simple*
    # command. A redirection error on a brace group is printed but does not
    # fail the group -- measured on /bin/bash 3.2.57 and on Homebrew 5.x alike,
    # so `if ! { …; } > "$file"` is a guard that can never fire. `cat src > f`
    # does report it, which is what makes the write's status checkable.
    full="$(mktemp "$ROOT/.build/golden-full.XXXXXX")"
    {
      echo "# prompt:      $PROMPT"
      echo "# max-new:     $MAX_NEW"
      echo "# temperature: 0 (greedy)"
      echo "# seed:        $SEED"
      echo "# target:      $t (${q}-bit)"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD)"
      echo "---"
      cat "$work"
    } > "$full"
    if ! cat "$full" > "$file"; then
      echo "  FAILED to write ${file#$ROOT/} — nothing was captured"
      rm -f "$work" "$work.err" "$full"; status=1; continue
    fi
    rm -f "$full"
    # The size of the file that now exists, not of the temp file that fed it:
    # the header is part of every stored baseline, and the old line reported
    # only the body.
    echo "  captured -> ${file#$ROOT/} ($(wc -c < "$file" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at ${file#$ROOT/}; run without --check first"; status=1
    elif [ "$(sed '1,/^---$/d' "$file" | wc -c | tr -d ' ')" = 0 ]; then
      # Same vacuity, from the other side: a baseline whose body is empty can
      # only be matched by a run that produced nothing. Either way the check
      # has stopped being a check, so refuse before the diff.
      echo "  REFUSED: ${file#$ROOT/} has an empty body — it captures no output"
      echo "  Re-capture it: tools/golden-baseline.sh $t"
      status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok — output identical to baseline"
    else
      echo "  MISMATCH against ${file#$ROOT/}:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
