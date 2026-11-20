#!/usr/bin/env bash
#
# The whole-model check for the dense CPU models on disk: `TinyTitanBench cpu35`
# must print "all continuations correct" for each.
#
# The equivalence gate (tests/TinyTitan/CPUEngine/DenseSSDAIEquivalenceTests.swift,
# driven by tools/repack_dense.sh) compares logits against the snapshot a repack
# came from. This asks the other question -- does the engine continue real text
# correctly end to end -- so the two are complements, not duplicates.
#
# Each model is faulted into memory (up to 4.5 GB for the 4B at 8 bits) and runs
# on the performance cores, so it refuses to run beside a live model server
# rather than evicting that server's pages or skewing a measurement it is part
# of. That is the guard AGENTS.md states for any model run.
#
# Usage:
#   tools/verify_cpu_models.sh [model ...]        # default: the four below
#
# Model names are directory names under models/.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
# shellcheck source=model-guard.sh
. "$ROOT/tools/model-guard.sh"
DEFAULT_MODELS=(qwen3.5_2B_4Bit qwen3.5_2B_8Bit qwen3.5_4B_4Bit qwen3.5_4B_8Bit)
GUARD='TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  targets=("${DEFAULT_MODELS[@]+"${DEFAULT_MODELS[@]}"}")
fi

# Each model is faulted into memory on the performance cores, so the run is
# refused rather than raced. The guard answers three things and all three but
# one are a refusal: the old `if pgrep ... >/dev/null 2>&1` could only act on a
# match, so an erroring pgrep read as "nothing running" and the sweep started
# beside the live server anyway (AUD-268). It also asked pgrep a second time to
# print the matches; the guard hands back the lines it already collected.
model_guard_status=0
busy="$(model_guard_matches "$GUARD")" || model_guard_status=$?
case "$model_guard_status" in
  0)
    echo "a model process is already running; not starting:" >&2
    echo "$busy" | sed 's/^/  /' >&2
    exit 2
    ;;
  1) : ;;
  *)
    echo "not starting: the model-process guard could not answer, so nothing is" >&2
    echo "known about what is running. See the model-guard message above." >&2
    exit 2
    ;;
esac

cd "$ROOT" || exit 1
for model in "${targets[@]+"${targets[@]}"}"; do
  [ -f "$MODELS/$model/manifest.json" ] || {
    echo "no $MODELS/$model (an installed .ssdai)"; exit 1; }
done

swift build -c release --product TinyTitanBench --product TinyTitanCLI || exit 1
BIN="$(swift build -c release --show-bin-path)"
BENCH="$BIN/TinyTitanBench"
CLI="$BIN/TinyTitanCLI"

status=0
for model in "${targets[@]+"${targets[@]}"}"; do
  echo "== $model"
  out="$(/usr/bin/time -l "$BENCH" cpu35 "$MODELS/$model" 2>&1)"
  # /usr/bin/time -l prints a page of counters per run; keep the verdict and
  # the peak footprint, which is what a reader compares between models.
  echo "$out" | grep -vE '^ +[0-9]+ +[a-z]' || true
  echo "$out" | grep 'maximum resident set size' || true
  grep -q "all continuations correct" <<<"$out" || status=1

  # The shipped CPU path is the library's `CPUEngine`, not `TinyTitanBench`'s
  # own 6.x runner: exercise the CLI's `--engine cpu` as well, one short
  # deterministic generation, and require a real timing footer.
  echo "   CLI --engine cpu"
  cli_out="$("$CLI" --model "$MODELS/$model" --prompt "The capital of France is" \
    --temperature 0 --max-new 4 --engine cpu 2>&1)"
  cli_rc=$?
  echo "$cli_out" | tail -1
  if [ "$cli_rc" -ne 0 ] || ! grep -q 'tok/s=' <<<"$cli_out"; then
    echo "   FAILED: CLI --engine cpu on $model"
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "VERIFIED: ${#targets[@]} model(s)"
else
  echo "FAILED: see above"
fi
exit "$status"
