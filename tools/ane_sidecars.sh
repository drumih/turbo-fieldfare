#!/usr/bin/env bash
# Give every installed GPU-path model the ANE sidecar it needs, and verify it.
#
# A model without a sidecar has no ANE prefill at all: the switch is on by
# default, the runtime asks for the sidecar, and finds nothing — which is how
# the dense Qwen 3.5 family spent its whole life on the GPU with the feature
# apparently enabled. So the sidecar is part of installing a model, not an
# experiment to remember afterwards.
#
# Exporting compiles one Core ML program per full-attention layer, per
# history variant, so it takes minutes per model and is skipped when the
# sidecar already exists. Verification is cheap and always runs: it compares the
# exported graph against an independent NumPy implementation of the same
# attention block, which is the check that a wrong geometry cannot pass.
#
#   tools/ane_sidecars.sh                 # every install, export what is missing
#   tools/ane_sidecars.sh qwen3.5_4B_4Bit # one install
#   tools/ane_sidecars.sh --chunk 1024    # the width for the band under 4,096
#   tools/ane_sidecars.sh --force         # re-export even where one exists
#   tools/ane_sidecars.sh --verify-only   # no export, just check what is there
#
# Exits non-zero if any model it was asked about is still without a verified
# sidecar, so it can gate an install. Three statuses, the tree's table: 0 every
# candidate the walk found is verified, 1 one it was asked about is not, 2 it did
# not get to run -- an option it does not take, a chunk the runtime would not
# route, an interpreter it cannot use, or a models directory holding no install
# to ask about. Reporting that last shape as 0 would be a pass over a check that
# looked at nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${TINYTITAN_MODELS_DIR:-models}"
# coremltools is only in the exporter's documented interpreter; the system
# python does not have it.
COREML_PYTHON="${TINYTITAN_COREML_PYTHON:-$HOME/.venvs/coreml-py311/bin/python}"
CHUNK=4096
MAX_HISTORY=12288
FORCE=0
VERIFY_ONLY=0
REQUESTED=()

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --chunk) CHUNK="$2"; shift 2 ;;
    --max-history) MAX_HISTORY="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) REQUESTED+=("$1"); shift ;;
  esac
done

[ -x "$COREML_PYTHON" ] || {
  echo "error: no coremltools interpreter at $COREML_PYTHON" >&2
  echo "       set TINYTITAN_COREML_PYTHON, or install one:" >&2
  echo "       python3 -m venv ~/.venvs/coreml-py311 && ~/.venvs/coreml-py311/bin/pip install coremltools" >&2
  exit 2
}

# The runtime only accepts these as a prefill chunk, and only routes a chunk to
# the sidecar when the two match.
allowed_chunk=0
for choice in 32 64 128 256 512 1024 2048 4096; do
  [ "$CHUNK" = "$choice" ] && allowed_chunk=1
done
if [ "$allowed_chunk" -eq 0 ]; then
  echo "error: --chunk $CHUNK is not a prefill chunk the runtime accepts" >&2
  echo "       (32, 64, 128, 256, 512, 1024, 2048, 4096)" >&2
  exit 2
fi

# 4,096 keeps the historical directory; any other width gets its own, so a model
# can carry several and the configured chunk picks one.
if [ "$CHUNK" = 4096 ]; then SIDECAR="ane_prefill"; else SIDECAR="ane_prefill-$CHUNK"; fi

if [ "${#REQUESTED[@]}" -gt 0 ]; then
  candidates=("${REQUESTED[@]+"${REQUESTED[@]}"}")
else
  candidates=()
  for dir in "$MODELS_DIR"/*/; do
    [ -f "$dir/manifest.json" ] && candidates+=("$(basename "$dir")")
  done
fi

# A walk that found no candidate is not a set of installs that all passed: it is
# the instrument reading nothing. The shapes that get here are a fresh clone with
# no `models/`, a directory of directories none of which is an install, and a
# `TINYTITAN_MODELS_DIR` pointing at the wrong place — and `--verify-only` over
# any of them is exactly how a check comes back green without having checked.
if [ "${#candidates[@]}" -eq 0 ]; then
  echo "error: no installed model to give a sidecar: $MODELS_DIR" >&2
  echo "       a candidate is a subdirectory holding a manifest.json, and a fresh clone" >&2
  echo "       has no models/ at all. Installing one is the operator's job" >&2
  echo "       (docs/adding-a-model.md), never this script's." >&2
  echo "       nothing was exported and nothing was verified: 0 here would be a claim" >&2
  echo "       about installs this walk never found." >&2
  exit 2
fi

skipped=() ; exported=() ; verified=() ; failed=()

for name in "${candidates[@]+"${candidates[@]}"}"; do
  model="$MODELS_DIR/$name"
  if [ ! -f "$model/manifest.json" ]; then
    echo "== $name: no manifest at $model — not installed"
    failed+=("$name: not installed"); continue
  fi

  # The family decides whether this install is exported, skipped or refused, and
  # it is read through an interpreter that may be present but useless. A failed
  # substitution under `set -e` used to end the run with status 1 and no word
  # about which install or which manifest, so the reader could not tell a broken
  # coremltools venv from a corrupt install.
  family="$("$COREML_PYTHON" -c "
import json,sys
print(json.load(open('$model/manifest.json'))['arch']['family'])")" || {
    echo "!! $name: $COREML_PYTHON could not read the family from $model/manifest.json" >&2
    failed+=("$name: manifest unreadable"); continue
  }

  # Two kinds of install are reported rather than exported.
  #
  # `qwen38flash` is measured *not to pay*: the fold is wired and verified, and
  # the ANE still loses (0.72x on the 4-bit install at 4,333 tokens). Its GPU
  # path already attends to only the indexer's ~2,051 selected keys, while the
  # ANE graph is dense over the context and its per-variant Core ML load measured
  # 7-14 s. Export one explicitly to re-measure; do not install one expecting a
  # win. See benchmark/ane-prefill/README.md.
  #
  # The MTP draft is one the runtime never routes: it is verified rather than
  # prefilled on the ANE, so a sidecar for it would never be loaded.
  case "$family" in
    qwen36|qwen3_5_dense) ;;
    qwen38flash|qwen4flash)
       echo "== $name: $family — the ANE is measured slower on this model; skipping (export explicitly to re-measure)"
       skipped+=("$name ($family: measured slower)"); continue ;;
    *)
       echo "== $name: $family — the exporter does not build a sidecar for this family; skipping"
       skipped+=("$name ($family)"); continue ;;
  esac

  sidecar_dir="$model/$SIDECAR"
  if [ "$VERIFY_ONLY" -eq 0 ] && { [ "$FORCE" -eq 1 ] || [ ! -f "$sidecar_dir/ane_prefill.json" ]; }; then
    echo "== $name: exporting the $CHUNK-token sidecar (minutes; compiles every variant)"
    # Not piped: a pipeline's status is the last command's, so `| tail` would
    # report success for a failed export.
    log="$(mktemp)"
    if "$COREML_PYTHON" tools/export_ane_prefill.py --model "$model" \
         --chunk "$CHUNK" --max-history "$MAX_HISTORY" > "$log" 2>&1; then
      grep -vE "passes/s|passes\]" "$log" | tail -3
    else
      grep -vE "passes/s|passes\]" "$log" | tail -5 >&2
      rm -f "$log"
      echo "!! $name: export failed; the previous sidecar (if any) is untouched" >&2
      failed+=("$name: export failed"); continue
    fi
    rm -f "$log"
    exported+=("$name")
  else
    echo "== $name: sidecar present, not re-exporting"
  fi

  if [ ! -f "$sidecar_dir/ane_prefill.json" ]; then
    failed+=("$name: still no sidecar"); continue
  fi

  # Cheap, and the only check that a wrong geometry cannot pass. The width
  # matters: `--chunk` selects which sidecar directory is opened, so a
  # non-4,096 run must verify the one it just exported.
  echo "== $name: verifying the graph against NumPy"
  log="$(mktemp)"
  if "$COREML_PYTHON" tools/verify_ane_sidecar.py --model "$model" \
       --chunk "$CHUNK" > "$log" 2>&1; then
    tail -2 "$log"
    verified+=("$name")
  else
    tail -5 "$log" >&2
    failed+=("$name: verification failed")
  fi
  rm -f "$log"
done

echo
echo "verified: ${#verified[@]}${verified:+ — ${verified[*]+"${verified[*]}"}}"
[ "${#exported[@]}" -gt 0 ] && echo "exported: ${exported[*]+"${exported[*]}"}"
[ "${#skipped[@]}" -gt 0 ] && echo "skipped : ${skipped[*]+"${skipped[*]}"}"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'FAILED  : %s\n' "${failed[@]+"${failed[@]}"}" >&2
  exit 1
fi
exit 0
