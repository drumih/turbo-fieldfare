#!/usr/bin/env bash
# Build and benchmark the Qwen3.8 4E-8R-16N install (experts 4-bit, rest 8-bit,
# n-gram fp16) from the two installs already on disk. Run from the repo root:
#
#   bash start.sh              # move, build, verify, smoke test, RAM matrix,
#                              # then offer to delete the two old installs
#   SKIP_MATRIX=1 bash start.sh
#
# The experts and n-gram table are MOVED out of the 4-bit install, so that
# install stops working from step 2 on. BF16Rest stays intact until step 5: it
# is the source of the 8-bit rest.
#
# Log of everything: .build/start-4E-8R-16N.log
if ((BASH_VERSINFO[0] < 5)); then
  [[ -x /opt/homebrew/bin/bash ]] && exec /opt/homebrew/bin/bash "$0" "$@"
  echo "error: bash 5 required (brew install bash)" >&2
  exit 1
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f Package.swift && -d models ]] || { echo "error: run from the TinyTitan repo root" >&2; exit 1; }
mkdir -p .build
exec > >(tee -a .build/start-4E-8R-16N.log) 2>&1

step() { printf '\n=== [%s] %s ===\n' "$(date '+%H:%M:%S')" "$*"; }

step "1/5 build release binaries"
swift build -c release 2>&1 | tail -3

step "2/5 move experts + n-gram table into the new folder"
# Idempotent: a rerun after a finished move is a no-op (link script exits 0).
bash tools/link_qwen38_shared.sh --move

step "3/5 requantise the rest to 8-bit, verify, smoke test"
bash tools/build_qwen38_8bitrest.sh

if [[ "${SKIP_MATRIX:-0}" == 1 ]]; then
  step "4/5 RAM matrix skipped"
else
  # The 4-bit rows were measured on 2026-11-13 (docs/qwen38-flash-next-port.md);
  # that install has no experts any more, so only the new one runs.
  step "4/5 RAM matrix: 4E-8R-16N (~30 min)"
  MODELS="qwen3.8-flash-next_125B_A6B_4E-8R-16N" bash tools/ram-matrix.sh
fi

step "5/5 delete the two old installs"
# Step 3 leaves the new install with its own receipt; the 4-bit install is
# gutted by the move in step 2 (empty packed_experts/), BF16Rest is still
# complete but no longer needed (its n-gram name is shared, the new install
# keeps the data). Deleting BF16Rest frees ~72 GB; the n-gram table stays.
OLD=(models/qwen3.8-flash-next_125B_A6B_4Bit models/qwen3.8-flash-next_125B_A6B_4Bit_BF16Rest)
du -sh "${OLD[@]}" 2>/dev/null || true
if [[ -t 0 ]]; then
  read -r -p "Permanently delete both? This cannot be undone. Type yes: " answer
else
  answer=no
fi
if [[ "$answer" == "yes" ]]; then
  rm -rf "${OLD[@]}"
  step "deleted; done"
else
  echo "kept. To delete later: rm -rf ${OLD[*]}"
  step "done"
fi
