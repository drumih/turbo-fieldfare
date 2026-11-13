#!/usr/bin/env bash
# Put the 4-bit install's routed experts and n-gram table into the 4E-8R-16N
# install folder, without copying a byte.
#
#   bash tools/link_qwen38_shared.sh           # hardlink (default)
#   bash tools/link_qwen38_shared.sh --move    # move: the 4-bit install stops working
#
# A hardlink is a second name for the same file: it is instant, uses no disk,
# and both installs keep working. The two builds' experts and n-gram table are
# byte-identical (both are the 4-bit converter's output), so sharing is exact.
# `--move` takes the files away from the 4-bit install instead.
#
# Afterwards tools/build_qwen38_8bitrest.sh (or requantize_qwen38_rest.py)
# accepts these files in place and writes only model_weights.bin and the
# manifest. Needs bash 5 (Homebrew: /opt/homebrew/bin/bash).
set -euo pipefail

if ((BASH_VERSINFO[0] < 5)); then
  echo "error: bash 5 required (this is $BASH_VERSION); run: /opt/homebrew/bin/bash $0" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${TINYTITAN_MODELS_DIR:-$ROOT/models}"
SRC="$MODELS/qwen3.8-flash-next_125B_A6B_4Bit"
DEST="$MODELS/qwen3.8-flash-next_125B_A6B_4E-8R-16N"
MODE="link"
[[ "${1:-}" == "--move" ]] && MODE="move"

[[ -d "$SRC/packed_experts" && -f "$SRC/ngram_table.bin" ]] || {
  echo "error: $SRC has no packed_experts/ or ngram_table.bin" >&2
  exit 1
}
mkdir -p "$DEST/packed_experts"
if [[ "$(stat -f %d "$SRC")" != "$(stat -f %d "$DEST")" ]]; then
  echo "error: $SRC and $DEST are on different volumes; hardlinks and instant moves need one" >&2
  exit 1
fi

declare -i placed=0 skipped=0
declare -i bytes=0
place() {
  local from="$1" to="$2"
  if [[ -e "$to" ]]; then
    if [[ "$from" -ef "$to" ]]; then
      # Linked by an earlier run: a move just drops the old name.
      if [[ "$MODE" == "move" ]]; then
        rm "$from"
        placed+=1
      else
        skipped+=1
      fi
      return
    fi
    echo "error: $to exists and is a different file" >&2
    exit 1
  fi
  bytes+=$(stat -f %z "$from")
  if [[ "$MODE" == move ]]; then mv "$from" "$to"; else ln "$from" "$to"; fi
  placed+=1
}

shopt -s nullglob
for f in "$SRC"/packed_experts/*; do
  [[ -f "$f" ]] && place "$f" "$DEST/packed_experts/${f##*/}"
done
place "$SRC/ngram_table.bin" "$DEST/ngram_table.bin"
# Small companion the n-gram table is addressed by; always copied.
[[ -f "$SRC/ple_constants.json" && ! -e "$DEST/ple_constants.json" ]] &&
  cp -p "$SRC/ple_constants.json" "$DEST/ple_constants.json"

verb=$([[ "$MODE" == move ]] && echo moved || echo linked)
printf '%s %d files (%d GB), %d already in place -> %s\n' \
  "$verb" "$placed" $((bytes / 1000000000)) "$skipped" "$DEST"
du -sh "$DEST/packed_experts" "$DEST/ngram_table.bin"
echo "next: bash tools/build_qwen38_8bitrest.sh"
