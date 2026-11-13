#!/usr/bin/env bash
# Build the Qwen3.8-Flash-Next 4/8/16 install: routed experts 4-bit, every
# other quantised tensor 8-bit, the n-gram table at its original fp16.
#
#   bash tools/build_qwen38_8bitrest.sh            # local, from the two installs
#   bash tools/build_qwen38_8bitrest.sh --download # fresh 258 GB conversion
#
# Evaluation build next to the 4-bit install; the installer default is not
# changed.
#
# Local route (default, minutes, ~7 GB of new disk): when the BF16Rest install
# is present, tools/requantize_qwen38_rest.py quantises its bf16 rest to 8 bits
# and hardlinks the experts, n-gram table and tokenizer from the 4-bit install.
# Every tensor is self-checked first (its bf16 re-quantised at the base width
# must reproduce the 4-bit install's bytes).
#
# Download route: tools/prepare_qwen38.py --rest-8bit fetches Qwen's checkpoint
# (the n-gram shards are skipped by reusing the 4-bit install's table), then
# TinyTitanRepack --consume-snapshot writes the install.
#
# Both end with --verify-install (which writes the receipt) and a short greedy
# generation as a smoke test.
#
# Env: FETCH_WORKERS (default 6), SMOKE_RAM (default 12G),
#      TINYTITAN_MODELS_DIR / TINYTITAN_WORK_DIR as for install_models.sh.
# Log: .build/build-qwen38-8bitrest.log (also printed to the terminal).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODELS="${TINYTITAN_MODELS_DIR:-$ROOT/models}"
WORK="${TINYTITAN_WORK_DIR:-$ROOT/.build}"
BASE_INSTALL="$MODELS/qwen3.8-flash-next_125B_A6B_4Bit"
BF16_INSTALL="$MODELS/qwen3.8-flash-next_125B_A6B_4Bit_BF16Rest"
DOWNLOAD=0
[[ "${1:-}" == "--download" ]] && DOWNLOAD=1
DEST="$MODELS/qwen3.8-flash-next_125B_A6B_4E-8R-16N"
SNAP="$WORK/qwen38-affine-4bit-8bitrest"
SHARDS="$WORK/qwen38-shards-8bitrest"
MODEL_ID="qwen3.8-flash-next-8bitrest"
FETCH_WORKERS="${FETCH_WORKERS:-6}"
SMOKE_RAM="${SMOKE_RAM:-12G}"
LOG="$WORK/build-qwen38-8bitrest.log"
mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
free_gb() { df -g "$1" | awk 'NR==2 {print $4}'; }

stamp "Qwen3.8 4/8/16 build -> $DEST"

# --- preflight --------------------------------------------------------------
[[ -d "$BASE_INSTALL" ]] || { echo "error: $BASE_INSTALL missing (needed for the n-gram table)"; exit 1; }
if [[ -f "$DEST/manifest.json" ]]; then
  stamp "install already present; skipping to verification"
elif (( DOWNLOAD == 0 )); then
  [[ -f "$BF16_INSTALL/manifest.json" ]] || {
    echo "error: $BF16_INSTALL missing; run with --download for a fresh conversion"
    exit 1
  }
  # shellcheck source=tools/lib/python.sh
  source "$ROOT/tools/lib/python.sh"
  PY="$(tinytitan_resolve_python)" || exit 1
  [[ -x .build/release/TinyTitanCLI ]] || swift build -c release
  stamp "local requantisation: rest bf16 -> 8-bit (python: $PY)"
  "$PY" tools/requantize_qwen38_rest.py --base "$BASE_INSTALL" --source "$BF16_INSTALL" \
    --output "$DEST" --bits 8
else
  # shellcheck source=tools/lib/python.sh
  source "$ROOT/tools/lib/python.sh"
  PY="$(tinytitan_resolve_python)" || exit 1
  stamp "python: $PY"

  if [[ ! -x .build/release/TinyTitanRepack || ! -x .build/release/TinyTitanCLI ]]; then
    stamp "building release binaries"
    swift build -c release
  fi

  # Peak: the in-flight checkpoint shards (2 x workers x ~3.4 GB) plus the
  # ~77 GB snapshot; --consume-snapshot keeps the repack from adding a second
  # copy on top.
  need=$((77 + 2 * FETCH_WORKERS * 4 + 10))
  have="$(free_gb "$WORK")"
  stamp "free disk: ${have} GB, needed: ~${need} GB"
  if (( have < need )); then
    echo "error: not enough free disk. Free ${need} GB (for example by removing"
    echo "  models/qwen3.8-flash-next_125B_A6B_4Bit_BF16Rest, ~78 GB not shared)"
    echo "  or lower FETCH_WORKERS, then run this script again."
    exit 1
  fi

  # --- 1. convert -----------------------------------------------------------
  if [[ ! -f "$SNAP/model.safetensors.index.json" ]]; then
    stamp "download step 1/2: fetch + quantise (rest 8-bit, experts 4-bit), $FETCH_WORKERS workers"
    "$PY" tools/prepare_qwen38.py --bits 4 --rest-8bit \
      --reuse-ngram-table "$BASE_INSTALL" \
      --fetch-workers "$FETCH_WORKERS" \
      --output "$SNAP" --work "$SHARDS" ${HF_ENDPOINT:+--endpoint="$HF_ENDPOINT"}
    rmdir "$SHARDS" 2>/dev/null || true
  else
    stamp "download step 1/2: snapshot already complete"
  fi

  # --- 2. repack ------------------------------------------------------------
  stamp "download step 2/2: repack into $DEST (consumes the snapshot)"
  .build/release/TinyTitanRepack --input-snapshot "$SNAP" --model-id "$MODEL_ID" \
    --share-ngram-table --consume-snapshot --output "$DEST"
fi

# --- 3. verify + smoke ------------------------------------------------------
stamp "verify install (hashes every file, writes the receipt)"
.build/release/TinyTitanRepack --verify-install --input-ssdai "$DEST"

stamp "quant slots in the manifest:"
/usr/bin/python3 - "$DEST/manifest.json" <<'PY'
import json, sys
q = json.load(open(sys.argv[1]))["quant"]
for slot in ("embedding", "attention", "router", "sharedExpert", "routedExpert"):
    print(f"  {slot:13s} {q[slot]['weightBits']}-bit")
widths = {}
for k, v in q.items():
    if isinstance(v, dict) and k not in ("embedding", "attention", "router", "sharedExpert", "routedExpert"):
        widths[v["weightBits"]] = widths.get(v["weightBits"], 0) + 1
print(f"  per-tensor overrides by width: {widths or 'none'}; promoted bf16: {len(q.get('promoted', []))}")
PY

stamp "smoke test: 64 tokens greedy at --ram-budget $SMOKE_RAM"
TINYTITAN_RUNNER_STATS=1 TINYTITAN_EXPERT_PROFILE=0 .build/release/TinyTitanCLI \
  --model "$DEST" --prompt "Write a Swift function that returns the n-th Fibonacci number." \
  --max-new 64 --temperature 0 --seed 1234 --ram-budget "$SMOKE_RAM"

stamp "done. Disk use:"
du -sh "$DEST"
stamp "next: SMOKE_RAM-style matrix -> MODELS=\"qwen3.8-flash-next_125B_A6B_4Bit qwen3.8-flash-next_125B_A6B_4E-8R-16N\" bash tools/ram-matrix.sh"
