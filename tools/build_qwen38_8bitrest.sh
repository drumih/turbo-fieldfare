#!/usr/bin/env bash
# Build the Qwen3.8-Flash-Next 4/8/16 install: routed experts 4-bit, every
# other quantised tensor 8-bit, the n-gram table at its original fp16.
#
#   bash tools/build_qwen38_8bitrest.sh
#
# Evaluation build next to the 4-bit install; the installer default is not
# changed. Steps, each resumable by running the script again:
#   1. tools/prepare_qwen38.py --rest-8bit: fetch Qwen's bf16 checkpoint and
#      quantise it. The n-gram table is hardlinked from the 4-bit install, so its
#      ~102 GB of shards are not fetched (~258 GB download instead of ~360).
#   2. TinyTitanRepack --input-snapshot --consume-snapshot --share-ngram-table:
#      the snapshot is deleted shard by shard as the install is written.
#   3. --verify-install, then one short greedy generation as a smoke test.
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
DEST="$MODELS/qwen3.8-flash-next_125B_A6B_4Bit_8BitRest"
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
    stamp "step 1/3: fetch + quantise (rest 8-bit, experts 4-bit), $FETCH_WORKERS workers"
    "$PY" tools/prepare_qwen38.py --bits 4 --rest-8bit \
      --reuse-ngram-table "$BASE_INSTALL" \
      --fetch-workers "$FETCH_WORKERS" \
      --output "$SNAP" --work "$SHARDS" ${HF_ENDPOINT:+--endpoint="$HF_ENDPOINT"}
    rmdir "$SHARDS" 2>/dev/null || true
  else
    stamp "step 1/3: snapshot already complete"
  fi

  # --- 2. repack ------------------------------------------------------------
  stamp "step 2/3: repack into $DEST (consumes the snapshot)"
  .build/release/TinyTitanRepack --input-snapshot "$SNAP" --model-id "$MODEL_ID" \
    --share-ngram-table --consume-snapshot --output "$DEST"
fi

# --- 3. verify + smoke ------------------------------------------------------
stamp "step 3/3: verify install"
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
stamp "next: SMOKE_RAM-style matrix -> MODELS=\"qwen3.8-flash-next_125B_A6B_4Bit qwen3.8-flash-next_125B_A6B_4Bit_8BitRest\" bash tools/ram-matrix.sh"
