#!/usr/bin/env bash
#
# A local, non-shippable Apple M3 build of every product.
#
# The shipped artifacts are built for the portable arm64 baseline: the target
# is `arm64-apple-macos26.0` with no CPU flag, so the codegen baseline is
# clang's default for the triple (apple-m1) and an M3's instructions are unused
# (AGENTS.md). This script opts into the M3 codegen for a machine that will not
# ship the result: `-target-cpu apple-m3` for Swift, `-mcpu=apple-m3` for C.
#
# It is deliberately not wired into `tools/release.sh` or
# `tools/build_library.sh`: a binary built here may use instructions an M1/M2
# does not have. Never hand it to another Mac or publish it.
#
# Usage:  tools/build_m3.sh
# Env:    M3_SCRATCH=<dir>  scratch path (default .build/m3)
#         M3_KEEP=1         reuse the scratch instead of rebuilding from clean
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${M3_SCRATCH:-$ROOT/.build/m3}"
SWIFT_FLAGS=(-Xswiftc -target-cpu -Xswiftc apple-m3)
C_FLAGS=(-Xcc -mcpu=apple-m3)
PRODUCTS=(TinyTitanCLI TinyTitanServer TinyTitanRepack TinyTitanBench ContinuityDemo tinytitan-memory)
LIBRARIES=(TinyTitanLib TinyTitanLibDynamic)
LOG="$SCRATCH/m3-build.log"

die() { echo "build_m3: $*" >&2; exit 1; }

[ "$(uname -m)" = arm64 ] || die "this build targets arm64; host is $(uname -m)"
BRAND="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
case "$BRAND" in
  "Apple M3"*) ;;
  *) die "this build targets Apple M3; host reports '${BRAND:-unknown}'" ;;
esac

# A clean build so the compiler flags below are actually exercised and can be
# read out of the verbose log; M3_KEEP=1 keeps an existing scratch.
if [ "${M3_KEEP:-}" != 1 ]; then
  case "$SCRATCH" in
    "$ROOT/.build/m3"|"$ROOT"/.build/m3-*) rm -rf "$SCRATCH" ;;
    *) die "refusing to remove M3_SCRATCH=$SCRATCH; keep it under $ROOT/.build/m3*" ;;
  esac
fi
mkdir -p "$SCRATCH"

build() {
  swift build -c release --scratch-path "$SCRATCH" \
    "${SWIFT_FLAGS[@]+"${SWIFT_FLAGS[@]}"}" "${C_FLAGS[@]+"${C_FLAGS[@]}"}" "$@"
}
BIN="$(swift build -c release --scratch-path "$SCRATCH" --show-bin-path)"

echo "== native Apple M3 build into $SCRATCH"
echo "   swift: -target-cpu apple-m3   C: -mcpu=apple-m3"

# The first build is verbose: the flags have to be visible in the compiler
# commands, not merely intended, or the "optimized" claim is untested. With
# M3_KEEP=1 an up-to-date scratch would print no compile command at all, so
# touch one source to force the one build the log is read from.
if [ "${M3_KEEP:-}" = 1 ]; then
  touch "$ROOT/sources/TinyTitanCLI/Run.swift"
fi
build --product TinyTitanCLI -v 2>&1 | tee "$LOG"
grep -q -- "-target-cpu apple-m3" "$LOG" \
  || die "the Swift compiler did not receive -target-cpu apple-m3"
grep -q -- "-mcpu=apple-m3" "$LOG" \
  || die "the C compiler did not receive -mcpu=apple-m3"

for product in "${PRODUCTS[@]+"${PRODUCTS[@]}"}"; do
  [ "$product" = TinyTitanCLI ] && continue
  build --product "$product"
done
for library in "${LIBRARIES[@]+"${LIBRARIES[@]}"}"; do
  build --product "$library"
done

# Every product is present, arm64 and macOS 26+. The flags are the M3 part;
# this is the "all binaries" part.
for product in "${PRODUCTS[@]+"${PRODUCTS[@]}"}"; do
  [ -x "$BIN/$product" ] || die "no executable at $BIN/$product"
  ARCHS="$(lipo -archs "$BIN/$product")"
  [ "$ARCHS" = arm64 ] || die "$product is '$ARCHS', expected arm64"
  vtool -show-build "$BIN/$product" 2>/dev/null | grep -q "platform MACOS" \
    || die "$product has no macOS build command"
done
for artifact in libTinyTitanLib.a libTinyTitanLibDynamic.dylib; do
  [ -f "$BIN/$artifact" ] || die "no library artifact at $BIN/$artifact"
  ARCHS="$(lipo -archs "$BIN/$artifact")"
  [ "$ARCHS" = arm64 ] || die "$artifact is '$ARCHS', expected arm64"
done

echo
echo "== built (bytes)"
for product in "${PRODUCTS[@]+"${PRODUCTS[@]}"}"; do
  printf '  %-20s %s\n' "$product" "$(stat -f%z "$BIN/$product")"
done
printf '  %-20s %s\n' libTinyTitanLib.a "$(stat -f%z "$BIN/libTinyTitanLib.a")"
printf '  %-20s %s\n' libTinyTitanLibDynamic.dylib \
  "$(stat -f%z "$BIN/libTinyTitanLibDynamic.dylib")"

if [ -x "$ROOT/.build/release/TinyTitanCLI" ]; then
  echo
  echo "== portable ('.build/release') is still the shippable build; this tree"
  echo "   was not touched. Compare with:"
  echo "   lipo -archs .build/release/TinyTitanCLI $BIN/TinyTitanCLI"
fi
echo
echo "Flags verified in $LOG; all products are arm64. Do not ship this build."
