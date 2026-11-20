#!/usr/bin/env bash
#
# Build the shipping form of the library product: the static archive, the
# dynamic library, the Swift module that imports them, and the resource bundle
# the runtime loads its Metal kernels from.
#
#   tools/build_library.sh <version> [--out <dir>] [--scratch-path <dir>]
#
# `tools/release.sh` calls this and publishes what it writes, so every release
# carries `libTinyTitanLib.a` and `libTinyTitanLib.dylib` beside the engine
# binaries. It is a separate script so it can be run on its own (and so the
# packaging can be inspected without cutting a release).
#
# Why the archive is merged: `swift build` writes one `.a` per target, and
# `libTinyTitanLib.a` alone does not contain the runtime, the format reader, the
# C kernels or the tokenizer package. A consumer linking that single file would
# fail on every symbol one layer down, so this collects every archive the
# release build produced into one, which is what a static distribution has to be.
#
# Why the dylib is renamed: the dynamic product is built under its own product
# name (`TinyTitanLibDynamic`, so that the SwiftPM-facing `TinyTitanLib` stays
# static for source consumers), and it is linked with an install name of
# `@rpath/libTinyTitanLib.dylib` so the shipped file can carry the plain name.
# `otool -D` is asserted below rather than trusted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The arm64 assertion RELEASE.md rule 2 promises. `tools/release.sh` re-checks
# the archive it builds from this directory, but this script also runs standalone
# from a terminal, and a hand-packaged library is still a library we handed out.
# shellcheck source=assert-arch.sh
. "$(dirname "$0")/assert-arch.sh"
# shellcheck source=assert-build-log.sh
. "$(dirname "$0")/assert-build-log.sh"
PRODUCT="TinyTitanLib"
DYNAMIC_PRODUCT="TinyTitanLibDynamic"
INSTALL_NAME="@rpath/lib$PRODUCT.dylib"
OUT="$ROOT/.build/library-dist"
SCRATCH=""

usage() {
  sed -n '3,8p' "$0" | sed 's/^# \{0,1\}//'
}

die() { echo "build_library: $*" >&2; exit 1; }
step() { printf '\n== %s ==\n' "$1"; }

VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
    --scratch-path) SCRATCH="${2:?--scratch-path needs a directory}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    -*) die "unknown flag: $1" ;;
    *) VERSION="$1"; shift ;;
  esac
done
[[ -n "$VERSION" ]] || die "usage: tools/build_library.sh <version> [--out <dir>] [--scratch-path <dir>]"

SCRATCH_ARGS=()
if [[ -n "$SCRATCH" ]]; then
  SCRATCH_ARGS=(--scratch-path "$SCRATCH")
else
  # A caller that named a scratch path already owns it (release.sh wipes and
  # rebuilds its own). Without one, use a private scratch and start it empty:
  # `swift build` never removes the archive of a target that has since been
  # renamed, and a stale `lib*.a` beside the real one is merged into the
  # distribution by the glob below — duplicate symbols in the shipped archive.
  SCRATCH="$ROOT/.build/library-scratch"
  rm -rf "$SCRATCH"
  SCRATCH_ARGS=(--scratch-path "$SCRATCH")
fi

step "build $PRODUCT (static) and $DYNAMIC_PRODUCT (dynamic)"
cd "$ROOT"
BUILD_LOG="$OUT.buildlog"
swift build -c release "${SCRATCH_ARGS[@]+"${SCRATCH_ARGS[@]}"}" --product "$PRODUCT" \
  2>&1 | tee "$BUILD_LOG" | tail -1
swift build -c release "${SCRATCH_ARGS[@]+"${SCRATCH_ARGS[@]}"}" --product "$DYNAMIC_PRODUCT" \
  -Xlinker -install_name -Xlinker "$INSTALL_NAME" 2>&1 | tee -a "$BUILD_LOG" | tail -1
# The same rule the engine's release build enforces: a warning in the library is
# a failure, not a note in a log nobody reads. `assert_clean_build_log` refuses a
# log it cannot read and one that holds no finished build, because the `grep -q`
# this replaces could only act on a warning it found — a missing, empty, unreadable
# or truncated log left `die` unreached and the library got staged anyway (AUD-266).
assert_clean_build_log "$BUILD_LOG" "library build"
BIN="$(swift build -c release "${SCRATCH_ARGS[@]+"${SCRATCH_ARGS[@]}"}" --show-bin-path)"

step "stage into $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

# 1. The static archive, merged with every archive the build produced. This is
#    the only form in which the one file a consumer links is self-contained.
ARCHIVES=()
while IFS= read -r archive; do
  ARCHIVES+=("$archive")
done < <(find "$BIN" -maxdepth 1 -name '*.a' | sort)
[[ ${#ARCHIVES[@]} -gt 0 ]] || die "no .a archives in $BIN; the static build did not run"
xcrun libtool -static -o "$OUT/lib$PRODUCT.a" "${ARCHIVES[@]+"${ARCHIVES[@]}"}"

# 2. The dynamic library, checked for the install name the rename depends on.
DYLIB_SRC="$BIN/lib$DYNAMIC_PRODUCT.dylib"
[[ -f "$DYLIB_SRC" ]] || die "no $DYNAMIC_PRODUCT.dylib in $BIN"
cp "$DYLIB_SRC" "$OUT/lib$PRODUCT.dylib"
FOUND_INSTALL_NAME="$(otool -D "$OUT/lib$PRODUCT.dylib" | tail -1)"
[[ "$FOUND_INSTALL_NAME" == "$INSTALL_NAME" ]] \
  || die "install name is '$FOUND_INSTALL_NAME', expected '$INSTALL_NAME'"

# 3. The modules. A Swift module does not stand alone: the binary
#    `.swiftmodule` records its imports, and the compiler has to deserialize
#    every one of them — `swiftc -I <out>` on the library's own module alone
#    fails with "missing required module 'yyjson'". So the whole module set the
#    build produced travels, exactly as the archives do. Measured, not assumed:
#    a one-module distribution does not compile.
[[ -d "$BIN/$PRODUCT.swiftmodule" ]] || die "no $PRODUCT.swiftmodule in $BIN"
for module in "$BIN"/*.swiftmodule; do
  [[ -e "$module" ]] || continue
  case "$module" in *Tests.swiftmodule) continue ;; esac
  cp -R "$module" "$OUT/"
done
#    The C targets in the graph (our kernels, yyjson, NIO's shims, BoringSSL)
#    need their generated module maps, and SwiftPM normally passes each one
#    explicitly. They are collected here so a consumer can do the same. Note the
#    search root: `--show-bin-path` resolves inside `out/Products/Release`, and
#    the maps are a sibling of `Products`, not a child of it.
MODULE_MAPS="$(find "$SCRATCH" -maxdepth 4 -type d -name GeneratedModuleMaps 2>/dev/null | head -1)"
[[ -n "$MODULE_MAPS" ]] || die "no GeneratedModuleMaps under $SCRATCH: C modules would not import"
cp "$MODULE_MAPS"/*.modulemap "$OUT/"
[[ -f "$OUT/TinyTitanKernelsC.modulemap" ]] \
  || die "the staged module maps do not include TinyTitanKernelsC.modulemap"

# 4. The resource bundle. The runtime reaches its Metal shader library through
#    `Bundle.module`, which resolves beside the consumer's executable, so it has
#    to travel with the library and be documented. Test bundles stay behind.
FOUND_BUNDLE=0
for bundle in "$BIN"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  case "$bundle" in *Tests.bundle) continue ;; esac
  cp -R "$bundle" "$OUT/"
  FOUND_BUNDLE=1
done
[[ "$FOUND_BUNDLE" == 1 ]] || die "no resource bundle in $BIN: the Metal kernels would not ship"
# The bundle carries the shader *sources*. `MetalContext` compiles them with
# `device.makeLibrary(source:)` at load time — there is no `.metallib` step to
# check for, so the assertion is that the sources travelled, not a binary. The
# sources live in whichever target carries the Metal context (TinyTitanCore since
# the 7.0 extraction), so check the staged non-test bundles by content, not name.
[[ -n "$(find "$OUT"/*.bundle -name '*.metal' -print -quit 2>/dev/null)" ]] \
  || die "the staged resource bundle carries no Metal shader sources"

# 5. Licensing travels with any binary distribution (Apache-2.0).
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/THIRD_PARTY_NOTICES.md" "$OUT/"

# 6. The demo apps, with the source they are built from. An archive whose whole
#    point is "link this instead of building the package" should arrive with a
#    program that does, and their build scripts find the library one directory
#    up from where they land here.
DEMO_SRC="$ROOT/examples/library-demo"
[[ -f "$DEMO_SRC/main.swift" ]] || die "no examples/library-demo/main.swift to ship"
mkdir -p "$OUT/demo"
cp "$DEMO_SRC/main.swift" "$DEMO_SRC/build-static.sh" "$DEMO_SRC/build-dynamic.sh" \
  "$DEMO_SRC/README.md" "$OUT/demo/"
chmod +x "$OUT/demo/build-static.sh" "$OUT/demo/build-dynamic.sh"

# 7. A consumer needs one `-Xcc -fmodule-map-file=` pair per C module, and those
#    paths only exist after the archive is extracted somewhere, so the flags are
#    generated on the consumer's side rather than baked in here. `swiftc @file`
#    reads them back as ordinary arguments (verified: it compiles and runs).
cat > "$OUT/make-flags.sh" <<'FLAGS'
#!/usr/bin/env bash
# Write swiftc-flags.txt next to this script: one -Xcc -fmodule-map-file=...
# pair per module map, which is what a swiftc build of a consumer needs.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
: > "$DIR/swiftc-flags.txt"
for map in "$DIR"/*.modulemap; do
  printf -- '-Xcc\n-fmodule-map-file=%s\n' "$map" >> "$DIR/swiftc-flags.txt"
done
echo "wrote $DIR/swiftc-flags.txt"
FLAGS
chmod +x "$OUT/make-flags.sh"

cat > "$OUT/README-library.txt" <<TXT
TinyTitan $VERSION — the library product (macOS, Apple Silicon / arm64)

Built from tag $VERSION with: tools/build_library.sh
Requires macOS 26+. Apple Silicon only; there is no x86_64 build.

Contents
  libTinyTitanLib.a          static archive: the library and every dependency
                             it needs, merged into one file
  libTinyTitanLib.dylib      the same library, dynamically linked
  TinyTitanLib.swiftmodule   the Swift module — plus every module it imports,
                             because a binary module does not stand alone
  TinyTitanVariants.swiftmodule
                             the 7.0 engine-factory surface (variant selection,
                             variant factories, dense CPU selection)
  *.modulemap                the generated module maps for the C targets in the
                             graph (the kernels, yyjson, NIO's shims, ...),
                             which SwiftPM normally passes one by one
  make-flags.sh              writes swiftc-flags.txt naming those maps, because
                             the paths only exist once you have extracted this
  demo/                      two terminal apps built from one source — one
                             linked against the static archive, one against the
                             dylib — with the scripts that build them
  TinyTitan_TinyTitanCore.bundle
                             the Metal shader sources, compiled when the library
                             loads; the bundle must sit next to the executable
                             that uses the library
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md

Linking. The C module maps have to be named explicitly, so run the generator
once (it writes swiftc-flags.txt with the paths of *this* directory), then:

  ./make-flags.sh

  static:   swiftc -I . @swiftc-flags.txt mytool.swift libTinyTitanLib.a -o mytool
  dynamic:  swiftc -I . @swiftc-flags.txt -L . -lTinyTitanLib mytool.swift \\
              -Xlinker -rpath -Xlinker "\$PWD" -o mytool

Both lines were run against this archive before it was published. Run the result
from a directory that holds TinyTitan_TinyTitanCore.bundle (and the two dependency
bundles), or copy them beside the executable: the runtime finds its shaders
through Bundle.module, which resolves next to the binary.

The API is the public surface of the TinyTitanLib module: Engine, Session,
EngineConfiguration, ChatMessage, GenerationOptions, GenerationEvent,
GenerationSummary and TinyTitanError. See the wiki page "Library and Engine".

This is the binary form for a consumer that cannot or will not build from
source. Building the package as a SwiftPM dependency remains the supported
route and needs none of these flags. The module is built by this release's
toolchain (Xcode 27 / Swift 6.4), which is the only supported one: it is not
module-stable across toolchains and does not need to be, because no other
toolchain is supported.

No model weights are included: the library reads a .ssdai install and never
downloads one.

These binaries are NOT code-signed or notarized. Clear the quarantine attribute
after verifying the checksum published with this archive:

  xattr -dr com.apple.quarantine /path/to/this/directory
TXT

# Both shipping forms are checked before the directory is declared staged: the
# static archive and the dynamic library are the files a consumer links, and
# `lipo -archs` reads an `!<arch>` container (measured: `arm64`), so the `.a` is
# not a hole in the assertion. The 29 `.swiftmodule` files beside them are Swift
# bitcode and are excluded by magic rather than by name.
assert_arm64_dir "$OUT" "library dist"

step "staged"
ls -1 "$OUT"
echo
echo "  lib$PRODUCT.a       $(wc -c < "$OUT/lib$PRODUCT.a" | tr -d ' ') bytes"
echo "  lib$PRODUCT.dylib   $(wc -c < "$OUT/lib$PRODUCT.dylib" | tr -d ' ') bytes"
