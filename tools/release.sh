#!/usr/bin/env bash
# Verify, clean-build, package, and optionally publish an TinyTitan release.
# with a checksum, and publishes a GitHub Release from an existing tag.
#
#   tools/release.sh v4.0                  # dry run: verify, build, package, stop
#   tools/release.sh v4.0 --publish        # same, then create the Release
#   tools/release.sh v4.0 --publish --notes path/to/notes.md
#
# Dry run is the default on purpose: publishing is public and irreversible in
# the sense that watchers are notified immediately. Run it once without
# --publish, inspect the staged archive, then re-run with it.
#
# Two mistakes this script exists to prevent:
#
#   1. `gh` in a fork defaults to the PARENT repo. `gh release list` here shows
#      drumih/turbo-fieldfare, not this repo, and `gh release create` refuses
#      with a confusing message about an unpushed tag. Every gh call below pins
#      --repo.
#   2. An incremental `swift build` compiles nothing when the tree is unchanged,
#      so a warning gate over its output passes vacuously. The release build
#      always goes to a fresh scratch path.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${TINYTITAN_RELEASE_REPO:-Pummelchen/TinyTitan}"
PRODUCTS=(TinyTitanServer TinyTitanCLI TinyTitanRepack TinyTitanBench)

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

# What counts as a passing suite. `swift test --no-parallel` prints one summary
# line per test target -- six on this tree -- so a verdict that asks whether
# *some* line says passed is a verdict on one target out of six, and the run's
# own status knows more than any grep does.
test_gate_verdict() {
  local log="$1" status="$2" summaries failures
  summaries=$(grep -c 'Test run with' "$log")
  failures=$(grep -c 'Test run with .* failed' "$log")
  [ "$summaries" -gt 0 ] \
    || die "swift test exited $status and reported no test-target run at all; $log holds no 'Test run with' line"
  [ "$failures" -eq 0 ] \
    || die "$failures of $summaries test-target runs reported failure in $log; every one must pass"
  [ "$status" = 0 ] \
    || die "swift test exited $status over $summaries passing test-target run(s): see $log"
  echo "  $summaries test-target run(s) passed, swift test exited 0"
}

# The two questions the preconditions ask outside this repository, and the one
# thing that must never be inferred from a failed command. `gh` answers with exit
# status 1 for every outcome it dislikes, and an invisible repository answers
# `release not found` identically to a missing release (measured on gh 2.102.0),
# so "is there a Release for this tag yet?" has to be two queries where the first
# proves the second is worth reading: `gh api repos/$REPO` establishes that the
# credentials work and the repository can be seen, and only a 404 from the release
# query means absent. `gh api` rather than `gh release view` because it labels its
# failures with the HTTP status and `gh release view` does not. The six answers
# these refusals are written against, and the stub that replays them:
# benchmark/test_release_precondition_gate.py.
release_exists_verdict() {
  local repo="$1" tag="$2" body status code
  command -v gh >/dev/null 2>&1 \
    || die "gh is not installed; nothing is known about a Release for $tag on $repo. Install it, then check: gh auth status"
  if ! body="$(gh api "repos/$repo" --jq .full_name 2>&1)"; then
    die "cannot ask whether a Release for $tag exists on $repo: the repository query itself failed, so nothing about its releases is known. gh said: $body"
  fi
  status=0
  body="$(gh api "repos/$repo/releases/tags/$tag" --jq .tag_name 2>&1)" || status=$?
  if [ "$status" -eq 0 ]; then
    die "a Release for $tag already exists on $repo"
  fi
  # Absent has to be an answer, not the absence of one.
  code="$(printf '%s' "$body" | sed -n 's/.*(HTTP \([0-9][0-9][0-9]\)).*/\1/p' | tail -1)"
  [ "$code" = "404" ] \
    || die "cannot tell whether a Release for $tag exists on $repo: gh exited $status${code:+ with HTTP $code} instead of answering 404. gh said: $body"
  echo "  no Release for $tag on $repo (repository readable, release query answered 404)"
}

# Whether the tag is on origin. This was
# `git ls-remote --tags origin 2>/dev/null | grep -q "refs/tags/$TAG$"`, which
# fails closed but reports the wrong cause: git's own failure -- a remote that
# cannot be read, rc 128 with `fatal: 'origin' does not appear to be a git
# repository` -- arrives as an empty list and is announced as "not pushed", which
# sends the operator to `git push` when the push is not the problem. The pattern
# was the second defect: the tag is interpolated into a regex, so `v5.18` matches
# a remote carrying only `v5X18` (measured). The match is on the whole ref field
# now, tab-delimited, quoted so a dot is a dot.
git_tag_pushed_verdict() {
  local tag="$1" refs status=0 line found=0
  refs="$(git ls-remote --tags origin 2>&1)" || status=$?
  [ "$status" -eq 0 ] \
    || die "$tag is not known to be on origin: git ls-remote exited $status without answering. git said: $refs"
  while IFS= read -r line; do
    case "$line" in
      *$'\t'refs/tags/"$tag"|*$'\t'refs/tags/"$tag"^{}) found=1 ;;
    esac
  done <<< "$refs"
  [ "$found" -eq 1 ] \
    || die "$tag is not pushed to origin; run: git push origin $tag"
  echo "  $tag is on origin ($(printf '%s\n' "$refs" | grep -c 'refs/tags/') tag ref(s) listed)"
}

# RELEASE.md rule 2 is a promise about bytes — `lipo -archs <binary>` must report
# exactly `arm64` — and until now nothing in the release path ran lipo, so the only
# arm64 claim in a shipped artifact was its filename. Every name-based assertion
# below (`tar tzf | grep -q libTinyTitanLib.dylib`) passes on the wrong
# architecture, Rosetta hides an x86_64 slice on this host, and the sha256 in the
# release notes would certify those bytes as the Apple Silicon build. The scan is
# `tools/assert-arch.sh`, which `tools/build_library.sh` also uses.
# shellcheck source=assert-arch.sh
. "$SCRIPT_DIR/assert-arch.sh"
# The build log's compiler-warning gate, shared with tools/build_library.sh: the
# two callers each had their own copy of the grep, which is how a fix to one would
# have left the other exactly as broken (AUD-266).
# shellcheck source=assert-build-log.sh
. "$SCRIPT_DIR/assert-build-log.sh"

TAG="${1:-}"
[ -n "$TAG" ] || die "usage: tools/release.sh <tag> [--publish] [--notes <file>]"
shift
PUBLISH=0
NOTES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --publish) PUBLISH=1; shift ;;
    --notes)   NOTES="${2:-}"; [ -n "$NOTES" ] || die "--notes needs a file"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

VERSION="${TAG#v}"
STAGE_ROOT="$ROOT/.build/releases/tinytitan-release-$VERSION"
STAGE="$STAGE_ROOT/tinytitan-$VERSION-macos-arm64"
ARCHIVE="$STAGE_ROOT/tinytitan-$VERSION-macos-arm64.tar.gz"
SCRATCH="$STAGE_ROOT/build"

cd "$ROOT" || die "cannot cd to $ROOT"

# --- preconditions ----------------------------------------------------------
step "preconditions"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty; commit or stash first"
# The golden gate runs before the clean scratch build and drives the release CLI
# in .build (golden-baseline.sh exits 2 without it), so a missing release build
# used to surface as per-target "golden baseline mismatch" lines. Demand it
# first, where the message can say what to actually run.
[ -x "$ROOT/.build/release/TinyTitanCLI" ] \
  || die "no release build at .build/release/TinyTitanCLI; run: swift build -c release (the golden gate drives that binary)"
# The K2 targets drive the probe instead of the CLI.
[ -x "$ROOT/.build/release/TinyTitanBench" ] \
  || die "no release build at .build/release/TinyTitanBench; run: swift build -c release (the k2-4/k2-8 golden targets drive that binary)"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || die "tag $TAG does not exist locally"
[ "$(git rev-parse "$TAG^{commit}")" = "$(git rev-parse HEAD)" ] \
  || die "HEAD is not $TAG; check out the tagged commit before releasing"
git_tag_pushed_verdict "$TAG"
release_exists_verdict "$REPO" "$TAG"

# --- CI green on the commit being tagged ------------------------------------
# tools/ci-green.sh holds the rule and the reasons -- that CI is the only run
# which sees a clean clone on the pinned toolchains, that a short sha queries
# nothing, and that "no runs" and "still running" are refusals rather than
# warnings. What belongs here is where the answer is demanded: before the gates,
# the build and the golden baselines, and with a recorded reason for going over
# it that then has to survive into the published notes.
TAG_SHA="$(git rev-parse "$TAG^{commit}")"
CI_ALLOW_RED="${TINYTITAN_RELEASE_ALLOW_RED_CI:-}"
CI_ALLOW_RED_REASON="${TINYTITAN_RELEASE_ALLOW_RED_CI_REASON:-}"
if [ -n "$CI_ALLOW_RED" ] && [ -z "$CI_ALLOW_RED_REASON" ]; then
  die "TINYTITAN_RELEASE_ALLOW_RED_CI=$CI_ALLOW_RED without TINYTITAN_RELEASE_ALLOW_RED_CI_REASON; a release over a red CI must record why, and say it in the notes"
fi
CI_ARGS=()
if [ -n "$CI_ALLOW_RED_REASON" ]; then
  CI_ARGS=(--allow-red "$CI_ALLOW_RED_REASON")
fi
if ! CI_REPORT="$("$SCRIPT_DIR/ci-green.sh" "$REPO" "$TAG_SHA" ${CI_ARGS[@]+"${CI_ARGS[@]}"})"; then
  die "CI does not pass on $TAG_SHA: $CI_REPORT"
fi
# The helper's last line is `result<TAB>green|overridden<TAB><first red run URL>`,
# which is how a green run stays quiet and an overridden one learns which URL the
# notes must then quote.
CI_STATUS="$(printf '%s\n' "$CI_REPORT" | tail -1 | cut -f2)"
CI_RED_URL="$(printf '%s\n' "$CI_REPORT" | tail -1 | cut -f3)"
if [ "$CI_STATUS" = "overridden" ]; then
  CI_NOTES_REQUIRE="$CI_RED_URL"
  printf '%s\n' "$CI_REPORT" | sed '$d'
else
  # Both branches have to write it. The notes guard reads this under `set -u`, so
  # leaving the green path unassigned is an `unbound variable` abort at the last
  # step of a publish, after the archives are built. Assigning `$CI_RED_URL`
  # outside the branch is no better: on a green run that is the helper's `-`
  # placeholder, and the notes would then be required to quote a dash.
  CI_NOTES_REQUIRE=""
fi

# A skipped baseline needs its reason before anything expensive starts, so a
# forgotten TINYTITAN_RELEASE_SKIP_GOLDENS_REASON fails here and not an hour later.
if [ -n "${TINYTITAN_RELEASE_SKIP_GOLDENS:-}" ] && [ -z "${TINYTITAN_RELEASE_SKIP_GOLDENS_REASON:-}" ]; then
  die "TINYTITAN_RELEASE_SKIP_GOLDENS=${TINYTITAN_RELEASE_SKIP_GOLDENS} without TINYTITAN_RELEASE_SKIP_GOLDENS_REASON; a skipped baseline must record why"
fi
echo "  tag $TAG at $(git rev-parse --short HEAD), tree clean, no existing Release"

# Identity is single-sourced: `ServerVersion.current` is the one version literal
# in the tree, and it is what the running server prints in its ready banner. A
# tag that disagrees with it would publish binaries that name a different
# version than the Release does, so refuse before anything expensive starts.
# Preparing a release is bumping that one literal -- docs/release-process.md
# step 1 -- and committing it with the tag.
SERVER_VERSION="$(sed -n 's/.*static let current = "\([^"]*\)".*/\1/p' \
  "$ROOT/sources/TinyTitanServer/Core/ServerVersion.swift")"
[ -n "$SERVER_VERSION" ] \
  || die "could not read ServerVersion.current from sources/TinyTitanServer/Core/ServerVersion.swift"
[ "$SERVER_VERSION" = "$VERSION" ] \
  || die "tag $TAG is version $VERSION but ServerVersion.current is $SERVER_VERSION; bump that literal (docs/release-process.md step 1) and commit it before tagging"

rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT"

# --- gates ------------------------------------------------------------------
step "gates"
"$SCRIPT_DIR/lint.sh" || die "tools/lint.sh failed"
test_log="$STAGE_ROOT.testlog"
swift test --no-parallel 2>&1 | tee "$test_log" 2>/dev/null | grep -E 'Test run with'
test_gate_verdict "$test_log" "${PIPESTATUS[0]}"

# The golden baseline is the only check that exercises real inference.
#
# VERIFICATION USES ONLY THE MODELS ALREADY INSTALLED UNDER models/. That
# directory is deliberately kept below the full supported set to save disk, so a
# target with no install is *reported as not checked* -- here and in the release
# notes -- and is never resolved by downloading, converting, repacking or
# re-installing a model. Nothing in this script fetches a model, and the guard
# below re-checks that the golden phase left models/ exactly as it found it.
#
# A baseline the host can see but cannot *read* is a different case and stays a
# documented exception with a mandatory reason. 5.3 was cut on a machine where
# Dropbox had left seven installs online-only and the disk could not hold the
# 134 GB the largest one needed to materialize: every expert read failed, which
# this phase reports as `mismatch (4)` and which has nothing to do with the
# runtime. Deleting a target from the list below would hide that from every
# future reader of this file, so the skip is explicit, carries a reason, prints
# it beside the skip, and must be repeated in the release notes -- --publish
# refuses when it is not:
#
#   TINYTITAN_RELEASE_SKIP_GOLDENS=qwen38-8 \
#   TINYTITAN_RELEASE_SKIP_GOLDENS_REASON="install is Dropbox online-only; 134 GB
#     needed, 123 GB free" tools/release.sh v5.3
GOLDENS_CHECKED=0
GOLDEN_SKIPPED=""
GOLDEN_ABSENT=""
GOLDEN_DECLARED=""
SKIP_GOLDENS="${TINYTITAN_RELEASE_SKIP_GOLDENS:-}"
SKIP_GOLDENS_REASON="${TINYTITAN_RELEASE_SKIP_GOLDENS_REASON:-}"

# Installs that are deliberately NOT golden targets. The coverage guard below
# errors on any installed model missing from check_golden, so this list is how
# an intentional exception is declared instead of being silently unchecked.
#
#   * the MTP draft head -- a sidecar to a target whose own baseline already
#     exercises it, not a served model.
#   * the K2-Horizon installs are golden targets of their own (k2-4/k2-8), but
#     they are driven by `TinyTitanBench k2gpu` rather than the CLI: the family
#     has no CLI path. tools/golden-baseline.sh owns that branch and the bench's
#     `TINYTITAN_K2_GOLDEN` capture/compare.
#
# The dense Qwen 3.5 2B/4B/9B were listed here too, because they had no stored
# baseline at all: a release verified them through neither path, and capturing
# one was an open item. They now have targets of their own, so they moved into
# check_golden and out of this list.
# The names may be laid out one per line for readability; NON_GOLDEN_SET folds
# the whitespace to single spaces first, because the match below is a
# space-delimited substring test and a name that ends a line has no trailing
# space to match on. That bug survived a full dry run once -- the names are
# wrapped, not guessed at.
NON_GOLDEN_INSTALLS="
  ornith-1.5_35B_A3B_MTP_4Bit
"
NON_GOLDEN_SET=" $(printf '%s' "$NON_GOLDEN_INSTALLS" | tr -s '[:space:]' ' ') "

# The gate must not change the machine to pass. Fingerprint what `models/`
# contains before the golden phase and require the same after, so installing,
# removing, renaming or leaving junk behind inside the gate is a failure rather
# than a way through it:
#
#   * every top-level entry -- an install, a sidecar, a stray lock file -- by
#     name, type, size and mtime. A receipt-only fingerprint missed exactly this:
#     an aborted install left `ornith-1.5_35B_A3B_8Bit.install.lock` in models/
#     and the guard could not see it.
#   * every install receipt's bytes, which is what catches a rewritten receipt.
#
# Neither is a payload hash: hashing 461 GB is not a gate, and the receipt the
# runtime verifies is what attests the payload.
install_fingerprint() {
  [ -d "$ROOT/models" ] || return 0
  find "$ROOT/models" -mindepth 1 -maxdepth 1 \
    -exec stat -f '%N %HT %z %m' {} \; | LC_ALL=C sort
  find "$ROOT/models" -maxdepth 2 -name verified-install.json \
    | LC_ALL=C sort | while IFS= read -r f; do
        printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print $1}')" "${f#"$ROOT"/}"
      done
}
INSTALLS_BEFORE="$(install_fingerprint)"

check_golden() {  # <install dir> <golden target>
  GOLDEN_DECLARED="$GOLDEN_DECLARED $1"
  if [ ! -f "$ROOT/models/$1/verified-install.json" ]; then
    echo "  -- NOT CHECKED golden baseline $2 ($1): no install under models/"
    GOLDEN_ABSENT="$GOLDEN_ABSENT $2"
    return 0
  fi
  case " $SKIP_GOLDENS " in
    *" $2 "*)
      echo "  !! SKIPPED golden baseline $2 ($1)"
      echo "  !! reason: $SKIP_GOLDENS_REASON"
      GOLDEN_SKIPPED="$GOLDEN_SKIPPED $2"
      return 0 ;;
  esac
  "$SCRIPT_DIR/golden-baseline.sh" --check "$2" || die "golden baseline mismatch ($2)"
  GOLDENS_CHECKED=$((GOLDENS_CHECKED + 1))
}
# The canonical target names, not the bare `4`/`8` aliases golden-baseline.sh
# still accepts: these strings are what the absence report prints and what
# --publish greps the release notes for, so a bare `8` would match almost any
# notes and make the requirement meaningless.
check_golden ornith-1.5_35B_A3B_8Bit ornith-8
check_golden ornith-1.5_35B_A3B_4Bit ornith-4
# The bf16-dense Ornith build: attention, GDN and the shared expert at the
# checkpoint's bf16 with 4-bit routed experts. A different install from the
# 4-bit row, so it needs its own stored text rather than inheriting ornith-4's.
check_golden ornith-1.5_35B_A3B-bf16dense_4Bit ornith-bf16dense-4
check_golden qwen3.6_35B_A3B_4Bit qwen36-4
check_golden qwen3.6_35B_A3B_8Bit qwen36-8
# The 4-bit type carries the 4/8/16 recipe (experts 4-bit, the rest 8-bit)
# under the canonical `_4Bit` spelling; 8-bit is uniform.
check_golden qwen3.8-flash-next_125B_A6B_4Bit qwen38-4
check_golden qwen3.8-flash-next_125B_A6B_8Bit qwen38-8
check_golden k2-horizon-4bit.ssdai k2-4
check_golden k2-horizon-8bit.ssdai k2-8
check_golden qwen-agentworld_35B_A3B_4Bit agentworld-4
check_golden qwen-agentworld_35B_A3B_8Bit agentworld-8
# KAT-Coder-V2.5-Dev. Declared before its install existed so the first release
# that ships it cannot pass without its baseline.
check_golden kat-coder-v2.5_35B_A3B_4Bit katcoder-4
check_golden kat-coder-v2.5_35B_A3B_8Bit katcoder-8
# The dense Qwen 3.5 models. They used to have no baseline at all, which made
# them the one supported shape a release never verified; the targets exist now
# and are inert on a machine that has not installed them, exactly like every
# other target here.
check_golden qwen3.5_2B_4Bit qwen35-2b-4
check_golden qwen3.5_2B_8Bit qwen35-2b-8
check_golden qwen3.5_4B_4Bit qwen35-4b-4
check_golden qwen3.5_4B_8Bit qwen35-4b-8
check_golden qwen3.5_9B_4Bit qwen35-9b-4
check_golden qwen3.5_9B_8Bit qwen35-9b-8
# Ornith 1.5 122B, the wider member of the Ornith family: hidden 3072 over 48
# layers in the same qwen36 graph. Declared before this machine had the
# installs, so the first release that ships them cannot skip the check.
check_golden ornith-1.5_122B_4Bit ornith122-4
check_golden ornith-1.5_122B_8Bit ornith122-8

# An installed model that no check_golden line covers would be silently
# unchecked. The old guard caught that only when *no* baseline had been checked
# at all, so it could not see a straggler beside a passing target.
for dir in "$ROOT"/models/*/; do
  [ -f "$dir/verified-install.json" ] || continue
  name="$(basename "$dir")"
  case " $GOLDEN_DECLARED $NON_GOLDEN_SET " in
    *" $name "*) continue ;;
  esac
  die "installed model $name has no golden target; add it to check_golden, or declare it in NON_GOLDEN_INSTALLS with a reason"
done

INSTALLS_AFTER="$(install_fingerprint)"
[ "$INSTALLS_BEFORE" = "$INSTALLS_AFTER" ] \
  || die "the golden phase changed the install set under models/; a gate verifies what is installed and never installs, removes or rewrites a model"

if [ "$GOLDENS_CHECKED" = 0 ]; then
  echo "  no golden baseline could be checked on this machine (state this in the notes)"
else
  echo "  $GOLDENS_CHECKED golden baseline(s) identical"
fi
if [ -n "$GOLDEN_ABSENT" ]; then
  echo "  NOT CHECKED — no install in models/, and none may be fetched to fix that:$GOLDEN_ABSENT"
fi
if [ -n "$GOLDEN_SKIPPED" ]; then
  echo "  NOT CHECKED — documented skip:$GOLDEN_SKIPPED"
fi
if [ -n "$GOLDEN_ABSENT$GOLDEN_SKIPPED" ]; then
  echo "  the release notes must name every baseline that was not checked"
fi

# --- clean build ------------------------------------------------------------
step "clean release build"
rm -rf "$SCRATCH"
swift build -c release --scratch-path "$SCRATCH" 2>&1 | tee "$STAGE_ROOT.buildlog" | tail -1
# The status this replaces could only report a warning it *found*: a build log
# that was missing, empty, unreadable or truncated left `die` unreached and the
# release staged anyway. assert_clean_build_log refuses all four.
assert_clean_build_log "$STAGE_ROOT.buildlog" "release build"
BIN="$SCRATCH/release"
[ -x "$BIN/TinyTitanServer" ] || die "build produced no TinyTitanServer"
# Resolve the products directory physically before anything globs it. SwiftPM's
# `$SCRATCH/release` is a symlink to `out/Products/Release`, and `find` does NOT
# follow a symlink given as its own starting point — so `find "$BIN" -name
# '*.bundle'` matched nothing while `cp "$BIN/$product"` worked through the same
# link. 5.6 shipped without its Metal shader library that way, and the release
# binaries died with "unable to find bundle named TinyTitan_TinyTitan".
BIN="$(cd "$BIN" && pwd -P)"

# --- stage ------------------------------------------------------------------
step "stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
for p in "${PRODUCTS[@]+"${PRODUCTS[@]}"}"; do
  [ -x "$BIN/$p" ] || die "missing product: $p"
  cp "$BIN/$p" "$STAGE/"
done
# .bundle resources carry the Metal shader library; without them beside the
# executables the runtime cannot load its kernels. Fail closed rather than
# shipping an archive that dies on the first model load. Test bundles are
# excluded: they hold fixtures the tests read, and 5.5's archive carried the six
# that the executables of the day needed.
[ -d "$BIN/TinyTitan_TinyTitan.bundle" ] \
  || die "no TinyTitan_TinyTitan.bundle in $BIN: the Metal shader library would not ship"
find "$BIN" -maxdepth 1 -name '*.bundle' ! -name '*Tests.bundle' -exec cp -R {} "$STAGE/" \;
# LICENSE and NOTICE are what Apache-2.0 requires to travel with a binary
# distribution; THIRD_PARTY_NOTICES.md carries the upstream attributions.
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"

cat > "$STAGE/README-binaries.txt" <<TXT
TinyTitan $VERSION — prebuilt binaries (macOS, Apple Silicon / arm64)

Built from tag $TAG with: swift build -c release
Requires macOS 26+. Apple Silicon only; there is no x86_64 build.

Contents
  TinyTitanServer          OpenAI-compatible local server (binds 127.0.0.1 only)
  TinyTitanCLI             one-shot prompt CLI
  TinyTitanRepack          model installer / repacker
  TinyTitanBench           benchmark driver
  *.bundle             Metal shader library and other runtime resources — keep
                       these next to the executables or the runtime cannot
                       load its kernels
  LICENSE              Apache License 2.0
  NOTICE               copyright and upstream attribution
  THIRD_PARTY_NOTICES.md

These binaries are NOT code-signed or notarized. macOS Gatekeeper will refuse
them on first run. Either build from source, or clear the quarantine attribute
yourself after verifying the checksum published with this archive:

  xattr -dr com.apple.quarantine /path/to/tinytitan-$VERSION-macos-arm64

No model weights are included. TinyTitanRepack defaults to Ornith 1.5 8-bit (about
36.9 GB); 4-bit remains available explicitly. The runtime defaults to standard
answers with thinking off, as described in the README and Wiki.
TXT

step "package"
( cd "$STAGE_ROOT" && tar czf "$ARCHIVE" "$(basename "$STAGE")" )
# Assert on the *archive*, not the staging directory: a bundle that failed to
# stage is invisible until someone runs the binary somewhere else, which is how
# 5.6's first upload shipped without the Metal shader library.
tar tzf "$ARCHIVE" | grep -q '^[^/]*/TinyTitan_TinyTitan\.bundle/' \
  || die "the archive carries no TinyTitan_TinyTitan.bundle: the runtime could not load its kernels"
# Every product the build produced, by name, in the archive. The stage loop above
# proves each one was executable *before* the copy, and `set -e` is off here, so a
# `cp` that fails partway -- out of disk is the realistic one -- would otherwise
# leave an archive with three of four binaries and a release that never mentions
# the missing one. The arch assertion below counts Mach-O artifacts and reports
# that count; this is what makes the count mean four rather than "whatever arrived".
for p in "${PRODUCTS[@]+"${PRODUCTS[@]}"}"; do
  tar tzf "$ARCHIVE" | grep -q "^[^/]*/$p$" \
    || die "the archive carries no $p, which the build produced and the stage copied"
done
# The checksum is the release notes' promise about these bytes, so the arch of
# what the archive actually holds is asserted before it is taken, not after.
assert_arm64_archive "$ARCHIVE" "engine"
shasum -a 256 "$ARCHIVE" | sed "s|$STAGE_ROOT/||" > "$ARCHIVE.sha256"
SHA="$(awk '{print $1}' "$ARCHIVE.sha256")"
BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"
echo "  $(basename "$ARCHIVE")  $BYTES bytes"
echo "  sha256 $SHA"

# --- the library ------------------------------------------------------------
# Every release carries the library product in both of its link forms, so this
# is not a step that can be skipped or remembered: it runs here, and the two
# `die`s below are what stop a release that staged an archive without them.
#
# It is a second archive rather than more files in the first because the two
# audiences want different things — the engine archive is executables, this one
# is `libTinyTitanLib.a`, `libTinyTitanLib.dylib`, the Swift module and the
# resource bundle they have to travel with. `tools/build_library.sh` owns the
# build and its own assertions; this stages, names and checksums what it wrote.
step "library"
LIB_STAGE="$STAGE_ROOT/library"
LIB_ARCHIVE="$STAGE_ROOT/tinytitan-lib-$VERSION-macos-arm64.tar.gz"
"$SCRIPT_DIR/build_library.sh" "$VERSION" --out "$LIB_STAGE" --scratch-path "$SCRATCH"
( cd "$STAGE_ROOT" && tar czf "$LIB_ARCHIVE" "$(basename "$LIB_STAGE")" )
# Assert on the *archive*, not the staging directory: a dylib that failed to
# stage is invisible until somebody links against the download.
for member in libTinyTitanLib.a libTinyTitanLib.dylib TinyTitanLib.swiftmodule/ \
  TinyTitanKernelsC.modulemap TinyTitan_TinyTitan.bundle/Contents/Resources/Metal/ \
  demo/main.swift demo/build-dynamic.sh; do
  tar tzf "$LIB_ARCHIVE" | grep -q "$member" \
    || die "the library archive carries no $member"
done
# The member loop above proves the names shipped; this proves the bytes under
# those names are the Apple Silicon ones. `libTinyTitanLib.a` is an `!<arch>`
# container and `lipo -archs` reads it (measured: `arm64`), so the static form
# an embedder links is covered alongside the dylib. There is no demo *binary* in
# this archive — `demo/` ships main.swift and the two build scripts — so if a
# compiled demo is ever added here, the magic scan checks it without being told.
assert_arm64_archive "$LIB_ARCHIVE" "library"
shasum -a 256 "$LIB_ARCHIVE" | sed "s|$STAGE_ROOT/||" > "$LIB_ARCHIVE.sha256"
LIB_SHA="$(awk '{print $1}' "$LIB_ARCHIVE.sha256")"
LIB_BYTES="$(wc -c < "$LIB_ARCHIVE" | tr -d ' ')"
echo "  $(basename "$LIB_ARCHIVE")  $LIB_BYTES bytes"
echo "  sha256 $LIB_SHA"

# --- the tools ---------------------------------------------------------------
# `tools/install_tinytitan.sh` downloads this, verifies it and then *runs* what
# it extracted: the launcher, the model installer and the DSH setup. So it is a
# released artifact with a released digest, exactly like the other two archives,
# and the installer refuses a release that has no checksum for it.
#
# `git archive` of the tag rather than a hand-picked list of files, because the
# installed layout is meant to look like a checkout apart from the build, and a
# list of what the tools need would be a second place to remember that. The
# members asserted below are the ones the installer itself names.
#
# This archive gets no `assert_arm64_archive`: `git archive` of a tag is source
# and scripts, so the scan would find no Mach-O and the gate would die on a
# correctly built tools archive. It is the one artifact here with no binary in it.
step "tools"
TOOLS_STAGE_PREFIX="tinytitan-$VERSION-tools"
TOOLS_ARCHIVE="$STAGE_ROOT/$TOOLS_STAGE_PREFIX.tar.gz"
git archive --format=tar --prefix="$TOOLS_STAGE_PREFIX/" "$TAG" | gzip > "$TOOLS_ARCHIVE" \
  || die "git archive produced no tools snapshot for $TAG"
for member in tools/install_tinytitan.sh tools/server_launcher.sh tools/install_models.sh \
  tools/tinytitan_models.sh plugins/dsh-tinytitan/ Package.swift \
  sources/TinyTitanLib/; do
  tar tzf "$TOOLS_ARCHIVE" | grep -q "^$TOOLS_STAGE_PREFIX/$member" \
    || die "the tools archive carries no $member, which an installed copy needs"
done
shasum -a 256 "$TOOLS_ARCHIVE" | sed "s|$STAGE_ROOT/||" > "$TOOLS_ARCHIVE.sha256"
TOOLS_SHA="$(awk '{print $1}' "$TOOLS_ARCHIVE.sha256")"
TOOLS_BYTES="$(wc -c < "$TOOLS_ARCHIVE" | tr -d ' ')"
echo "  $(basename "$TOOLS_ARCHIVE")  $TOOLS_BYTES bytes"
echo "  sha256 $TOOLS_SHA"

# --- notes ------------------------------------------------------------------
# Checked whenever --notes is given, not only for --publish. Compaction and its
# budget are cheap to check here and painful to discover after a full gate run.
if [ -n "$NOTES" ]; then
  [ -f "$NOTES" ] || die "notes file not found: $NOTES"

  # A baseline that was not checked -- skipped by name, or absent because its
  # model is not installed under models/ -- is only acceptable when the notes name
  # it: the point is that a reader of the Release learns what was not re-checked.
  # Naming an absent target never means fetching it; the model stays absent.
  for notchecked in $GOLDEN_SKIPPED $GOLDEN_ABSENT; do
    grep -q "$notchecked" "$NOTES" \
      || die "notes do not mention the unchecked baseline $notchecked; every baseline that was not checked must be named in the notes"
  done

  # Two values in the notes are only knowable here. --publish rebuilds from
  # scratch, so the binaries carry fresh mtimes and the archive both hashes and
  # weighs differently from any dry run; a number copied out of a dry run is a
  # false claim waiting to be published. (5.4's notes quoted the dry run's
  # 24,770,128 bytes for an archive that published at 24,770,200.) So the notes
  # carry SHA256_PENDING and ARCHIVE_BYTES_PENDING and both are filled in here,
  # which makes the invariant hold by construction.
  #
  # A wrong digest is worse than none: it tells a careful user their download is
  # corrupt, which is how 3.7 shipped for a few minutes. Both are enforced.
  RENDERED_NOTES="$STAGE_ROOT/notes-rendered.md"
  # Order matters. `SHA256_PENDING` is a *substring* of both
  # `LIBRARY_SHA256_PENDING` and `TOOLS_SHA256_PENDING`, so substituting the
  # engine's token first rewrites those names into `LIBRARY_<engine digest>` and
  # `TOOLS_<engine digest>` — their own rules then match nothing and the notes
  # publish a digest that belongs to the other archive. The guard below catches
  # it (it did, on the first release that carried these placeholders), but the
  # fix is not to write it that way: specific token first.
  sed -e "s/LIBRARY_SHA256_PENDING/$LIB_SHA/g" -e "s/TOOLS_SHA256_PENDING/$TOOLS_SHA/g" \
    -e "s/SHA256_PENDING/$SHA/g" \
    -e "s/LIBRARY_BYTES_PENDING/$LIB_BYTES/g" -e "s/TOOLS_BYTES_PENDING/$TOOLS_BYTES/g" \
    -e "s/ARCHIVE_BYTES_PENDING/$BYTES/g" \
    "$NOTES" > "$RENDERED_NOTES" \
    || die "failed to render notes"
  grep -q 'SHA256_PENDING' "$NOTES" && echo "  filled SHA256_PENDING with $SHA"
  grep -q 'ARCHIVE_BYTES_PENDING' "$NOTES" && echo "  filled ARCHIVE_BYTES_PENDING with $BYTES"
  grep -q 'LIBRARY_SHA256_PENDING' "$NOTES" && echo "  filled LIBRARY_SHA256_PENDING with $LIB_SHA"
  grep -q 'LIBRARY_BYTES_PENDING' "$NOTES" && echo "  filled LIBRARY_BYTES_PENDING with $LIB_BYTES"
  grep -q 'TOOLS_SHA256_PENDING' "$NOTES" && echo "  filled TOOLS_SHA256_PENDING with $TOOLS_SHA"
  grep -q 'TOOLS_BYTES_PENDING' "$NOTES" && echo "  filled TOOLS_BYTES_PENDING with $TOOLS_BYTES"
  grep -q "$SHA" "$RENDERED_NOTES" \
    || die "the notes neither contain SHA256_PENDING nor quote this archive's sha256 ($SHA)"
  grep -q "$BYTES" "$RENDERED_NOTES" \
    || die "the notes neither contain ARCHIVE_BYTES_PENDING nor quote this archive's size ($BYTES bytes)"
  # The library ships on every release, so its digest is enforced exactly like
  # the engine archive's: a release that publishes a `.a` and a `.dylib` the
  # notes never mention leaves its readers unable to verify the download.
  grep -q "$LIB_SHA" "$RENDERED_NOTES" \
    || die "the notes neither contain LIBRARY_SHA256_PENDING nor quote the library archive's sha256 ($LIB_SHA)"
  grep -q "$LIB_BYTES" "$RENDERED_NOTES" \
    || die "the notes neither contain LIBRARY_BYTES_PENDING nor quote the library archive's size ($LIB_BYTES bytes)"
  # Same argument for the tools archive, and one more: the installer dies without
  # its published checksum, so notes that do not quote it describe a download
  # nobody can verify or explain.
  grep -q "$TOOLS_SHA" "$RENDERED_NOTES" \
    || die "the notes neither contain TOOLS_SHA256_PENDING nor quote the tools archive's sha256 ($TOOLS_SHA)"
  grep -q "$TOOLS_BYTES" "$RENDERED_NOTES" \
    || die "the notes neither contain TOOLS_BYTES_PENDING nor quote the tools archive's size ($TOOLS_BYTES bytes)"

  # The Release page gets the COMPACT form: the same claims as bullets, one
  # sentence each, wrapped narrow. The full notes stay in the repo as the record
  # of why each change exists and how it was verified.
  #
  # Two reasons this is a build step rather than a request to the author:
  # compaction is mechanical (re-lay-out, never reword), so it should not depend
  # on remembering; and every string the greps above rely on is passed back in as
  # --require, so a compaction that would drop an unchecked baseline fails HERE,
  # before the Release exists, rather than publishing one that is quietly
  # missing a target.
  #
  # The character budget is the part that actually keeps notes short -- the
  # compactor can only reformat, so a draft that says too much still says too
  # much. Raise it deliberately with TINYTITAN_RELEASE_NOTES_MAX_CHARS.
  COMPACT_NOTES="$STAGE_ROOT/notes-compact.md"
  NOTES_MAX_CHARS="${TINYTITAN_RELEASE_NOTES_MAX_CHARS:-12000}"
  # A budget is a size, not a switch. Zero is the compactor's documented
  # "no limit", so handing one to it would publish the release with the notes
  # gate removed and say nothing about it -- unlike the other two deliberate
  # narrowings on this path, each of which dies without its reason and names
  # itself in the notes. The shapes that are not sizes are refused here rather
  # than at the compactor, where an argparse failure and an over-long note
  # would both arrive as the same "the notes did not survive compaction".
  case "$NOTES_MAX_CHARS" in
    '' | *[!0-9]*)
      die "TINYTITAN_RELEASE_NOTES_MAX_CHARS=$NOTES_MAX_CHARS is not a character budget; name a whole number of characters, 1 or more"
      ;;
  esac
  if [ "$NOTES_MAX_CHARS" -lt 1 ]; then
    die "TINYTITAN_RELEASE_NOTES_MAX_CHARS=0 is the compactor's no-limit, which removes the notes budget; raise it by naming a number of characters"
  fi
  REQUIRE_ARGS=()
  for required in $GOLDEN_SKIPPED $GOLDEN_ABSENT "$SHA" "$BYTES" "$LIB_SHA" "$LIB_BYTES" \
    "$TOOLS_SHA" "$TOOLS_BYTES"; do
    REQUIRE_ARGS+=(--require "$required")
  done
  # When CI was red and the operator chose to release over it, the run that says
  # so has to survive into the published notes: the override is honest while it is
  # written down, and a reader of the Release page is the audience that matters.
  if [ -n "$CI_NOTES_REQUIRE" ]; then
    REQUIRE_ARGS+=(--require "$CI_NOTES_REQUIRE")
  fi
  python3 "$SCRIPT_DIR/compact-release-notes.py" "$RENDERED_NOTES" \
    --out "$COMPACT_NOTES" \
    --max-chars "$NOTES_MAX_CHARS" \
    "${REQUIRE_ARGS[@]+"${REQUIRE_ARGS[@]}"}" \
    || die "the notes did not survive compaction, or are over the ${NOTES_MAX_CHARS}-character budget (raise it with TINYTITAN_RELEASE_NOTES_MAX_CHARS)"
fi

# --- publish ----------------------------------------------------------------
if [ "$PUBLISH" -ne 1 ]; then
  step "dry run complete"
  echo "  staged: $STAGE"
  echo "  library: $(basename "$LIB_ARCHIVE")"
  echo "  tools: $(basename "$TOOLS_ARCHIVE")"
  if [ -n "$NOTES" ]; then
    echo "  release page: the compact form checked above is what --publish would carry"
  else
    echo "  (pass --notes docs/release-notes-vX.Y.md to check the notes here too)"
  fi
  echo "  re-run with --publish to create the Release on $REPO"
  exit 0
fi

[ -n "$NOTES" ] || die "--publish needs --notes <file> (see the previous release for the shape)"

step "publish"
gh release create "$TAG" "$ARCHIVE" "$ARCHIVE.sha256" "$LIB_ARCHIVE" "$LIB_ARCHIVE.sha256" \
  "$TOOLS_ARCHIVE" "$TOOLS_ARCHIVE.sha256" \
  --repo "$REPO" \
  --title "TinyTitan $VERSION" \
  --notes-file "$COMPACT_NOTES" \
  --latest || die "gh release create failed"
gh release view "$TAG" --repo "$REPO" --json url,assets \
  --jq '"  \(.url)\n  assets: \([.assets[].name] | join(", "))"'
