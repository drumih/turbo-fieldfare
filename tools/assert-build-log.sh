#!/usr/bin/env bash
#
# The compiler-warning gate for a release build log.
#
#   . tools/assert-build-log.sh
#   assert_clean_build_log <logfile> <label>
#
# `tools/release.sh` and `tools/build_library.sh` both need it, and both shipped
# the same line for it: a `grep -qE ... && die` over the build log. grep answers
# with three different statuses -- 0 found a warning, 1 found none, 2 could not
# read the file -- and that shape can only act on 0, so a log that was missing,
# empty, unreadable or truncated left `die` unreached and the script carried on to
# stage and publish. AUD-266. The workflow ran the same line a third time as
# `if grep -qE ...; then exit 1; fi`, which fails open the same way, so its build
# step sources this file too.
#
# Four refusals, four answers. "This log holds no warning", "this log records no
# finished build", "I cannot read this log" and "the pattern did not run against
# this log" are different findings, and only the first may be a pass. The
# completion marker is what separates a log that holds a build from one that holds
# a fragment: every real build log on the machine this was measured on -- `find
# .build -name '*.buildlog'`, eleven files: the six release logs, the three library
# logs under their release directories, and the two library-dist logs -- is UTF-8
# text carrying at least one line that starts `Build complete!`, and all eleven
# pass this gate.
#
# Deliberately not changed here: the pattern's extension set, which is exactly the
# set this package compiles (no `.cpp`, `.cc`, `.cxx` or `.S` under `sources/`).
# Widening it to every line that contains `warning:` would refuse the three most
# recent releases, each of whose logs carries SwiftPM's own
# `warning: 'swift-nio': skipping cache due to an error:` -- a build-management
# message, not a diagnostic -- so whether a release should stop on one is a
# decision, not a fix.
set -uo pipefail

# The diagnostics a compiler prints as `file:line:col: warning: `. Single-sourced:
# the two callers each had their own verbatim copy, which is how one gets fixed and
# the other stays broken.
BUILD_WARNING_PATTERN='^[^ ]+\.(swift|metal|c|h|m|mm):[0-9]+:[0-9]+: warning:'

bl_die() { echo "error: $*" >&2; exit 1; }

assert_clean_build_log() {  # <logfile> <label>
  local log label status lines
  [ $# -eq 2 ] || bl_die "usage: assert_clean_build_log <logfile> <label>"
  log="$1"
  label="$2"
  [ -f "$log" ] \
    || bl_die "$label: no build log at $log — the warning gate has nothing to read"
  # Readability is settled once, here, so the two reads below can treat "no match"
  # and "cannot read" as different answers without each re-deciding the first.
  [ -r "$log" ] \
    || bl_die "$label: cannot read $log — a log this gate may not open is not a log it has checked"
  grep -q '^Build complete!' "$log" \
    || bl_die "$label: $log holds no 'Build complete!' line — an empty or truncated log records no build, so it proves nothing about warnings"
  status=0
  grep -qE "$BUILD_WARNING_PATTERN" "$log" || status=$?
  case "$status" in
    0) bl_die "$label: the build emitted compiler warnings — see $log" ;;
    1) : ;;
    # Reachable without a race: grep answers an invalid expression with 2, so a
    # corrupted pattern refuses instead of reporting a clean build.
    *) bl_die "$label: cannot check $log (grep status $status) — the warning pattern did not run against it" ;;
  esac
  # Say what was read, so a pass over one line is not mistaken for a pass over a build.
  lines="$(wc -l < "$log")"
  lines="${lines//[[:space:]]/}"
  echo "  $label: build log read, $lines line(s), no compiler warnings"
}

# --- generic Apple Silicon ---------------------------------------------------
# The shipped artifacts are one arm64 build for every Apple Silicon Mac, so the
# release path must never opt into a CPU-specific codegen. `tools/build_m3.sh`
# is the deliberate local exception and is not referenced here. Both release
# builders call this before they build: the manifest and the two build commands
# are the places a `-target-cpu`/`-mcpu` flag could enter, and a release that
# carries one would hand an M1/M2 a binary it may not run.
assert_generic_arm64_build() {  # <root>
  local root="${1:-}" hits
  if [ -z "$root" ] || [ ! -d "$root" ]; then
    bl_die "usage: assert_generic_arm64_build <root>"
  fi
  hits="$(grep -nE -- '(-target-cpu|-mcpu)' "$root/Package.swift" 2>/dev/null || true)"
  [ -z "$hits" ] \
    || bl_die "Package.swift opts into CPU-specific codegen; the shipped build must stay generic: $hits"
  hits="$(grep -nE -- '(-target-cpu|-mcpu)' "$root/tools/release.sh" "$root/tools/build_library.sh" 2>/dev/null || true)"
  [ -z "$hits" ] \
    || bl_die "a release builder opts into CPU-specific codegen; the shipped build must stay generic: $hits"
  echo "  portable arm64: no CPU-specific codegen in Package.swift or the release builders"
}

# A library: sourced, never run. Direct invocation says so rather than doing nothing
# and exiting 0, which is the shape this file exists to remove.
case "$0" in
  */assert-build-log.sh|assert-build-log.sh)
    bl_die "assert-build-log.sh is a library — source it and call assert_clean_build_log"
    ;;
esac
