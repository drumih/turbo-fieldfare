#!/usr/bin/env bash
# Production gates that the compiler cannot express. Run locally before a PR;
# CI runs the same script, so a green run here is a green run there.
#
#   tools/lint.sh              # all checks
#   tools/lint.sh force-cast   # one check
#
# Checks:
#   force-cast          no `as!` / `try!` in sources/ without an audited opt-out
#   unbounded-read      no whole-file read in sources/ without an audited bound
#   func-length         no NEW function longer than MAX_FUNC_LINES (ratcheted)
#   file-length         no production source under sources/ over MAX_FILE_LINES
#   unchecked-sendable  new `@unchecked Sendable` must document its invariant
#   converter           routed experts must land at their own index
#   arch-path           no hardcoded SwiftPM triple in a build path (see below)
#   stdout-clean        no stdout write anywhere in the library's target closure
#   silent-test-skip    no env/capability-shaped early return in tests/ (see below)
#   test-hollow         no @Test body that cannot fail (see below)
#   library-facade      TinyTitanLib public surface allowlisted; no NIO import,
#                       no stdout write (AGENTS.md "Two products" rules 1 and 3)
#   docs                a documented count, mode name, tag->commit sha, ledger
#                       commit reference or table shape must match what the
#                       repository computes (see below)
#   shell-portability   scripts run on the system bash (3.2), not just the dev one
#   shell-lint          shellcheck warnings-as-errors over every script, pinned version
#   swiftlint           SwiftLint violations-as-errors under the committed config
#   swift-format        formatting enforced under the committed .swift-format
#   javascript          eslint + prettier --check over the plugin packages, pinned
#   python              ruff check + ruff format --check under pyproject.toml,
#                       with the pinned ruff version
#   core-purity         TinyTitanCore carries no model knowledge (7.0 spec §1.3)
#   variant-isolation   a variant target imports only the generic library (7.0 spec §3)
#
# Opting out of force-cast: put `lint:allow-force <reason>` in a comment on
# the line immediately above. The reason is mandatory and is what a reviewer
# reads — an opt-out without one fails the same as no opt-out at all.
#
# Opting out of arch-path: `lint:allow-arch-path <reason>` on the line above,
# for a deliberate compatibility fallback rather than a build path.
#
# Opting out of stdout-clean: `lint:allow-stdout <reason>` on the line or the
# line above the write, for a stdout this process legitimately owns. There are
# none in the closure today.
#
# Opting out of unbounded-read: `lint:allow-unbounded-read <reason>` in the comment
# block above, for a read whose input is already bounded by something other than a
# size cap. See the check for what counts as a reason.
#
# Opting out of the converter check: `ALLOW_MISSING_CONVERTER_DEPS=1`. The gate
# fails when python3 or the converter's three pinned dependencies are absent,
# because a check that does not run must not read as a pass; the variable is the
# documented way to say "this machine cannot install them", and it prints the
# skip instead of quieting it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT
BASELINE="$SCRIPT_DIR/func-length-baseline.txt"
FACADE_BASELINE="$SCRIPT_DIR/library-facade-baseline.txt"
FACADE_UPDATE="${FACADE_UPDATE:-}"
export FACADE_BASELINE
MAX_FUNC_LINES="${MAX_FUNC_LINES:-120}"
export MAX_FILE_LINES="${MAX_FILE_LINES:-500}"

status=0
want="${1:-all}"

# --- force-cast / force-try -------------------------------------------------
check_force_cast() {
  echo "== force-cast: as! / try! outside tests =="
  local found=0 scanned
  # grep's own failure is invisible here by construction -- the walk is a process
  # substitution, so its status never reaches the loop, and stderr goes to
  # /dev/null. The count of what the scan could open is the only thing that tells
  # "no offender" apart from "no input", and the siblings already insist on it.
  scanned="$(find "$ROOT/sources" -name '*.swift' -type f 2>/dev/null | grep -c . || true)"
  if [ "${scanned:-0}" -eq 0 ]; then
    echo "  FAIL: force-cast read 0 Swift files under $ROOT/sources."
    echo "        Expected ~370; check ROOT and the directory, and read this as the"
    echo "        gate not having run rather than as code that has no force casts."
    status=1
    return
  fi
  while IFS= read -r hit; do
    local file line
    file="${hit%%:*}"
    line="$(echo "$hit" | cut -d: -f2)"
    # Walk up the contiguous comment block directly above the hit, looking for
    # an opt-out marker followed by a reason. Scanning the whole block (not
    # just the previous line) lets the reason wrap naturally.
    local n=$((line - 1)) text ok=0
    while [ "$n" -ge 1 ]; do
      text="$(sed -n "${n}p" "$file")"
      echo "$text" | grep -qE '^[[:space:]]*//' || break
      if echo "$text" | grep -qE 'lint:allow-force[[:space:]]+[^[:space:]]'; then
        ok=1
        break
      fi
      n=$((n - 1))
    done
    [ "$ok" -eq 1 ] && continue
    echo "  ${file#$ROOT/}:$line: $(echo "$hit" | cut -d: -f3- | sed 's/^[[:space:]]*//')"
    found=1
  done < <(grep -rnE '(\bas!\s|\btry!\s)' --include='*.swift' "$ROOT/sources" 2>/dev/null)

  if [ "$found" -ne 0 ]; then
    echo "  FAIL: force cast/try without an audited 'lint:allow-force <reason>' comment above it"
    status=1
  else
    echo "  ok ($scanned Swift files scanned, no unexempt force cast/try)"
  fi
}

# --- unbounded whole-file reads ---------------------------------------------
# A metadata document read by `Data(contentsOf:)` or `String(contentsOf:)`
# materializes the whole file before anything looks at it, so a bound applied to
# the bytes afterwards is a bound applied after the cost. Measured here: a 2 GiB
# sparse file (0 B allocated on disk, so the read is pure allocation) took 0.350 s
# and raised the process footprint by 2,049 MB on a 24 GB Mac — enough to page a
# resident model out mid-generation. The nine load-path sites that had that shape
# now read through `BoundedMetadataRead` (engine) or `Posix.readBoundedData`
# (converter), which `fstat` the descriptor being read and refuse before
# allocating.
#
# This gate keeps that from regressing: every whole-file read under `sources/`
# needs an audited reason. Opting out: `lint:allow-unbounded-read <reason>` in the
# comment block directly above, where the reason names the bound the input already
# carries — a range this process itself requested, a resource the package ships, an
# operator-named file outside every trust boundary, or an mmap that is the point of
# the read.
#
# The call is found by joining each line with its continuations, not by matching
# one line: `swift-format` wraps a long argument list, and a line-oriented grep
# reported `ok` over `Data(\n    contentsOf: …)` -- which is exactly how one of the
# nine reads AUD-142 bounded survived it. A gate that cannot see its own input
# reads as a pass over code it never looked at, the failure this file already
# refuses for `func-length` by printing UNRESOLVED instead of shrugging.
unbounded_read_hits() {
  ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    root = File.join(ENV.fetch("ROOT"), "sources")
    paths = Dir.glob(File.join(root, "**", "*.swift")).sort
    # A search that found no files has read no code, and silence here would print
    # `ok` over the whole tree. `ROOT` is exported above; this is the guard for the
    # day it is not, or the day the directory moves.
    raise "no Swift files under #{root}" if paths.empty?
    paths.each do |path|
      lines = File.readlines(path, chomp: true)
      lines.each_with_index do |line, i|
        # The walk starts at the call, not at the statement around it: an outer
        # line would otherwise report the same read a second time, at a line no
        # exemption can be attached to.
        next unless line =~ /(Data|String)\(/
        depth = line.count("(") - line.count(")")
        joined = line
        j = i
        while depth > 0 && j + 1 < lines.length
          j += 1
          joined += " " + lines[j]
          depth += lines[j].count("(") - lines[j].count(")")
        end
        next unless joined =~ /(Data|String)\(\s*contentsOf:/
        puts "#{path}:#{i + 1}:#{line}"
      end
    end
  '
}

check_unbounded_metadata_read() {
  echo "== unbounded-read: whole-file reads under sources/ must state their bound =="
  local found=0
  local hits
  if ! hits="$(unbounded_read_hits)"; then
    echo "  FAIL: the walk over sources/ did not run"
    echo "        a gate that searched nothing would report a pass over code it never read"
    status=1
    return
  fi
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    local file line text n ok=0
    file="${hit%%:*}"
    line="$(echo "$hit" | cut -d: -f2)"
    text="$(sed -n "${line}p" "$file")"
    # A doc comment that only names the shape is not a read.
    case "$(echo "$text" | sed 's/^[[:space:]]*//')" in
      //*|'/*'*) continue ;;
    esac
    n=$((line - 1))
    while [ "$n" -ge 1 ]; do
      text="$(sed -n "${n}p" "$file")"
      echo "$text" | grep -qE '^[[:space:]]*//' || break
      if echo "$text" | grep -qE 'lint:allow-unbounded-read[[:space:]]+[^[:space:]]'; then
        ok=1
        break
      fi
      n=$((n - 1))
    done
    [ "$ok" -eq 1 ] && continue
    echo "  ${file#$ROOT/}:$line: $(echo "$hit" | cut -d: -f3- | sed 's/^[[:space:]]*//')"
    found=1
  done <<< "${hits}"

  if [ "$found" -ne 0 ]; then
    echo "  FAIL: unbounded whole-file read without a 'lint:allow-unbounded-read <reason>' comment above it"
    echo "        bound the read instead (BoundedMetadataRead / Posix.readBoundedData), or say why it needs no bound"
    status=1
  else
    echo "  ok"
  fi
}

# --- function length --------------------------------------------------------
# Indentation-anchored: a function runs from its `func` line to the first line
# that closes a brace at the same indent. Brace-depth counting drifts on braces
# inside strings and comments; this codebase is consistently formatted, so
# indent is the more reliable anchor.
#
# Every `func` must land in exactly one of three buckets: no body (a protocol
# requirement), a body that opens and closes on one line, or a body with a
# closer at its own indent. Anything else is printed as UNRESOLVED and fails
# the check. A gate that silently skips what it cannot parse reports "ok" for
# code it never looked at, which is worse than no gate — so unparsed input is
# an error, not a shrug.
measure_functions() {
  ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    limit = Integer(ENV.fetch("MAX_FUNC_LINES", "120"))
    root = ENV.fetch("ROOT")
    scanned = 0
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      lines = File.readlines(path, chomp: true)
      rel = path.delete_prefix(root + "/")
      lines.each_with_index do |line, i|
        next unless (m = line.match(/^(\s*)(?:[\w@\(\)]+\s+)*func\s+([A-Za-z_]\w*)/))
        scanned += 1
        indent, name = m[1], m[2]

        # An inline opt-out in the contiguous comment block above, mirroring
        # lint:allow-force. Preferred over a baseline row for a function that is
        # long on purpose: the reason sits next to the code instead of in a
        # separate file, so it is reviewed whenever the function is.
        k = i - 1
        exempt = false
        while k >= 0 && lines[k] =~ /^\s*(\/\/|\/\/\/)/
          if lines[k] =~ /lint:allow-long\s+\S/
            exempt = true
            break
          end
          k -= 1
        end
        next if exempt

        # Walk the (possibly multi-line) signature looking for the body brace.
        # Stop at the next declaration or at a closer no deeper than us, which
        # is what a bodyless protocol requirement runs into.
        open_at = nil
        j = i
        while j < lines.length
          text = lines[j].sub(%r{//.*$}, "")
          if j > i && text =~ /^\s{0,#{indent.length}}(\}|func\s|var\s|let\s|case\s)/
            break
          end
          if text.include?("{")
            open_at = j
            break
          end
          j += 1
        end

        if open_at.nil?
          next # no body: protocol requirement or bodyless declaration
        end

        opener = lines[open_at].sub(%r{//.*$}, "")
        if opener.count("{") == opener.count("}") && opener.rstrip.end_with?("}")
          next # body opens and closes on one line
        end

        closer = /^#{indent}\}/
        stop = ((open_at + 1)...lines.length).find { |k| lines[k] =~ closer }
        if stop.nil?
          puts "UNRESOLVED:#{rel}:#{name}:#{i + 1}"
          next
        end
        length = stop - i
        next unless length > limit
        puts "#{rel}:#{name}:#{length}"
      end
    end
    # Coverage receipt. Without it an empty result is indistinguishable from
    # "scanner never ran", and the gate would report ok for an unexamined tree.
    puts "SCANNED:#{scanned}"
  '
}

check_func_length() {
  echo "== func-length: no NEW function over $MAX_FUNC_LINES lines =="
  local measured current new unresolved raw rc scanned

  # Capture without a pipe so the scanner's exit status survives, then check it.
  # A gate whose measurement step died must fail, not report "ok (0 new)" —
  # that is how an unexported ROOT once let this check pass while looking at
  # nothing at all.
  raw="$(measure_functions)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the length scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the length scanner reported no functions scanned."
    echo "        Expected ~1000 under sources/; check ROOT and the glob."
    status=1
    return
  fi
  measured="$(echo "$raw" | grep -v '^SCANNED:' | sort)"

  # Coverage first: if the scanner could not resolve a function, the ratchet
  # below is reporting on an unknown subset of the tree. Fail loudly rather
  # than let an "ok" stand for code that was never measured.
  unresolved="$(echo "$measured" | grep '^UNRESOLVED:' || true)"
  if [ -n "$unresolved" ]; then
    echo "$unresolved" | sed 's/^UNRESOLVED:/  UNRESOLVED: /'
    echo "  FAIL: the length scanner could not find these functions' bounds."
    echo "        Fix tools/lint.sh — do not silence this by ignoring them."
    status=1
    return
  fi
  current="$(echo "$measured" | grep -v '^UNRESOLVED:' || true)"

  if [ ! -f "$BASELINE" ]; then
    echo "  no baseline at ${BASELINE#$ROOT/}; writing one"
    echo "$current" > "$BASELINE"
    echo "  ok (baseline created, $(echo "$current" | grep -c . ) entries)"
    return
  fi
  # Compare on file:function only, so shrinking a baselined function toward the
  # limit does not churn the file.
  local baseline_keys current_keys stale
  baseline_keys="$(cut -d: -f1,2 "$BASELINE" | sort -u)"
  current_keys="$(echo "$current" | grep -v '^$' | cut -d: -f1,2 | sort -u)"

  new="$(comm -13 <(echo "$baseline_keys") <(echo "$current_keys"))"
  if [ -n "$new" ]; then
    echo "$new" | sed 's/^/  NEW: /'
    echo "  FAIL: shorten it, or update ${BASELINE#$ROOT/} with a reason in the PR"
    status=1
    return
  fi

  # An exemption has to stay earned. Once a function is decomposed below the
  # limit it drops out of `current`, and leaving its baseline row behind would
  # let it silently grow back over the limit later under the old exemption.
  stale="$(comm -23 <(echo "$baseline_keys") <(echo "$current_keys"))"
  if [ -n "$stale" ]; then
    echo "$stale" | sed 's/^/  STALE: /'
    echo "  FAIL: these are no longer over $MAX_FUNC_LINES lines — drop them from"
    echo "        ${BASELINE#$ROOT/} so the exemption cannot be reused."
    status=1
    return
  fi

  echo "  ok ($(echo "$current" | grep -c .) baselined, 0 new, $scanned scanned)"
}

# --- production file length -------------------------------------------------
# `AGENTS.md` states the standard: a file under `sources/` stays at 500 physical
# lines or fewer, comments and blanks included, and an oversized file is split
# along a cohesive seam as pure code motion with the public API preserved.
# `tests/` is deliberately not covered — it is organised by the suite each file
# covers, so a test file over the number is a readability question rather than a
# gate finding. `.metal` shader sources are not covered either: one kernel file
# is one compiled artifact per pass family, so cutting `prefill.metal` (1,377
# lines) at 500 would move a kernel between files without making either easier
# to read. The rule as written says "a file under `sources/`", which is wider
# than that; where the scope is narrower than the prose, the prose is the thing
# to amend, and this comment is where the difference is stated rather than
# hidden.
#
# What the gate is for: the standard was carried by habit alone, and habit lost.
# Measured 2026-10-06, three files were over it (529, 521 and 519) and had been
# for a week after the layout doc last claimed none were; this sweep's own
# AUD-142 commit added 4 lines to one of them and 7 to another, and neither the
# compiler nor any of the other thirteen gates objected. A number nobody checks is
# a suggestion.
#
# There is no opt-out and no baseline file on purpose. func-length ratchets
# because 1,449-line functions cannot all be decomposed in one pass; a 501-line
# file can be split in one, so an exemption row would only be a way to leave one
# behind. `MAX_FILE_LINES` exists so the number is stated in one place.
measure_file_lengths() {
  python3 - <<'PY'
import os
import pathlib

root = pathlib.Path(os.environ["ROOT"])
limit = int(os.environ.get("MAX_FILE_LINES", "500"))
counted = []
for path in sorted((root / "sources").rglob("*.swift")):
    # Physical lines, as the written rule counts them: every newline, plus an
    # unterminated final line. A trailing newline alone is not a line.
    lines = len(path.read_bytes().splitlines())
    counted.append((lines, path.relative_to(root).as_posix()))
print("SCANNED:%d" % len(counted))
if counted:
    largest = max(counted)
    print("LARGEST:%d %s" % largest)
# Exit status is reserved for "this did not run". Finding offenders is a
# successful measurement, so the caller never has to tell a crash apart from a
# verdict -- which is exactly the mistake this gate's first draft made, when a
# missing environment variable surfaced as "found no Swift sources".
for lines, path in sorted(counted, reverse=True):
    if lines > limit:
        print("OVER %d %s" % (lines, path))
PY
}

check_file_length() {
  echo "== file-length: no production source over $MAX_FILE_LINES physical lines =="
  local raw rc over scanned largest
  raw="$(measure_file_lengths)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the length counter exited $rc; it measured nothing."
    echo "        Expected python3 on PATH; do not read this as a pass."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the length counter found no Swift sources under sources/."
    echo "        Expected ~370; check ROOT and the glob."
    status=1
    return
  fi
  over="$(echo "$raw" | grep '^OVER ' || true)"
  largest="$(echo "$raw" | sed -n 's/^LARGEST://p' | tail -1)"
  if [ -n "$over" ]; then
    echo "$over" | while IFS= read -r row; do
      echo "  $row"
    done
    echo "  FAIL: $(( $(echo "$over" | grep -c .) )) file(s) over $MAX_FILE_LINES lines."
    echo "        Split each along a cohesive seam as pure code motion, keeping the"
    echo "        public API (docs/repository-layout.md)."
    status=1
    return
  fi
  echo "  ok (largest: $largest, $scanned files scanned)"
}

# --- unchecked Sendable -----------------------------------------------------
# `@unchecked Sendable` is a promise to the compiler that a type is safe to
# share across threads. Unlike the checked kind, nothing verifies it — so the
# reasoning has to be written down where the next reader will find it, or the
# promise is unreviewable. Existing sites are baselined; new ones must explain
# themselves.
SENDABLE_BASELINE="$SCRIPT_DIR/unchecked-sendable-baseline.txt"

check_unchecked_sendable() {
  echo "== unchecked-sendable: new conformances must document their invariant =="
  local raw current new stale scanned rc
  raw="$(ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    root = ENV.fetch("ROOT")
    scanned = 0
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      scanned += 1
      lines = File.readlines(path, chomp: true)
      rel = path.delete_prefix(root + "/")
      lines.each_with_index do |line, i|
        next unless line.include?("@unchecked Sendable")
        next if line =~ /^\s*(\/\/|\/\/\/)/      # a comment mentioning it
        # The invariant comment sits above the declaration, and the declaration
        # may wrap over several lines (swift-format breaks a long inheritance
        # clause), so walk the whole contiguous non-blank block above and collect
        # its comments. The marker is still required; only its distance from the
        # `@unchecked Sendable` token changed.
        j = i - 1
        block = []
        while j >= 0 && !lines[j].strip.empty?
          block << lines[j] if lines[j] =~ /^\s*(\/\/|\/\/\/)/
          j -= 1
        end
        text = block.join(" ").downcase
        next if text =~ /unchecked-invariant:/
        # Name the type so the row survives line-number churn.
        name = line[/(?:class|struct|enum|actor)\s+([A-Za-z_]\w*)/, 1] || "line#{i + 1}"
        puts "#{rel}:#{name}"
      end
    end
    puts "SCANNED:#{scanned}"
  ')"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the sendable scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  # Read before the baseline is consulted: an empty scan plus a missing allowlist
  # used to write an empty baseline and report `ok (baseline created, 0 entries)`,
  # which blessed whatever the tree actually contains.
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: unchecked-sendable read 0 Swift files under $ROOT/sources."
    echo "        Expected ~375; check ROOT and the glob. An empty scan has not found"
    echo "        an undocumented conformance, and must not be read as a pass."
    status=1
    return
  fi
  current="$(echo "$raw" | grep -v '^SCANNED:' | sort -u)"

  if [ ! -f "$SENDABLE_BASELINE" ]; then
    echo "$current" > "$SENDABLE_BASELINE"
    echo "  ok (baseline created, $(echo "$current" | grep -c .) entries)"
    return
  fi
  new="$(comm -13 <(sort -u "$SENDABLE_BASELINE") <(echo "$current"))"
  if [ -n "$new" ]; then
    echo "$new" | sed 's/^/  NEW: /'
    echo "  FAIL: document the invariant above it in a comment containing"
    echo "        'unchecked-invariant: <what makes this safe>'"
    status=1
    return
  fi
  stale="$(comm -23 <(sort -u "$SENDABLE_BASELINE") <(echo "$current"))"
  if [ -n "$stale" ]; then
    echo "$stale" | sed 's/^/  DOCUMENTED: /'
    echo "  These now carry an invariant — drop them from"
    echo "  ${SENDABLE_BASELINE#$ROOT/} so the exemption cannot be reused."
    status=1
    return
  fi
  echo "  ok ($scanned files scanned, $(echo "$current" | grep -c .) undocumented, 0 new)"
}

# --- converter: expert placement ---------------------------------------------
# A routed-expert checkpoint may ship experts one tensor at a time, and the
# converter stacks them into the fused axis the repacker packs. The experts
# arrive in whatever order the shards and the (lexicographic) index put them --
# KAT's arrive 0, 1, 10, 100, ... -- so filing them by arrival order puts expert
# k's weights in expert j's slot. Everything downstream still passes: the bytes
# match the checkpoint, the shapes are right, `validateRoleUniformity` passes,
# the receipt verifies, and the model answers fluently from the wrong experts.
# It has to be caught here, because no Swift test can see a converter bug.
#
# Fails closed. The probe imports `numpy` and the converter module, so a machine
# without them used to print "SKIP: ... (converter deps unavailable)" and exit 0
# — which is the exact shape of the defect this gate hunts: everything reads
# green while nothing was checked. Missing dependencies now name the install
# command, the way the javascript gate names `npm ci`.
#
# Opting out: `ALLOW_MISSING_CONVERTER_DEPS=1 tools/lint.sh` for a run that
# cannot install them (a machine with no numpy for the width this check needs).
# The skip is then printed as a skip, loudly, and is the only case in which this
# gate goes quiet without running the probe.
check_converter_expert_order() {
  echo "== converter-expert-order: experts file at their own index =="
  local py
  py="$(command -v python3.13 || command -v python3)"
  if [ -z "$py" ]; then
    if [ "${ALLOW_MISSING_CONVERTER_DEPS:-0}" = "1" ]; then
      echo "  SKIPPED by ALLOW_MISSING_CONVERTER_DEPS=1: no python3, the probe did not run"
      return
    fi
    echo "  FAIL: no python3 on PATH; the converter check cannot run"
    status=1
    return
  fi
  local out rc
  out="$("$py" - <<'PY' 2>&1
import os
import sys
sys.path.insert(0, "tools")
try:
    import numpy as np
    import prepare_agentworld as P
except ImportError as exc:
    # 3 is the documented "missing dependency" exit; any other non-zero code is
    # a genuine failure of the probe.
    if os.environ.get("ALLOW_MISSING_CONVERTER_DEPS") == "1":
        print("SKIPPED by ALLOW_MISSING_CONVERTER_DEPS=1: {} — the probe did not run".format(exc))
        sys.exit(0)
    print("FAIL: converter deps unavailable ({}); run: python3 -m pip install -r "
          "benchmark/requirements.txt".format(exc))
    sys.exit(3)

class W:
    def __init__(self): self.added = {}
    def add(self, n, v): self.added[n] = np.asarray(v)
    def flush(self): pass

NE = 8
order = [3, 0, 7, 1, 5, 2, 6, 4]          # deliberately not ascending
w = W(); f = P.FusedExperts()
for e in order:
    f.add("model.language_model.layers.0.mlp.experts.%d.gate_proj.weight" % e,
          np.full((512, 2048), float(e), dtype=np.float32), 4, w)
f.release(NE, {4: w})
key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases"
bi = w.added.get(key)
if bi is None:
    print("FAIL: the fused tensor was never emitted")
    sys.exit(1)
got = [float(bi[e].astype(np.float32).mean()) for e in range(NE)]
want = [float(e) for e in range(NE)]
# Each expert carries its index as its value, so the axis must read 0..NE-1.
if [round(x, 3) for x in got] != [round(x, 3) for x in want]:
    print("FAIL: experts landed by arrival order: %s (want %s)" % (got, want))
    sys.exit(1)
print("ok")
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- arch-triple build paths ------------------------------------------------
# SwiftPM's product directory is not a fixed name. The layout moved from
# `.build/<triple>/release` to `.build/release -> out/Products/Release`, so a
# script that pins the triple points at nothing on a machine that built with a
# newer toolchain — or, worse, at a stale binary left behind by an older one,
# where an `-x` guard passes and the wrong executable runs quietly. Both
# happened: tools/install_models.sh refused to install on a fresh clone
# (issue #8) while this checkout would have run a two-day-old TinyTitanRepack.
# `.build/release` is the stable spelling on both layouts.
check_arch_path() {
  echo "== arch-path: no hardcoded SwiftPM triple in a build path =="
  local out rc
  out="$(cd "$ROOT" && python3 - <<'PY'
import pathlib
import re
import sys

PATTERN = re.compile(r"arm64-apple-macosx")   # lint:allow-arch-path the gate names the literal it forbids
ALLOW = re.compile(r"lint:allow-arch-path\s+\S+")
SKIP = {".build", ".qwen", ".git", "releases", "__pycache__"}
bad = []
scanned = 0
for ext in ("*.sh", "*.py", "*.swift"):
    for path in pathlib.Path(".").rglob(ext):
        if SKIP & set(path.parts):
            continue
        scanned += 1
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not PATTERN.search(line) or ALLOW.search(line):
                continue
            if index and ALLOW.search(lines[index - 1]):
                continue          # the reason sits on the line above
            bad.append(f"{path}:{index + 1}: {line.strip()[:90]}")
# An empty walk and a clean one both leave `bad` empty, so the count is the only
# thing that separates them; "ok (none)" over 0 files was a pass over a checkout
# the gate never opened.
if scanned == 0:
    print("FAIL: arch-path read 0 script files under the checkout root.")
    print("      Expected ~810; check ROOT, and that the walk was not filtered away.")
    sys.exit(1)
if bad:
    print("FAIL: hardcoded SwiftPM triple in a build path; use .build/release")
    for entry in bad:
        print("  " + entry)
    sys.exit(1)
print("ok (%d files scanned, none name the triple)" % scanned)
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- stdout-clean -----------------------------------------------------------
# `AGENTS.md` rule 1: the library keeps stdout clean, because "stdout belongs
# to the embedding program, and a stray `print` is how a consumer's output
# stops being its own". `TinyTitanLib` itself has never held a `print`, so the
# letter of the rule was met -- and the purpose was not: the diagnostics the
# engine writes under `TINYTITAN_LAYER_TRACE` and `TURBO_FIELDFARE_PHASES` went
# to stdout from `TinyTitan`, which every library call runs through. An
# embedder that turned one on got trace lines interleaved with its own answers.
#
# The scope is therefore the closure the manifest gives the library, not the
# library directory: `library-facade-rules.py` already enforces rule 1 over
# `sources/TinyTitanLib/` alone, and nothing reached past it. The closure is read
# from `Package.swift` rather than listed here, so a new dependency enters the
# gate on the same commit it enters the library -- and a gate that walked no
# target reports a failure instead of a pass.
# Front ends keep their own stdout: measured on the tree as it stands, 91
# `print`/`debugPrint`/`standardOutput` lines in 14 files, every one in an
# executable outside the closure (the CLI and server `Command`/main entries, the
# bench driver, the installer's progress, the memory tool, the fleet manager, the
# continuity demo). The scan is Swift-only; the one C target in the closure has
# no stdout write today, measured by the same grep (`printf`/`fputs`/`stdout`
# across `sources/TinyTitanKernelsC` finds nothing).
check_library_stdout() {
  echo "== stdout-clean: no stdout write in the library's target closure =="
  local out rc
  out="$(cd "$ROOT" && python3 - <<'PY'
import pathlib
import re
import sys

manifest = pathlib.Path("Package.swift").read_text()
DECL = re.compile(r"^        \.(target|executableTarget|testTarget)\(\s*$", re.M)
spans = [(m.group(1), m.end()) for m in DECL.finditer(manifest)]
targets = {}
for index, (kind, start) in enumerate(spans):
    end = spans[index + 1][1] - 1 if index + 1 < len(spans) else len(manifest)
    body = manifest[start:end]
    name = re.search(r'name:\s*"([^"]+)"', body)
    path = re.search(r'path:\s*"([^"]+)"', body)
    if not name or not path:
        continue
    # Dependencies are bare strings ("TinyTitan"), `.target(name: "...")`, or
    # `.product(name: ..., package: ...)` for something outside this package.
    # The array is read by balancing its brackets because it spans lines, and
    # products simply resolve to nothing below -- they are not in `targets`.
    deps, array = [], re.search(r"dependencies:\s*\[", body)
    if array:
        depth, at = 1, array.end()
        while at < len(body) and depth:
            if body[at] == "[":
                depth += 1
            elif body[at] == "]":
                depth -= 1
            at += 1
        text = body[array.end():at]
        deps = re.findall(r'"([^"]+)"', text)
        deps += re.findall(r'\.target\(name:\s*"([^"]+)"', text)
    targets[name.group(1)] = (kind, path.group(1), deps)

ROOT_TARGET = "TinyTitanLib"
if ROOT_TARGET not in targets:
    print("FAIL: %s is not declared in Package.swift; the gate read nothing" % ROOT_TARGET)
    sys.exit(1)

closure, queue = {ROOT_TARGET}, [ROOT_TARGET]
while queue:
    for dep in targets.get(queue.pop(), (None, None, []))[2]:
        if dep in targets and dep not in closure:
            closure.add(dep)
            queue.append(dep)

WRITE = re.compile(r"(^|[^.\w])(print|debugPrint)\s*\(|standardOutput")
ALLOW = re.compile(r"lint:allow-stdout\s+\S+")
scanned = 0
dirs = []
bad = []
for name in sorted(closure):
    kind, path, _ = targets[name]
    directory = pathlib.Path(path)
    if not directory.is_dir():
        print("FAIL: Package.swift declares %s at %s, which does not exist" % (name, path))
        sys.exit(1)
    dirs.append("%s (%s)" % (name, path))
    for swift in sorted(directory.rglob("*.swift")):
        scanned += 1
        lines = swift.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not WRITE.search(line) or ALLOW.search(line):
                continue
            if index and ALLOW.search(lines[index - 1]):
                continue
            bad.append("%s:%d: %s" % (swift, index + 1, line.strip()[:90]))
if not scanned:
    print("FAIL: stdout-clean walked no Swift file; a scan that read nothing is not a pass")
    sys.exit(1)
if bad:
    print("FAIL: the library's closure writes stdout; diagnostics go to stderr")
    for entry in bad:
        print("  " + entry)
    sys.exit(1)
print("ok (%d targets in %s's closure, %d Swift files, no stdout write)" % (
    len(closure), ROOT_TARGET, scanned))
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- shell-portability ------------------------------------------------------
# Every user-facing script carries `#!/usr/bin/env bash`, which on a factory Mac
# resolves to `/bin/bash` **3.2.57** — not the Homebrew 5.x a developer has, and
# which `bash` finds first on a development machine. Two classes of bug reached
# main before this gate existed:
#
#   * a single-quoted heredoc containing an apostrophe inside `$( )`, which 3.2
#     refuses to *parse* — the whole script dies before its first line;
#   * bash-4 expansions and builtins (`${v^^}`, `mapfile`), which 3.2 parses and
#     then fails on at run time, halfway through a menu; and
#   * a whole-array expansion `"${a[@]}"` on an **empty** array, which 3.2 makes
#     `a[@]: unbound variable` under the `set -u` every script here sets. 5.x
#     accepts it, so it is invisible on a development machine. Write it
#     `${a[@]+"${a[@]}"}` (and likewise `[*]`), which means the same thing for a
#     non-empty array on both shells.
#
# The parse half needs the old shell, so it runs `/bin/bash` whatever that is; the
# two pattern halves are version-independent and catch the other classes anywhere.
# Opting out: `lint:allow-shell <reason>` on the line immediately above, for a
# deliberate use rather than an oversight.
check_shell_portability() {
  echo "== shell-portability: runs on the system bash, not only the developer's =="
  local old_bash="/bin/bash" version scripts=() f hit failed=0
  version="$("$old_bash" --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -1)"

  # Every shell script in the tree, not only the ones the installer runs: the
  # rule is a property of the shell, and `docs/paper/build.sh` is a script too.
  while IFS= read -r f; do scripts+=("$f"); done < <(
    find "$ROOT/tools" "$ROOT/benchmark" "$ROOT/docs" "$ROOT/examples" -name '*.sh' -not -path '*/.build/*' 2>/dev/null | sort)

  # The count has always been printed and never tested, so the day those four
  # directories are not there the line reads `ok (0 scripts...)` -- a pass over a
  # tree this gate did not open. check_shellcheck builds the same list and refuses
  # it; only this gate had no such branch.
  if [ "${#scripts[@]}" -eq 0 ]; then
    echo "  FAIL: shell-portability found 0 shell scripts under tools/, benchmark/, docs/, examples/."
    echo "        Expected ~29; check ROOT, and that find could read those directories."
    status=1
    return 1
  fi

  for f in "${scripts[@]+"${scripts[@]}"}"; do
    if ! "$old_bash" -n "$f" >/dev/null 2>&1; then
      echo "  FAIL: $f does not parse under $old_bash"
      "$old_bash" -n "$f" 2>&1 | head -2 | sed 's/^/    /'
      failed=1
    fi
  done

  # lint:allow-shell the rule's own pattern, not a use of the rule
  local pattern='\$\{[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?(\^\^|,,|\^\}|,)\}|\b(mapfile|readarray)\b|declare -A|local -A|declare -n|local -n|&>>|\|&|wait -n|;;&|\[\[ -v |shopt -s globstar'
  if [ "${#scripts[@]}" -gt 0 ]; then
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      local file line code trimmed
      file="${hit%%:*}"
      line="$(printf '%s' "$hit" | cut -d: -f2)"
      code="$(printf '%s' "$hit" | cut -d: -f3-)"
      # A comment that names the rule is not a use of it, and this repository
      # explains these constructs in prose right where it avoids them.
      trimmed="$(printf '%s' "$code" | sed 's/^[[:space:]]*//')"
      case "$trimmed" in \#*) continue ;; esac
      if [ "$line" -gt 1 ] && sed -n "$((line - 1))p" "$file" | grep -q 'lint:allow-shell'; then
        continue
      fi
      echo "  FAIL: $file:$line uses a bash 4+ feature; the system bash is 3.2"
      echo "    $trimmed"
      failed=1
    done < <(grep -rnE "$pattern" "${scripts[@]+"${scripts[@]}"}" 2>/dev/null)
  fi

  # A whole-array expansion with no `+` guard. A bare `"${a[@]}"` is correct on
  # 5.x for any array, so the only way to catch it is by shape: mask the guarded
  # `${a[@]+"${a[@]}"}` down to its test, and anything still expanding a whole
  # array was written bare.
  local array_pattern='\$\{[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}'
  if [ "${#scripts[@]}" -gt 0 ]; then
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      local file line code trimmed masked
      file="${hit%%:*}"
      line="$(printf '%s' "$hit" | cut -d: -f2)"
      code="$(printf '%s' "$hit" | cut -d: -f3-)"
      trimmed="$(printf '%s' "$code" | sed 's/^[[:space:]]*//')"
      case "$trimmed" in \#*) continue ;; esac
      if [ "$line" -gt 1 ] && sed -n "$((line - 1))p" "$file" | grep -q 'lint:allow-shell'; then
        continue
      fi
      masked="$(printf '%s' "$code" \
        | sed -E 's/[+]"\$\{[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}"/+GUARDED/g')"
      printf '%s' "$masked" | grep -qE "$array_pattern" || continue
      echo "  FAIL: $file:$line expands a whole array without the empty-array guard"
      echo "    $trimmed"
      echo "    on bash 3.2 an empty array there is 'unbound variable' under set -u;"
      echo "    the guard is the form documented above this check"
      failed=1
    done < <(grep -rnE "$array_pattern" "${scripts[@]+"${scripts[@]}"}" 2>/dev/null)
  fi

  if [ "$failed" -eq 0 ]; then
    echo "  ok (${#scripts[@]} scripts, system bash ${version:-unknown})"
  else
    echo "  fix: use a tr/while-read equivalent, or put"
    echo "       'lint:allow-shell <reason>' on the line above a deliberate use"
    # The gates share one exit status; a returned non-zero from the `all` list
    # would be swallowed and the gate would report a failure while exiting 0.
    status=1
  fi
  return $failed
}

# --- silent-test-skip --------------------------------------------------------
# A test that returns early has run nothing, and reports as **passed**. The
# audit found three (AUD-127 and the two siblings it swept): a checkpoint-config
# test whose env var was unset, a TensorOps test whose GPU family was absent,
# and a KV suite whose Metal device was missing. Each was green in the run log
# while checking nothing, which is worse than a skip because a skip is honest —
# `.enabled(if:)` records one, and `LibraryContractTests` and the MPP suite
# already gate that way.
#
# The gate therefore flags a bare `else { return }` (block form too) in `tests/`
# when the condition above it reads an environment variable, a file's existence,
# or a device capability: those are the three ways a suite goes environment-shaped.
# A gate helper that returns `nil` is the sanctioned idiom and is not flagged; only
# the unreported early exit is. Opting out: `lint:allow-silent-skip <reason>` on
# the line above, for a body that deliberately has nothing to assert on that path.
check_silent_test_skip() {
  echo "== silent-test-skip: no env/capability-shaped early return in tests =="
  local out rc
  out="$(cd "$ROOT" && python3 - <<'PY'
import pathlib
import re
import sys

TRIGGER = re.compile(
    r"ProcessInfo\.processInfo\.environment|fileExists\(|supportsFamily\("
    r"|MTLCreateSystemDefaultDevice")
INLINE = re.compile(r"\belse\s*\{\s*return\s*\}")
BLOCK_OPEN = re.compile(r"\belse\s*\{\s*$")
BARE_RETURN = re.compile(r"^\s*return\s*$")
ALLOW = re.compile(r"lint:allow-silent-skip\s+\S+")
bad = []
scanned = 0
for path in sorted(pathlib.Path("tests").rglob("*.swift")):
    scanned += 1
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        previous = lines[max(0, index - 3):index]
        if INLINE.search(line):
            exit_line, window = line, previous + [line]
        elif BLOCK_OPEN.search(line) and any(
            BARE_RETURN.match(n) for n in lines[index + 1:index + 3]
        ):
            exit_line, window = line, previous
        else:
            continue
        if not any(TRIGGER.search(text) for text in window):
            continue
        if any(ALLOW.search(text) for text in previous + [exit_line]):
            continue
        bad.append(f"{path}:{index + 1}: {exit_line.strip()[:90]}")
if scanned == 0:
    print("FAIL: silent-test-skip read 0 Swift files under tests/.")
    print("      Expected ~230; check ROOT and the glob -- a scan that opened no")
    print("      test body has not found an early return, and is not a pass.")
    sys.exit(1)
if bad:
    print(
        "FAIL: a test that returns early reports green while asserting nothing; "
        "gate it with .enabled(if:) so the skip is recorded")
    for entry in bad:
        print("  " + entry)
    sys.exit(1)
print("ok (%d test files scanned, no unreported early exit)" % scanned)
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- test-hollow -------------------------------------------------------------
# A @Test body with no assertion, no `try` and no call into an asserting helper
# cannot fail, whatever the code under test does. The audit found three of these
# (AUD-128): a body whose only statement was `_ = (aFunction, "a string")`, a
# comment that conceded "Documentation-only", and two runners built and then
# discarded so the test asserted only that construction did not throw. Each
# recorded coverage that did not exist, which is the same injury as a silent
# skip with a harder edge: it is invisible even in a green run's line-by-line log.
#
# `try` counts as an assertion because Swift Testing fails the test when the body
# throws, so the "this accepted" bodies the repo already has pass the gate on
# purpose. The rule is:
#   assertion token -> fine
#   `try`           -> fine (a throw fails the test)
#   calls a helper whose body asserts, directly or through another helper -> fine
#   none of those   -> hollow
# Comments and string literals are blanked before the scan, so a `#expect` inside
# a doc comment is not an assertion, and brace matching cannot be thrown off by a
# brace in a string.
#
# Opting out: `lint:allow-hollow-test <reason>` on a line above the `@Test`, for
# a body that deliberately asserts nothing. There are none in the tree.
measure_hollow_tests() {
  python3 - <<'PY'
import os
import pathlib
import re

ASSERT = ("#expect", "#require", "Issue.record", "withKnownIssue",
          "confirmation(", "XCTAssert")
ALLOW = re.compile(r"lint:allow-hollow-test\s+\S+")
CALLS = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")


def blank_noise(src):
    """Blank comments and string contents; keep every newline in place."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and src[i + 1:i + 2] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    out.append("\n" if src[j] == "\n" else " ")
                    j += 1
            i = j
            continue
        hashes = len(src[i:]) - len(src[i:].lstrip("#"))
        k = i + hashes
        if hashes <= 2 and src[k:k + 1] == '"':
            triple = src[k:k + 3] == '"""'
            delim = '"""' if triple else '"'
            end, closed = k + len(delim), False
            while end < n:
                if not triple and src[end] == "\\":
                    end += 2
                    continue
                if src.startswith(delim, end):
                    closed = True
                    break
                end += 1
            tail = end + len(delim) if closed else n
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:tail]))
            i = tail
            continue
        out.append(c)
        i += 1
    return "".join(out)


def span(text, brace_at):
    depth = 0
    for j in range(brace_at, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return brace_at, j
    return None


def body_after(text, at):
    found = re.search(r"\{", text[at:])
    if not found:
        return None
    return span(text, at + found.start())


def declared(text):
    """(name, body) for every func in the file."""
    rows = []
    for match in re.finditer(r"\bfunc\s+([A-Za-z_]\w*)", text):
        got = body_after(text, match.end())
        if got:
            rows.append((match.group(1), text[got[0] + 1:got[1]]))
    return rows


scanned = 0
hollow = []
root = pathlib.Path(os.environ["ROOT"])
for path in sorted((root / "tests").rglob("*.swift")):
    raw = path.read_text(errors="replace")
    text = blank_noise(raw)
    rows = declared(text)
    asserting = {name for name, body in rows if any(t in body for t in ASSERT)}
    grew = True
    while grew:                 # a helper that asserts only through another helper
        grew = False
        for name, body in rows:
            if name in asserting:
                continue
            if asserting & set(CALLS.findall(body)):
                asserting.add(name)
                grew = True
    lines = raw.splitlines()
    for match in re.finditer(r"@Test\b", text):
        found = re.search(r"\bfunc\s+([A-Za-z_]\w*)", text[match.end():])
        if not found:
            continue
        got = body_after(text, match.end() + found.end())
        if not got:
            continue
        body = text[got[0] + 1:got[1]]
        scanned += 1
        if any(t in body for t in ASSERT) or re.search(r"\btry\b", body):
            continue
        if asserting & set(CALLS.findall(body)):
            continue
        line = text.count("\n", 0, match.start()) + 1
        window = lines[max(0, line - 4):line]
        if any(ALLOW.search(entry) for entry in window):
            continue
        hollow.append("HOLLOW %s:%d %s" % (
            path.relative_to(root).as_posix(), line, found.group(1)))
print("SCANNED:%d" % scanned)
for row in hollow:
    print(row)
PY
}

check_test_hollow() {
  echo "== test-hollow: no @Test body that cannot fail =="
  local raw rc scanned hollow
  raw="$(measure_hollow_tests)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the hollow-body counter exited $rc; it measured nothing."
    echo "        Expected python3 on PATH; do not read this as a pass."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the hollow-body counter found no @Test bodies under tests/."
    echo "        Expected ~1600; check ROOT and the glob."
    status=1
    return
  fi
  hollow="$(echo "$raw" | grep '^HOLLOW ' || true)"
  if [ -n "$hollow" ]; then
    echo "$hollow" | while IFS= read -r row; do
      echo "  $row"
    done
    echo "  FAIL: $(( $(echo "$hollow" | grep -c .) )) test body(ies) cannot fail."
    echo "        Assert what the name promises, gate it with .enabled(if:),"
    echo "        or delete it -- an unassertive body records coverage that does"
    echo "        not exist."
    status=1
    return
  fi
  echo "  ok ($scanned test bodies scanned, none hollow)"
}

# --- python -----------------------------------------------------------------
# Ruff is the Python standard: the rules and the format are pinned in the
# repository's pyproject.toml. A missing or different ruff FAILS rather than
# skipping — a gate that quietly does nothing is the failure mode this check
# exists to prevent.
RUFF_PIN="0.16.7"
# The oldest Python the scripts must run on; pyproject.toml's target-version is
# the same number, and CI installs this interpreter for the gates.
PYTHON_FLOOR="3.13"

check_python() {
  echo "== python: ruff check + ruff format --check (pinned $RUFF_PIN) =="
  if ! command -v ruff >/dev/null 2>&1; then
    echo "  FAIL: ruff is not installed; this gate needs the pinned version:"
    echo "        pipx install ruff==$RUFF_PIN   (or: python3 -m pip install --user ruff==$RUFF_PIN)"
    status=1
    return 1
  fi
  local version
  version="$(ruff --version | awk '{print $2}')"
  if [ "$version" != "$RUFF_PIN" ]; then
    echo "  FAIL: ruff $version is installed, this gate pins $RUFF_PIN"
    echo "        the binary in use is $(command -v ruff)"
    echo "        pipx install --force ruff==$RUFF_PIN"
    echo "        if that path is not pipx's, an earlier PATH entry shadows it"
    echo "        (brew install ruff puts one in /opt/homebrew/bin)"
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && ruff check . 2>&1)"; then
    printf '%s\n' "$output" | tail -25
    echo "  FAIL: ruff check (fix: ruff check --fix .)"
    status=1
    return 1
  fi
  if ! output="$(cd "$ROOT" && ruff format --check . 2>&1)"; then
    printf '%s\n' "$output" | tail -10
    echo "  FAIL: ruff format --check (fix: ruff format .)"
    status=1
    return 1
  fi
  # The floor is enforced, not trusted: every script must PARSE under the pinned
  # Python, not merely under whichever interpreter is on this Mac. Ruff's
  # formatter rewrites to the target version's syntax (targeting 3.14 turned
  # `except (A, B):` into PEP 758's `except A, B:`, a syntax error on 3.13), so
  # the check is on the real files with the real floor.
  if ! output="$(cd "$ROOT" && python3 - "$PYTHON_FLOOR" <<'PY' 2>&1
import ast, pathlib, sys

floor = tuple(int(part) for part in sys.argv[1].split("."))
roots = [pathlib.Path("benchmark"), pathlib.Path("tools"), pathlib.Path("docs")]
bad = []
scanned = 0
for root in roots:
    for path in sorted(root.rglob("*.py")):
        if ".build" in path.parts:
            continue
        scanned += 1
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
                      feature_version=floor)
        except SyntaxError as error:
            bad.append(f"{path}:{error.lineno}: {error.msg}")
print("SCANNED:%d" % scanned)
if bad:
    print("\n".join(bad))
    sys.exit(1)
PY
)"; then
    printf '%s\n' "$output" | tail -10
    echo "  FAIL: a script does not parse under Python $PYTHON_FLOOR"
    status=1
    return 1
  fi
  # rglob on a directory that is not there yields nothing and raises nothing, so an
  # empty walk used to leave the floor unapplied while ruff answered
  # "All checks passed!" -- measured on an empty tree: `ok`, exit 0. ruff is not the
  # thing that walks this tree, so its clean line says nothing about this count.
  local parse_scanned
  parse_scanned="$(printf '%s\n' "$output" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$parse_scanned" ] || [ "$parse_scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the parse-floor scan read 0 .py files under benchmark/, tools/, docs/."
    echo "        Expected ~165; check ROOT -- a scan that opened no script has not"
    echo "        applied the $PYTHON_FLOOR floor, whatever ruff had to say about it."
    status=1
    return 1
  fi
  echo "  ok (ruff $version, check and format clean, $parse_scanned scripts parse under $PYTHON_FLOOR)"
  return 0
}


# --- shellcheck -------------------------------------------------------------
# The scripts are the installer, the launcher and the release tooling: a real
# bug here is a user's disk or a published artifact, not a style point. The
# pinned version matters because shellcheck's checks change between releases.
SHELLCHECK_PIN="0.11.0"

check_shellcheck() {
  echo "== shellcheck: warnings are errors over every script (pinned $SHELLCHECK_PIN) =="
  if ! command -v shellcheck >/dev/null 2>&1; then
    echo "  FAIL: shellcheck is not installed; this gate needs the pinned version:"
    echo "        brew install shellcheck  (CI downloads $SHELLCHECK_PIN from the release page)"
    status=1
    return 1
  fi
  local version
  version="$(shellcheck --version | awk '/^version:/ {print $2}')"
  if [ "$version" != "$SHELLCHECK_PIN" ]; then
    echo "  FAIL: shellcheck $version is installed, this gate pins $SHELLCHECK_PIN"
    status=1
    return 1
  fi
  local scripts=() f output
  while IFS= read -r f; do scripts+=("$f"); done < <(
    find "$ROOT/tools" "$ROOT/benchmark" "$ROOT/docs" "$ROOT/examples" -name '*.sh' -not -path '*/.build/*' 2>/dev/null | sort)
  if [ "${#scripts[@]}" -eq 0 ]; then
    echo "  FAIL: no shell scripts found to check"
    status=1
    return 1
  fi
  if ! output="$(shellcheck -S warning -f gcc "${scripts[@]+"${scripts[@]}"}" 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: shellcheck found warnings (see above)"
    status=1
    return 1
  fi
  echo "  ok (shellcheck $version, ${#scripts[@]} scripts, no warnings)"
  return 0
}

# --- swiftlint --------------------------------------------------------------
# The committed `.swiftlint.yml` is the Swift standard here: the safety rules
# (force_unwrapping, implicitly_unwrapped_optional) are on and had to reach zero
# before this gate could be wired, layout is delegated to swift-format and
# size/complexity to the ratchet above, and every remaining decision is
# justified in the config itself. `--strict` promotes every warning to a
# failure. The pinned version matters because the rule set changes between
# releases.
SWIFTLINT_PIN="0.65.1"

check_swiftlint() {
  echo "== swiftlint: violations are errors under the committed config (pinned $SWIFTLINT_PIN) =="
  if ! command -v swiftlint >/dev/null 2>&1; then
    echo "  FAIL: swiftlint is not installed; this gate needs the pinned version:"
    echo "        brew install swiftlint  (or the $SWIFTLINT_PIN release binary)"
    status=1
    return 1
  fi
  local version
  version="$(swiftlint version)"
  if [ "$version" != "$SWIFTLINT_PIN" ]; then
    echo "  FAIL: swiftlint $version is installed, this gate pins $SWIFTLINT_PIN"
    echo "        the binary in use is $(command -v swiftlint)"
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && swiftlint lint --strict --no-cache --quiet 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: swiftlint found violations (fix them; an exclusion needs a written reason)"
    status=1
    return 1
  fi
  echo "  ok (swiftlint $version, --strict clean)"
  return 0
}

# --- swift-format -----------------------------------------------------------
# The committed `.swift-format` is the formatting standard: 4-space indentation
# (the tree's actual style; swift-format defaults to 2) and
# `AlwaysUseLowerCamelCase` off, because the numerical vocabulary (`D`, `N`,
# `Dv`, `FmoE`, `qwen36_8bit`, ...) is the same deliberate naming the
# `.swiftlint.yml` identifier_name decision already records (AUD-018).
# Everything else is the formatter's default. The binary is the one bundled with
# the pinned Xcode 27 / Swift 6.4 toolchain, so the toolchain pin IS the version
# pin -- that build's `swift-format --version` reports the branch ("main"), which
# is why there is no separate SWIFT_FORMAT_PIN.
check_swift_format() {
  echo "== swift-format: formatting is enforced under the committed .swift-format =="
  if ! command -v xcrun >/dev/null 2>&1 || ! xcrun --find swift-format >/dev/null 2>&1; then
    echo "  FAIL: swift-format is not available from the toolchain (needs Xcode 27 / Swift 6.4)"
    status=1
    return 1
  fi
  # `examples/` holds consumer packages, which are SwiftPM roots of their own.
  # swift-format has no ignore file, so recursing into `examples/` would descend
  # into the checkouts SwiftPM puts under their `.build`; collect the fixture's
  # own files instead. (System bash 3.2 runs this, hence the empty-array guard.)
  local example_files=()
  while IFS= read -r path; do
    example_files+=("$path")
  done < <(find "$ROOT/examples" \( -name .build -o -name .swiftpm \) -prune -o \
      -name '*.swift' -print 2>/dev/null | sort)
  # swift-format reads a path argument that is missing or empty without complaint,
  # so on a checkout whose ROOT moved this gate certified `--strict clean` over a
  # tree it had opened no file in — measured on an empty tree: `ok`, exit 0. The
  # argument list is the gate's whole input, so counting it is what separates a
  # clean tree from no tree, which is the rule this file already holds elsewhere.
  local swift_scanned
  swift_scanned="$(find "$ROOT/sources" "$ROOT/tests" "$ROOT/benchmark" "$ROOT/examples" \
      \( -name .build -o -name .swiftpm \) -prune -o -name '*.swift' -print 2>/dev/null | grep -c . || true)"
  if [ "${swift_scanned:-0}" -eq 0 ]; then
    echo "  FAIL: swift-format found 0 Swift files under sources/, tests/, benchmark/, examples/."
    echo "        Expected ~610; check ROOT, and that the four directories named below are there."
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && xcrun swift-format lint --strict --recursive \
      sources tests benchmark Package.swift "${example_files[@]+"${example_files[@]}"}" 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: swift-format found formatting drift"
    echo "        fix: xcrun swift-format format --in-place --recursive sources tests benchmark Package.swift examples"
    status=1
    return 1
  fi
  echo "  ok ($swift_scanned Swift files in view, --strict clean)"
  return 0
}

# --- javascript -------------------------------------------------------------
# The two DSH plugin packages are npm packages with no runtime dependencies.
# ESLint and Prettier are pinned exactly in each package.json and locked in its
# package-lock.json, so `npm ci` reproduces the toolchain byte-for-byte; this
# gate FAILS when the installed versions differ from those pins instead of
# skipping, and it fails when a package has no toolchain installed at all.
ESLINT_PIN="10.11.0"
PRETTIER_PIN="3.9.9"
NODE_FLOOR="22"

check_javascript() {
  echo "== javascript: eslint + prettier --check over the plugin packages (pinned $ESLINT_PIN/$PRETTIER_PIN) =="
  if ! command -v node >/dev/null 2>&1; then
    echo "  FAIL: node is not installed; the plugin packages need Node $NODE_FLOOR or newer"
    status=1
    return 1
  fi
  local node_version node_major
  node_version="$(node --version)"
  node_major="${node_version#v}"
  node_major="${node_major%%.*}"
  if [ "$node_major" -lt "$NODE_FLOOR" ]; then
    echo "  FAIL: node $node_version is installed, the packages declare engines.node >=$NODE_FLOOR"
    status=1
    return 1
  fi
  local package name eslint prettier version output checked=0
  for package in "$ROOT"/plugins/dsh-*; do
    [ -f "$package/package.json" ] || continue
    name="$(basename "$package")"
    eslint="$package/node_modules/.bin/eslint"
    prettier="$package/node_modules/.bin/prettier"
    if [ ! -x "$eslint" ] || [ ! -x "$prettier" ]; then
      echo "  FAIL: $name has no installed toolchain; run: (cd ${package#"$ROOT"/} && npm ci)"
      status=1
      return 1
    fi
    version="$("$eslint" --version | tr -d 'v')"
    if [ "$version" != "$ESLINT_PIN" ]; then
      echo "  FAIL: $name has eslint $version, this gate pins $ESLINT_PIN"
      status=1
      return 1
    fi
    version="$("$prettier" --version)"
    if [ "$version" != "$PRETTIER_PIN" ]; then
      echo "  FAIL: $name has prettier $version, this gate pins $PRETTIER_PIN"
      status=1
      return 1
    fi
    if ! output="$(cd "$package" && "$eslint" . 2>&1)"; then
      printf '%s\n' "$output" | head -20
      echo "  FAIL: $name: eslint findings (fix: npm run lint, then npm run format)"
      status=1
      return 1
    fi
    if ! output="$(cd "$package" && "$prettier" --check . 2>&1)"; then
      printf '%s\n' "$output" | head -20
      echo "  FAIL: $name: formatting drift (fix: npm run format)"
      status=1
      return 1
    fi
    checked=$((checked + 1))
    echo "  ok: $name (node $node_version, eslint $ESLINT_PIN, prettier $PRETTIER_PIN)"
  done
  if [ "$checked" -eq 0 ]; then
    echo "  FAIL: no plugin package found under plugins/"
    status=1
    return 1
  fi
  return 0
}

# --- library facade ----------------------------------------------------------
# `AGENTS.md`'s "Two products, one repository" makes three promises about
# `sources/TinyTitanLib/`, the surface an embedder depends on: `public` there is
# deliberate ("nothing becomes `public` by accident", rule 3), the library
# "imports no NIO" and keeps stdout clean (rule 1). Measured 2026-10-06 all
# three held and none of them was enforced anywhere: `tools/lint.sh` had
# fifteen checks and not one of them looked at access level, imports or stdout
# in that target, and swiftlint's committed config has no such rule either. An
# unenforced promise is a reviewer's memory, and the day a fifth front end
# needs one more `public` type nobody will notice the surface has stopped being
# a surface.
#
# The public surface is a ratchet over a committed allowlist, compared as a
# multiset rather than a set of keys: `Session.respond` has two overloads, and
# `sort -u` would let one of them vanish while its row stayed "earned". `open`
# counts as surface too -- it promises more than `public`, because it also
# allows overriding outside the module -- and leading attributes are stripped
# before the modifier is tested, so `@discardableResult public func` cannot hide
# from a scanner that only looks at what a line starts with. The two absolute
# rules have no baseline and no opt-out, because there is nothing to ratchet --
# the correct count is zero.
#
# A missing allowlist is a failure, not a fresh baseline. func-length may write
# its own because an over-long function is already known and counted; here an
# absent file means the audited list is gone, and blessing whatever is in the
# tree is the exact accident the check exists to catch. Regenerate deliberately
# with `FACADE_UPDATE=1 tools/lint.sh library-facade`.
measure_library_facade() {
  python3 - <<'PY'
import os, re, sys

root = os.environ["ROOT"]
lib = os.path.join(root, "sources", "TinyTitanLib")
DECL = re.compile(
    r"\b(func|let|var|init|struct|class|enum|actor|protocol|typealias|"
    r"extension|subscript)\b[ \t]*([A-Za-z_][A-Za-z0-9_]*)?")
# `public` is the surface, and `open` is a wider one than public -- it also
# promises overridability outside the module -- so both are counted. Attributes
# precede the access modifier, so `@_spi`-style prefixes are stripped before the
# modifier is tested; otherwise `@discardableResult public func` would be
# invisible to the very check that is supposed to catch it.
ACCESS = re.compile(r"^(public|open)\b[ \t]*(.*)$", re.S)
ATTR = re.compile(r"^@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?[ \t]*")
COMMENT = ("//", "/*", "*", "*/")

if not os.path.isdir(lib):
    print("SCANNED:0")
    print("UNRESOLVED:%s is not a directory" % lib)
    sys.exit(0)

rows, unresolved, scanned = [], [], 0
for dirpath, _dirs, names in os.walk(lib):
    for name in sorted(names):
        if not name.endswith(".swift"):
            continue
        path = os.path.join(dirpath, name)
        scanned += 1
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                text = line.strip()
                if text.startswith(COMMENT):
                    continue
                while True:
                    prefix = ATTR.match(text)
                    if not prefix:
                        break
                    text = text[prefix.end():]
                access = ACCESS.match(text)
                if not access:
                    continue
                match = DECL.search(access.group(2))
                if not match:
                    unresolved.append("%s:%d:%s" % (rel, number, line.strip()[:70]))
                    continue
                kind = match.group(1)
                rows.append("%s:%s:%s" % (rel, kind, match.group(2) or kind))

print("SCANNED:%d" % scanned)
for item in unresolved:
    print("UNRESOLVED:%s" % item)
for row in sorted(rows):
    print(row)
PY
}

check_library_facade() {
  echo "== library-facade: the TinyTitanLib surface must be chosen, not inherited =="
  local raw rc scanned current unresolved
  raw="$(measure_library_facade)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the facade scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: no Swift files were scanned under sources/TinyTitanLib."
    echo "        Expected ~30; check ROOT and the target's location."
    status=1
    return
  fi
  unresolved="$(echo "$raw" | grep '^UNRESOLVED:' || true)"
  if [ -n "$unresolved" ]; then
    echo "$unresolved" | sed 's/^UNRESOLVED:/  UNRESOLVED: /'
    echo "  FAIL: the scanner could not name these public declarations."
    echo "        Fix tools/lint.sh — do not silence this by ignoring them."
    status=1
    return
  fi
  current="$(echo "$raw" | grep -v '^SCANNED:' | grep -v '^$')"

  local baseline new stale
  if [ -n "$FACADE_UPDATE" ]; then
    # The deliberate way the allowlist changes: the surface is measured here and
    # now, written out, and the diff of that file is what a reviewer reads.
    echo "$current" > "$FACADE_BASELINE"
    echo "  allowlist rewritten: $(echo "$current" | grep -c .) public declarations"
  else
    if [ ! -f "$FACADE_BASELINE" ]; then
      echo "  FAIL: no allowlist at ${FACADE_BASELINE#$ROOT/}."
      echo "        The audited public surface is missing, so nothing here is"
      echo "        checked. Restore it, or regenerate it deliberately with"
      echo "        FACADE_UPDATE=1 tools/lint.sh library-facade."
      status=1
      return
    fi
    baseline="$(sort "$FACADE_BASELINE" | grep -v '^$')"
    new="$(comm -13 <(echo "$baseline") <(echo "$current"))"
    if [ -n "$new" ]; then
      echo "$new" | sed 's/^/  NEW: /'
      echo "  FAIL: these are public in TinyTitanLib and not in the allowlist."
      echo "        Make them package, or add them to ${FACADE_BASELINE#$ROOT/}"
      echo "        in the same PR that argues for them."
      status=1
    fi
    stale="$(comm -23 <(echo "$baseline") <(echo "$current"))"
    if [ -n "$stale" ]; then
      echo "$stale" | sed 's/^/  STALE: /'
      echo "  FAIL: these are no longer in the surface — drop them from"
      echo "        ${FACADE_BASELINE#$ROOT/} so the exemption cannot be reused."
      status=1
    fi
    [ -n "$new" ] || [ -n "$stale" ] || \
      echo "  ok ($(echo "$current" | grep -c .) public declarations, all allowlisted)"
  fi

  # Rule 1, twice over: no server concept may enter the library, and stdout
  # belongs to the program that embedded it. The scanner skips comments,
  # because `ServerLog.swift`'s own doc comment states this rule back at
  # itself -- a gate that fails on prose is a gate people learn to distrust.
  local violations rc
  violations="$(python3 "$SCRIPT_DIR/library-facade-rules.py")"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the rule-1 scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  if [ -n "$violations" ]; then
    echo "$violations" | sed 's/^/  /'
    echo "  FAIL: rule 1 -- TinyTitanLib may import no NIO and write nothing to"
    echo "        stdout, because stdout belongs to the embedding program."
    echo "        Diagnostics go through ServerLog.diagnostic(), on stderr."
    status=1
  else
    echo "  ok (no NIO import, no stdout write, in $scanned files)"
  fi
}


# --- 7.0 isolation gates ------------------------------------------------------
# The 7.0 design contract (docs/plan-tinytitan-7-engine.md) makes isolation the
# product: the generic core carries no model knowledge (§1.3), and a variant
# target imports only the generic library (§3). Both gates fail closed — an
# empty scan is a failure, never a pass (the AUD-271 family rule).

check_core_purity() {
  echo "== core-purity: TinyTitanCore carries no model knowledge =="
  local dir="$ROOT/sources/TinyTitanCore"
  local scanned
  scanned=$(find "$dir" -name '*.swift' -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$scanned" -eq 0 ]; then
    echo "  FAIL: no Swift files under sources/TinyTitanCore; the scan read nothing"
    status=1
    return
  fi
  local hits found=0
  hits=$(grep -rnEi 'qwen36|qwen38|qwen35|qwen4flash|ornith|agentworld|katcoder|laguna|nemotron|k2horizon|deepseek|ModelFamily|ModelProfile|ModelCatalog' "$dir" --include='*.swift' 2>/dev/null)
  local hit file rest line above
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    file="${hit%%:*}"; rest="${hit#*:}"; line="${rest%%:*}"
    above=$(sed -n "$((line-1))p" "$file")
    case "$above" in *"lint:allow-core-name "*) continue ;; esac
    echo "  FAIL: $hit"
    found=$((found+1))
  done <<< "$hits"
  if [ "$found" -gt 0 ]; then
    echo "        TinyTitanCore is generic-only (spec §1.3); move the model knowledge into the variant or the runtime."
    status=1
    return
  fi
  echo "  ok ($scanned files scanned, no model knowledge)"
}

check_variant_isolation() {
  echo "== variant-isolation: a variant imports only the generic library =="
  local dir="$ROOT/sources/Engines"
  if [ ! -d "$dir" ]; then
    # No variant targets exist yet (pre-7.0): there is nothing to isolate, and
    # saying so is not a silent pass. The fail-closed case is a directory that
    # exists but holds no Swift — a scan that should have read files and read none.
    echo "  ok (no variant targets yet; nothing to isolate)"
    return
  fi
  local scanned
  scanned=$(find "$dir" -name '*.swift' -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$scanned" -eq 0 ]; then
    echo "  FAIL: sources/Engines exists but holds no Swift files; the scan read nothing"
    status=1
    return
  fi
  local found=0 f variant imports bad
  while IFS= read -r f; do
    variant=$(basename "$(dirname "$f")")
    imports=$(grep -E '^import ' "$f" | sed 's/^import //;s/ .*//')
    local imp
    while IFS= read -r imp; do
      [ -z "$imp" ] && continue
      bad=0
      case "$imp" in
        TinyTitanCore|TinyTitanFormat|Metal|MetalKit|Foundation|Accelerate|Compression|SystemPackage|Darwin|Synchronization|IOKit|TinyTitanKernelsC) ;;
        TinyTitan|TinyTitanLib|TinyTitanServerCore|TinyTitanRepackCore|TinyTitanCLICore) bad=1 ;;
        *)
          # Another variant target: a sibling directory under sources/Engines.
          if [ -d "$dir/$imp" ] && [ "$imp" != "$variant" ]; then bad=1; fi
          ;;
      esac
      if [ "$bad" -eq 1 ]; then
        echo "  FAIL: $f imports $imp (a variant may import only the generic library, spec §3)"
        found=$((found+1))
      fi
    done <<< "$imports"
  done < <(find "$dir" -name '*.swift' -type f | sort)
  if [ "$found" -gt 0 ]; then
    status=1
    return
  fi
  echo "  ok ($scanned files scanned, no cross-variant or runtime import)"
}

# --- documented facts --------------------------------------------------------
# A fact the repository can compute and a document restates is a defect waiting
# for the next commit that changes the computation. Measured 2026-10-06: the gate
# count was written down in four documents and derived in none, so all four
# disagreed with the script -- `AGENTS.md` thirteen, `CONTRIBUTING.md` eleven, the
# handover and `RELEASE.md` both fifteen -- four mode names in the usage header
# were not runnable at all, and the release brief cited three tag-object shas
# where every consumer of a sha needs the tagged commit.
#
# `tools/docs-facts.py` derives the gate set from this script's own `all` chain,
# `case` arms, usage header and unknown-check message -- they must agree with each
# other before any document is judged -- then compares every tracked Markdown
# and workflow YAML against it. This audit's own ledger prose is excluded on
# purpose: it quotes wrong numbers and shas verbatim as the thing it later refutes,
# and it is corrected forward, so failing on a quotation would demand an edit to the
# record instead of an addition to it. The boundary is prose versus reference — the
# ledger's structured `commit` field is not a quotation, it is the pointer a reader
# follows to check a fix, and `ledger_commit_evidence` does judge it (AUD-209).
check_docs() {
  echo "== docs: a documented count, name, sha or table must match the repository =="
  local out rc
  out="$(python3 "$SCRIPT_DIR/docs-facts.py")"
  rc=$?
  if [ -n "$out" ]; then
    echo "$out" | sed 's/^/  /'
  fi
  if [ "$rc" -ne 0 ]; then
    status=1
  fi
}

# Modes come in two shapes: one per check under its short name, plus the longer
# names the usage header above and `AGENTS.md` teach (`unchecked-sendable`,
# `silent-test-skip`, `shell-portability`, `shell-lint`, `format`, `js`). Four of
# them were documented and *not runnable* — `tools/lint.sh shell-portability`, the
# exact name in `AGENTS.md`, exited 2 — which is what `tools/lint.sh docs` now
# refuses. Every spelling must appear in the unknown-check message below.
case "$want" in
  all)         check_force_cast; check_unbounded_metadata_read; check_func_length; check_file_length; check_unchecked_sendable; check_converter_expert_order; check_arch_path; check_library_stdout; check_silent_test_skip; check_test_hollow; check_library_facade; check_docs; check_shell_portability; check_shellcheck; check_swiftlint; check_swift_format; check_javascript; check_python; check_core_purity; check_variant_isolation ;;
  force-cast)  check_force_cast ;;
  unbounded-read) check_unbounded_metadata_read ;;
  func-length) check_func_length ;;
  file-length)   check_file_length ;;
  sendable)    check_unchecked_sendable ;;
  unchecked-sendable) check_unchecked_sendable ;;
  converter)   check_converter_expert_order ;;
  arch-path)   check_arch_path ;;
  stdout-clean) check_library_stdout ;;
  test-skip)   check_silent_test_skip ;;
  silent-test-skip) check_silent_test_skip ;;
  test-hollow) check_test_hollow ;;
  library-facade) check_library_facade ;;
  docs)        check_docs ;;
  shell)       check_shell_portability ;;
  shell-portability) check_shell_portability ;;
  shellcheck)  check_shellcheck ;;
  shell-lint)  check_shellcheck ;;
  swiftlint)   check_swiftlint ;;
  swift-format) check_swift_format ;;
  format)      check_swift_format ;;
  javascript)  check_javascript ;;
  js)          check_javascript ;;
  python)      check_python ;;
  core-purity) check_core_purity ;;
  variant-isolation) check_variant_isolation ;;
  *) echo "unknown check: $want (all|force-cast|unbounded-read|func-length|file-length|sendable|unchecked-sendable|converter|arch-path|stdout-clean|test-skip|silent-test-skip|test-hollow|library-facade|docs|shell|shell-portability|shellcheck|shell-lint|swiftlint|swift-format|format|javascript|js|python|core-purity|variant-isolation)" >&2; exit 2 ;;
esac

exit $status
