# Cutting an TinyTitan release

This is the runbook for turning a green `main` into a tagged, published release
with prebuilt binaries. It exists because the sequence has traps that cost time
and one that can publish the wrong thing. `CONTRIBUTING.md` has the short
version; this file is the one to follow, including what to do when it stalls.

Every release is a **tag and a published GitHub Release with binaries**. The
mechanism is `tools/release.sh`, the shape is the previous release, and the
release notes are `docs/release-notes-vX.Y.md`.

## 0. What you need before starting

| Check | Command | Why |
| --- | --- | --- |
| macOS 26+, Xcode 27 / Swift 6.4 | `sw_vers`, `swift --version` | The runtime's floor and the only supported toolchain; the release notes state both |
| Disk | `df -h .` | A clean scratch build plus the staged archive wants ~10 GB |
| Memory | `memory_pressure -Q` | The golden baselines load real models |
| **No model process** | `pgrep -fl 'TinyTitanServer\|TinyTitanCLI\|TinyTitanPackageTests\|swiftpm-testing-helper\|mlx_lm\|mlx-lm'` | The golden gate refuses to run beside one; see §5 |
| `gh` authenticated | `gh auth status` | Publishing uses it; it must be the repo owner's account |
| Compile/test CI green on the commit to be tagged (CodeQL advisory) | `tools/ci-green.sh Pummelchen/TinyTitan "$(git rev-parse vX.Y^{commit})"` | `release.sh` refuses without it; the gates in §4 are this machine's view, and only CI sees a clean clone on the pinned toolchains — see §3 |
| A release build exists | `ls .build/release/TinyTitanCLI` | The golden gate drives that binary and runs *before* the clean scratch build; `release.sh` refuses to start without it |
| The installs to verify | `ls models/*/verified-install.json` | The gate verifies only what is installed, and never fetches a model; see §5 |
| Clean tree, HEAD on the tag | `git status --porcelain` | `release.sh` enforces both |

Never terminate a process you did not start. If one of these is alive and not
yours, stop and ask the human — §5 is the long version.

## 1. Prepare the version

Three places, and only the first is a literal:

1. **`sources/TinyTitanServer/Core/ServerVersion.swift`** — `ServerVersion.current`.
   That is the only version literal in the tree, and the server prints it in its
   ready banner; `tools/release.sh` refuses to publish a tag that disagrees with
   it. Grep for the previous version before believing this: `grep -rn "5\.1\b"
   --include="*.sh" --include="*.swift" sources/ tools/`.
2. **The wiki `Changelog.md`** (`.qwen/wiki/Changelog.md`) — a new `## X.Y — <headline>`
   section at the top, with `[Release vX.Y](https://github.com/Pummelchen/TinyTitan/releases/tag/vX.Y)`
   and user-facing bullets. Keep it compact: what a *user* can do now that they
   could not before, and the numbers that back it.
3. **`README.md`** — **no release callout.** The README is the stable front
   page: what TinyTitan is, the benchmark table, the supported-model list and the
   links. A version's announcement belongs in the wiki `Changelog.md` above, so a
   reader finds it once instead of the README accumulating a section per release.
   The README changes only when a fact it states changes — a new benchmark row, a
   model joining or leaving the supported list — and the only version string in
   the tree is `ServerVersion.current` above.

## 2. Write the release notes

`docs/release-notes-vX.Y.md`, modelled on the previous one:

- a `## TinyTitan X.Y — <headline>` title, then one paragraph saying what the
  release is for;
- one `###` section per user-visible change, each naming the check that backs
  it (a gate, a measurement, a real-model run);
- `### Also in this release` for the smaller items;
- `### Performance` — only numbers measured on *this* commit, and say plainly
  when a previous table was not re-run;
- a final section:

  ```
  ### Checksum

  `tinytitan-X.Y-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
  `tinytitan-X.Y-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
  `tinytitan-lib-X.Y-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
  `tinytitan-lib-X.Y-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
  `tinytitan-X.Y-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
  `tinytitan-X.Y-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
  ```

No placeholder is one to forget: `release.sh --publish` substitutes all six
with the archives it just built, and **refuses to publish unless the notes carry
the placeholder or quote the real value** for each. A release whose notes quote
the wrong digest is worse than one quoting none — 3.7 shipped that way for a few
minutes — and a size copied out of a dry run is wrong for the same reason, because
the publish pass rebuilds from scratch and the archive differs. 5.4's notes
quoted the dry run's 24,770,128 bytes for an archive that published at
24,770,200, which is why the size has a placeholder too.

The library archive is the second one the release carries (`libTinyTitanLib.a`,
`libTinyTitanLib.dylib`, the Swift module and its resource bundle — see
`tools/build_library.sh`). It is published on **every** release, so its digest is
enforced exactly like the engine archive's rather than left to the author to
remember.

The tools archive is the third: `git archive` of the tag, which is what
`tools/install_tinytitan.sh` downloads into `~/.tinytitan/src` and then *runs* —
the launcher, the model installer, the DeepSeek Harness setup. It therefore has to
carry a published checksum for the installer to accept the release at all: the
installer verifies both downloaded trees and stops if either cannot be verified, so
**a release that skips this asset is a release nobody can install.** That is why
`release.sh` stages it, asserts its members, publishes it with the Release, and
enforces its digest in the notes here rather than leaving it to be remembered.

The last two sections are a claim about what was verified. Do not write a gate
result you have not seen; add it after the dry run if you want it in the notes.

#### The Release page carries the compact form

`release.sh` runs `tools/compact-release-notes.py` over the rendered notes and
publishes the result, so the Release page is bullets — one sentence each, wrapped
narrow — while `docs/release-notes-vX.Y.md` stays the full record of why each
change exists and how it was checked.

The compactor only re-lays-out; it never rewords, so **it cannot make a long
draft short**. That is what the budget is for: the compact form must come in
under `TINYTITAN_RELEASE_NOTES_MAX_CHARS` (default 12000) or the run stops. Write
the notes compactly to begin with — bullets over paragraphs, the claim first and
the elaboration only where it changes the claim.

It also fails when compaction would drop a string `--publish` needs — every
unchecked baseline, the digest, the byte count — which is the whole point of
running it in the build rather than trusting the transformation: an omission is
caught before the Release exists instead of after it is public.

Both checks run whenever `--notes` is given, including the dry run, so a budget
or coverage failure costs a notes edit rather than a full gate run.

Note that the tag still holds whatever was committed at cut time. Editing the
notes afterwards — to shorten them, say — moves `main` and the Release page, and
cannot move the tag without rewriting it, which is not done.

## 3. Commit, tag, push

The annotated tag's message is the starting point for the notes, so make it the
headline plus the lead paragraph.

```bash
git add -A && git commit -m "Prepare X.Y: release notes, the changelog entry, and the version literal"
git push origin main
git tag -a vX.Y -m "TinyTitan X.Y — <headline>

<lead paragraph>"
git push origin vX.Y
```

Push the wiki's Changelog too (`git -C .qwen/wiki commit && git -C .qwen/wiki push`).
`release.sh` requires HEAD to *be* the tag and the tag to be *pushed*; a tag
that exists only locally fails the precondition with a clear message.

### The compile/test CI has to be green on the commit the tag names

`release.sh` refuses to continue until the **compile/test CI** on the tagged
commit is `completed` with a conclusion of `success` or `skipped`. **CodeQL is
advisory:** its run may still be in flight, or red, without blocking a patch
release; `ci-green.sh` prints its state so a finding is handled as a follow-up.
The rule lives in `tools/ci-green.sh <repo> <full-commit-sha>`, which is runnable
on its own and is tested by `python3 -m unittest test_release_ci_green`. This is not a formality: the
gates in §4 are one machine's view of the tree, and only CI runs against a clean
clone on the runner's pinned toolchains. v5.18 was published an hour after its own
tag commit came back `CI | completed | failure` (run 37270213543, the Installer
gates job) because nothing in this runbook asked.

Three answers are refusals, and each says which it is:

- **a run that has not finished** — wait for it. An in-flight run is not what the
  override below is for.
- **no run at all for that commit** — CI never saw it. A query that matches nothing
  must not read as a pass, which is also why the helper insists on the *full* sha:
  `head_sha` with a short one returns no runs, and the release would be published on
  an answer nobody collected.
- **a conclusion that is not success or skipped** — including `cancelled`,
  `timed_out` and `action_required`, each of which means CI did not check this
  commit and report it fine.

The test job can take hours; that is expected and the release waits for it. The
security scan does not: it is reported, not required. Going over a red compile/test
CI has to be deliberate, so it takes both the flag and a reason, and
`--publish` then refuses unless the failing run's URL is quoted in the release notes —
the same bargain as a skipped golden baseline (§5): it is a documented skip, not an
invisible one.

```bash
TINYTITAN_RELEASE_ALLOW_RED_CI=1 \
TINYTITAN_RELEASE_ALLOW_RED_CI_REASON="the only failing job is the runner's disk, reran green at ..." \
  tools/release.sh vX.Y --publish --notes docs/release-notes-vX.Y.md
```

## 4. Dry run, then publish

```bash
tools/release.sh vX.Y                     # verify, build, stage — no publish
tools/release.sh vX.Y --publish --notes docs/release-notes-vX.Y.md
```

The dry run is the default because publishing notifies watchers. Read its
output; then look at `.build/releases/tinytitan-release-X.Y/` — the staged tree and
the tarball — before re-running with `--publish`.

What the dry run does, in order:

1. **Preconditions** — clean tree, `vX.Y` exists locally, HEAD is that commit,
   the tag is on `origin`, no Release for it exists yet, and **CI is green on the
   commit the tag names** (`tools/ci-green.sh`, see §3b).
2. **Gates** — `tools/lint.sh`; `swift test --no-parallel`, which must print
   `Test run with N tests in M suites passed`; then **every installed model that
   has a golden target** (`benchmark/golden/`), each through
   `tools/golden-baseline.sh --check <target>`. The gates need their pinned
   toolchains on the machine (the plugin packages' `npm ci`, and
   `python3 -m pip install -r benchmark/requirements.txt` for the converter
   probe): a gate whose dependency is absent **fails**, because a check that
   could not run is not a pass. Install it, do not opt out of it —
   `ALLOW_MISSING_CONVERTER_DEPS=1` exists for a machine that genuinely cannot,
   and the release notes must then say the converter gate was skipped.
3. **A clean scratch build** — `swift build -c release --scratch-path
   .build/releases/.../build`, with the log scanned for compiler warnings. It is
   deliberately not an incremental build: an incremental one compiles nothing
   and the warning gate passes vacuously.
4. **Stage and package** — the four executables, the `.bundle` resources (the
   Metal shader library — the runtime cannot load kernels without them), the
   licence and notices, `README-binaries.txt`, then the tarball. Each of the four
   products is then asserted to be *in* the archive by name (the script runs
   without `errexit`, so a `cp` that fails partway is otherwise silent and leaves
   a three-binary release), and the unpacked archive is run through
   `tools/assert-arch.sh`, every Mach-O required to be exactly `arm64`
   (RELEASE.md §1.2.2) — both **before** its `.sha256` is taken, so a digest is
   never published over bytes of the wrong platform or over an incomplete set.
5. **The two accompanying archives** — the library (`tools/build_library.sh`,
   asserted member by member, and its staging directory asserted by arch) and the
   tools snapshot (`git archive` of the tag, asserted for the scripts an installed
   copy runs; it is the one archive with no binary in it, so no arch check). Each
   gets its own `.sha256`,
   because the installer downloads and verifies all three: a release without the
   tools archive cannot be installed at all, and a release whose notes do not
   quote one of the digests is refused by `--publish`.

`release.sh` requires HEAD to *be* the tag, so if a commit landed on `main`
after tagging (a documentation fix is the usual reason), run the release from
the tag itself — the binaries are built from the tagged commit either way:

```bash
git checkout vX.Y          # detached HEAD on the tag
tools/release.sh vX.Y      # then --publish
git checkout main
```

`--publish` repeats all of that and then creates the Release with `--repo`
pinned. Every `gh` call is pinned because in a fork `gh` defaults to the
*parent* repository: `gh release list` would show another project's releases and
`gh release create` fails with a misleading "tag has not been pushed".

## 4b. Internal-speed benchmark (optional, owner decision 2026-10-01)

**Releases no longer gate on timing.** The owner removed the mandatory
internal-speed step on 2026-10-01: a release is verified by the lint gates, the
serial suite, every installed golden and the clean scratch build, and the timing
record is a diagnostic anyone may take — not a blocker, and not a claim the notes
have to carry. The reason is the machine rather than the metric: this
workstation's numbers move by far more than the 10% threshold whenever Chrome,
WindowServer or a game holds the GPU, which is why the 5.13 attempts had to be
discarded twice before a quiet window could be found.

The tooling stays, and a record is worth taking when the machine is quiet:

```bash
# on the release machine, with the release build and the 4B install present
tools/internal-speeds.py --record --label vX.Y \
  --baseline benchmark/internal-speeds/<previous>.json
```

It measures GPU kernel bandwidth (QKV GEMV, routed MoE, GDN in-projection), CPU
int8 affine GEMV bandwidth, prefill and decode tok/s, time to first token, the
effective decode bandwidth, and a quality proxy on the fixed prompt
*"difference swift vs c++ in detail"*. The full field list is in
`benchmark/internal-speeds/README.md`.

**Records are per (model, prompt).** The mandatory record is the 4B; an ANE
prefill number exists for any install whose family the sidecar exporter can
describe — the qwen36 MoE family and the dense Qwen 3.5 family, since the
exporter now reads its geometry from the model's manifest. Record such an
install as an additional record when a sidecar is present:

```bash
tools/internal-speeds.py --record --label vX.Y-<model> \
  --model models/<install with a sidecar>
```

The one family the ANE does not serve at all is the one-layer MTP draft: the
runtime verifies it rather than prefilling it on the Neural Engine, so the record
says so rather than quoting a number. Sparse-indexed attention is *not* an
exception any more — Qwen 3.8's QSA indexer selects keys rather than changing the
arithmetic, and the runtime folds that selection into the additive mask the
sidecar already takes — but it is also **not installed**: measured, the ANE runs
0.72× the GPU's prefill on that model, so its row reports the measurement instead
of a speedup. The ANE also requires a 4,096-token prefill chunk and a prompt that
fills one; a model left on a smaller chunk cannot reach it at all.

With no `--baseline`, the comparison picks the newest previous record for the
**same model and prompt**, so extra records never become the 4B's baseline. A
model with no install is reported **not checked** — never fetched to fill a row.
An ANE row with no sidecar is recorded not applicable, with the reason.

**When you take one:** the command exits non-zero when any bandwidth or
tokens-per-second metric regressed by more than 10%, or when TTFT/decode/total
seconds rose by more than 10% (`--threshold` to change it). Read a non-zero exit
as a signal to re-measure on a quiet machine — a contended one moves these
numbers further than the threshold — and only if it reproduces does it belong in
`### Verification` with the metric, both values and the reason. A *changed*
greedy response hash is reported as a note, not a failure — a deliberate
numerics change moves it, and `### Verification` should say so.

Commit the new `benchmark/internal-speeds/<label>.json` with the release. Do
not overwrite an older record: the diff against it is the point, and the record
is only valid for one (machine, build, model) triple. If the 4B install is
absent, the step is reported as **not checked** like a missing golden — never
"fixed" by installing a model for it, and never skipped silently.

## 5. When a golden gate refuses (the trap that looks like a failure)

If the log shows this, **no golden was compared**:

```
refusing to start: these processes match the model-process guard
  91687 .../swiftpm-testing-helper --test-bundle-path .../WebTransport...
stop them yourself, or re-run when they are gone. This script never terminates a process it did not start.
error: golden baseline mismatch (qwen38-4)
```

`release.sh` reports the refusal as `golden baseline mismatch (<target>)`,
because its helper exits non-zero for both. Check the line above it: a real
mismatch prints an output diff instead. Do not re-capture a baseline to make
this go away — the baseline is valid for one (machine, build, model) triple.

A third shape exits non-zero without a diff, and it is a finding rather than a
race: `REFUSED: … has an empty body` means the stored baseline holds no captured
output, so no comparison could have meant anything, and `REFUSED: exit 0 with no
output` or `FAILED to write …` means a capture landed nothing. Those lines name
the file; re-capture it deliberately, or fix why the run printed nothing.

The common blocker on a shared machine is another project's
`swiftpm-testing-helper` (a Dropbox-resident checkout, in this workspace). It
can be a *loop* that respawns every couple of minutes; the guard is re-checked
by **each** golden invocation, so a loop with a 40% duty cycle means the phase
cannot complete. What to do:

1. Tell the human which process is blocking, with its parent and how long it
   has been alive (`ps -o pid,ppid,etime,%cpu -p <pid>`).
2. Ask them to stop it, or wait for it to finish. Never terminate it yourself,
   and never `pkill` by pattern.
3. When the machine has been quiet for ~90 s, re-run the dry run.

Waiting for a *single* short gap is not enough: the gate must start cleanly once
for every golden in the list. Verify a real quiet window before spending another
attempt:

```bash
for i in $(seq 1 6); do
  pgrep -f 'swiftpm-testing-help[e]r' >/dev/null
  status=$?
  case "$status" in
    0) echo busy ;;
    1) echo quiet ;;
    *) echo "cannot ask — pgrep exited $status, which is not quiet" ;;
  esac
  sleep 30
done
```

Only `1` means nothing matched. `pgrep` exits 2 when it cannot answer (a pattern
it cannot compile, an unreadable process table), and `&& echo busy || echo quiet`
prints *quiet* for that too — the same conflation `tools/model-guard.sh` removes
from the scripts (AUD-268). Treat a `cannot ask` line as a window you did not
measure, and wait on one you did.

### Only the installs already in `models/` are verified

`models/` is deliberately kept below the full supported set — it is hundreds of
gigabytes and the operator prunes it to save disk. The golden phase therefore
verifies **every golden target that has an install there, and nothing else**, and
says out loud which ones it could not check:

```
  -- NOT CHECKED golden baseline ornith-4 (ornith-1.5_35B_A3B_4Bit): no install under models/
  4 golden baseline(s) identical
  NOT CHECKED — no install in models/, and none may be fetched to fix that: ornith-8 ornith-4 qwen36-4 qwen36-8 agentworld-4 agentworld-8
  the release notes must name every baseline that was not checked
```

**A missing install is never resolved by downloading, converting, repacking or
re-installing a model.** No release step fetches a model: `release.sh`
fingerprints the install set under `models/` — every top-level entry by name,
type, size and mtime, plus every receipt's bytes — before the golden phase and
fails if any of it changed, so a gate cannot quietly install one to go green.
(That is deliberately not a payload hash: hashing 461 GB is not a gate, and the
receipt the runtime verifies is what attests the payload.) A release that would
need a model the machine does not have waits for the
operator to install it deliberately — that is the
[adding-a-model](adding-a-model.md) runbook, and it is a decision, not a
side-effect of cutting a release.

`--publish` refuses unless the notes name **every** target that was not checked,
absent ones included. That is the whole point: a reader of the Release learns
what was and was not re-verified, and "not checked, no install" is a different
sentence from "checked and byte-identical".

An installed model that no `check_golden` line covers is a hard error rather than
a silent omission, so a model cannot join the fleet unchecked. An intentional
exception is declared, with its reason, in `NON_GOLDEN_INSTALLS` in
`tools/release.sh`: today that is only the MTP draft head, a sidecar a covered
target's baseline already exercises. (The dense Qwen 3.5 2B/4B/9B used to be
listed here because they had no stored baseline at all, so a release verified
them through neither path. They have targets of their own now —
`qwen35-{2b,4b,9b}-{4,8}` — captured while the installs were present.) The guard
exists to catch the *undeclared* case, so add to that list deliberately, not to
silence it.

**A baseline can only be captured while its model is installed**, and `models/`
is pruned for disk. So the set the gate actually checks moves with what is on the
machine: a target with no install is reported as not checked (above), and its
stored file stays in the repository for whenever the install returns. Capture
while an install is present — that is the only window — and never delete a stored
baseline because its model is currently absent.

### A baseline the host cannot check at all (a documented skip)

Some installs live in a synced folder and the provider can leave them
**online-only**: the file lists its full size with zero blocks allocated, and a
read either blocks while it is fetched or fails outright. 5.3 hit this on a
machine where Dropbox had made seven installs online-only; every expert read
failed with

```
error: parallel expert read failed: Operation timed out
```

which surfaces through the gate as `mismatch (4)` and has nothing to do with
the runtime — `cat` on the file reproduces it, and `fileproviderctl evaluate
<path>` shows `isDownloaded = 0`. Materializing is the fix, but the largest
install needs 134 GB and the disk must hold *every* checked install at once, so
it can be impossible. Diagnose before concluding anything:

```bash
find models -type f -size +1M -exec stat -f "%b %z %N" {} \; |
  awk '$1*512 < $2*0.9 {print $3, $2}'        # online-only files, by size
```

Never delete a target from `check_golden`'s list to get past this: the list is
what stops a release from silently skipping an installed model. Name the target
instead, with its reason, and `release.sh` prints it in the golden phase and
**refuses to publish unless the notes repeat it**:

```bash
TINYTITAN_RELEASE_SKIP_GOLDENS=qwen38-8 \
TINYTITAN_RELEASE_SKIP_GOLDENS_REASON="install is Dropbox online-only; 134 GB needed, 123 GB free" \
  tools/release.sh v5.3
```

The env var is per-invocation and the reason is mandatory. The notes get a
`### Verification` sentence naming the target, the reason, and what *was*
checked — a release that says plainly which baseline went unverified is worth
more than one that implies all of them ran.

(Bracket the pattern — `help[e]r` — or `pgrep` matches the shell running it.)

## 6. After publishing

```bash
gh release view vX.Y --repo Pummelchen/TinyTitan --json url,assets \
  --jq '"\(.url) \([.assets[].name] | join(", "))"'
```

Check: the notes on the Release quote the digest in the archive's `.sha256`
next to it; the assets are **three** tarballs with their checksums — the engine
archive, the library archive (`tinytitan-lib-X.Y-macos-arm64.tar.gz`) and the
tools archive (`tinytitan-X.Y-tools.tar.gz`, what `install_tinytitan.sh` extracts
and runs) — and the wiki Changelog points at the same tag. The binaries are
**not** signed or notarized, and the two `README` files in the archives say so and
tell the user how to clear the quarantine attribute after verifying the checksum —
keep that honest rather than implying a notarized build.

If you wrote a `### Performance` table, make sure it says which commit and
machine it was measured on, and leave previous releases' tables alone.

## 7. Checklist

- [ ] Installer version bumped; no stale version literal left (`grep` for it)
- [ ] Wiki `Changelog.md` has the new section, pushed
- [ ] No release callout added to the README — the Changelog section **is** the
      announcement, and the README changed only if a fact in it changed
- [ ] `docs/release-notes-vX.Y.md` ends with a checksum block carrying the
      placeholders for **all three** archives — `SHA256_PENDING` and
      `ARCHIVE_BYTES_PENDING`, `LIBRARY_SHA256_PENDING` and
      `LIBRARY_BYTES_PENDING`, `TOOLS_SHA256_PENDING` and `TOOLS_BYTES_PENDING` —
      never a size copied out of a dry run
- [ ] Tree clean, `git tag -a vX.Y`, tag pushed, `release.sh` preconditions pass,
      and the **compile/test CI is green on the commit the tag names** (`release.sh`
      asks `tools/ci-green.sh`; CodeQL is advisory; see §3)
- [ ] If CI was red and the release went over it anyway: both
      `TINYTITAN_RELEASE_ALLOW_RED_CI` and `..._REASON` were set, and the failing
      run's URL is quoted in the notes (`release.sh --publish` enforces the quote)
- [ ] A release build exists (`.build/release/TinyTitanCLI`) and
      `models/` holds exactly the installs you intend to verify
- [ ] Dry run green: lint, the serial suite, **every installed golden**, a
      warning-free clean build
- [ ] *(Optional since 2026-10-01)* An internal-speed record, if one was taken on
      a quiet machine (`tools/internal-speeds.py --record --label vX.Y
      --baseline …`), committed with the release. A record that was not taken is
      not a gap, and a contended machine is not a reason to hold the release
- [ ] **No model was downloaded, converted, repacked or re-installed** to make a
      check run; the golden phase left `models/` byte-identical (`release.sh`
      enforces this)
- [ ] Every baseline that was not checked is named in `### Verification` — both
      the ones absent from `models/` and any skipped via
      `TINYTITAN_RELEASE_SKIP_GOLDENS` (`release.sh --publish` enforces this)
- [ ] The notes are compact: the Release page is bullets, and the compact form is
      under the `TINYTITAN_RELEASE_NOTES_MAX_CHARS` budget (`release.sh` compacts
      and enforces this on any run that passes `--notes`)
- [ ] Staged archive inspected (four executables, bundles, licence, notices)
- [ ] Library archive inspected (`.a`, `.dylib`, `TinyTitanLib.swiftmodule`,
      the Metal bundle and `README-library.txt`; `build_library.sh` asserts them)
- [ ] `--publish --notes docs/release-notes-vX.Y.md`, then `gh release view`
- [ ] No model process left running afterwards
