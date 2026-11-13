# Contributing

TinyTitan welcomes focused fixes, documentation improvements, and
benchmark reports from Apple Silicon Macs.

## Before opening a change

- Keep the package compatible with macOS 26, Swift 6.4, and Metal 4.
- Preserve the bounded-memory model path. Never load a complete checkpoint,
  shard, or large model tensor into Swift heap memory.
- Keep public runtime controls limited to those documented in
  [Runtime controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls).
- Add or update a focused test for behavior changes.

Run the release build, the production gates, and the serial tests:

```bash
swift build -c release
tools/lint.sh
swift test --no-parallel
```

CI runs the same three, so a green run here is a green run there. `tools/lint.sh`
enforces eighteen checks the compiler cannot — project-specific probes and the
pinned third-party linters (shellcheck, swiftlint, swift-format, the plugin
packages' eslint + prettier, and ruff). That number is not free to rot:
`tools/lint.sh docs` compares every documented count, gate name and tag citation
against the script and the repository, and fails on a disagreement. The full list
is the script's own usage header; the probes that most often bite a first
change:

- **force-cast** — no `as!` / `try!` under `sources/`. To keep one, put
  `lint:allow-force <reason>` in the comment block directly above it; a marker
  without a reason fails exactly like no marker.
- **func-length** — a ratchet, not a limit: no function over 120 lines without an
  inline `lint:allow-long <reason>` above it. The ratchet file
  `tools/func-length-baseline.txt` carries the audited exemptions (13 rows, all
  from the formatter sweep that expanded their bodies; the as-is GGUF decoder's
  AUD-025 row was dropped when its pipeline split into per-layer helpers, and
  `advanceMTP`'s AUD-024 row when its embedding lookup moved to a helper). If you ever add a row,
  drop it once the function shrinks: the gate fails on a stale exemption so it
  cannot be reused later.
- **unchecked-sendable** — every `@unchecked Sendable` under `sources/` must
  carry an `unchecked-invariant: <what makes this safe>` note above it.
- **converter** — feeds routed experts to the converter in shuffled order and
  fails unless each lands at its own index. This catches a class of bug that
  produces installs which load, pass every byte check, and answer from the wrong
  weights; no Swift test can see it, which is why it lives here.

- **arch-path** — no hardcoded SwiftPM target triple (for example
  `arm64-apple-macosx`) in a build path. Such a path points at nothing on a newer
  toolchain, or at a stale binary on this one. To keep one deliberately, put
  `lint:allow-arch-path <reason>` on the line above.

`tools/lint.sh <mode>` runs a single gate — `tools/lint.sh force-cast` after a
cast, `tools/lint.sh swift-format` before pushing — and the script's usage header
lists every mode name, including the longer spellings each gate also answers to.

These checks do not download or load the model. For a change to the runtime or
the model-load path, also run the golden baseline, which is the only check that
exercises real inference:

```bash
tools/golden-baseline.sh --check 8        # Ornith 1.5 8-bit, the default target
tools/golden-baseline.sh --check 4        # Ornith 1.5 4-bit
```

It compares greedy, fixed-seed output against `benchmark/golden/`. A baseline is
valid for one (machine, build, model) triple — capture your own with
`tools/golden-baseline.sh 8` before making changes, and re-capture only for a
deliberate numerics change. **Only models already installed under `models/` can
be checked.** That directory is deliberately kept below the full supported set to
save disk: a target with no install is reported as *not checked*, and it is never
resolved by downloading, converting or re-installing the model. For a real-model
change also report the prompt, generated token count, output, timing footer, Mac
model, memory, macOS version, Swift version, and any protocol change.

## Benchmark reports

Follow the [community benchmark protocol](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarking-Guide). Review
all captured files before sharing them, and remove personal paths or unrelated
process details.

## Pull requests

Keep each pull request narrow. Explain the behavior change, tests run, and any
remaining limitation. By contributing, you agree that your work is licensed
under the repository's [Apache License 2.0](LICENSE).

## Releasing (maintainers)

Every release is a tag *and* a published GitHub Release with prebuilt binaries.
`tools/release.sh` does the whole sequence — 3.6 is the reference shape. The full
runbook, including the preconditions that stall it and what each failure message
actually means, is [`docs/release-process.md`](docs/release-process.md).

```bash
# 1. add the release's section to the wiki Changelog — that is where a version is
#    announced; the README carries no release callout
git tag -a vX.Y -m "..."      # annotation is the starting point for the notes
git push origin main vX.Y

# 2. dry run: preconditions, gates, clean build, staged archive — no publish
tools/release.sh vX.Y

# 3. inspect the staged archive, then publish
tools/release.sh vX.Y --publish --notes path/to/notes.md
```

The dry run is the default because publishing notifies watchers immediately.

What the script enforces, and why each check is there:

- **Clean tree, HEAD on the tag, tag pushed, no existing Release.** Cheap
  guards against releasing something other than what you think you tagged.
- **`tools/lint.sh`, the full serial suite, and the golden baseline.** The
  baseline runs against every golden target that has an install under `models/`;
  a target with no install is reported as *not checked* and named in the release
  notes. It is never resolved by downloading a model — the gate verifies what is
  on the machine, and never changes what is on the machine to pass.
- **A clean scratch build.** An incremental `swift build` compiles nothing when
  the tree is unchanged, so scanning its output for warnings passes vacuously.
  The shipped binaries are always built fresh from the tagged commit.
- **`--repo` on every `gh` call.** In a fork, `gh` defaults to the *parent*
  repository: `gh release list` here lists drumih/turbo-fieldfare's releases,
  and `gh release create` fails with a misleading "tag has not been pushed"
  error. 3.6 was nearly published against the wrong repo because of this.

The archive ships the four executables plus the `.bundle` resources carrying the
Metal shader library — the runtime cannot load its kernels without them beside
the executables — along with `LICENSE` and `THIRD_PARTY_NOTICES.md`. It contains
no model weights.

Binaries are **not code-signed or notarized**, and the bundled
`README-binaries.txt` says so. Signing would need a Developer ID in CI; until
then the release notes should keep pointing users at building from source or
clearing the quarantine attribute themselves after checking the published
SHA-256.
