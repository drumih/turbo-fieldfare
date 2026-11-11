# Handover: the as-is GGUF route (2026-11-11)

**Read this block first; the release handover below it is older.** The Unsloth
`Qwen3.8-Flash-Next-UD-IQ3_XXS` GGUF in `models/qwen3.8-flash-next-unsloth-iq3xxs`
runs in place, CLI only: `.build/release/TinyTitanCLI --model
models/qwen3.8-flash-next-unsloth-iq3xxs --prompt "..."` (a directory of `.gguf`
shards with no `manifest.json` routes to `runGGUFAsIs`). The engineering record
is `docs/qwen38-gguf-backend-plan.md`, its last three sections in particular.

- **Correct against llama.cpp** since `203673c`: same top-1/top-5 and top-5
  logits within 1.0 on the pinned prompt (`GGUFLogitProbeTests`, model-gated by
  `TINYTITAN_GGUF_MODEL`). Five defects were fixed; the traps that hid them are
  in the plan doc. Any change to this path re-runs that test.
- **Speed**: 1.07 -> 2.6-3.8 tok/s decode on the 24 GB M3 (`e539c42`: one
  concurrent fetch per layer, vectorised Q6_K/Q8_0 GEMV). The fetch now runs
  near the drive's read limit; speculative prefetch was measured and rejected
  (`f74d8c2`). `TINYTITAN_GGUF_TRACE=1` prints the per-phase split.
- **Parity tooling**: `TINYTITAN_GGUF_DUMP_DIR` (+ `_POSITION`) captures every
  stage; the llama.cpp side needs `llama-debug` with a local tensor-dump patch
  (the plan doc names it). Comparison scripts used on 2026-11-11 lived in the
  gitignored `.build/parity/`.
- **Open** (wiki Project Tracker): TT-056 chunked prefill and the ANE sidecar
  (the prompt still runs token by token), TT-057 decode speed, TT-058 serving
  through `TinyTitanLib` and the server, TT-059 QSA beyond 2,048 tokens, TT-060
  a measured noise floor (needs a Metal llama.cpp build).
- **Machine**: this Mac is shared with other projects' builds and an ANE
  compile; check `uptime` and the top CPU users before calling a number a
  measurement, and A/B interleaved in one build.

# Handover: after release 5.18, the pre-production audit

**Paste this into the next session:**

> Continue the TinyTitan work in this checkout. Read `AGENTS.md`, then
> `docs/handover-tinytitan.md`, then the audit ledger
> `docs/audit-ledger.jsonledger.md`, then the wiki `Project-Tracker`.
> **5.18 is cut and published** (`v5.18` → commit `ea5de8c`, 2026-10-05: the
> launcher fetches a model that is not on disk, the menu offers the ones the
> checkout lacks, and every long download, conversion and repack draws a real
> percentage). Before it, **5.17** (`v5.17` → commit `701bb2e`, 2026-10-04) fixed
> the converter guard that keyed the per-expert duplicate check on the tensor name
> instead of name + width, so `tools/install_models.sh katcoder [both]`
> installs again (issue #19), and moved the user documentation into the wiki;
> **5.16** (`v5.16` → commit `551522a`, 2026-10-03) made the engine a library —
> `TinyTitanLib` is a shipped product, the CLI and the server both generate
> through it, and the toolchain is pinned **exactly**: Xcode 27 / Swift 6.4,
> nothing else. These three citations name the **tagged commit**, not the tag
> object — `git rev-parse v5.18` is `d30de44`, which is a tag object and is what
> this brief said until 2026-10-06 — because CI, `tools/ci-green.sh` and
> `tools/release.sh` all key on the commit, and `git show <tag-object-sha>` does
> not print a diff.
> `tools/lint.sh` runs **eighteen** pinned gates (measured 2026-10-08: the modes
> in its own usage header), and every one
> of them **fails** when its dependency is missing rather than reporting a
> skip — a check that did not run is not a pass. Release notes in the tree keep
> `SHA256_PENDING`; `tools/release.sh` fills them in the staged copy at publish
> time, so a placeholder there is the house pattern, not a defect.
>
> The open work is the **pre-production audit** in
> `docs/audit-ledger.json` — `ledger.json` is the single source of truth and
> `ledger.md` is rendered from it (recompute `counts` by hand before
> `render_ledger.py`, which refuses stale ones). Row discipline: verify against
> the code before fixing anything, fix only what is true and unfixed, test the
> fix, re-verify on current `main`, then audit for the sibling defect and close
> with the measured evidence and the commit hash. **Do not read a list of open
> rows out of this brief** — name the severity you want and ask the ledger:
> `python3 -c "import json;[print(t['id'],t['severity'],t['title']) for t in
> json.load(open('docs/audit-ledger.json'))['tasks']
> if t['status']=='OPEN']"`. This block used to point at **AUD-143** as the
> hardest open row; it closed at `07e867b`, and a session that trusted the prose
> would have re-learned that for nothing. One row needs a human and not a fix:
> **AUD-139** (S1) is blocked on the repository owner, because `v5.18` published
> no `tinytitan-5.18-tools.tar.gz` (confirmed against the release assets on
> 2026-10-06), so the installer's closed check refuses the newest release until
> someone re-publishes it.
>
> Phase D is at convergence on the model-free surface. The ledger is 210 rows / 209
> closed / 0 open / 1 blocked, and the last eight closes (AUD-284, AUD-293, AUD-259,
> AUD-311, AUD-312, AUD-313, AUD-314 and AUD-310, at `154dc68`, `9ea1368`, `cc7ef48`, `e1d651d`,
> `36b4a81`, `1ff2fb3`, `ec1d546` and `92a48c6`) each landed with the test written first, an anchored
> mutant sweep, and the model-free gates clean. The three were the operator's
> calls and the operator answered them: 284 moved the test's assertion to the cause the
> run actually produced, 293 joined the tree's exit table rather than keeping its own,
> and 259 turned out not to need a wiring at all -- the concurrent GPU-plus-CPU client
> generation its doc priced cannot happen, because a catalog holds one model resident and
> a switch drains the resident one first. The fourth was not one of the three the operator
> answered: it is what reading the run said. `main` has been red on every push since the
> ledger grew past the checkout's history, because `.github/workflows/ci.yml` takes
> `actions/checkout`'s default depth of one commit and `tools/docs-facts.py` resolves every
> ledger sha against that history -- 211 real commits reported as `no such commit`, a refusal
> whose stated cause had not happened (AUD-311, S1). Fixed on both sides: the gate's checkout
> takes `fetch-depth: 0`, and a shallow clone is now refused as a shallow clone, with its
> `checked` count left at zero so a run that resolved nothing cannot read as a pass. The
> `thread-sanitizer` checkout stays at depth 1 -- measured, it runs no gate that reads history.
> The fifth close is the run behind that one: once `fetch-depth: 0` let the later CI steps
> execute again, run 38124699999 on `3404269` failed exactly one test the tree had never
> seen pass on that host -- `test_launcher_concurrency.test_enter_keeps_one` forbids any
> WARNING on the launcher's stderr, and on the 7 GB CI runner the launcher's *correct* RAM
> warning had to fire, because the fixture model plus `--ram 4` crosses that machine's 3 GB
> recommendation; AUD-311's red had masked this step since `fe19203`. The suite's own
> comment claimed `--ram 4` was "under 30% of any Mac this runs on" -- an assumption about
> hardware, not a property of the code (AUD-312, S2). The fix pins the launcher's documented
> seam, `TINYTITAN_PHYSICAL_RAM_BYTES`, at 16 GB in both helpers, so the host cannot decide
> what the tests read; no assertion moved. The sweep over the pin is 2 killed of 4, with the
> survivors recorded rather than hidden: deleting the pty helper's pin survives because no
> test asserts RAM-warning absence through that path yet, and a too-comfortable 256 GB pin
> survives because the test's subject is the *absence* of a warning, so it pins "not the CI
> host", not "exactly 16 GB" -- and deleting the non-pty pin survives on a big host with no
> hostile export, which is inherent to that subject.
> The sixth close is the run behind that one, and the argument was in the assertion's
> argument order: run 38130598551 on `3e925b5` failed the converter step with
> `AssertionError: 8 != 9`, and `assertEqual(threading.active_count(), baseline)` puts
> actual first -- the count ended BELOW baseline, which a leaked fetcher cannot produce.
> The fetcher-shutdown test measured the whole process and assumed the threads of the two
> suites that run before it in the same CI process stay put; one died inside its window.
> The fix changes only the instrument: the set of threads alive before the pool starts,
> and after close nothing outside that set may survive -- printed by name when it does.
> The mutant that removes the shutdown signal is killed with the leak named
> (`['Thread-1 (fetcher)', ...]`); removing the join survives, which is guarding liveness,
> not join-ness (AUD-313, S2).
> What remains is Phase E -- the
> fresh-clone verification on an independent host -- and the README, wiki and
> Project-Tracker sync, which is **two pushes**: `main` here and `master` on the wiki
> remote. **That changed on 2026-10-11: the operator gave the word and `main` is pushed.**
> `origin/main` tracks this tree through every close since -- the push history is the
> `main` table row below -- and the ahead count, re-measured with
> `git rev-list --count origin/main..main` now, is 0: the AUD-314 close `d9b4317`,
> the handover wiki-sync `340ccac` and the two rebase-merged dependabot bumps
> `57a7098` and `7086953` are all on the remote, and the run on `7086953`
> (CI 38144887362, CodeQL 38144887368) is green in both jobs. The wiki half has
> landed too:
> `master` on `TinyTitan.wiki.git` carries `e62c041` (the plugin guide's
> 0.2.1-alpha.2 pin and the mount-on-missing autonomy section) and `ce5a7aa` (the
> Project-Tracker paragraph), ahead re-measured **0** after the push -- publishing is
> two pushes, and both halves of the dsh move are now on their remotes.
> **The two passes the audit's task list still carried are closed as of the last open
> row closing:** the L5/L6 pass (performance + tests) and the L7/§5/§6 pass (ops,
> facade, unused code) drained into the ledger's rows across Phases C and D, and the
> operator's call on 2026-10-11 was to close them once AUD-310 closed -- which
> `92a48c6` and `8837f27` did. **The audit's Phase A tooling task is done in the same
> window:** a third `supply-chain` job now runs beside `test` and `thread-sanitizer`
> -- gitleaks 8.30.1 over the full history (verified clean before it was wired: 1643
> commits, no leaks found, 2.5 s), pip-audit 2.10.1 over `benchmark/requirements.txt`
> (no known vulnerabilities) and npm audit over both plugin lockfiles (0
> vulnerabilities each, verified to read the lockfile alone) -- and
> `swift test --enable-code-coverage` rides the existing serial-test run, and
> `tools/coverage-export.sh` merges its seven test bundles into one llvm-cov JSON
> uploaded as the `swift-code-coverage` artifact: reported, not gated, per the
> operator's call, with `if-no-files-found: error` so a coverage run that wrote no
> report cannot read as a pass. Uploading SwiftPM's own `<bin-path>/codecov/*.json`
> instead was the defect: that file names one test bundle, not the package
> (AUD2-031, corrected after this paragraph was first written). CodeQL was already
> the SAST half. The supply-chain
> checkout takes `fetch-depth: 0` because its gitleaks step reads history; the
> `thread-sanitizer` checkout stays at depth 1 -- measured, it runs no gate that reads history.
> What the list carries now is Phase E only, which the operator answered the same
> day: the close is recorded on CI as the fresh clone once the run on the final head
> is green, with the residual (the release build and the install flow, which CI does
> not run) named as a permanent finding in
> `docs/audit-ledger.jsonenvironment.md`.
>
> The closure invariant was re-measured at `92a48c6`: all 209 DONE rows are `owner: ""` with
> a commit, the blocked AUD-139 is the only row left naming an owner with no commit, and all
> 198 distinct
> shas named in a DONE row's `commit` field resolve to a commit that is an ancestor of HEAD** -- that last check is not one `tools/docs-facts.py`
> does today, since it resolves that a referenced commit exists and not that it landed, so a
> row citing a commit on a branch that was never merged would pass the gate as it stands; a
> future pass could add the ancestry test beside the resolution test. Measuring it here once
> found nothing to file, and a first attempt reported seven false violations by taking the
> comma out of a two-commit field as part of the sha, which is the reason to strip punctuation
> before calling `git merge-base`.
>
> The product is the engine plus its loopback server — the Mac app is gone — and
> `tools/install_tinytitan.sh` downloads a built release instead of compiling
> one; the browser chat window arrives the same way, from the release tag's
> source archive, with no registry account on either side. The installs under
> `models/` have receipts **valid for this folder**, because a rename invalidates
> them; re-issue in place with `--verify-install` if the folder moves again, and
> never hand-edit a receipt. **Verification uses only the installs already under
> `models/`** — never download, convert, repack or re-install a model to make a
> gate pass, and never fetch one of the installs the operator deleted; a gate
> that cannot run is reported *not checked*. The open model row is still
> **TT-048, a fresh KAT-Coder install**: the install-path bug is fixed and its
> failure point verified on the real checkpoint (2,075 per-expert tensors, both
> widths, fused `[256, …]` stacks), while the ~69 GB fetch and both repacks
> remain. Report measurements, not assurances.

This is the only current brief; the 5.12 handover it replaces is superseded. The
traps that one named still bite and are folded in below.

> **The product shape changed: the GUI is gone.** The Mac app, the out-of-process
> decode service, the app's library and test targets and `tools/make_app_icon.py`
> were all removed. TinyTitan is an **LLM engine plus its loopback server**:
> `tools/install_tinytitan.sh` downloads the published executables, optionally
> downloads a model, installs `~/.local/bin/tinytitan`, and offers to start the
> server; that command runs `tools/server_launcher.sh`, which prints the base URL a
> client is pointed at. **Do not add a GUI, a desktop front end, or any work that
> only one needs** — `AGENTS.md` states the policy and the reason, and this is not
> a pause. The app's `Info.plist` was also the tree's only version literal; that
> literal now lives in `ServerVersion.current`
> (`sources/TinyTitanServer/Core/ServerVersion.swift`), is printed in the server's
> ready banner, and is checked against the release tag by `tools/release.sh`.

> **The replacement window is DeepSeek Harness, installed not built.**
> `tools/dsh_local.sh` installs a **pinned** `@deepseek-ai/dsh` (`0.2.0-rc.2`,
> into `~/.tinytitan/dsh`, with our `plugins/dsh-tinytitan` bundle from this
> checkout, and the launcher's `--web` starts the server and opens it in the
> browser. Two things are load-bearing. **Isolation:** our copy uses its own
> `DSH_HOME`, npm prefix, pnpm store, and port (7788, stepping up when taken), so a
> DeepSeek Harness the user already runs — their `~/.dsh`, their `dsh` on PATH,
> their 3080 UI — is never read, written or stopped. **No fork:** it is upstream's
> code plus our plugin; the harness is in developer preview and says it will break
> compatibility, which is exactly why the version is pinned and why nothing here
> may start depending on a window existing. Two traps cost time and are recorded in
> the script: a fresh `DSH_HOME` has no `settings.yaml`, so we create it before
> `tools/dsh_route.sh --write` will touch it — 0.2.0 then imports that file into the
> profile patch at boot and renames it to `settings.yaml.imported`, so the plugin's
> own refresh goes through the `settings` service instead; and pnpm's npm-installed
> shim has no shebang, which macOS refuses to `exec` (`spawnSync pnpm ENOEXEC`) —
> the private shim execs `@pnpm/exe.darwin-arm64` instead. `tools/dsh_local.sh
> status` says what is installed.
>
> **Since 5.7 the pin is enforced rather than declared.** Both plugins support
> exactly `0.2.0-rc.2` — `dsh-tinytitan`'s peers are exact, not ranges — and
> **refuse to run** on any other harness, including one whose version cannot be
> read. A refusal never throws: it writes one line to stderr and returns, so DSH
> boots, every other plugin loads, and removing ours leaves nothing to undo. stderr
> is not a preference — the harness prints a plugin's log record only when the boot
> itself fails, so a host-logger line would be invisible (tracker, plugins section).
>
> **Amended 2026-10-17: the two sentences above name `0.2.0-rc.2` because that is
> what they were written against; the pin is now `0.2.1-rc.2`** — `0.2.1-alpha.2`
> between them, from 2026-10-11. Nothing about the enforcement changed: the release
> is still one, exact, refused on sight, and read from the harness package rather
> than assumed. What moved is the four sites that carry it — both
> `SUPPORTED_DSH_VERSION` constants, `dsh-tinytitan`'s three harness
> `peerDependencies`, the launcher's `DSH_VERSION` default and both plugin READMEs.
> See the #230 entry under "Where the work stands" for why it moved and what was
> re-verified on the way.

## The active thread after the audit: Qwen3.8 on GGUF (2026-11-09)

**The audit below is complete (210/210, 0 open); the live work is this.** The
checkpoint `models/qwen3.8-flash-next-unsloth-iq3xxs` (83.63 GB base + 4.14 GB
MTP + 0.90 GB mmproj) is the only copy on this machine, and the engine cannot
serve a GGUF today.

The operator chose **route B: convert it to the engine's affine `.ssdai`
format**, starting with prototypes. Landed, each with a gated test against the
real file and a commit:

| commit | component |
| --- | --- |
| `e76ad10f` | split GGUF v2/v3 directory reader (1,224 tensors, 83,615,413,760 bytes) |
| `7e6a852f`, `33336798` | block decode for all nine checkpoint types, bit-exact against ggml `7d7f028` |
| `eab6b227` | tokenizer sidecar (tokenizer.json/config/template), ten prompts id-identical to `llama-tokenize` |
| `c2e7e8e7` | per-expert addressing (73,728 slices tile 50.29 GB) |
| `da9d0ee9` | host decode measured: 0.21–0.36 G elements/s, so decode-on-fetch is rejected |
| `c45fa8ca` | affine probe: 8.7–10.1% added weight error; experts 50.29 → 67.95 GB |
| `36f958b8` | one layer written as a `.ssdai` blob and read back |
| `9476bbbf` | full layout plan: 48×512×9 records, layout.json 19.2 MB, payload 68.05 GB |
| `556bd588` | manifest: accepted by `ManifestReader.decode` against `qwen38FlashNext` |

**Where it stands:** the expert payload, the layout and the manifest are proven.
What remains is the **resident** half — `model_weights.bin`. Per-tensor it is a
role map, not a composition: the GGUF already fuses `[query ; gate]` into
`attn_q` for the 12 full-attention layers (12288 rows = 2·24·256), and on the 36
GDN layers `attn_gate` (6144) is the linear attention's z gate, not an
attention gate. A first reading here said "concatenate `attn_q` and
`attn_gate`"; the shapes disproved it. Still to do: the role table for the
hyper-connection pairs, the norms and the PLE, the width per role, then the full
~100 GB write and KLD/top-1 against a llama.cpp run of the same GGUF.

**Facts that shape it.** `CPUModelFamily` serves only `qwen3_5_dense`, so there
is no CPU forward for this architecture; experts are affine-only in the engine
(`DequantInt4/8GEMV`), while `BF16GEMV` exists for non-experts. `expertStride`
must be a multiple of 16 KiB (2,764,800 → 2,768,896). The PLE table's 160-wide
rows are `embedDim / ngramHeads`, not the preset's 2,560. The full run needs the
83.6 GB source on external storage: ~100 GB of output against 79 GiB free.
Details, measurements and open questions: `docs/qwen38-gguf-backend-plan.md`.

## Where the work stands

**#230 closed 2026-10-17 — the systemwide harness moved to `0.2.1-rc.2` and the pin
moved with it, test-first.** The operator upgraded the harness they run in their own
`~/.dsh` (not the pinned private one under `~/.tinytitan`, which the launcher owns and
nothing here touched) and the web boot answered `Failed to load plugins / 22 entries did
not activate`. The gate did its job: `dsh-tinytitan` declares exact peers, rc.2 is not
the release they name, so profile startup denies the plugin instead of letting it write
into a harness home it was not written against. Four pin sites and two READMEs moved to
`0.2.1-rc.2`, and the lockfile mirror with them — 3 lines. **Written first, and it failed
first:** `0.2.1-alpha.2` moved into `OTHER_VERSIONS` in both plugin suites (4 titan + 3
lan failures, exit 1 each), because that list is what "one supported release" means —
the old release is now a refusal, not a permission. Then 157/157 and 141/141, exit 0,
and `shell` (29 scripts, system bash 3.2.57), `shellcheck` 0.11.0, `docs` (87 documents,
18 gates) and `javascript` (eslint 10.11.0 / prettier 3.9.9) all exit 0 on the final tree.

**The boot is verified, not assumed, and the verification is functional.** Against a
real rc.2 install the skip line is gone; what replaces it is the preset doing its job on
rc.2's own shipped composition — `standard.patch.yml` parsed through the plugin's loader
(22 rows), the compaction row repointed at `dsh-tinytitan/backend`, the chat-noise rows
stripped, and the `optional-ralph` group **appended**, because rc.2 ships no ralph row
either, so the alpha.2-era adaptation still holds. The installed copy in the operator's
`~/.dsh/profiles/web/node_modules` was refreshed from this tree; the Oct-4 copy kept as
`dsh-tinytitan.pre-rc2.bak` was later pruned by pnpm during the webhook install recorded
below, and the old pins remain recoverable from git at `d137a9e`.

**Three things this move did not settle, and one claim I retracted.** (1) The boot now
reports **1** entry not activating, not 22 — the one left is a `webhook` insert in
`~/.dsh/profiles/web/cordis.patch.yml` pointing at `@deepseek-ai/dsh-webhook`, which is
not installed. That is a different defect and the operator did not approve touching
their profile, so it was left untouched **then** — settled later the same day, see the
amendment below. (2) The 22-entry report is therefore **not
reproducible** on a fresh boot. The leading explanation was their own long-lived harness
process (PID 39037, alive since Oct 4) with `patchReload: "live"` re-mounting the old
plugin against rc.2's renamed packages — a hypothesis then, **now confirmed**: the
restart was made with the operator's approval, and the live page reproduced the report
before it and cleared it after (see the amendment below). (3) 13 dangling
links in the legacy `.dsh-module-fallback` farm stay as they are; rc.2 resolves through
installation scope and does not read them. **Retracted:** the launcher comment first
said `/dsh-lan/*` answers on this pin. It does not say that any more — `dsh-lan-manager`
is certified by its passing suite, not by a boot I did not run, and its only harness
peer is the range `@deepseek-ai/cordis ^4.0.2`, so the exact-version gate has nothing to
deny it on. Also disclosed: the first boot attempt measured nothing — `dsh web` selects
its own profile, so `--profile web` was rejected before it booted.

**Amended 2026-10-17, after the operator asked for the whole harness to work.** The
webhook row is settled: `dsh plugin --profile web add @deepseek-ai/dsh-webhook@0.2.1-rc.2`
exited 0 — that is the version rc.2 publishes, its peers name `0.2.1-rc.2` and cordis
`~4.0.5`, and the harness bundles cordis 4.0.5, so the exact-version gate has nothing to
deny. A fresh `dsh web --no-open --port 0` on the real profile now prints **no warning at
all**: zero entries fail, not one. Backups `package.json.bak-webhook-20261017` and
`pnpm-lock.yaml.bak-webhook-20261017` sit beside the patched profile manifests, and the
install added three `minimumReleaseAgeExclude` entries to the profile's
`pnpm-workspace.yaml`. The private launcher harness was still on `0.2.1-alpha.2` with
alpha.2-pinned plugin copies, so `tools/dsh_local.sh ensure` was run: exit 0, the private
marker is now `0.2.1-rc.2`, both private plugin copies (`dsh-tinytitan`,
`dsh-lan-manager`) read `0.2.1-rc.2`, and a boot of the private web profile prints only
its URL — no skip line, which is also the first boot check of `dsh-lan-manager` that the
paragraph above said this record did not have. npm 11.20 reported six install scripts
"not yet covered by allowScripts" on that private reinstall; this was checked, not
assumed — node-pty's darwin-arm64 `pty.node` and `spawn-helper` ship prebuilt in the
tarball (the blocked postinstall only chmods the helper, already 0755), `node-pty` loads
through Node, and koffi's darwin-arm64 binary is present, so the warning is immaterial
here. The operator's long-lived systemwide process (PID 39037, port 3080, started Oct 4)
still ran code loaded before any of this until the end of that same day, when it was
restarted **with the operator's approval** through its launchd service
(`launchctl kickstart -k gui/501/com.andreborchert.dsh-web`). It came back as PID 33574
with a boot log holding only the URL. The live page was then driven in headless Chrome
through its printed token: HTTP 200, no plugin banner, the full UI rendered. Before the
restart the same probe reproduced the operator's report exactly — `web boot: 22 entries
did not activate` — so the report is confirmed as the stale process, not a fresh-boot
defect. The only console errors left are seven 404s from the community
`dsh-memory-evolve` plugin's own `/memory-evolve/api/*` routes, outside this repository.

**#224 closed 2026-10-17, recorded as §5bn — 52 live arms answered the operator's
combination question, and the answer is a no.** The two levers §5ba–§5bm that did
not measure slower than the shipped default (`TINYTITAN_GPU_KEEPWARM_US`, +0.48 %
§5bl; `TINYTITAN_DECODE_EXPERT_EXECUTION=gpu-residency`, +2.07 % §5bj) had never
run one beside the other. Batch 1 was 4 bracketed rounds × 7 arms, batch 2 6 rounds
× 4, one artifact (`e8535a85…`, clean tree at `b6da2d7`, mains, one model process,
qwen38 4-bit at the shipped `cache=11.88G slots=96` tier, 384 tokens greedy), all 52
exit 0 and every one byte-identical (`c172890d`), including all seven with both
knobs live. Both falsifiers were written before measuring and both returned no:
the drift-free contrast `both − kw` inside the same round is +1.58, +1.76, −1.81,
−10.63, +0.21, −10.94 — **mean −3.31 %, best positive +1.76 %, against control bands
of 3.2–11.4 %** (the direct arm ratio gives −3.70 %, best +0.62 %: no estimator in
this design puts the pair above its own bands) — and batch 2's three type-B rounds
give the stacked arm *more*
`wait_ms` (+10.18 %) and *more* `body_ms` (+7.65 %) per token than keep-warm alone.
**They are one mechanism**: two knobs that poke the same GPU queue do not compound,
and the number to carry is "+0.48 % from keep-warm alone", never `both`'s +1.38 %.

**The finding that outranks the clock: `gpu-residency` changes no executed work.**
Grouping all 52 arms by kind, the three counters that cannot move unless the
assignment moves are equal to the unit in every arm — `expert_evictions` 31 293,
`expert_reloads` 22 986, ring `begins` 18 001 — and every kind's `expert_misses`,
`expert_read_mib`, `hit_fixup_layers` and `prefetch_adopted_per_token` sits inside
the *base arms' own* spread. The source says why:
`RealForwardRunner+DecodeMoE.swift:128-182` reads the device's lists back, sorts,
checks the superset property, adds three `gpu_classified_*` counters, and then its
own comment executes the CPU partition regardless, because the GPU's list cannot see
prefetch adoptions and running it would re-fetch an expert already resident. Its
marginal allocation is **zero bytes** (the seven classification buffers and the
residency table are built with no mode branch — `BuildScratch.swift:104-126`,
`PreadExpertStreamer.swift:239-248`), so §5bj's four affected sentences are amended
in place, and the mode's only executed delta is one 1-lane dispatch per layer per
token — keep-warm's mechanism at a third of its density.

**Four method points this batch cost, so the next session does not pay them again.**
(1) Two arms were withdrawn *on counters, not on clock*: batch-1 `r3_base2` at 2.319
tok/s and batch-2 `r5_p4` at 2.783, each with its round mates' work counters and
2–3× the host-only durations — that is this host thread being descheduled under deep
swap, not a knob. One of them moved the single-batch verdict by the whole size of the
effect (−0.30 % ↔ +2.12 %), which is why the falsifiers and not the means carry it.
(2) The slow controls that fell just *under* the 1.20× screen (1.175×, 1.124×) were
deliberately not withdrawn: lowering a contamination threshold after seeing the data
is threshold shopping. (3) Batch 2 ran in a bad regime (swap 8 682 → 10 778 MB,
controls 3.81–4.98 against batch 1's 4.85–5.57), so **no clock crosses the two
batches** — only within-round pairing is quoted, per the standing rule. (4) The
keep-warm dose is now at least three doses across 52 arms (achieved period 1 313–2 042
µs as `Thread.sleep`'s quantisation moved with the swap level), so §5bl's deviation 1 —
the best dose was never runnable — still stands. Also filed and deliberately **not
edited**: `moe.metal:117` says "Only eight positions are touched" where qwen38's
`topKExperts` is 10 (`ModelTypes.swift:282`); the kernel is correct (`:134` loops to
the passed `top_k`), so it is a one-word doc fix that belongs on its own commit after
the campaign, not inside a measurement that every later arm is hashed against.

**What is left on this line is unchanged.** The 52.8 ms `phase1_hit →
phase1_miss_fixup` gap: §5bm closed its bytes at 0.09 ms and §5bn closes its knobs,
so nothing switchable remains on it. Earlier-bytes and the 26.8/16.1 ms gap pairs
stay open and are bounded by landing slots (`free=0.99`) and the operator's 12 GiB
cap, not by a control. No knob on the decode expert path is left unpriced, the
shipped default is unchanged, and both levers stay opt-in and documented.
**Landed and published:** this commit carries §5bn, the four §5bj amendments and
this entry on `main`; the wiki's Runtime-controls page carries both knobs and the
combination result, published as `e766bf9` on `master`. `5357c33` was that page's
first form, printed before the estimator correction above, and `e766bf9` is the
one carrying −3.31 %. The wiki's Project-tracker narrative deliberately does not
gain a §5bn row — it has never carried §5ba–§5bm either, and the knob
measurements belong on the page a user actually reads.

**#163 closed 2026-10-17 — the wiki now publishes measured levels for this model.**
The operator chose to publish the campaign's own CLI protocol rather than wait for a
gate0 window this Mac's desktop will not give it, so `Benchmarks.md` loses its
**6.82 tok/s** row (v4.7, 2026-08-30 — above the `5.5` the same page quotes as that
model's peak) and gains a section of its own: greedy 384-token `TinyTitanCLI` at the
shipped 96-slot default, **median 5.40 tok/s** over §5bn's 15 surviving control runs
(mean 5.34, 4.85–5.57, sd 4.1 % of mean), with §5bn's second batch (12 controls,
median 4.13) and §5r's five runs (median 4.63) printed beside it as the same
configuration measured three times. The page carries the two rules that make those
three lines one band — the first arm after a load is systematically slow, and a
qwen38 clock does not travel between batches — plus the reason the table's own
instrument stays unpublished here (§5bd's floor-is-the-desktop finding, and §5t's
contended 5.713 median). **Published as wiki `c2591e7`**, which also removes TT-049
from `Project-Tracker` and leaves one open row; the page's head is now `5e13783`, a
same-day in-place re-word of that section's gate0 sentence — it said "needs a
different Mac", which is stronger than §5bd measured, and now says the preflight has
to measure TinyTitan's own occupancy. Every number on that page reproduces
from `/tmp/c224/b1/manifest.tsv` and `/tmp/c224/b2/manifest.tsv` while this session
lives; the campaign record is §5bn and §5r.

**#223 closed 2026-10-17, recorded as §5bm — an offline screen, no model run, no clock.** The
one direction §5bl left standing on the 52.8 ms `phase1_hit → miss_fixup` gap was *fewer bytes
per miss*, and every closure on that gap had inherited the premise unexamined: §5w proved a miss
reads exactly `expertStride`, which is not the same as the stride being all payload. Scanned
**all 24 576 experts** of the installed qwen38 4-bit layout: stride 2 768 896 = **2 764 800 B of
payload + 4 096 B of page padding**, zero interior holes, `offset == expert × stride` in every
case, no header and no metadata inside the blob. So the entire bytes lever is worth **0.09 ms** of
the 52.8 — and the 4 096 B is the alignment the §5c sub-range split reads *through*, so taking it
back trades a byte-equality property its own test pins for a saving two orders below this
machine's run-to-run band. What that leaves on the gap is *earlier bytes*, bounded by landing
slots (`free=0.99`) and by the operator's 12 GiB cap, and nothing in the format.

**#222 closed 2026-10-17, recorded as §5bl — the batch finished, nothing here
awaits data.** Task #222 priced `TINYTITAN_GPU_KEEPWARM_US` live on qwen38 4-bit
at this Mac's derived `cache=11.88G slots=96` tier (no budget flag was passed, and
the CLI takes none), mains, one model process: twenty arms, both sentinels
(`BATCH1DONE`, `BATCH2DONE`) present, every arm exit 0 and every response text
the same md5 (`c172890d`), so the second command queue is proven numerics-neutral
at ~35 000 dispatches per request. The clock says **+0.48 %** on the
position-balanced confirmation batch (four of five rounds land inside their own
base-vs-base drift band), and the lever is **not adopted** — the shipped default
is unchanged and the knob stays opt-in. Two things it leaves named: the dose the
bench measured best was never run in the engine (`Thread.sleep` achieves ~2 ms
when asked 200 µs, where §5bk achieved 616 µs, so a true sub-ms spinner needs
`mach_wait_until`), and this batch transcribed no runner-stats gap counters, so
it priced the wall consequence without touching the mechanism claim.
**Landed and published:** `248f795` (engine), `8d46b15` (park note) and `7971271`
(§5bl) are on `origin/main`; CI 39002656047 and CodeQL 39002656025 both came back
**success** on that head. The knob is documented on the wiki's Runtime-controls
page, published as `2faa854` on `master`.

Every measured cell below carries the date it was measured and the command that
re-measures it. That is the point of the format: this table was a hand-copied
snapshot, and six of its eleven rows had drifted by 2026-10-06 — it named 5.15 as
the current release after 5.16, 5.17 and 5.18 shipped, "8 installs, 244 GB"
after the model set was pruned to two, "7 checked" goldens when one target has
an install here, "level with `origin/main`" on a tree with work no remote has
seen, "28 findings, all closed" beside an open audit, and "level with
`origin/master`" for a wiki with an unpushed commit. **When a fact here changes,
re-run the command and rewrite the cell with the new value and today's date** —
do not edit the number to match what you expect, and do not trust it as current
if the date is old.

| Piece | State |
| --- | --- |
| Repository | `Pummelchen/TinyTitan` (renamed 2026-09-14; the old URL redirects) |
| Checkout folder | `~/Downloads/TinyTitan` — **renamed from `~/Downloads/NVMAI`**, which invalidated every receipt and `.build`'s debug half |
| `main` | as of 2026-10-11, **pushed to `origin/main` on 2026-10-11, twice: `182f770` then `fe19203`** (363 audit commits plus a merge of four upstream traffic-badge commits that landed while the audit ran; ahead count re-measured after that push: **0**; re-measured again now: **4** -- `154dc68` (AUD-284), `9ea1368` (AUD-293), `cc7ef48` (AUD-259) and `e1d651d` (AUD-311, the red-CI fix), none of them on the remote yet; then pushed a third time on 2026-10-11 with the AUD-311 close `3404269` (ahead re-measured **0**, CI run 38124699999 got past the `docs` gate and named AUD-312), then a fourth time with the AUD-312 close `3e925b5` (ahead re-measured **0**; its CI run 38130598551 got past the `docs` gate and the launcher step and named one failure, AUD-313, fixed at `1ff2fb3`), and a fifth push carried `1ff2fb3` (AUD-313's fix), `7d9b09a` (the combined codeql-action bump to v4.38.3) and the AUD-313 close `4fd77a7` (ahead re-measured **0**; its CI run 38132374149 named a fourth failure, AUD-314's tsan fixture, fixed through `d0d688c`, `a6ffdc2` and `ec1d546` with the probe runs of the `tsan-fixture-probe` workflow this close deletes), then the dsh 0.2.1-alpha.2 move `6461e9f`, then the AUD-314 close `d9b4317` (ahead re-measured **0** after its push; its CI run 38142231539 and CodeQL run 38142231436 both success), then the handover wiki-sync `340ccac`, then the two rebase-merged dependabot bumps `57a7098` (actions/setup-python v7.0.0, PR #20) and `7086953` (actions/setup-node v7.1.0, PR #22) (ahead re-measured **0** after the last push; the run on `7086953` -- CI 38144887362, CodeQL 38144887368 -- is green in both jobs, the run on `57a7098` cancelled by the merge push behind it, so the certification of the range lives on the final head) (`git rev-list --count origin/main..main`, measured at 343 at AUD-303's fixes `19e2a23` and `5af9805`, 344 at that close, 345 at AUD-304's fix `954ff87`, 346 at that close and 347 at AUD-305's fix `e60d950` -- 349 at AUD-306's fix `79308fe`, 350 at AUD-307's fix `b1f4260` -- 352 at AUD-296's fix `1b4eccc` -- 353 at that close, 354 at AUD-308's fix `4f9ebd7` -- 354 at AUD-308's fix `4f9ebd7`, 355 at that close, 356 at `e31381d`, 357 at `ef254d5`, 358 at AUD-309's fix `9c50d4b`, 359 at `b287f21` whose M4 claim did not hold, 360 at `6eb050f` which killed M4 for real -- `4df0951` closes AUD-309 as the 361st, this row's pickup paragraph is the 362nd, `6634e9d`'s successor records the closure invariant as the 363rd, and `182f770` is the merge that landed the stack -- a merge deliberately, because the ledger cites 201 fixes by sha and rebasing 363 commits would rewrite every one of them; read wrong here four times in a row — the AUD-242 close wrote 216/217 from a count taken before its own fix commit landed, when the truth was 217/218, and AUD-233, AUD-237 and AUD-246 each moved the number without re-reading it. The pattern is that the close row predicts its own landing correctly and then the *next* fix commit moves the count before anyone re-reads it, so this row's successor must measure again rather than add one. `tools/lint.sh docs` does not check this number, so it drifts); newest tag `v5.19` (tagged commit `906456c`; the tag was force-moved once during prep, `0d8ac3b` -> `906456c`, before any Release existed), HEAD at this close. CI now covers everything through `6461e9f` green end to end (run 38124699999: the `docs` gate green, one test failure, AUD-312, fixed at `36b4a81`; run 38130598551: the `docs` gate and the launcher step green, one test failure, AUD-313, fixed at `1ff2fb3`; runs 38132374149 on `4fd77a7` and 38136569050 on `a6ffdc2` red on the tsan fixture, AUD-314, fixed through `d0d688c`, `a6ffdc2` and `ec1d546` -- the run on `ec1d546`, 38138725190, was cancelled by the push behind it with its thread-sanitizer job already success; run 38140028728 on `6461e9f` is green in both jobs and every step; CodeQL run 38130598556 succeeded on the old pin, which is why `7d9b09a` is consistency, not a red-fix; run 38142231539 on `d9b4317` (the AUD-314 close) and run 38144887362 on `7086953` (the merged dependabot bumps, with CodeQL 38144887368) are each green in both jobs, the runs on `57a7098` cancelled by the merge push behind them), so the stale-coverage warning is retired through `7086953` and re-arms for the commits ahead of it until their run is read. `tools/release.sh` still refuses to tag until `tools/ci-green.sh` sees a run on the commit -- the release prep then landed: `ca8cdb7` (Prepare 5.19), `0d8ac3b` (the pip-audit step fix), `906456c` (the notes' measured numbers), tag `v5.19` forced onto `906456c` with CI 38155915391 and CodeQL 38155915358 green, the Release published 2026-10-11, and the AUD-139 ledger close behind it as the next push |
| Release | **5.19 published** 2026-10-11 (`gh release list` — it is the latest), six assets: `tinytitan-5.19-macos-arm64.tar.gz` + `.sha256`, `tinytitan-lib-5.19-macos-arm64.tar.gz` + `.sha256` and `tinytitan-5.19-tools.tar.gz` + `.sha256` — the first release the closed installer accepts, because the tools asset it checksums exists and its tag is ≥ 5.19 (`tools/install_tinytitan.sh` refuses anything older). `ServerVersion.current` is `5.19`, and `tools/release.sh:118` refuses a tag that disagrees with it; the notes quote the publish-pass digests, which the dry run's differ from — the runbook's warning, re-proven |
| Models | as of 2026-10-06, **2 installs, 163 GB** (`du -sh models/*`): `qwen3.8-flash-next_125B_A6B_4Bit` (162 GB) and `qwen3.8-flash-next_125B_A6B_MTP_4Bit` (1.4 GB). The rest were pruned for disk and **must not be re-fetched** to satisfy a gate; every receipt here is bound to this path, so both load |
| Goldens stored | 21 files under `benchmark/golden/`, 21 targets in `tools/golden-baseline.sh`; as of 2026-10-06 **1 is checkable** on this host — `qwen38-4`, the only target whose directory exists under `models/`. The other 20 (`ornith-{4,8}`, `ornith-bf16dense-4`, ornith122-{4,8}, `qwen38-8`, `qwen36-{4,8}`, `agentworld-{4,8}`, `katcoder-{4,8}`, `qwen35-{2b,4b,9b}-{4,8}`, `k2-{4,8}`) are reported *not checked* and named in the notes; the default `ornith-8` is among them. The MTP install maps to no golden target at all |
| Audit | **this audit**: `docs/audit-ledger.json` — as of 2026-10-11, `counts` in `ledger.json` is **210 rows / 210 closed / 0 open / 0 blocked** (`python3 -c "import json;print(json.load(open('docs/audit-ledger.json'))['counts'])"`), and the eighteen gates in `tools/lint.sh` are partly what it left behind. That command is not a suggestion here: `tools/lint.sh docs` compares this very string against `ledger.json` and fails, so any ledger close that moves a count edits this line in the same commit — which is the point, because a restated count is how AUD-122 shipped a commit message its own file contradicted. **AUD-280 (closed 2026-10-10 at `1b16d7a`) is the eighth driver along that seam, and the first whose verdict was not a status but a string comparison**: `tinytitan_hit_fixup_ab.py` decided whether hit/fixup decode agrees with the barrier by testing the two arms' text for equality, and required a server footer by *line count* while parsing no value out of it. Measured model-free against the real `main()`, with only the module's leaves patched, five shapes exited 0 or read as clean: two arms answering nothing hashed to one agreement, `decode_tok_s=nan` printed beside a rate, 128 completion tokens reported `0.00 tok/s`, an answer sat beside a `0.00` footer, and a response that claimed no `completion_tokens` left the length control -- the control the A/B exists to run -- unrun. Post-fix those answer 2, 2, 1, 2 and 1 with the reason named, the control case still answers 0 over `cases compared 6 of 6`, a cell with no figure prints `-- tok/s` rather than a zero no run measured (AUD-276's rule applied to a driver that predates it), a refusal writes no record and names the cases that ran, and a sweep with no rows -- or, the shape the first guard missed, no cases to compare -- refuses instead of certifying `0 of 0`. `benchmark/test_hit_fixup_ab_verdict.py` (29 tests) is registered in CI before this line; the 31-mutant anchored sweep kills 30, the survivor being the CONTESTED summary for a disagreement, which duplicates the per-case line the page already prints and so is pinned as text, not as a status. Two process notes this close made visible: the `if not cases` guard written before its test was *unreachable*, because `_cases()` indexes every (prompt, warmth, mode) triple and an empty sweep died in `KeyError` first -- that is what the test caught, and what moved the guard to `rows`; and one expectation in the new test was wrong before the driver's was (`4 of 6` quoted for a six-case page with one case disabled, where the figure is `5 of 6`), so the assertion was corrected and the correction is stated here rather than left silent. `hit_fixup_layers` is deliberately NOT gated: a legitimate hit-fixup run can report 0, because a layer whose experts all hit takes no fixup phase, so refusing on `> 0` would rest on a premise no run here may test -- the operator's server (PID 17514) holds the machine. Whether that field should gate the arm at all, and whether a refusal that arrives after some cases ran should also leave a record (today it does not, but names them), are **the operator's call**. The **1 open** then was AUD-259, deliberately unfixed — `CPUModelBackend.setContention` documents a caller that does not exist, and wiring it would change measured decode performance, so it waited on the operator rather than on turns (the operator's answer came 2026-10-11: the wiring was wrong and the deletion right, closed at `cc7ef48`). AUD-261 itself closed on 2026-10-09 and came from this audit's own mistake: a `--filter` that matched nothing exited 0 having run no test, which is how `tools/repack_dense.sh` could print `all checks passed` over an equivalence gate that had compared no logits. **AUD-262 (closed 2026-10-10 at `15808fe` and `ba0ddaf`) was the same shape in the release path**: `tools/release.sh` decided its whole-suite gate with `grep -q 'Test run with .* passed'` over a log that a real unfiltered run shows carries **seven** summary lines, one per test target, while the run's own exit status was discarded before that grep ever ran — so one target passing was enough for the release to proceed while six reported failures. `test_gate_verdict` now takes the log and `${PIPESTATUS[0]}`, refuses a run with no summary line and a run where any target's line says failed, and says how many targets it read when they all pass; `benchmark/test_release_test_gate.py` (7 tests) drives the extracted function over the seven real lines and the two wiring claims, and six mutants were each killed. Its own mutation run is what removed a seventh guard no test could reach. **AUD-263 (closed 2026-10-10 at `cb08776` and `05bb0f5`) came from that sweep's coverage pass** — the tracked scripts no test file names — and `tools/compact-release-notes.py` was one of them: it is the only author of what the Release page shows, its docstring promises nothing is reworded and only re-laid-out, and it merged every block whose line breaks *are* its content into one bullet. Measured on the shipped notes: v5.18's two-line progress-format example published as a single fenced sentence, and v5.1 lost two tables, two blockquotes and one fence the same way. `release.sh`'s `--require` could not see it because every token survives the merge. The fold was wrong in both of its cases, and the same branch shredded a numbered list and glued a nested item onto its parent. `benchmark/test_release_notes_compact.py` (16 tests, in the CI gate group) asserts each block travels contiguous and blank-delimited, over the fixtures, over every `docs/release-notes-v*.md`, and over the checksum, `--require` and budget refusals; eleven mutants, all killed — two of them (folding a capitalised sentence, a fence published above its own lead-in) survived the first sweep and are the reason those two pins exist. **AUD-264 (closed at `ab39375`) is this audit's own bookkeeping**: `ledger_commit_evidence` tolerated a blank `commit` field for every status, though the exception was written for rows that fixed nothing, so AUD-262 closed with its fix on `main`, an empty pointer, and a report claiming the pointer had been kept. A DONE row must now name a resolvable commit — AUD-262's field names `15808fe` and `ba0ddaf` — and the gate then caught this audit's own close on its first run over new rows, which is the proof that it works. **AUD-265 (closed 2026-10-10 at `b8287c6`) is the arm64 assertion itself** — RELEASE.md rule 2 made executable, and the script no test named. Its magic read was a command-substituted `od` piped through `tr`, so the status the gate tested was `tr`'s: an artifact it may not open, or one whose header is truncated to 1-3 bytes (what a `cp` that filled the disk leaves), returned an empty magic and took the same skip as a README, then printed "each exactly arm64" over a set it had read only half of. The magic set was also missing the 64-bit fat magics while the comment above it said "fat 32/64", so a universal binary — the one artifact that rule exists to stop — was skipped by the directory scan. `benchmark/test_assert_arch.py` (10 tests, registered in the CI gate group) puts each subject beside a real arm64 binary so a refusal cannot come from the vacuity guard, and nine mutants were each killed. Its re-measurement is also what found the gate-group tally restated in AUD-263's row to be stale; a test count is checked by no gate, so the ledger rows now carry the command to re-run rather than a figure that rots behind them. **AUD-266 (closed 2026-10-10 at `b67cf1e` and `42fedc7`) is that same root cause in the next script over**: the release's compiler-warning gate was one `grep -qE … && die` duplicated verbatim in `tools/release.sh` and `tools/build_library.sh`, and it can only act on the status that *found* a warning — measured in both callers' own option shapes, because a failing command left of `&&` never triggers errexit, so a missing, 0-byte or unreadable build log left the refusal unreached and the script walked on into staging, while grep's own "Permission denied" sat two lines above a `== staged ==` header reading as a pass. A truncated log needed no fault at all: it records no build, and "no warning line in it" was a pass. The sibling sweep then found the same line a **third** time, in CI's own build step, written as `if grep -qE …; then exit 1; fi` — and driving that step's *extracted text* under the `bash -e` a `run:` block with no `shell:` key gets showed all four shapes exiting 0, in a step whose `swift build … piped to tee` has no pipefail either, so the build's own status never reaches the gate and only its log does. `tools/assert-build-log.sh` now owns the decision for all three readers with four distinct refusals (not there, may not open, no `Build complete!` line, the pattern did not run) and a pass that prints the line count it read; the marker was measured rather than invented — `find .build -name '*.buildlog'` returns eleven real logs on this host, all UTF-8 text each carrying it, and all eleven pass the new gate, so nothing already published is refused retroactively. `benchmark/test_build_log_gate.py` (17 tests, in the CI gate group) went in first, and its caller test is what keeps the pattern in one file. Fourteen mutants across the two commits: thirteen killed — two of those only after the design moved (a redundant second read, and an assertion that checked the words of a pass instead of its number) — and one that survived because it is equivalent: a fallback that swallows the gate's status with `or true` cannot neuter a gate that `exit`s inside a subshell, measured. **Not done, on purpose:** widening the pattern to any line containing `warning:` would refuse v5.16, v5.17 and v5.18, each of whose logs carries SwiftPM's own cache-skip notice, so whether a release stops on one is the operator's decision, reported as pending rather than assumed here. **AUD-267 (closed 2026-10-10 at `9094502`) is that conflation in the release's own preconditions**: `gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1` chained to a `die` could only act on the status that *found* a release, and both streams were discarded with it. Measured on gh 2.102.0: an invisible repository answers `release not found` with the identical status *and* message a missing release gets, a revoked token answers `gh: Bad credentials (HTTP 401)`, an unreachable API answers `dial tcp ... connection refused`, and a host with no `gh` answers 127 -- and the precondition line as written exited 0 on every one of them, so a release carried on past the clean scratch build and the golden baselines to re-learn the cause at `gh release create`. The fix asks twice (`gh api repos/$REPO` proves the repository before anything is asked of its releases) and accepts absence only from a 404. Its sibling in the same block piped `git ls-remote --tags origin` through `2>/dev/null` into a grep: it fails closed but throws away git's reason, so an unreadable remote (rc 128) is announced as `not pushed to origin`, and its pattern interpolated the tag into a regex, measured to accept `refs/tags/v5X18` for `v5.18`. `benchmark/test_release_precondition_gate.py` (18 tests, in the CI gate group) drives the two extracted verdict functions against stub `gh` and `git` replaying the measured answers byte for byte -- no socket, no token, no network -- and eleven mutants were each killed. Two of those kills say more about the tests than the first sweep did: deleting the gh-installed guard still exited non-zero, because bash's own `gh: command not found` was captured into the *repository-query* refusal and contained the word the assertion looked for, and relaxing the whole-field ref match to a prefix survives the dot test, since only a shorter tag (`v5.1` asked of a remote holding `v5.18`) shows that end of the conflation. The two functions were then driven against the real gh and the real origin, read-only, and each refusal named its true cause. That sweep found the next finding: the model-process preflight (`tools/golden-baseline.sh:130`, `tools/install_tinytitan.sh:184`, and six python benchmark drivers) reads only pgrep's success, so a pgrep that errors (rc 2, measured) answers `no model process running` and the run starts beside a live server -- filed as AUD-268. **AUD-268 (closed 2026-10-10 at `bfaa6fd`) is that conflation again, and its widest surface**: the precondition AGENTS.md itself states — *no process from that pgrep* — was asked at **eleven sites across nine files**, every one of them a shape that can only act on the answer it wanted: `if busy=$(pgrep -fl '...' 2>/dev/null)`, `if pgrep ... >/dev/null 2>&1`, a `wc -l` line count over pgrep's own output, and six python drivers that read stdout and threw the status away. Measured against a stub pgrep answering 2, each shell shape reached the run anyway (`reached the run (busy=[])`, `proceeded without warning`, `reached the build`) and exited 0 — and status 2 needs no race: `/usr/bin/pgrep -fl '('` answers it on this host with *parentheses not balanced*, while a machine without pgrep adds the shell's 127. Two of the six drivers are a different consequence worth naming: `pgrep -n` read an error as *no pid*, so `tinytitan_vs_competitors.py` published a running competitor's RSS as absent, which mislabels a published comparison rather than racing anything. `tools/model-guard.sh` now owns the tri-state for shell — 0 with the lines it matched, 1 nothing matches, 2 could not ask quoting pgrep's status and message — and `tinytitan_profile.pgrep_answer` gives python the same three verdicts; the installer carries the guard as a verbatim inline copy because it arrives through `bash -c "$(curl ...)"` with no repository beside it, and a test pins that the two copies are the same text. `benchmark/test_model_process_guard.py` (32 tests, in the CI gate group) runs the *real* scripts against a stub `pgrep` and a stub `swift`, so it starts no model and no build. Ten mutants, and the two that survived the first sweep are the reason the suite is stronger than the fix: dropping `2>&1` from the capture was caught *only* by the copy-equality pin, because a stub's stderr reaches the terminal whether or not the guard read it — so the refusal is now pinned to carry pgrep's reason inside its own line; and deleting the unknown branch at the ollama pid site survived outright, because the wiring pins covered the drivers' busy guards and not the pid probes, so three behaviour tests now pin all three answers at both competitor sites, each confirmed RED against the code the mutant removed. shellcheck then found the dead assignment this audit would otherwise have left in place — the installer collected the matched lines into a variable and warned a fixed sentence — so the warning now names the process it found, and that assertion was proven RED against the previous installer before being kept. The sibling sweep's first find was the same conflation in a **documented** command: `docs/release-process.md`'s quiet-window loop printed `quiet` for an error it could not read, which is a runbook step an operator trusts before spending a release attempt; it is now three-state, verified against a stub. The prose copies (AGENTS.md, the wiki's FAQ and Benchmarking-Guide) were inspected and left, because a human reading them sees pgrep's own error line, and the wiki is a separate repository whose change is a second push rather than part of this fix. **No model run happened or was possible**: the operator's TinyTitanServer is up on 127.0.0.1:8080, which bars every one of them by the very precondition this row fixes.AUD-269 is the same conflation, a sixth time, in the instrument that is supposed to say whether the forward pass is right: `tools/qwen38_parity.py`, `qwen38_full_forward.py` and `qwen38_sequence_parity.py` (the harness `docs/qwen38-flash-next-port.md` describes) computed a verdict for every stage and threw all of them away -- `report()` returns a bool and `main()` dropped the nine it made, and `full_forward` set `first_bad`, printed "first divergence at layer 3" and bare-`return`ed -- so eight `FAIL` lines and exit status 0 left together and `python3 tools/qwen38_parity.py ... && echo "parity ok"` printed its sentence over a run that agreed with nothing. The harness could not tell "these two implementations disagree" from "I had nothing to compare" either: a dump directory with no `posN`/`L*_entry.f16` files made the sequence script print nothing and exit 0, while the other two died on `np.fromfile` with status 1 -- the status a genuine disagreement now has to carry. The fix adopts the three statuses this tree already uses in `tools/model-guard.sh` and `tools/qwen35_reference.py:411` (0 compared and every stage agreed, 1 compared and at least one disagreed, 2 could not compare, with the reason on stdout), and `benchmark/test_parity_harness_status.py` (17 tests, in the CI gate group) pins all three by running the real scripts through their own `__main__` against a synthetic weight table and real float16 dumps: no model, no install, no socket. Tests-first -- 8 RED before the fix, 17 GREEN after -- and the suite carries a deliberate pass direction (`agree`, `sequence_agrees`, an agreeing stack output) so that an "always refuses" mutant is caught rather than rewarded: 17 mutants, zero survivors. It is the gate group's slowest suite (about three minutes) because the sequence cases run the whole 48-layer numpy reference before they reach the guard under test, and the script's own order is what the guard is. Two things are left open rather than decided here: whether the harness should now be wired into a gate or a release step, and whether `qwen38_sequence_parity.py` should carry a threshold verdict at all -- it stays a profile, because choosing a tolerance is numerics work nobody asked for. One record correction: the commit that closed AUD-268 is titled "file AUD-269" and filed only the queue item that named it; this row is where AUD-269 is filed and closed, at d862cdd.  **AUD-270** is the same root cause an eighth time, in `tools/reconcile_snapshot.py`, the pre-repack check that says whether an affine norm was folded (the build stores `weight = γ − 1` and adds the `+1` itself, and `prepare_qwen38.py:275-278` records that getting it wrong is silent and produces fluent nonsense): `--check` compared `fold_names[:6]` while its own docstring promised the fold was verified, printed `folded (sampled 6)` and `snapshot matches the converter policy` and returned 0 over the rest of the fold set it had not looked at (18 of 24 on the suite's fixture), and the repair path re-fetched all 24 whenever any one was wrong. The fix compares every fold tensor, gives a tensor it cannot decode or whose size differs its own bucket instead of a silent `continue`, and adopts the third status this tree already uses (`tools/model-guard.sh`, `tools/qwen35_reference.py:411`) -- 0 compared and matched, 1 drifted and named, 2 every check that could run passed but the fold is not proven -- so a snapshot the tool only half read no longer reports a clean pass, and the repair rewrites only the tensors that are wrong or uncomparable. `benchmark/test_reconcile_snapshot_verdict.py` (15 tests, the fifth suite in CI's converter gate group, which the five together run in 115 tests and 42.7 s) drives the real `main()` through `importlib` against a synthetic bf16 checkpoint served from 127.0.0.1 with real `curl` range requests, so no shard is fetched, no model is read and nothing touches huggingface.co; tests-first, six proven RED before the fix, and a 15-mutant sweep that first survived M11 (`drifted` minus `extra`) -- evidence about the tests, not the tool, so the stray-tensor pair was written and proven RED under that mutant and GREEN unmutated, and the re-sweep kills 15 of 15. The sibling sweep found one live-looking sibling and cleared it rather than filing it: `tools/patch_snapshot_precision.py` rewrites a snapshot's widths without folding, which would be this defect on the write path, but all ten `UNIT_OFFSET_NORM_SUFFIXES` are also members of `STRIP_WEIGHT_SUFFIXES`, so `rename()` drops the `.weight` that `quant_bits` requires in order to answer with a width at all -- every fold tensor is passthrough by construction, and that is pinned as a test over the converter's own name table rather than left as reasoning. Four other `[:N]` sites (`wrong[:8]`, `problems[:40]`, two inside error strings) truncate only the display while the verdict counts every item, so they are not this defect, and `tools/compare_qwen35_precision.py` and `tools/precision_plan_qwen35.py` were measured and left for their own rows. Two things stay open rather than decided here: whether `--check`'s full fold comparison -- one range request per norm tensor against the live checkpoint -- is an acceptable cost, and whether the repair path should re-verify before returning 0; neither is in the filed scope and the second would change what the tool prints.  **AUD-271 (closed 2026-10-10 at `787ef8b`) is this audit's own evidence base: the gate harness.** `tools/lint.sh` was the (f) queue's largest member at 1609 lines, it is the harness behind eighteen of this repository's gates, and no test file named it. Six of its checks already refused a scan that read nothing; **five had no guard at all**, and measured against a throwaway tree each exited **0**: `force-cast` ran its grep inside a process substitution with stderr hidden, so an absent `sources/` left `found` at 0 and printed `ok`; `arch-path` globbed with no counter and no expectation; `shell-portability` printed its script count and never tested it; `silent-test-skip` globbed a `tests/` that was not there and got an empty iterator, which raises nothing; and `unchecked-sendable` answered an empty scan by **writing an empty baseline** and calling that a pass. So every close above this one cites a green the harness could have produced without opening a single file, and the trigger is not hypothetical: this checkout was renamed once and `AGENTS.md` records that the receipts and `.build` broke, func-length's own comment records a pass-on-nothing that already happened here, and `ROOT` is derived from the script's own location, so a wrapper that copies `tools/` moves it with it. The sibling sweep -- every mode run against an empty tree -- then found two more unguarded checks, `swift-format` (`xcrun swift-format` reads a missing path argument without complaint) and the `python` parse floor (ruff answers `All checks passed!` over no files and a `rglob` over a missing directory yields an empty list), so the fix covers seven. `benchmark/test_lint_gate_scan_guard.py` (26 tests, registered in CI's `Gate-harness empty-scan guards` step) went in first and ran RED on the five; it drives a copy of the harness inside a throwaway tree with `xcrun` and `ruff` stubbed, and pins **both directions** -- an empty scan refused and named, and a non-empty tree that satisfies the rule still passing -- because a guard that refused everything would otherwise score as the fix. Each check now counts what its own scan opened, refuses at zero **before** any baseline or verdict branch (the sendable guard sits ahead of the branch that would write the empty baseline), and names the count in its `ok` line; `bash tools/lint.sh all` now prints `375 Swift files scanned`, `811 files scanned`, `29 scripts, system bash 3.2.57`, `232 test files scanned`, `611 Swift files in view`, `167 scripts parse under 3.13`, rc 0. The sweep broke one guard per mutant three ways (never fires, always fires, refuses but forgets its own status) across the seven and one per pre-existing control, 27 mutants **all killed**, and the first sweep survived one: `library-facade`, because its control asserted only a non-zero status plus a token that appears in *both* of that check's refusal branches, so an unrelated branch masked the neutered guard -- evidence about the tests, not the harness. The fixture was reshaped so the other two branches are satisfied and the assertion pinned the guard's own phrase; no assertion was weakened, and the change is recorded here rather than slipped in. Two more findings are about prose rather than code: `AGENTS.md`'s `silent-test-skip` entry already claimed that check "fails when its own counter measured no bodies" at a time when no such guard existed, so this fix made the sentence true instead of correcting it, which is exactly why a documented guarantee needs a test behind it; and `stdout-clean` still throws a raw traceback rather than printing a gate message when `Package.swift` is missing -- it fails closed today, so it is reported here and not filed, and whether it should answer in words is the operator's call.  **AUD-272 (closed 2026-10-10 at `7469d27`) is the gate that audits the other gates refusing a true document.** `tools/docs-facts.py`'s names rule says a paragraph listing three or more modes must not hold a hyphenated name that is not one -- right rule, wrong block: `paragraphs_of()` split on blank lines, a GFM table has none between its rows, and so this table's eleven rows were one paragraph. Validating the AUD-271 close is what proved it, because that close put a list of real gate names in the Audit row and the gate then answered with four refusals at `docs/handover-tinytitan.md:120`, each one reading "names so-and-so, which is neither a mode nor a header label of tools/lint.sh" -- three golden-target names in the Goldens row, a sanitizer run in the CI row four below it, and the triggering row sitting between them. The line it reported was the table's header, so even a correct refusal sent the reader to the wrong place. A gate that refuses correct documents trains its operator to edit prose to satisfy it, which is this audit's own failure mode approached from the other side. The fix makes a table row a block of its own -- the unit `delimiters()` in the same file already uses for the pipe-count rule -- and leaves the three-mode threshold, the history-heading exclusion and the four other checks that read paragraphs alone. `benchmark/test_docs_facts_gate_names.py` (5 tests, in CI's gate group) went in first and ran RED on the three granularity failures; its controls pin that prose which lists gates is still checked and that a *wrapped* prose paragraph still groups, which is what distinguishes row granularity from the over-broad per-line alternative -- and the sweep proves the difference matters: 3 mutants (rows not their own paragraph, every line its own paragraph, a refusal that forgets which row it read), all killed. Its own first gate-facing run failed for the reason AUD-211 exists: the workflow named the suite before the file was tracked. `bash tools/lint.sh docs` ok (83 documents against 18 gates, table shape in all 109, 72/76 python suites registered) and `bash tools/lint.sh python` ok (ruff 0.16.7 clean, 168 scripts parse under 3.13); the four docs suites together with AUD-271's harness suite run 55 tests, OK, 6.7 s. One blind spot is stated rather than fixed: a paragraph naming one or two modes is not read as a gate list at all, so a bogus gate name beside a single real one is still invisible -- the deliberate price of not flagging every hyphenated word in prose, and the reason the suite carries that case as a control.  **AUD-273 (closed 2026-10-10 at `5467792`) is the same conflation in the three drivers that answer the MTP and ANE questions.** `benchmark/tinytitan_mtp_phases.py` exists to say whether the draft arm's bytes match the control's; it printed `output identical mtp-on vs mtp-off: NO (off ['aaaa'], on ['bbbb'])` and reached `return 0` (:372), so `python3 benchmark/tinytitan_mtp_phases.py … && echo "mtp is identical"` printed its sentence over the run that proved the opposite. Its near copy `tinytitan_mtp_b3_qualification.py` computes the same expression (:121 against :360) and returns `0 if identical else 1`, so it carried the identity claim and not the gate it also prints: `DELTA: +1.00%   gate +10% at p>=0.65: FAIL` (:131-136) exited 0, and so did OUT OF DOMAIN. `tinytitan_ane_prefill_ab.py` printed `SPEEDUP: 1.25x (gate is >=1.5x: FAIL)` and the GPU fallback WARNING (:221-223, :234-236) and returned 0 (:244). All three closed the sweep with `median([r[k] for r in sel if k in r])` (:308, :112, :206), so an on arm run whose server logged no footer still enters the rate median beside the runs that engaged while the acceptance median uses the one survivor and no count prints anywhere (measured: `acceptance 90.0%` over 1 of 2 runs, exit 0), and when no run carries the key the comprehension is empty and the driver dies after its headline lines with `StatisticsError: no median for empty data`. The fourth shape is all three again: two arms that streamed nothing hash the same, so the digest of an empty answer certified `output identical: YES` at exit 0, which is the case `tinytitan_determinism_ab.py:156-165` names in its own docstring and refuses. Measured by patching each driver's `one_run`, `ROOT` and preflight and calling its real `main()`, so no model, no server and no socket was involved and every status quoted is one a driver computed. Each driver now has a module level `verdict(rows)` that returns the page and the status, the shape `tinytitan_determinism_ab.py:137` and `tinytitan_slots_ab.py:133` already use, on the three statuses this tree already uses (`tools/model-guard.sh`, `tools/qwen35_reference.py:411`): 0 both arms measured and every claim holds, 1 it measured and a claim failed, 2 it could not be compared; `main()` prints it, records it beside the rows and returns it. The counting moved into four primitives in `tinytitan_profile.py`, which all three already import, so a fourth driver cannot reintroduce a private `med()`: `arm_metric` answers the median with its count and returns None instead of raising, `metric_count` turns a short denominator into a PARTIAL or NOT MEASURED line, `arm_answered` requires content before an identity claim means anything, and `byte_claim` holds the digest comparison the two MTP drivers had duplicated. `benchmark/test_mtp_phases_verdict.py` (32 tests, registered in CI's model-free gate group, which this commit re-measures at 37 suites and 739 tests in 35.6 s) went in first and ran RED with 3 failures and 20 errors; the page that was already honest is asserted unchanged, and two controls pin both directions (a sweep that holds still answers 0, a divergence still answers 1). Twenty seven mutants, all killed. Three survived the first sweep, and each is the reason a test exists rather than evidence about the code: an arm that disagrees with itself while agreeing set for set with the other arm, the attribution table hiding that it medians one of two runs, and a prefill time of 0.0 s being divided into instead of refused; each was proven RED against its mutant and GREEN against the fixed file, and no assertion was weakened. Two decisions belong to the operator, not this row: a fallback ANE arm is treated here as a claim that failed (status 1) while `ane_prefill_ab_matrix.py` refuses such a ratio outright as unmeasurable, so the two files read one run differently and only one of them can be the house rule; and whether a benchmark that simply lost its published gate (ANE under 1.5x, MTP under ten percent) should exit non-zero at all is the question every gate in this family answers. The sibling sweep, every filtered median left in `benchmark/`, found four more drivers that median a subset without carrying its count (`ane_steady_state_decode.py:153`, `expert_cache_slots.py:223`, `tinytitan_gate0_profile.py:438`, `ane_gather_probe.py:299`), the first two also returning 0 whatever the arms did; those are greps and not measurements, so they are the next item on the queue rather than a claim made here.  **AUD-274 (closed 2026-10-10 at `0374436` and `09fde61`) is that queue item measured**: the four drivers AUD-273's sibling sweep named as greps — `ane_steady_state_decode.py`, `expert_cache_slots.py`, `tinytitan_gate0_profile.py`, `ane_gather_probe.py` — each driven through its own real `main()` with its run function stubbed, so no model, server, build, Core ML program or download was involved. Thirteen shapes, each ending in a printed answer and exit 0 or a traceback: eight failed runs printed `prompt ? tokens` eight times plus two `steady state unavailable` lines, so an arm that produced nothing read as a property of the model; a run set where every run stopped early still printed a prefill speedup built from rows the driver itself marked `[INCOMPLETE]`; a slot that produced nothing was simply absent from the published table behind `if not group: continue`; a group whose runs logged no IO line died with `StatisticsError: no median for empty data`; a run whose IO line parsed without a per-token figure died mid-sweep with `TypeError: unsupported format string passed to NoneType.__format__`, so the later slots never ran. The one worth naming is gate0, whose whole output is which optimization track to fund, decided by `busy >= 45` / `busy <= 35` over `m["busy_per_token_ms"] or 0`: a run set that never logged the metric printed `GPU busy/token 0.000 ms (0.0% of token)` and `VERDICT: DEPENDENCY-STALLED (Track B is the game)` and exited 0 — a track funded by the absence of a measurement. `--reps 0`, `--rounds 0` and `--repeats 0` reached an `IndexError` after the header or a `statistics.median([])` that landed in `ane_error` as `StatisticsError:` — an argument test mislabelled as a Core ML failure — and gather's ratio block sat behind four truthy lookups, so an arm that errored skipped it with no line at all: the same page shape as a probe that had answered its question. Each page is now a module-level `verdict(...) -> (lines, status)` that `main()` returns and its artifact records, counting through the primitives AUD-273 added rather than a private helper per driver, and a metric that was never logged is now a different answer from a measured zero: gate0's track rule refuses (status 2) instead of classifying from `or 0`, the slot table prints one row per requested slot with `no data` in every cell it has no figure for and names each uncomputable adjacent-pair comparison, and gather distinguishes an arm that logged no figure from a dense arm that logged `0.0 s` (`REFUSED: the dense arm logged a 0 s ANE figure, so the gather/dense ratio would divide by zero`). `arm_metric` itself was wrong underneath all of it — it counted a key present with a `None` value, so it returned a `None` median with a count above zero and every caller that formatted it died. `benchmark/test_sweep_driver_verdicts.py` (33 tests, in CI's model-free group, which this close re-measures at 39 suites and 802 tests in 34.5 s) went in first and measured RED at 19 failures and 11 errors against a clean `git archive` of the pre-fix tree — eleven `AssertionError: 0 != 2`, six `0 != 1`, two `'sweep status 0' not found`, and the drivers' own crashes — and the three that already passed are the three controls asserting that an honest page prints exactly what it printed. Twelve mutants, eleven killed: the survivor installs `for slot_count in [s for s in slot_list if any(r['slots'] == s for r in rows)]`, which selects every requested slot because the sweep records a row per planned run whether or not that run answered, so it is an equivalent mutant, and the faithful version of that shape (a survivor denominator reported as the whole) is killed by two tests. The other survivor was evidence about the tests: rewriting the gather refusal's own `NOT MEASURED` prefix survived because the assertion behind it read `assertIn("gather", out)`, which the per-arm note beside it already satisfies, so the assertion now names the line it is about and that mutant dies on exactly that test. Two test-side moves are recorded rather than buried, and neither weakened an assertion: a first draft expected status 2 from a `PARTIAL` denominator, the driver answered 1, and the **test** was the wrong side — a short denominator is a contested claim, which is AUD-273's convention and its still-open operator question — so the expectation moved and the `over 1 of 2` string assertion stayed; and the slots fixture omitted `swap_delta_gib`, a column the table prints, so the driver correctly refused and the fixture gained the key. Two runner bugs each made a measurement lie before they were fixed: a mutation runner that read `r.stdout` while `unittest` writes its results to stderr reported every mutant a survivor with empty output, and a first RED run over `git archive` of `benchmark/` alone had seven slot tests die on a `FileNotFoundError` for `docs/qwen38-flash-next-port.md`, a file the driver genuinely reads, which is why the 19-and-11 is measured against the whole tree. Three things this row decides and puts up for ratification rather than hides: gate0's track VERDICT now *refuses* rather than classifying from an unlogged metric, which changes what an operator reading that page by hand sees; a zero repeat count is a page-level status 2 rather than an argparse refusal, because the page is the product these drivers ship; and these four print no published performance gate, so AUD-273's question — whether a benchmark that lost its own gate should exit non-zero at all — does not arise here and stays with the operator. The sweep then measured the next driver along the seam, `tinytitan_sampler_ab.py`: an absent `busy_per_token_ms` prints `busy/token 0.00 ms` at exit 0, `--pairs 0` dies with `StatisticsError: no median for empty data`, two arms whose requests returned `completion_tokens: 0` still print `Output identical across every run of both arms: YES` at exit 0, and the page prints `DELTA: +0.00% (gate is +10%: FAIL)` while its status reads only `output_identical` — filed as AUD-275.**AUD-275 (closed 2026-10-10 at `61783f9` and `81344b4`) is the fifth driver along that seam**, and the one its neighbours' rows had already named: `tinytitan_sampler_ab.py` decides whether the tiled Top-K sampler path ships, and pre-fix its status read one thing -- `return 0 if out["output_identical"] else 1` -- so `DELTA: +0.00% (gate is +10%: FAIL)` printed and exited 0, `--pairs 0` and `--pairs -1` died with `StatisticsError: no median for empty data` before any page, a generic arm at 0 tok/s died with `ZeroDivisionError` at the delta line, one `busy_per_token_ms: null` killed the sweep in its first progress line with a `TypeError` after two runs of four, two arms whose requests streamed zero tokens printed `Output identical across every run of both arms: YES` at exit 0, and `row.get(key, 0)` meant a metric no run logged published as a measured `0.00` while a metric half the runs logged published the zero-padded median -- `busy/token 25.00 ms` for an arm whose two answering runs measured 50.00. It is now `summarize` + `verdict -> (lines, status)` on the three statuses this tree uses, counts and status in the JSON artifact beside the medians, `not logged` in a cell no run answered, `PARTIAL: ... over 2 of 4 runs` for a short denominator, `NOT MEASURED` for an unearned byte claim, a refusal page for `--pairs < 1` *before* preflight or any server launch, and the printed +10% gate now worth status 1 -- following `tinytitan_mtp_b3_qualification.py:125`, which already treats its own failed gate that way. `benchmark/test_sampler_ab_verdict.py` (22 tests) went in first, RED at 4 failures and 17 errors with one test passing (the control that `byte_claim`'s default key already worked), and fourteen of the fifteen drafted mutants were applied on the committed tree and all fourteen killed. The fifteenth matters more than the sweep: re-pointed at the code it meant, it *survived*, and so did its twin -- because the artifact's `rates` and `digests` filters drop runs that never logged those keys, and no run can: `parse_log` builds a record only from a regex match whose third group *is* the rate, and `generate()` returns a 16-hex-char digest even for an empty stream. Both filters were deleted rather than tested, which is the always-logged/optionally-logged line this whole seam turns on, and it is why the same file's bare `record['decode_tok_s']` in a progress line is not a defect. Three things are named here because they were wrong before they were right: an earlier run of this sweep reported 15/15 killed when its own log said 14 applied and 1 skipped; the probe's shape G was handed shape A's fixture and so proved nothing about the rate key until it was re-driven; and the first test harness handed rows out by call index in a driver that alternates arms, which is the harness, not the code, lying. Its sibling sweep found the `row.get(key, 0)` progress line still live in five neighbours -- `tinytitan_mtp_phases.py:421,427` prints `acc 0.0% passes 0` and `tinytitan_ane_prefill_ab.py:305,306` prints `prefill 0.00 s`, both measured, both with the *page* refusing correctly at 2 since AUD-273; `tinytitan_mtp_b3_qualification.py:204,207`, `tinytitan_gate0_profile.py:430,431` and `ane_prefill_ab_matrix.py:229` read the same way and are not yet driven. That is AUD-276, and its severity has to be measured rather than assumed: a live line that misreads while the verdict refuses is a smaller claim than the one this row just closed. **The 1 open was still AUD-259 at this point in the row** (it has since closed at `cc7ef48`, and its sibling-sweep finding AUD-310 closed at `92a48c6`), and the operator's decisions this row added were whether a benchmark that loses its own printed gate should exit non-zero at all, and whether `--pairs < 1` belongs to argparse or to the page. **AUD-276 (closed 2026-10-10 at `1ce890f`)** is that sibling lead, and it came out larger than the note predicted on one site and smaller on four: all four live lines do print a figure no run made (`acc 0.0% passes 0`, `prefill 0.00 s`, `busy_per_token=0.000 ms, occupancy=0.0%`) while the page refuses correctly at 2, but `ane_prefill_ab_matrix.py:229` puts `0.000` in a **published table** for every shape where `summarize()` refuses to compute a ratio -- a row that says nothing in the column the >=1.5x gate is decided by, beside `-` in the columns it also refuses to fill. The fix is one primitive, `tinytitan_profile.logged(value, spec, suffix)`, which is what gate0's private `_cell` and the sampler's private `_metric` already did for their own tables; both are gone and their call sites are byte-identical, so only the lines that were lying changed. `benchmark/test_logged_metric_cells.py` (21 tests) went in first and was measured RED on a `git archive` of the pre-fix tree -- 10 failures and 4 errors, with seven controls passing on both trees by design -- then GREEN at 21. Eighteen mutants were drafted and, unlike the last sweep, all eighteen applied: seventeen killed, and one survivor that is a proof rather than a gap (`"speedup" in r` and `if r.get("speedup")` coincide because `summarize()` writes the key only under `if off and on`, so no stored ratio is ever 0). Two mutants were re-attempted because the first log mis-recorded them: M13 genuinely SURVIVED until `Gate0TableCells` was added -- gate0's table cell had no test anywhere in the tree -- and M18's `ANCHOR NOT FOUND -- SKIP` meant it had never been applied, not that the code was untested. The false lead is recorded too: the matrix's `OFF arm reported an ANE fallback (unexpected)` note is correct, because `run_arm:128` makes the off arm report `used_ane: True` in every real run, so only a hand-built fixture fires it. Fifteen `.get(key, 0)` print sites remain in seven non-test `benchmark/` files, counted with `grep` and deliberately not claimed as defects without driving their producers; the one that needs no fixture is **AUD-277** -- `tinytitan_knob_sweep.py:267-271` publishes five optional counters and `:305-307` repeats three on the live line, while `main()` still ends on `return 0` at `:323`, so it is the pre-AUD-273 driver and the pre-AUD-276 line at once.  **AUD-277 (closed 2026-10-10 at `cac1877`)** is that driver measured, and it carries three of this audit's root causes in one file: `main()` ended on `return 0` at `:323` even though its docstring states the noise-floor rule twice, the five optional counters published through `.get(key, 0)` at `:267-271` and three more on the live line at `:305-307`, and `MAX_TOKENS = int(os.environ.get("SWEEP_TOKENS", "256"))` sat at module scope. Driven through the real `main()` with its `run_arm`, `pgrep_answer` and `ARMS` patched -- no model, no server, no socket, nothing fetched -- every shape below ended at exit 0: two arms that both failed; three measured runs whose servers logged no counter line, published as `hit% 0.0 io_hid% 0.0 io_ms 0.0 evict 0 waits 0` and a live `hit=0.0%`, which reads as "the cache never hit" for a cache that was never asked; `--arms bse` running zero arms and printing nothing; `--arms base,bse` running the one real arm and discarding the typo silently; a failed `base` leaving every `vs base` cell `--` -- correct on its own -- while the sweep claimed a pass over a comparison it had not made; a failed `base_again` leaving the drift line absent from the report entirely; and the shape the docstring exists for, `base_again` at +30.0% against a claimed +3.0% win, printing `baseline drift over the sweep: 30.0% (4.00 -> 5.20)` and `Any win smaller than this is inside the noise and is not a win.` without comparing that sentence to a single win. The import-time read is a fourth family: measured on a `git archive` of the pre-fix tree, `SWEEP_TOKENS=0` and `SWEEP_TOKENS=-5` import at rc 0 and are accepted, so a "measured" request asks for no tokens at all, while `SWEEP_TOKENS=abc` dies with `ValueError: invalid literal for int() with base 10: 'abc'` before the guard answers and before any refusal line can print. `benchmark/test_knob_sweep_verdict.py` (36 tests) went in first and ran RED against the unmodified driver -- 2 failures, the fabricated cells, and 29 errors naming the missing `verdict`, `measured_tokens`, `selected_arms` and `logged(refusal=...)`, plus 8 from the pre-fix `run_arm(name, delta)` taking no measured length -- then GREEN, and the boundary test was proven RED against a production line this row wrote: "every win claimed is larger than the drift" is false at equality, so the wording moved to `smallest win claimed +5.0% on sync_event, against 5.0% of drift`. The answer is now `verdict(results, selected)` returning page and status on the four statuses the family uses -- 0 every published cell measured and the drift bounding the wins claimed, 1 measured with the contested claim named, 2 the headline uncomputable with its reason, 3 still the guard's own -- with `measured_tokens` and `selected_arms` refusing by name before the guard runs, the table refusing with the `--` its own delta column already uses while the live line keeps the words, and the drift finally compared to the smallest win the page makes. Thirty mutants, and **the first run's figure is withdrawn**: it reported 30 applied / 28 killed / 2 survived, but M12 reproduced as a kill when run alone, because two mutants that leave the file the same size within the same second let CPython reuse the previous mutant's `.pyc`, so the run labelled M12 was executing M11's code. Re-run with `PYTHONDONTWRITEBYTECODE=1` it was 30 / 29 / 1 with M11 genuinely surviving -- `assertIn("no baseline", out)` is satisfied by two different refusal messages, which is evidence about the test -- so that assertion was **strengthened** to name the exact cause and the final sweep is 30 applied / 30 killed / 0 survived / 0 skipped. That strengthening is a test-side move awaiting ratification, and this is the audit's third false mutation figure, which is why every sweep now runs with bytecode writing disabled. CI registered the suite inside the fix commit, deliberately the reverse of AUD-276's order so AUD-211's gate is green at every commit rather than red between two: the 42-suite model-free group is 882 tests, OK, exit 0, measured five times between 35.9 s and 39.1 s reported at load 4.5 to 5.5, with the operator's server on 8080 and untouched; the whole `benchmark/` python tree is 1445 tests OK (skipped=27) in 291.4 s; and all eighteen gates are rc 0. Two things are left on purpose: `run_arm` still raises a bare `ValueError` when the release binary has not been built, because an existence check in `main()` would make this model-free suite fail in a fresh clone where `.build` legitimately holds no binary, and a traceback is loud rather than a false pass; and whether a sweep whose drift control never ran is status 1 (implemented -- a win nothing bounds is a contested claim) or 2 is the operator's call. The nine remaining module-scope `int(os.environ...)` sites and ten remaining zero-defaulted print cells were counted with grep and not driven, so none of them is claimed as a defect -- and one stale cite here was re-derived by this close: the module-scope read list below named `tinytitan_knob_sweep.py:35,39`, and after AUD-277 only `:52` survives in that file.**AUD-281 (closed 2026-10-10 at `0ca5387`) is the ninth driver along that seam and the first whose figure rule was the defect rather than the status**: `tinytitan_vs_competitors.py` publishes the claim *faster than any other similar project*, and it read none of its numbers as numbers -- the TinyTitan footer came back through `"" if rate else "no decode footer"`, a truthiness test on a float, so a footer printing `decode_tok_s=nan` was returned as a measurement and printed in the table, one printing `0.00` was reported as a footer that never existed and the note then discarded, `decode_tok_s=abc` raised ValueError and `decode_tok_s=` raised IndexError, llama.cpp/MLX/MLC raised on a `[\d.]+` token such as `..`, Ollama raised ZeroDivisionError on `eval_duration=0` and TypeError when the key was absent, and LM Studio raised on a string rate; `main()` ended on `return 0` whatever it measured, so an all-failed sweep and an all-skipped sweep both printed a table of dashes and exited 0. One finite-or-refused rule now covers all five engines, ours included, because one engine refusing a figure while another publishes it as a rate is the asymmetry that file's own header warns about; a measured zero stays a measurement and no longer also claims a missing footer, and `sweep_verdict` answers 0 / 1 / 2 with the failed engines, the unready engines and the no-engine-selected case each named. 31 tests, 16 of 27 red first, then a 37-mutant anchored sweep with 0 survivors and a byte-identical restore; two of the 37 kills are crash-kills (a refused cell formatted with `:.2f` raises TypeError inside `main()`), recorded as reachability evidence rather than assertion coverage. Two decisions are left on purpose: whether a measured `0.00 tok/s` should be refused as an unusable instrument the way AUD-280 refuses one -- it is not, because this driver discards the streamed completion and so has no token count to prove the run wasn't empty, and refusing a bare zero would invent a rule the evidence cannot support; and whether the head-to-head claim should be gated on a *comparison* (a sweep that measured one engine exits 1 today, which is the house table, not a threshold on TinyTitan's rate). The sibling sweep measured four more seams and fixed none of them: `expert_cache_slots.py:70` uses `float("nan")` as its absent sentinel while `_num` at `:149` refuses only `None`, `tinytitan_profile.py:391` `arm_metric` filters `is not None` without testing finiteness, `tinytitan_mtp_b3_qualification.py:67,106` test `off_rate` for truthiness, and `tinytitan_ane_prefill_ab.py:223` with `ane_steady_state_decode.py:230` each divide by an ANE median that can be zero -- each is a candidate row for the next sweep, with the lines measured and cited in the ledger row's sibling note. **AUD-282 (closed 2026-10-10 at `f0f37ad`) is the first of those four leads measured, and it was a live published defect rather than a reachability question**: `expert_cache_slots.py` answers TT-011, whose whole question is the memory column, and its `swap_used_gib()` answered `float("nan")` when `sysctl vm.swapusage` returned nothing it could read. Every guard that figure passes through tests for `None` -- the table cell and `arm_metric` behind it -- so the sentinel sailed through all of them and was published as a measurement: measured model-free, the startup line printed `swap now nan GiB`, each per-run line printed `swap   +nan GiB`, the published table printed `+nan` in the swap column for every slot, no refusal line appeared, the sweep exited **0**, and the JSON record wrote the bare literal `NaN`, which is not JSON. The fix adopts the `None` this tree already treats as absent, which is one line at the source rather than a new guard at each print: the `metric_count` guard wired to that column then answers `PARTIAL` for a slot whose runs split and status 1, and `NOT MEASURED` with status 2 when no run of the sweep answered at all, and the record carries `null`. The per-run cell prints `swap no data GiB` because a cell must match the column it is in -- the same shape the driver's existing `io no data MiB/tok` uses. `benchmark/test_sweep_driver_verdicts.py` grew 13 tests over the real `run_once` and the real `main()` with only the module's own subprocess call patched, so `sysctl` and the CLI are answered separately and the parse, the table and the exit status all run for real; 9 went in first and measured 5 RED against the pre-fix tree with 4 controls green by design. The first 17-mutant sweep left **7 survivors**, and each named an assertion the suite did not have rather than a mutant to delete: the exact table row for an unread swap column, the startup line's own refusal (the shared `logged()` helper in `tinytitan_profile.py` returns the refusal *without* its suffix, so an assertion written as `nan GiB` was testing nothing), a reading that carries the megabytes without the unit the divisor assumes, and a reading that fails on one side of a run only. Four tests and one added assertion later the sweep is 17 killed, 0 survivors, 7 of them crash-kills recorded as reachability evidence, and the restore proved byte-identical from a pristine copy re-taken after the formatter ran -- the first sweep's figures were measured against the pre-format file and are superseded, the 46-test suite is OK and the CI step that names it re-measures at 1011 tests OK in 34.6 s -- an earlier 321 for that group is withdrawn, it came from a hand-typed subset of suites and not from the workflow's list, which is why the row cites the command rather than a count. The fix commit's own message says five tests went in; the file says four plus an assertion added inside an existing test, so the number is corrected in the ledger row and not carried forward. That row's sibling sweep re-measured the other three leads and **withdrew two claims this audit had made**: `arm_metric`'s missing finiteness test is unreachable, because all 14 call sites take their figures from regex groups written `[0-9.]+` or `\d+`, which cannot spell `nan` or `inf` -- a non-numeric footer leaves the key absent, which is already refused -- and `tinytitan_longctx.py:298`'s `float("nan")` is dead code behind a refusal at `:291`; both are cleared, not fixed. And the "ZeroDivisionError over a zero ANE median" that AUD-281's row asserted is **false**: both drivers guard the divide, so AUD-281's ledger row has been corrected and what is left is worse than a crash -- `ane_prefill_ab.py:211` reads a measured `0.00 s` as a missing one and refuses with "a prefill time is missing", while `ane_steady_state_decode.py:227` and `:240` silently drop the speedup and gap lines with no reason and no status. (The `:240` half of that clause is false and AUD-283's row corrects it: `arm_steady_state` cannot return a zero, so that guard only ever fired on a value the page had already refused over.) That conflation, with `mtp_b3_qualification.py:67,106`, is filed as **AUD-283** and is a conflation and a silent omission, not a crash. **AUD-283 (closed 2026-10-10 at `e1fddc1`) is that conflation in the drivers AUD-282 named, measured in five shapes against HEAD and fixed in the four it could test.** A truthiness test on a figure the footer logged as 0 did two different damage: it refused with a cause that had not happened, and it deleted a published line while the status stayed clean. `ane_steady_state_decode.py:227` sat the whole prefill line behind `if prefill['gpu'] and prefill['ane']`, so rows whose ANE `prefill_s` is 0.0 printed neither arm's median and the page read `sweep status 0`, because `metric_count` had counted that zero as measured and raised no note; its `:262` did the same to a window's comparison line, and on the ANE side that was a real result dropped rather than published -- that arm 100 percent slower over the window. `tinytitan_mtp_b3_qualification.py:106` folded `delta` into its `None in (...)` tuple, and `delta` is None exactly when the control arm's rate is zero, so a scalar arm logging `tok/s=0.00` refused as `an arm carried no rate` and printed neither median. That driver's runs listing read `r['decode_tok_s']`, so an arm whose median was measured over part of its rows -- the shape `metric_count` exists to report as `over 1 of 2` -- raised `KeyError` out of the print line after its header had gone to stdout. `ane_prefill_ab_matrix.py:205` left no speedup and no note for an ANE median of 0 s, so the published row showed both medians beside a bare dash, and its `:201` skipped the under-one-chunk note for a prompt that logged `0tok` while still publishing a speedup for a prompt the ANE cannot serve. The fix separates the two cases by which side of the divide the figure sits on: a zero divisor refuses with the zero named (`the ANE arm's prefill median is 0 s`, `the GPU arm's 64-token window is 0 tok/s`, `the off arm's median is 0 tok/s, so no delta is computable` -- the last being the wording `tinytitan_mtp_phases.py:267` already used, so the siblings now agree rather than one of them inventing the rule), a zero numerator is published as the figure it is (`speedup 0.00x`, a 0.0 percent acceptance printing and refusing on its own out-of-domain grounds), a refusal no longer hides the cells it is about, and the runs listing reads with `r.get(...)` like its sibling at `tinytitan_ane_prefill_ab.py:216`. No new status vocabulary: each refusal reuses the NOT MEASURED token the driver's own status function already turns into a 2. Twelve tests went in first across three files -- 9 RED against HEAD, 3 controls green so the fix could not pass by deleting a claim (a measured sweep still prints `speedup 2.00x`, a zero numerator is still a computable ratio, and the sibling that names its zero keeps doing so). The 20-mutant anchored sweep is 12 kills and 8 crash-kills, five of those the driver itself raising and three the test's own KeyError on a field the mutant stopped writing, which pins the field rather than the arithmetic; the first sweep's 2 survivors were both assertion weaknesses and were closed by strengthening the assertion, not the mutant -- one quoted a zero without naming the arm, and one quoted `acceptance 0.0%`, which the refusal sentence also contains, so it now quotes the cell. The three suites run 105 tests, the CI verdict-group step re-measures at 1023 tests OK in 33.96 s (1011 in AUD-282's row plus these 12), and the pinned python gate is clean. **AUD-284 is filed OPEN from this row and is the only site it did not fix**: `tinytitan_ane_prefill_ab.py:210` has the same guard, and two tests were written for it, measured RED (the page answering `a prefill time is missing` over a 0.00 s median with neither median printed), and removed rather than committed red, because `test_a_zero_prefill_time_is_a_refusal_not_a_division` at `test_mtp_phases_verdict.py:484` asserts exactly that wording -- its own comment knows the figure is a logged zero, and its assertion line pins the absence sentence. The standing instruction here is that a test is not rewritten to reach green: *if the tests were wrong, say so before changing them and wait for my decision*. So the proposed two-line driver split and the one-line assertion change are written into AUD-284's row with the diff quoted, and the status 2 and the no-SPEEDUP half of that test are preserved either way. **Two operator decisions come out of this row**: whether a measured zero should refuse as NOT MEASURED (house precedent at `tinytitan_mtp_phases.py:268`, and what was implemented) or as a contradicted claim at 1; and which side moves on AUD-284. Also left by choice: `tinytitan_benchmark.py:451`'s `and decode_rate` excludes a zero that can only mean `completion_tokens == 0`, which is that driver's stated premise for keeping a row out of the warm-rate mean -- what it does wrong is print the same `n/a` its filtered case prints, a wording question and not a change to make unprompted. `ane_prefill_ab_matrix` still ends on `return 0` with no verdict of its own, which is the AUD-274 seam and needs its own row. The **3 open** are AUD-259 and AUD-284, neither unfixed for lack of a route -- AUD-259 needs a performance decision and AUD-284 needs a test moved, both the operator's to make -- plus AUD-293, filed by AUD-292's close: not an unfixed defect but the status-numbering call that fix deliberately left, the operator's to make. The label was one behind the ledger for one commit: the row filed after those three made four open while this sentence still read **3**, because `tools/lint.sh docs` compares the `counts` string in this table's Audit cell against `ledger.json` and nothing here checks a prose count -- so this close is what makes the number true again, and any future close that moves the open count has to re-read this sentence too.   **AUD-285 (closed 2026-10-10 at `36e27a1`) is the same seam in the driver that sits beside it, found by reading the six `benchmark/` drivers the ledger had never named rather than following the notes.** The pread-vs-Metal I/O A/B imports the hit/fixup driver for its protocol and then answers its own question with a raw comparison of the two arms' text, which is what AUD-280 removed from the parent -- and AUD-280's committed note says that comparison is unique to the parent, which this row disproves. Measured before filing, through the sibling's real `main()` with only `preflight`, `run_case` and `ROOT` patched: two backends answering nothing was `passed: true` at exit 0; the same words at 128 and 96 completion tokens was `passed: true`; footers reading `decode_tok_s=nan` was `passed: true`, because the driver takes `content` and never a value out of the footer `run_case` attached; the page printed nothing but its own output path; and a refused preflight escaped as a `RuntimeError` traceback at exit 1, the status a measured disagreement carries, with no line naming the cases that had already run. The fix threads the arm identity through the classifier the audit already has -- `arm_key` and `arms` on `_cases` and `verdict`, the parent's own `mode`/`MODES` as defaults, so the barrier-vs-hit/fixup page is unchanged -- rather than writing a second verdict beside the first: the sibling now gets the five states, the per-case measured cells, the three-status answer and `refuse()`, and its `passed` is derived from the status instead of from the absence of a text disagreement. 11 tests, 9 RED against HEAD and 2 controls green; 13 anchored mutants, 5 killed by assertions and 8 crash-kills recorded as reachability, 0 survivors, both drivers restored byte-identical. The first sweep's single survivor was `a dead arm escapes as a traceback`, and it survived because no test drove a case that died after others had run -- so a test went in and the mutant stayed. Two harness defects are recorded rather than hidden: `parse_args` was reading unittest's argv, and a fixture keyword that named no arm overrode nothing and now raises instead. 40 tests OK across the two suites; the CI verdict group, read from the workflow's own list, is 47 suites and 1035 tests OK, and it first failed on `test_docs_facts_suites` because this suite was named in CI a moment before it was tracked -- AUD-211's rule catching its own author. **AUD-286 (closed 2026-10-10 at `4daccf7`) is the same sweep's third caller**: `tinytitan_v42_control_plane_ab.py:73` compares every case's text against the first case's across a six-way matrix, parses no footer, has no refusal path and writes its artifact whatever happened -- measured at `passed: true` and exit 0 over 36 rows whose footers read nan. It was a bug and not a decision, filed open only because the turn budget reached AUD-285's close first. Fixed the way AUD-285 was: `benchmark.verdict` and `benchmark._cases` run once per reference-and-case pair through `arm_key` and `arms`, the statuses pool so that one pair reading nan contests the matrix instead of erasing the pairs that measured, and a refused preflight, a case that died mid-sweep and a single selected case all answer NOT MEASURED at 2 through `benchmark.refuse` and write no artifact. 14 model-free tests, 11 of them RED at HEAD. Two assertions I wrote were wrong and it is said here rather than fixed quietly: the page test read the reference's first pair instead of the pair it named, and the text test asserted a concatenated string the page never emits -- the tests moved where the driver was right. A guard the first draft had was deleted, not exempted, because the sweep showed it unobservable. 16 single-hit mutants and no survivor, 1049 tests OK across the CI verdict group's 48 suites, and the full gate run returned 0. Cleared on the way: the sibling's direct `by_case[...]` index is not the KeyError it looks like, because `run_case` refuses a log whose footer count differs from its row count. Of the six drivers this sweep opened on, `side_engine_recall.py` and `watchdog_calibrate.py` are still unread; `master_scenarios.py` is read and is where AUD-287 was found. **AUD-287 (closed 2026-10-10 at `19383bf`) is the first finding from the driver this sweep left unread.** `master_scenarios.hit()` matched containment in both directions, so a probe table against HEAD read `hit('crimson', '')` True, `hit('crimson', '   ')` True, `hit('crimson-blue', 'crimson')` True and `hit('', anything)` True, and `memory_master.py:261` sums exactly that function into the per-session recall score while :271 uses it to decide whether an answer changed -- a blank quiz field was a retrieval, and two blanks were stability. The answer-inside-the-expected half is deleted; the expected-inside-the-answer half stays, because `MatchingTests.test_strings_match_as_a_phrase` pins it by name as intended leniency and moving a pinned test is a decision, not a fix. Three tests were RED at HEAD and two controls held. The sweep's one survivor was my own test, not the code: `hit('none', None)` cannot collide because `str(None)` is capitalised, so the guard's test was rewritten to `hit('on', None)` and `hit('ne', None)` before that mutant died -- 5 mutants, no survivor, source restored byte-identical. 22 tests OK in the owner suite, the CI verdict group is 48 suites and 1053 tests OK, and the structural check still answers 0 over 10 scenarios, 70 sessions and 41 carryable keys, which is also the proof that no scenario's ground truth is a blank string. `memory_volume.matches`, `memory_sim`'s needles and the `normalise` helpers were probed for the same hole and are clear. Two questions are the operator's, because both change what a score means rather than fixing a false one: whether `hit('crimson', 'crimson-blue')` should remain a phrase match at all, and whether the numeric branch should keep accepting any number inside a sentence. **AUD-288 (closed 2026-10-10 at `498795a`) is `benchmark/side_engine_recall.py`, the second of the two drivers AUD-287's row left unread, and its defect was silence read as an answer. Every judgement came from `(row.get('completion') or '').strip().upper().startswith('YES')`, so a blank or absent completion was a NO: measured against HEAD, an engine that never ran and an engine that disagreed printed the same table, and because `t7_rank` is built over the whole `BIBLE` a fact with no judged row was ranked last by construction rather than left unranked, which scores a group that judged 3 of 10 facts as though it had judged all 10 and got 7 wrong. The rates then divided by however many questions produced rows -- AUD-233's denominator-is-its-survivors shape, in the one comparison the memory docs publish: an empty answer file printed `recall@1 0/0` twice and answered 0, and a path that does not exist escaped as a traceback at exit 1. `--score` now answers the house status table: 0 every fair question ran and every answer was there, 1 a named fair question ran no rows, left facts unanswered or carried silent rows, so the rates cover part of the matrix, 2 the comparison does not exist, reason printed and no rates. `judgement()` returns nothing for silence so it leaves the denominator instead of becoming a NO, and `exclusions()` names why a question left the score. The shipped leniency is preserved and pinned rather than quietly tightened: `strip().upper()` and `startswith('YES')` still accept lower-case, padded and rambling answers, because the recorded runs were produced with them. The module docstring's value weight is corrected 1 to 2, where it contradicted both its own code and `docs/side-engine-tasks.md:290`, and `deterministic_rank` now names the two terms the Swift scorer has and this mirror does not: tags, and `record.importance`, raised on top of the text score, which can reorder what the store returns. 17 tests in the new owner suite `benchmark/test_side_engine_recall_verdict.py`, tracked before CI named it as AUD-211 requires: 15 RED at HEAD and 2 controls pinning the preserved leniency. Sweep: 25 single-hit anchored mutants, 0 survivors, source restored byte-identical by md5, and the two survivors that appeared mid-sweep became tests rather than deleted mutants. The CLI statuses measured on fixtures are 0, 1, 1 and 2 for a full run, a question that did not run, a run with silent rows, and an empty file. `--baseline` is unchanged on what it publishes: `mechanical 10/10`, `paraphrased recall@1 3/4 recall@3 3/4`, `1 of 4 missed at rank 1`. The CI verdict group re-measures at 49 suites and 1070 tests OK (1053 in AUD-287's row plus these 17) and `tools/lint.sh all` returns 0. Cleared on the way: `side_engine_tasks.py:548-624` already refuses on no file, no rows and no completions and prints `--` for an absent variant arm, and `side_engine_judges.py:108-149` counts a row carrying an error as refused, excludes it from the denominator and prints the count. Two findings come out of this row's sibling sweep and are filed with it. **AUD-289** was the same seam in `benchmark/side_engine_wired_cases.py:147-168`, measured against HEAD and closed below: an empty judgement file prints the bare header and exits 0, because `failures = 0` over no rows is the pass status, and a missing path exits 1 with `FileNotFoundError` out of `read_text` instead of refusing. **AUD-290** was the figure this fix needed in order to check its own baseline, measured against HEAD and closed below: `--baseline` measures `recall@1 3/4` while `docs/side-engine-tasks.md:303`, `docs/clickhouse-long-session-verdict.md:50` and `docs/plan-fact-keeping-over-sessions.md:34` each still publish `1/4`, and `git show 878cea6 -- docs/side-engine-tasks.md` is an empty diff, so the number is stale rather than differently framed -- while `docs/agent-memory.md:174` and `docs/release-notes-v5.8.md:57` carry the corrected 3 of 4, which makes the tree publish both figures for one measurement. Neither the 4B nor the 9B is installed under `models/` here, so only the model-free token-match column is re-measurable and the side-engine columns stand as recorded on 2026-09-18; AGENTS.md bars installing a model to satisfy a document. `benchmark/watchdog_calibrate.py` is the last driver that sweep left unread, and it is checked for the same seam before AUD-289 closes.      **AUD-289 closes** its own row at fix commit `1ba59a1`. `side_engine_wired_cases.py --score` answered from whatever rows arrived rather than from the 12 cases the driver prepares, so an empty file exited 0 on `failures = 0` over an empty by-task map, a task that never ran printed `T5: 8/8`, a group that died mid-case printed `T5: 5/5`, a blank completion printed MISS and entered the denominator, a resumed run's duplicate row was judged twice, a foreign jsonl was scored against truths this file never authored, and a missing path, a half-written line or a row with no prompt each escaped as a traceback at exit 1 -- the status a contested comparison carries -- while a genuine disagreement answered 2, the code reserved for NOT MEASURED. `score()` now drives `cases()` and answers 0 every authored case ran and matched, 1 a named case ran nothing or carried no judgement or was repeated or came from elsewhere or disagreed, 2 NOT MEASURED with the reason printed and no totals. The first-word reading is byte-identical to what shipped, so `docs/side-engine-tasks.md`'s 4B `7/8` and 9B `8/8` re-measure the same way, and moving a miss from 2 to 1 contradicts no documented promise because that file names no exit code. Suite `benchmark/test_side_engine_wired_verdict.py` (17 tests, 15 RED at HEAD, `git add`ed before CI named it) is registered in CI; the verdict group is 50 suites and 1087 tests OK; the CLI statuses were measured end to end on fixtures as 0-1-1-1-2-2 with no traceback left; the sweep is 25 single-hit anchored mutants with 0 survivors and both files restored byte-identical; `bash tools/lint.sh all` is rc=0. **AUD-291 (closed 2026-10-10 at `1b25ef2`)** is what this row's sibling sweep then found in `benchmark/side_engine_judges.py:201-206`: `summarize()` returns an empty map over a done file with no rows, and the verdict `1 if any(entry refused) else 0` is False over nothing, so a judge that wrote no rows answers 0 while `line()` prints `measured nothing: every request refused, so none of the above is a score` -- naming refusals over a run in which zero requests ran; two answered rows out of the authored 60 give `T2 correct 2 of 2, refused 0` and also answer 0, because `len(jobs)` reaches `line()` for the s/judgement rate and never reaches the verdict; and a missing path (`FileNotFoundError`), a half-written line (`JSONDecodeError`), a row with no `task` (`KeyError`) and a row with a `task` but no `prompt` (`KeyError` out of `side_engine_tasks.py:167`) each raise out of `summarize()` unguarded, with `main()` reading the jobs file the same way. **AUD-291 closes** at fix commit `1b25ef2`: `refuse()` prints the reason and answers 2, `read_rows()` owns every unreadable shape -- a path that cannot be read, a line that is not json, a row naming no task/prompt/truth, a file with no rows -- and `main()` refuses on each before it reaches a table; `tally()` counts a row carrying neither a completion nor an error as silence that enters no denominator, where a blank completion used to enter the count as a judgement; `line()` says `every request refused` only where requests refused and prints a rate only over judgements that exist; and `main()` prints `judged N of M jobs, R refused, S with no answer`, names the shortfall or the surplus whenever `len(rows) != len(jobs)`, and returns 1 when `judged != len(jobs)`. The answer reading (`truth_of`, `strip().upper()`, the first word, the `.,:;"'` strip) is byte-identical, so the recorded figures still mean what they meant. Tests-first: `benchmark/test_side_engine_judges_verdict.py` is 17 tests, 15 RED at HEAD (8 failures, 5 errors), and the 2 that pass at HEAD are labelled preservation pins in their docstrings rather than guards; the pre-existing `test_side_engine_judges.py` had no assertion touched. The sweep is 33 single-hit anchored mutants, 0 bad anchors; the first run left 7 survivors and every one strengthened a test -- the no-rows reason naming the job count, the refused note absent over a clean run, a silent run naming silence rather than refusals, the 0.7 both-halves bar with a task put under it, the shortfall clause quoted whole (a `-58` sign swap survives `58 of the jobs wrote no row`, because it is that string's own substring), and a half-written jobs file -- re-run: 0 survivors with all three files md5-identical to their pristine copies. `ruff format`/`ruff check` clean on all three, the CI verdict group is **51 suites / 1104 tests OK**, `bash tools/lint.sh all` is rc=0 and `docs` resolves every ledger commit reference. Two findings the fix itself surfaced, recorded here rather than rowed: `test_side_engine_judges.py` kept its `if __name__ == "__main__": unittest.main()` guard *above* `RefusalTests`, so running that file directly collected 4 of its 9 tests and reported OK over the five it never ran -- the guard moved to the end of the file in the fix commit (measured: 4 before, 9 after), and a suite whose entry point silently under-collects is the class to look for; and the docs gate's own count reads **86 of 90 python suites registered in CI**, the four that stay out being the coremltools-gated ANE suites beside `test_export_ane_prefill`, which is the standing coremltools item and not a new one. **AUD-292 (closed 2026-10-11 at `5ddc1a8`)** is what this row's sibling sweep found reading `benchmark/watchdog_calibrate.py`, the last driver AUD-286's sweep left unread: `main()` prints `zero false positives on N replies` and answers 0 with the stub rule's own row reading `no corpus` -- measured against HEAD at `1b25ef2` with `replies`, `exchanges` and `tool_conversations` patched to fixture corpora, a full corpus and one missing every exchange answer the same 0; the pingpong watchdog is not in `clean` at all, so neither a trip there nor an absent corpus can ever contest that 0; and its codes run the other way from the table the drivers AUD-273 through AUD-291 answer -- a measured trip is 2, which those drivers reserve for NOT MEASURED, and a corpus that does not exist is 1 with `no recorded replies under <path>` printed, which is the status for a measurement that contested itself, the same 1 `selftest()` answers over a missing fixture. What the driver does hold: a tripping reply is counted and named, and the exchange shortfall is printed honestly in its own row -- the printout is honest and the status is not. **AUD-292 closes** at fix commit `5ddc1a8`: `clean` now reads `not loop_hits and not stub_hits and not pingpong`, a rule whose corpus is not there is collected and printed as `calibration incomplete: <rule> ran on no corpus` and answers 1 rather than certifying the run, and a measured trip is returned ahead of that note -- the precedence this row's first GREEN got wrong and a fourth RED test caught, because on this machine's own recorded corpus the loop rule trips once and the pingpong corpus is absent, so an answer of 1 for the absence left `NOT clean` unsaid. `main()` answers 0 the three calibrated rules ran on their corpora and tripped nothing, 1 the calibration did not get to run -- no replies, or a rule that saw no corpus, naming which, 2 it ran and a watchdog fired; the codes are pinned, not renumbered. Suite `benchmark/test_watchdog_calibrate_verdict.py` (11 tests; 4 RED against `git show 5ddc1a8^:benchmark/watchdog_calibrate.py`, the 7 that pass there being 2 labelled preservation pins and 5 holding behaviour the driver already had) is `git add`ed before CI named it (AUD-211). It reads the printed table row by row (`row(text, 'stub')`), which is what killed the sweep's only survivor: `assertIn('no corpus', text)` was satisfied by the incomplete note beside the table, the third time this audit has found a substring assertion matching a different line of output. Sweep: 17 single-hit anchored mutants, 0 bad anchors, 0 survivors, both files md5-identical to their pristine copies. Real corpus run, no model and nothing fetched: exit 2, `stub 0 0.0% over 30 recorded exchanges`, `pingpong no corpus`, one named loop trip, and both the incomplete note and `NOT clean` printed -- before the fix the same corpus hid the missing pingpong corpus from the status entirely. `--selftest` exit 0 over `8 fixture cases, 0 mismatches`; `ruff format`/`ruff check` clean on both files; the CI verdict group is **52 suites / 1115 tests OK**; `bash tools/lint.sh all` is rc=0. **AUD-293 (OPEN)** is the numbering call that close leaves, now a row of its own rather than a clause in another's owner field: the calibration answers 2 for a trip it measured and 1 for a corpus that does not exist, while all thirteen `benchmark/` drivers that print a status-2 table row mean `the run produced no measurement` (`side_engine_judges.py:33`, `guard_source_rate.py`, `side_engine_recall.py`, `side_engine_wired_cases.py`, `tinytitan_hit_fixup_ab.py`, `tinytitan_io_backend_ab.py:20`, `tinytitan_knob_sweep.py`, `tinytitan_sampler_ab.py`, `tinytitan_v42_control_plane_ab.py`, `capital_of_paris_smartness.py`, `tinytitan_ane_prefill_ab.py`, `tinytitan_mtp_phases.py`, `tinytitan_mtp_b3_qualification.py`), and `quant_perplexity_ab.py:119` refuses with 2 for a missing engine binary without printing the table at all. Measured: 51 non-test drivers answer a status (23 via `raise SystemExit(main())`, 28 via `sys.exit(main())`) and nothing in `tools/`, `.github/`, `sources/`, `tests/` or `benchmark/` invokes the calibration or branches on its code -- so the collision is latent, both readings are pinned by tests, and the row carries the auditor's recommendation (join the table) with what each direction moves. It also names, without rowing, `docs/plan-watchdogs.md:232`'s `zero false positives on 999 replies` against the `237 recorded replies, 1.3 MB, from 11 runs` and `loop 1 0.4%` the same command prints on this tree: a snapshot of a corpus that has since rotated, and the same gate question AUD-290 left. **AUD-294 (closed 2026-10-11 at fix commit `d09aa7b`)** is the unnamed-harness sweep's second finding -- the first, `benchmark/combos.sh`, it measured and cleared as a three-line `exec` forwarder whose status is its driver's. `tools/ane_sidecars.sh` is the sidecar gate `docs/adding-a-model.md:128` tells an operator to run, and its own header promises it `exits non-zero if any model it was asked about is still without a verified sidecar, so it can gate an install`. Measured with `TINYTITAN_MODELS_DIR` at a temp tree and `TINYTITAN_COREML_PYTHON` at `/usr/bin/false`, so nothing exported, nothing verified and no install touched: an empty models directory prints `verified: 0` and exits **0**; a directory holding one subdirectory with no `manifest.json` does the same by a different route; a models directory that does not exist does the same and names nothing, because the failed glob is silent, so an absent tree reads to the operator as a family the script chose to skip; and `--verify-only` over the empty directory repeats it, which is the mode a check runs against installs it believes are there. What the script does hold, measured beside it: a *named* install that is not there prints the path and `not installed` and exits 1, an unknown option exits 2, a `--chunk` the runtime would not route is refused before any work, the export is deliberately unpiped so a `tail` cannot borrow its status, and a verification failure lands in `failed`. So the finding is the empty walk alone: the class of AUD-210, AUD-233 and AUD-240, the failure `tools/lint.sh`'s `unbounded-read` and `test-hollow` gates already refuse for themselves, and the shape `tools/install_models.sh:794-797` already guards with `return 2` and the reason on stderr. The repair is that guard, shipped with the tests: zero candidates after the collection names `$MODELS_DIR`, says that a candidate is a subdirectory holding a manifest.json and that a fresh clone has no `models/` at all, and exits 2 -- the code the script already answers for a usage refusal, so the three statuses stay its table (0 every candidate the walk found is verified, 1 one it was asked about is not, 2 it did not get to run) and AUD-293's numbering question is untouched. A directory whose installs are all *skipped* families keeps exit 0 with `verified: 0`: those were asked about, and the script says why it will not export them. Written tests-first, and the last RED run measures **7 of 15** failing on the unpatched script; two earlier RED runs, 11 of 14 and 8 of 14, were my fixture rather than the driver's -- a family spelled `qwen_3_5_dense` where the case is `qwen3_5_dense`, and an assertion that a verification failure prints on stdout where the script correctly writes stderr -- both named as the test being wrong before the test was touched. GREEN took three runs, the single failure each time being the usage pin, because the header grew five lines and `usage()` is a `sed` line range that had to grow with it; the suite is now 16 OK with that range pinned from both sides. The sibling defect the fixture walked into is fixed in the same commit: the family read was a bare command substitution, so an interpreter that could not read a manifest ended the whole run under `set -e` with status 1 and no word about which install or which file, which the new test names. Nineteen single-hit anchored mutants, `PYTHONDONTWRITEBYTECODE=1`: the first sweep left one survivor, `manifest: the unreadable one is counted but never named`, because the assertion meant to catch it matched a phrase the `FAILED :` summary line also prints -- the AUD-292 lesson, an `assertIn` satisfied by a line other than the one that names the cause -- and strengthened to assert the per-model line, the re-run against the committed driver kills 19 of 19 with 0 bad anchors and reports the driver and suite identical to the pristine copies (`git diff` on both empty afterwards). Gates: `tools/lint.sh` shell, shellcheck, python and docs each 0, then the eighteen-gate `all` 0; the CI verdict group with the new suite registered runs 53 suites / 1,130 tests OK; and on the real tree `--help` exits 0 with the new header's own last line, `--chunk 777` exits 2, and a named `nope_123` install that is not there exits 1 printing `models/nope_123`. NOT CHECKED: `--verify-only` over the two real installs under `models/`, because coremltools is not installed here and the operator's server is resident -- the stub interpreter is what the suite and CI exercise; nothing was fetched and no model was run. The sibling sweep cleared the glob-looping harnesses it reached -- `benchmark/combos.sh` (a three-line `exec` forwarder), `tools/lint.sh` (fails on its own empty walks, `:1455`), `tools/install_models.sh` (`:796` returns 2 over an empty catalogue, and `staging_paths()` prints `nothing left staged` on purpose at `:269`), `tools/release.sh:342`, `tools/install_tinytitan.sh:441`, `tools/build_library.sh:137-148` (dies with `no resource bundle in $BIN: the Metal kernels would not ship`, then checks the staged bundle carries `*.metal`), `tools/dsh_local.sh` (grep for a `for x in <glob>` loop finds no candidate-collecting loop at all) and the two demo builders at `examples/library-demo/build-dynamic.sh:65` and `build-static.sh:56`, which exit 2 before the loop when the directory holds no library and whose producer already guards -- so their zero-bundle shape is a hand-assembled build input, not a gate reporting a pass. Read since and cleared, each for a measured reason: `tools/tinytitan_models.sh` already carries this guard in both catalog builders (:458-462 and :395-397 test `${#TINYTITAN_CAT_ID[@]} == 0`, set `TINYTITAN_CATALOG_ERROR`, return 1), `tools/golden-baseline.sh` cannot iterate zero targets (argv-empty defaults the target to ornith-8 at :111, an unknown target sets `status=1` at :305-306), `benchmark/memval_run.sh` (this row's first draft cited it under `tools/`, where the file is not) does reach 0 arms from a whitespace-only `TINYTITAN_MEMVAL_ARMS` under /bin/bash 3.2 but ends on the report, which exits 1 on an empty results tree with `NOT MEASURED: no results in <dir>`, and `benchmark/memval_master.sh` counts scenarios and refuses when all exited 0 with nothing to pool. Re-enumerating every `tools/*.sh` and `benchmark/*.sh` against CI, every suite and the README leaves one file unnamed -- the `exec` forwarder cleared above -- so this sweep has no remainder. **AUD-290 closes** at fix commit `da8e430`: the three documents that still published the pre-IDF token-match figure now carry `3/4`, re-measured on HEAD with `--baseline` (`mechanical 10/10`, `paraphrased recall@1 3/4 recall@3 3/4`, `1 of 4 missed at rank 1`), and the two sentences that rested on three misses name the one miss that remains -- `rules/ferry`, printed `lexical=no`, still the case no ranking can reach and the side-engine hint still answers. `docs/agent-memory.md:174` and the v5.8 note already carried 3 of 4, so the tree states one figure for the measurement and a docs-wide grep for the stale shapes returns nothing. The side-engine columns stand as recorded 2026-09-18 and the documents now say why: no 4B or 9B is installed here and AGENTS.md bars fetching a model to satisfy a document. What this row leaves sitting with the operator is the gate question it raised -- `tools/docs-facts.py` can pin the documented token-match row against `--baseline` for milliseconds and does not, because that puts a benchmark harness inside a gate, which is the standing question the parity harness already raised. `bash tools/lint.sh docs` rc=0 with 192 ledger commit references resolved, and the two docs-facts suites are 17 tests OK; the wiki's copy of the recall table was not read here and stays unchecked.     **AUD-295 (closed 2026-10-11 at fix commit `5b6c2e1`)** is the second half of AUD-294's environment sweep. `tools/lib/python.sh` screens every interpreter it searches for -- the version floor, the three analysis imports, executability -- and returned the one the operator names by hand with none of those screens applied. Measured model-free against the real library on system bash with stub interpreters in a temp directory: an override naming a path that does not exist, one reporting 3.9, one whose third-package import fails, and a plain directory all resolved at status 0 and came back verbatim, after which `tools/install_models.sh:446` assigns the answer unchecked and execs it as a converter at six later sites -- so a typo in the variable arrives mid-install instead of stopping where it is read. The same probe cleared the search: a newest candidate without the stack falls to the next, a candidate below the floor falls, a non-executable file is skipped, and the bare `python` name is tried and can win. The refusal when nothing qualifies named six candidates where the loop tries seven. The fix puts the three screens in one function, runs the override through it and refuses at status 1 naming the value, the floor, the packages and the cause the candidate itself gave, and lists all seven; the library header and the resolver paragraph of the model-adding doc both claimed the override skips the check entirely and now say it skips the search and not the test. Tests first: a new CI-registered suite of 16, 5 of 14 failing against the unpatched library. Two of those tests then needed changing themselves, and both say so in the ledger row rather than reaching green quietly -- a substring test satisfied because `python3.14` contains `python`, and an assertion satisfied by the requirement line rather than the cause line, so a refusal carrying no cause at all would have passed. The anchored mutation sweep: 18 mutants, 0 bad anchors, 16 killed, 2 survivors, both equivalent -- the executable-bit screen is still caught because the probe then answers with a permission error, and the stderr wrapper is redundant because each line carries its own redirection -- with the library identical to pristine afterwards. The portability and syntax gates and the python and documentation checks all ran rc=0, as did 109 neighbour tests across the new suite, the two installer suites that set this variable, the two docs-facts suites, the sidecar gate suite and the release verification suite; no existing fixture was edited. Recorded rather than fixed, and it is the operator's call: the check is by exit status alone, so `/usr/bin/true` passes it -- the two installer suites use exactly that as their override -- and a probe that demands an interpreter which answers would refuse those fixtures. The sibling sweep over every shell file finds these two interpreter variables and no other; the Python half of the same token is open as AUD-296: six `tools/` files print advice to point this variable at another interpreter, and no Python file in the repository reads it. **AUD-296 (closed 2026-10-11 at `1b4eccc`)** is that Python half: the six dependency refusals ended with `(or point TINYTITAN_PYTHON at another Python 3.10+)`, and re-running the identical command with the variable pointed at an interpreter that does import the whole stack is byte-identical, because the interpreter is chosen before this program prints anything -- the reader is `tools/lib/python.sh`, reached through `tools/install_models.sh`. Each message now names that program and says this file does not read the variable, while the line naming `sys.executable` -- the half that was always actionable -- is pinned unchanged. `benchmark/test_python_resolver_gate.py`, which already governs the reader, grew `DependencyRefusalMessageTests` from 16 to 19 tests and reached the refusal by shadowing `numpy` with a stub that fails to import, so no venv and nothing fetched; RED was one failure naming the file, GREEN 19 rc 0, and 5 of 5 anchored mutants killed. Two things this close deliberately leaves where it was: the wider gate over every `tools/` message that names an environment variable is still not built, which is the decision the row was filed open for, and one of those mutants -- the advice deleted outright from all six -- is killed by the single-wording pin rather than by the invariant, so removing the pointer is now a decision instead of a drift.     **AUD-297 (closed 2026-10-11 at fix commit `9e4981d`)** is the third member of the same class, and the one that was wrong in a way a user would feel. `tools/dsh_local.sh` hands the port it was given straight to a `for (( ))` loop, so the string is evaluated as bash arithmetic instead of read as a port. Measured by running the real `port` subcommand -- it reads the port table and prints a number, so nothing is launched -- on the system bash: `-1` exits 0 printing `-1`, and a single space exits 0 printing `1`, because port 1 fails the listening check and the walk then offers it as the choice; neither is bindable by an unprivileged process, so the browser UI this script exists to open dies on its own argument after the script has already reported a port. `abc` and `7788a` exit 1, but with bash's own words -- `unbound variable`, `value too great for base` -- which name no action, and `99999` and `65536` exit 1 claiming `ports 99999-100099 are all in use` when the loop's entry guard means not one port was asked about. `tools/server_launcher.sh` refuses all six shapes before use; this script had no equivalent, and no test covered its port path at all. The guard now mirrors the launcher's message and sits in the one function both entry points and all three callers go through; the digit-count bound is load-bearing rather than tidy, because bash's 64-bit arithmetic wraps silently and 2**64+1 evaluates to 1. Two tests of the twelve pass before the fix on purpose -- a legal port must not read as an illegal shape, and a genuinely occupied port must still walk -- and the suite drives that second one with a real local listener rather than a stub, so it would notice a guard that over-refuses. One instrument failure is on the record: the probe helper first passed the flag before the command name, which this script's dispatcher reads as an unknown option, so a flag test was passing without reaching the code; corrected before any conclusion was drawn from it, and the corrected run is the one quoted. The anchored sweep: 7 mutants, 0 bad anchors, 7 killed, 0 survivors, script identical to its pristine copy afterwards. The portability and syntax gates, the documentation and python checks and 90 neighbour tests across this suite and the six that cover the harness, the route writer, the server's own port and the docs facts all ran rc=0; no existing fixture was edited. Siblings measured and cleared rather than filed: the server-port seam in the same script is guarded where it lands, in the route writer's own numeric check, though that one has no ceiling and words its refusal differently, and the launcher's two numeric environment inputs both sanitise before their arithmetic. Two residues are recorded instead of quietly fixed -- the walk's exhausted message still names a hundred ports when a high base stops the scan at the end of the space, and the exit status this refusal uses (1, this script's own convention) differs from the launcher's (2) for the same defect.     **AUD-298 (closed 2026-10-11 at fix commit `1e68ece`)** is the same operator-value class seen from the release path, and the shape that makes it different is that the bad value is not a typo but a documented meaning. `tools/release.sh` read `TINYTITAN_RELEASE_NOTES_MAX_CHARS` and handed it to `tools/compact-release-notes.py`, whose own max-chars argument documents 0 as *no limit*; measured on the real tool with a 3,003-character note, 0 published the notes with the budget removed and said nothing, `-1` came back as an over-budget note, and a word came back as an argument-parsing failure that `release.sh` then reported as a note that was too long. The budget is never echoed, so a release made with the gate off looks like one made with it on -- unlike the two other deliberate narrowings on that path, each of which dies without its companion reason and names itself in the notes. The guard sits where the value is read and refuses anything that is not a whole number of characters, 1 or more; the tool keeps the zero its own help promises, and the new `benchmark/test_release_notes_budget_gate.py` (10 tests, registered in CI) pins both halves -- the refusal happens before the compactor is reached, asserted from the arguments it was recorded as being handed, and the compactor's own 0 still means no limit over a real note. RED was 3 of 10, which is exactly the three shapes the fix refuses, and GREEN 10; neighbours 70 tests OK; the nine-mutant sweep killed 8 and its one survivor is an equivalence: dropping the guard's empty-value arm cannot change any run, because the assignment one line above already turns an empty value into the default, and the mutant that breaks *that* dependency is killed by the blank-value test. The one change outside the new suite is disclosed additive fixture support: `test_release_ci_green.py`'s `run_notes_block` gained an optional extra-environment argument so the new suite can ask what the operator's environment does to the extracted block; absent that argument the generated script is what it was before and its 19 tests still pass. Sibling sweep, and the next row's raw material: `tools/server_launcher.sh:1826` puts `TINYTITAN_WARM_TOKENS` straight into an arithmetic expansion with no shape check under the script's own strict mode, so a word or a leading-zero value dies after the model has loaded and before the client opens, and :1847 hands `TINYTITAN_WARM_TIMEOUT` to curl's max-time, whose 0 means the request continues forever -- both sit behind a completed model load, so no live launch could exercise them on this host while the operator's server runs; the next row drove them by taking the two blocks as a script instead, and closed both.     **AUD-299 (closed 2026-10-11 at fix commit f17f632)**: the launcher's two warm-up numbers are checked where they are read. A web run given a word in `TINYTITAN_WARM_TOKENS` died inside the warm block with bash's own `abc: unbound variable` after the model had loaded and before the browser opened, `08000` died as `value too great for base` because bash reads a leading zero as octal and the size named is not the size used, and 0 and a negative both printed a warm of the size asked for and then sent a clamped one-repeat prompt. `TINYTITAN_WARM_TIMEOUT` went straight to curl's max-time, whose documented zero means the request continues forever -- measured against the real curl and a loopback listener that accepts and never answers -- and any curl failure came back as the same sentence a slow model gets. The guard sits beside the port check and on its exit status, scoped to the run that actually warms, and the warm block reads the checked values. The new suite is `benchmark/test_launcher_warm_gate.py`, 20 tests: the port block and the warm block taken verbatim as one script under the system bash with a curl that records each argument on its own line, and four live-path tests through the launcher fixture, so a guard that exists but is not on the launch path fails. RED 10 of 20 against the file as shipped, GREEN 20 in 3.5 s, neighbours 82 tests OK over seven launcher suites with no existing assertion edited, eleven anchored mutants and nine killed. Two survivors are true equivalents, the warm block and the curl call re-reading the environment rather than the checked value, because a guard that refuses instead of rewriting leaves the second read with the same answer -- and that is what disproved this suite's own docstring claim that such a re-read would show up as a request that reached curl, so the claim is withdrawn in the docstring and in the script's comment with the sweep record as the evidence, and the zero test now pins the words it names so a refusal from the wrong arm cannot pass. Gates rc=0: shell on 29 scripts under the system bash, shellcheck 0.11.0, python with ruff 0.16.7 over 188 scripts, docs on 83 documents. Sibling sweep: every other max-time in the repository is a literal, no other shell script puts an environment value into an arithmetic expansion, the port seams are guarded, and the memval repeat count is AUD-246 -- but the same file reads its RAM-ceiling test seam through a guard that rejects non-digits and not a leading zero, so a zero-padded `TINYTITAN_PHYSICAL_RAM_BYTES` prints three bash errors and still finishes the plan with exit status 0 and the ceiling rule switched off. Measured on the live launcher; that is the next row.     **AUD-300 (closed 2026-10-11 at fix commit f105fe9)**: the launcher's RAM-ceiling test seam is refused where it is read. A zero-padded `TINYTITAN_PHYSICAL_RAM_BYTES` passed the sanitizer, because it is all digits, and reached three arithmetic expansions, where bash reads it as octal: measured on a live dry-run it printed three lines of the script's own name and the words `value too great for base`, and still finished the whole plan with exit 0. The third of those is the one that matters, because it sits inside the guard for an unreadable machine, so the ceiling rule went to no rule and the run reported success anyway -- and every other malformed value arrived at the same place in silence through the sanitizer, including a unit suffix, a negative and a fraction. The seam now refuses anything with a non-digit and anything with a leading zero, on the port's exit status and naming the variable, while 0 keeps its documented meaning of a machine whose size cannot be read and the sysctl fallback keeps its silent one. Eight new tests and one helper in `benchmark/test_launcher_ram.py`, with no existing assertion edited: RED 6 of 8, GREEN 21 OK, neighbours 110 OK over the eight launcher suites. Seven anchored mutants, seven killed -- and the first pass at the guard taught two things the tests were right about: a case pattern of a digit then a star matches anything starting with a digit, so a word walked through it, and the refusal did not name the variable the operator had to go set. When the surviving mutant was the one that drops that name, the test gained the pin and the mutant died; no mutant was deleted and no assertion was softened. Gates rc=0: shell on 29 scripts under the system bash, shellcheck 0.11.0, python with ruff 0.16.7 over 188 scripts, docs on 83 documents. Nothing else in the tree sets this seam.     **AUD-301 (closed 2026-10-11 at fix commit 788d202)**: the engine-versus-shell reader sweep that followed AUD-299 and AUD-300 listed every `TINYTITAN_*` name the engine reads out of `sources/` -- 70 of them -- and compared each one a shell script also reads against its parser. One pair disagreed: `sources/TinyTitanMemory/MemoryConfiguration.swift:236` takes `1`, `on` and `true` case-insensitively, with `on` pinned by a Swift test (`tests/TinyTitanMemory/MemoryConfigurationTests.swift:61`), while `tools/tinytitan_models.sh:509` compared the literal `1` alone. `TINYTITAN_MEMORY=on` therefore started a server with memory ON and none of the three exports that function owns: the store fell through to the engine's own defaults, `~/.tinytitan/memory` instead of the `<TinyTitan>/memory` beside `models/` the docs promise, and workspace `default` -- one journal for every checkout. Worse than the misplaced files: the junk-drawer refusal reaches `junkDrawerReason` only through `TINYTITAN_WORKSPACE_DIR` (:261-266), the export the early return skipped, so launching from `$HOME` with `on` -- the exact case the refusal exists for -- exited 0 with a running server from both sides, and the `--memory` flag never showed any of it because that path exports the literal `1`. The library is the wrong side, not the engine: the alias set is deliberate and mirrored by the sibling tools switch, so no Swift and no Swift test was touched; the gate now lowercases with `tr` -- not `${v,,}`, which /bin/bash 3.2 parses and then dies on -- takes the same three words, and leaves a word the engine reads as off reading off here too rather than gaining a refusal only the wrapper has. `benchmark/test_models_memory_gate.py`, 10 tests as filed, RED on 4, and a fifth the moment the operator-renamed-store test was strengthened -- it had been passing over a function that returned early and left an inherited variable standing. 13 green; 11 of 11 mutation shapes killed, but two survivors were evidence about the test and became tests of their own: the default `export TINYTITAN_MEMORY=${...:-0}` was invisible until the child's environment was observed, and the named-workspace escape hatch had no test at all. 91 neighbours across the launcher family; four gates green with the suite `git add`ed before CI named it, 93/97 registered. The rest of the sweep is recorded rather than filed: the ANE prefill switch throws outside off/on and `tools/golden-baseline.sh:99` exports the exact rawValue so the reference cannot silently follow the default-on, `TINYTITAN_MEMORY_TOOLS`, `_CONSOLIDATION`, `_JOURNAL` and `_GUARD` have no shell reader that interprets them, and `TINYTITAN_THINKING_MODE`'s launcher check is stricter than the engine's `1`/`true`/`yes` but refuses with a named message, which is the sanctioned direction. One residue stays on the same seam, filed nowhere: `tools/server_launcher.sh:1506` prints its `Memory: on` line only when the flag was used, so an env-spelled launch says nothing about memory in the plan       **AUD-302 (closed 2026-10-11 at fix `d6b9b7f`)**: the memory switch's third reader -- the engine takes `1`, `on` and `true`, AUD-301 made the export gate agree, and the launcher's plan and banner still read only the `--memory` flag, so `TINYTITAN_MEMORY=1 tools/server_launcher.sh` (the spelling docs/agent-memory.md documents under Setup) started memory, exported the repo-scoped store, and printed nothing about either. Measured as a `--dry-run` matrix first: the flag printed its line, `1`/`on`/`TRUE` printed none. The repair is one predicate in `tools/tinytitan_models.sh`, called by the gate and both report sites, with the flag path arriving through the :1301 export it already had; the docs row naming the accepted words came with it. 22 tests in `benchmark/test_models_memory_gate.py`, nine new, eleven of eleven mutation shapes killed -- and the one survivor (a predicate without its `:-0`, which still answers OFF but prints `unbound variable` under the launcher's `set -u`) is why the suite now asserts on empty stderr. Sibling sweep: no other reader of the name exists in shell, python or JS, and the plan's other lines read resolved values both input paths write. Two residues named and not widened: `tools/golden-baseline.sh:223` reads `READY_TIMEOUT` as a `test` operand, and `benchmark/memval_run.sh:233` puts a matrix label straight into the switch. **AUD-303 (closed 2026-10-11 at fix commit `19e2a23`)**: the four numbers `tools/golden-baseline.sh` takes from the operator's environment were read far from anywhere they could be checked. A word for READY_TIMEOUT made `[ "$waited" -lt "$READY_TIMEOUT" ]` error and take the false branch, so the readiness poll never ran and the gate printed `not ready after 1800s` — the typo returned as a slow model — while MAX_NEW, SEED and PORT reached a minutes-long load first. Fixed at the read: one loop refuses a non-digit for all four, zero is refused where it means run-nothing and not where it is a value (SEED=0 still runs), and PORT is held to 1-65535 without a leading zero because the launcher's own range check is `(( ))` arithmetic, which reads `08757` as an octal error and accepts the port unevaluated — that shape is filed OPEN as AUD-304, measured on :1230-1240 verbatim under /bin/bash 3.2.57. 20 tests OK in `benchmark/test_golden_baseline_capture` (RED first: twelve refusals that were not refusals), seventeen of seventeen mutants killed, and the residue this sweep leaves is `benchmark/memval_run.sh:108`, whose idle number the engine defaults to in silence while the run header prints the word. **AUD-304 (closed 2026-10-11 at `954ff87`)**: the port guard that read the port as arithmetic. `(( ))` treats a leading zero as octal, so `--port 08757` errored on the range check, the privileged check and the default comparison, answered neither branch of any of them, exited 0 and printed the whole plan; `01024` is the shape that costs something, a launch with no range verdict and no privileged note beside a summary naming a port nobody typed. Refused at the read with one case arm beside the digit glob, above all three arithmetic reads, so all three become decimal-only at once; refused rather than normalised with a forced base, because that would quietly launch on a port the operator did not type. `benchmark/test_launcher_port.py` went 12 to 17 tests through the launcher's own `--dry-run` (RED was six failures — the flag, the environment and the typed answer, each also asserting that no bash arithmetic error reaches stderr; two of the five new tests are neighbours that passed before the change and are labelled as such in the file). Six of seven anchored mutants killed; the survivor narrows the arm to a leading zero followed by a digit, which differs only for a bare `0` that the range arm already refuses and an existing test already pins, so it is an equivalent mutant rather than a gap. One expectation of mine was wrong and was corrected from measurement before the fix landed: I had written that `--port 1` must refuse, and 1 is the guard's own stated lower bound, so the pin is now that it launches with its privileged note. Gates `shell`, `shellcheck`, `python` and `docs` rc 0, model-free. **AUD-305 and AUD-306 (filed 2026-10-11)** are where that sweep stopped, both the same arithmetic one step further along an operator's number. **AUD-305 (closed 2026-10-11 at `e60d950`)** is a menu reply: the installer's guard at `tools/install_models.sh:820` asks the range question in the negative, so an errored octal comparison answers false and means ACCEPTED — `08` and `09` then die at the catalogue index at `:824` having installed nothing, and `010` reads as 8 on a menu the shipped catalogue makes 16 rows long, taking a 20.0 GB KAT-Coder 4-bit install for a person who pointed at row 10, the 5.3 GB Qwen 3.5 4B 8-bit, with no error anywhere. The launcher's model pick at `:609` and its index at `:610` are the same arithmetic with the same reach, and measured through `--dry-run` against an empty `TINYTITAN_MODELS_DIR`, which puts all 16 catalogue rows on the menu in a fixed order on any machine, `010` named the row 8 8-bit KAT-Coder at 38.0 GB for row 10's 220.0 GB, `013` named row 11 for row 13, `016` named row 14 for row 16, and `017` and `020` were accepted as rows 15 and 16 where the plain 17 and 20 refuse. `benchmark/test_install_models_menu.py` was green today precisely because none of its replies carry a leading zero. **The repair is not the one this row was filed with.** It forces the decimal base at the range test and at the index in both menus rather than refusing a leading zero, because `01` to `07` name the right row today and a refusal would newly break a reply that works — the same rule `ram_tier` and `valid_concurrency` follow for a number consumed on the spot, while a port stays refused (AUD-304) because it flows outward to readers one site cannot fix. Tests first: the installer suite went 17 to 19 with five RED failures on the sourced script and `install_one` stubbed, and `benchmark/test_launcher_install.py` went 17 to 18 with a new 2-test class that had six RED failures. Seven launcher and installer suites together: 117 tests at rc 0. Six of seven anchored mutants killed; the survivor deletes the blank `echo` before the install line, spacing no test claims — and it is the line my own first edit dropped and restored, which is why it is named here rather than smoothed over. Gates `shell`, `shellcheck`, `python` and `docs` rc 0, model-free, and nothing was downloaded: every launcher run is a `--dry-run` that declines the fetch, and `models/` still holds its two installs with their September mtimes. **AUD-306 (closed 2026-10-11 at `79308fe`)** is a run count: `memval_count` refuses a blank, a word and a zero by name and accepts a leading zero, then `benchmark/memval_run.sh:206` reads it as octal — measured against the shipped helper, `010` runs eight arms beside record names that say ten, which is AUD-245's class, and `00` is the countdown the zero arm exists to stop, spelled with two digits. **The filed premise was wrong in one detail worth keeping:** `09` does not kill the script at the loop. Measured with that loop line alone under both `set -euo pipefail` and `set -uo pipefail`, the `for` word list expands to nothing, bash writes `value too great for base` to stderr, the loop takes zero iterations, and the script walks on past it and exits 0 -- quieter than dying, and the same wrong-report half of the defect as `010`. The repair is the `0*` arm the owner note asked for: one arm in `memval_count`, refusing by name before either count is echoed back, and a refusal rather than a forced decimal base because a run count flows outward the way a port does -- into `{arm}-r$RUN` record names, report figures and matrix labels. 5 of 7 anchored mutants killed; the two survivors are the arm's own `return 1` (both call sites turn any non-zero return from the helper into exit 2, so the helper's own code is not observable from the script's exit status) and the padded message reworded toward the zero message, which still names the leading zero further along. shellcheck caught a defect in the fix itself first pass: a `printf` with a conversion-free format and two string arguments drops the second line, so half of the new refusal was never printed. Four readers in the sweep were measured and are clear: `ram_tier` echoes its decimal-forced value so the RAM target reaches its comparison already canonical, `valid_concurrency` forces the decimal base at each read, the warm-up guard AUD-299 shipped refuses a leading zero by name, and no other script reads the memval counts arithmetically. The sweep's third menu of the same shape — the thinking-level pick at `tools/server_launcher.sh:895` indexed at `:896` — is clear by size, as are the client and engine menus: at most 4 thinking levels, 2 engines and 6 client rows, so a padded reply there is either `01` to `07`, where octal equals decimal, or unreadable octal that the positive-polarity guard refuses. The wrong-row shape needs 10 or more rows, and only the two model menus have that. **AUD-307 (closed 2026-10-11 at `b1f4260`)** is what that same sweep found still live one script over: `tools/tsan-storm.sh:60-61` refuses a bare `0` and accepts `00` and every leading zero, and the two readers of its counts disagree in one script -- `$((runs * parallel))` at `:86`, `:142` and `:147` takes octal while `[ "$round" -le "$runs" ]` at `:91` takes decimal. Measured end to end through the suite's stub helper: `--runs 010 --parallel 6` announced 48 runs, emitted 10 round lines, and exited 0 on `Clean: no report in 48 instrumented runs` with 60 runs behind it; `--runs 09` printed `value too great for base` twice, ran 9 rounds and certified them clean with no count stated; `--runs 00` printed `Running 0 instrumented runs`, launched nothing, and exited 0 over `Clean: no report in 0 instrumented runs` -- AUD-218's verdict reached by a spelling its status check cannot see, because with no run launched the statuses file is empty and `incomplete_count` is 0. Both counts now refuse a leading zero by option name before the first expansion reads them, 6 of 6 anchored mutants killed, and the rest of the class is clear by reader: release's `TINYTITAN_RELEASE_NOTES_MAX_CHARS` has no arithmetic reader (`[ -lt 1 ]` and the compactor's argparse both read decimal), `tools/dsh_route.sh:118-120` interpolates its three numbers into a URL and the route file with no arithmetic reader at all, `tools/golden-baseline.sh:102-121` already documents the split and its `[ -eq 0 ]` reads `00` as 0, the launcher's `ram_tier` and `valid_concurrency` force `10#`, and every remaining `seq` in the tree is a hardcoded literal -- `benchmark/memval_run.sh:223` was the only operator-driven one, and AUD-306 guarded it. **the 2026-09 audit**: 28 findings, all closed. The wiki's archive page was removed on 2026-09-29 when the wiki became user-only — the record is in the wiki repository's history at `6acaa8f` | **AUD-308 (closed 2026-10-11 at `4f9ebd7`)** is the one file the ledger had never named, found by sweeping for exactly that: `docs/paper/figures.py` averages a fixed grid of 96 run files and shortened it silently, so one absent repeat left a chart built from two, an absent server log answered zero consolidated tokens, and a wholly absent tree died with a ZeroDivisionError three functions from the directory it could not read. A `require()` helper refuses at the read and names the file, pinned by `benchmark/test_paper_figures_inputs.py` -- 5 tests, 4 RED and an acceptance test that was already green before the fix, which is what keeps a guard from becoming a refusal of every build; 6 of 6 anchored mutants killed, and the suite is registered in CI beside the two that stub a script rather than load a model. The filed premise was corrected the same way AUD-246 and AUD-306's were: the suspected exit-0-over-nothing is in fact a crash, and the silent shape is the partial one. The same sweep's two candidates were then measured, and both cleared on the pass that found them: `benchmark/quant_perplexity_ab.py:59` skips an absent default corpus file, but `corpus_text` returns the files it really read and `main()` prints that list, which its suite pins against the disk; and `tools/qwen38_sequence_parity.py:48` skips a layer entry that no real run omits, because `RealForwardRunner+DecodeLayers.swift:44` dumps `L{L}_entry` outside the `dumpLayerLimit` block, so the reader's 48-layer loop is contiguous and its `largest single-layer drop` label bridges nothing. That pass therefore added no row -- the sweep that surfaced AUD-308 now returns nothing actionable, which is what convergence looks like on this surface. AUD-309 (closed 2026-10-11 at `9c50d4b`) is the site that sentence left open, and it was the last file no row had ever named: the paper's `build.sh` printed a page count taken off whatever PDF sat on disk after a Chrome launch it never checked, and the launch's `/dev/null` also swallowed bash's own message about a binary that could not be executed, so a failed build reported nothing at all. It now tests the launch's status, stamps the artifact with `stat -f '%m-%z'` before and after so a launch that left it byte-identical is refused as the previous build's paper, refuses a missing artifact by name, prints the count only when it is non-zero, and takes the browser from an override whose default is still the absolute Chrome path -- a test pins that default, because the knob exists for the tests and must not become the shipping behaviour. `benchmark/test_paper_build_pdf.py`, 8 tests, rc 0, registered in CI beside AUD-308's suite. The RED is disclosed as one false start: the first run had no knob yet, so real Chrome rendered and every test saw `pages: 1`. Two disclosures this row exists to carry. The first is the discipline: `b287f21` claimed a test killed the missing-artifact mutant, the re-run showed it SURVIVED at 5 of 6, and the cause was arithmetic -- with nothing on disk before and after, both stamps are the empty string, so the before/after refusal fires anyway and that test could not tell the two guards apart; `6eb050f` adds the state the equality guard truly cannot reach, an artifact present at start and gone at exit 0, asserted by its refusal message and by the absence of a traceback, and the sweep is 6 of 6. The second is a test-file change: the same commit added one new test and one new stub constant and altered no existing assertion, and it also briefly joined two lines of an unrelated test, which the re-run caught and repaired. Gates rc 0: `shell` over 29 scripts on bash 3.2.57, `shellcheck` 0.11.0 with no warnings, `python` on ruff 0.16.7 with 191 scripts parsing, and `docs`. **AUD-139 (closed 2026-10-11 at `906456c`) is the audit's last open row, and it closed by shipping**: v5.18 had published no tools archive, so the closed installer refused every tag up to it, and the two options were backfilling a published release -- refused, an action on a shared surface -- or cutting v5.19, which publishes the tools archive by construction. The release went the runbook way: the prep commit `ca8cdb7` single-sourced the version, the first dry run caught the gate doing its job (ruff 0.17.0 from Homebrew shadowing the pinned 0.16.7, so every gate and release command on this host runs under `PATH="$HOME/.local/bin:$PATH"`; the shell config is the operator's to fix permanently), and the publish pass re-measured all six digests because a size copied from a dry run is wrong by construction -- the runbook's own warning, re-proven. CI's pip-audit step had to be called in the step that installs it (`0d8ac3b`, GITHUB_PATH reaches only later steps), the notes' measured verification numbers landed at `906456c`, and the tag was force-moved `0d8ac3b` -> `906456c` -- a docs-only delta, no Release existing at either point, CI 38155915391 and CodeQL 38155915358 green on the final tag. Published 2026-10-11 with six verified assets, the notes quoting the publish-pass digests. **The audit is complete: 210 findings, 210 closed, 0 open, 0 blocked**, every row's evidence in `ledger.json`, and Phase E's independent-host requirement closed the same day on CI-as-fresh-clone (`environment.md`): GitHub Actions has been the clean checkout on the pinned toolchain all along, the green run on `906456c` is that evidence, and the release build plus installer flow, which CI does not run, stays named there as the permanent residual.
| `.build` | release build of current `main` (`swift build -c release`, 2026-10-06); a clean scratch release build is part of each dry run |
| Wiki | `.qwen/wiki`, remote `TinyTitan.wiki.git`, **pushed to `origin/master` on 2026-10-11** (`5780057`, the AUD-136 server-guide page, ahead re-measured **0** after), then the dsh 0.2.1-alpha.2 doc sync landed the same day (`e62c041` the plugin guide's pin and mount-on-missing autonomy section, `ce5a7aa` the Project-Tracker paragraph; ahead re-measured **0** after), and the 0.2.1-rc.2 move was synced on 2026-10-17 (`434dcb3`: the plugin guide's pin plus the tracker paragraph, ahead re-measured **0** after) (`git -C .qwen/wiki rev-list --count origin/master..master`) — publishing is **two pushes**, and both halves of the dsh move are on their remotes. User-facing only since 2026-09-29 |
| DeepSeek Harness | **one plugin, no bundled install.** `dsh-tinytitan` pins `0.2.2-rc.1` exactly — peers, gate constant and package version, with its suite failing on drift — and is installed into the user's **own** DSH profile. The private harness under `~/.tinytitan/dsh`, the launcher's `--web` hand-over and the dsh-bundle isolation/node/port tests were removed on 2026-10-24; `tools/dsh_route.sh` remains, because a user's harness still needs the TinyTitan route. The second bundle, `dsh-lan-manager`, and the `ttlanmanager` client were removed on 2026-10-28. No harness version, npm prefix or cache lives in this repository's install any more. |
| CI | every `main` push runs the `test` job (eighteen gates, plugin/ converter/ installer/ release gates, the embedded-dependency check, serial tests, Markdown links) and the `thread-sanitizer` job, plus CodeQL (`languages: swift`); the release commit's push is the run to watch (`gh run list`). As of 2026-10-08 nothing has been pushed, so **no run covers the 120 commits above** (`git rev-list --count origin/main..HEAD`, measured at `b5e0c4e` plus this commit — the row is wrong the moment anything lands, and the `main` row above carries the live count) — that is the `main` row, not a secret. Four CI arms have been run by hand on this tree because no push can cover them: `tools/embedded-dependency-check.sh` is exit 0 at `87ace2c` (`Build complete! (29.43 sec)`, `EmbeddedDemo: TinyTitanLib linked (Engine)`, `== ok: the package is consumable as a dependency ==`, `/tmp/embedded188.log`); the plugin suites pass through `tools/lint.sh javascript` on the pinned 10.11.0/3.9.9, and each package's own suite passes run directly at `c7e2159` — `npm test` in `plugins/dsh-tinytitan` 154/154 and in `plugins/dsh-lan-manager` 141/141, both exit 0 (`/tmp/js_tinytitan.log`, `/tmp/js_lan.log`); and the `thread-sanitizer` job's exact command (`swift test --no-parallel --sanitize=thread`, `ci.yml:259-273`) completed at `c3d8d58` with **exit 0 and zero `WARNING: ThreadSanitizer` lines** over all 7 targets, 1711 tests in 253 suites — the same counts as the unsanitized run, so nothing was skipped (`/tmp/tsan_all.log`; a `--scratch-path /tmp/tt-tsan` keeps `.build` warm). ASan has since been run over the C-kernel call sites (`swift test --no-parallel --sanitize=address --scratch-path /tmp/tt-asan`, scoped to the CPU suites that reach the kernels — `Int8AffineGEMVTests`, `CPUExpertFFNTests`, `CPUEngineTests`, `CPUServingTests`): **exit 0, 55 tests in 5 suites passed, zero `ERROR: AddressSanitizer` lines** (`/tmp/asan191.log`). UBSan has **not** been run through SwiftPM: `swift test --no-parallel --sanitize=undefined --scratch-path /tmp/tt-ubsan` fails before any test executes, because linking `TinyTitanBench` under that scratch path dies with `ld: symbol(s) not found for architecture arm64` on `_tinytitan_expert_reader_create in TinyTitanKernelsC.o` (`/tmp/ubsan191.log`) — an artifact of the sanitizer's separate build, not a defect in the kernels. It has since been run **directly**, which is how all three C files are now covered: `/tmp/probe_wide.c` (both affine GEMVs, 257 groups past the hoisted-sum table) and `/tmp/probe_io.c` (the reader: EINVAL shapes, thread clamping, exact-stride destinations, an offset past EOF, a `UINT32_MAX` expert id, two concurrent submitters) compiled with `-fsanitize=address,undefined,bounds,alignment,shift,signed-integer-overflow -fno-sanitize-recover=all` against the real kernel sources — **both exit 0, zero `runtime error` and zero `ERROR: AddressSanitizer` lines** (`/tmp/gemv_asan.log`). The reader driver is not vacuous: replacing `read_one`'s short-file `return EIO` with `return 0` flips three of its assertions and exits 1, and `expert_io.c` was restored from `/tmp/expert_io_backup.c` with `git diff` on `sources/TinyTitanKernelsC/` empty afterwards. Neither is evidence about the *pushed* tree; they are evidence that this tree would pass Since then every push is covered on its remote: the audit added a `supply-chain` job (`d587d66`, its pip-audit step fixed `0d8ac3b` to run in the step that installs it) running gitleaks, `pip-audit` over `benchmark/requirements.txt` and a coverage artifact, and the run on the release head `906456c` (CI 38155915391, CodeQL 38155915358) is green in both jobs -- the run the release was cut on |.

## What has landed

- **5.15** (`4ff6041`) — `.gturbo` → `.ssdai`: the format's **name only**.
  `manifest.json`'s magic is written as `"SSDAI"`, and reads accept the legacy
  `"GTURBO"` for one release (`SSDAIFormatV1.isSupportedMagic`), so every install
  built before the rename keeps loading untouched: verified with
  `TinyTitanRepack --verify-install --input-ssdai models/qwen3.5_4B_4Bit`
  ("Verified 7 files", legacy manifest) and by loading that install through the
  CLI. Nothing rewrites an existing manifest — the receipt
  (`verified-install.json`) binds the manifest's digest and the directory path,
  so editing the magic would invalidate every receipt for a string. The Swift
  vocabulary (`SSDAIFormatV1`, `SSDAIDirectoryAccess`, `SSDAIBinary`, …), the
  `--input-ssdai` flag (`--input-gturbo` stays a deprecated alias),
  `tools/ssdai_reader.py`, `tools/ssdai_diff_snapshot.py` and
  `docs/ssdai-format.md` carry the new name; the directory suffix is a
  convention the reader never checks. Dated records — release notes, the v4.x
  plans, `plan-dense-gturbo-installs.md` — keep the old word on purpose, with
  their links to the spec updated. Verification: eleven gates clean, 1,496 tests
  in 224 suites, 45/45 local links, plugin suites green.
- **5.14** (`10e5d0f`) — the browser window and both plugins on DeepSeek Harness
  **0.2.0-rc.2**. The `llm-pi-ai` route is applied through the harness's `settings`
  service (0.2.0 imports a legacy `settings.yaml` once and renames it
  `.imported`), the `tinytitan` preset is registered with the preset registry and
  becomes the default only while the profile names no selection, the `models/`
  watcher shares the boot refresher so a mid-session install reaches the picker,
  and the optional `compactionHeadroomTokens` sets the compaction trigger's
  headroom (the stock 65,536 holds it near 62% of a 262,144 window and stops
  compaction entirely below roughly `cap + 65,536`). `tools/dsh_local.sh status`
  now reports the installed harness version rather than the pin. Also: the
  internal-speed record became optional in `docs/release-process.md` §4b and
  `RELEASE.md` (owner decision 2026-10-01 — this machine's timings swing past the
  10% threshold under a browser or a game), so 5.14 carries no timing record.
  Verification: eleven gates clean, **1,496 tests in 224 suites**, **7 of 16
  goldens byte-identical**, the nine absent baselines named in the notes, a
  warning-free clean scratch build, both plugin suites green (84 tests: 83 pass,
  1 skipped; the LAN manager's 107), and CI green on `f77f786` and `bc3136e`.
- **5.13** (`89d3317`) — one diagnostic plus a documentation rewrite. A `--model`
  path that does not exist now fails with `model directory not found: <absolute
  path>` instead of `installed tokenizer is missing chat_template.jinja`, at the
  CLI, the server's session load and the router's token counter; three tests pin
  it, and the CLI test fails on the pre-fix tree at `7535ebc`. The wiki became a
  user guide (21 → 16 pages: new Installation and Configuration, rewritten
  Runtime Controls and Local Server and API, a larger Cookbook) and the README a
  verified quickstart. 50 source files were split under the 500-line rule as pure
  code motion. Verification: eleven gates clean, **1,496 tests in 224 suites**,
  7 goldens byte-identical, a warning-free clean scratch build, and the 4B
  internal-speed record inside the 10% gate on all thirteen metrics — after two
  contended passes were discarded because a 3D game held ~78% of a core and the
  GPU.
- **5.12** (`fc23691`) — the pre-production audit, drained to zero open findings:
  every force unwrap gone (171 in `sources/`, 139 in tests and benchmarks), the
  chunked-prefill force cast now the existing `chunkedUnsupported` error, a
  fleet-scanner test that could never fail made real, Swift SAST (CodeQL) running
  again after failing before it compiled anything, and the Markdown-link and
  Python-suite CI regressions closed. `tools/lint.sh` now runs **eleven** pinned
  gates (SwiftLint `--strict` and swift-format joined it, plus eslint/prettier for
  the plugin packages with committed lockfiles), the tree is at 0 SwiftLint and 0
  swift-format findings after a 442-file sweep, and all 28 audit findings are closed
  (archived in the wiki's history at `6acaa8f`; the wiki page itself was removed on
  2026-09-29). Verification: eleven gates clean, **1,493 tests
  in 223 suites**, 7 goldens byte-identical, a warning-free clean scratch build,
  and a 4B speed record whose first pass read prefill low (28.0 → 23.3 tok/s) and
  whose repeat read the baseline exactly with all thirteen metrics inside the gate
  — both records and the reason are in `docs/release-notes-v5.12.md`.
- **5.11** (`6e7cfc3`) — the DSH bundle is the delivery: the installer's source
  archive carries `plugins/`, the bundle is added from there with a `file:`
  install and no registry account exists on either side, the private harness
  keeps npm's cache/logs/user config, pnpm's home and the XDG cache/state inside
  `~/.tinytitan/dsh`, and both installed-layout resolutions are fixed
  (`dsh_route.sh` checks `../bin`/`../models`; `dsh_local.sh` resolves the models
  directory once and prints it in `paths`). The model installer refuses a download
  that cannot finish (staging: size × 1.25 + 12 GB; models: size + 3 GB, both
  printed, `TINYTITAN_SKIP_DISK_CHECK=1` to override), stages under the install
  root rather than the caller's cwd, reclaims staging as widths complete, and an
  EOF at the model menu no longer installs anything. Memory distillation is
  chained per scope, so a later session reads memory only after the earlier one
  wrote (TT-035). `--ram` help now says it accepts any whole GB from 4.
  Gates: six lint gates clean (2,031 functions, 20 scripts), **1,493 tests in 223
  suites**, 7 goldens byte-identical, a warning-free scratch build, and a 4B
  speed record with every metric inside the 10% gate and none regressed
  (`docs/release-notes-v5.11.md`).
- **5.10** (`a89255e`) — `--ram` is a target for the whole server process rather
  than the expert cache alone (4 GB floor, printed estimate; `--ram 8` now buys 32
  slots and `--ram 12` reproduces the old 64), Qwen3.8's two sampling rows are
  implemented (`--presence-penalty`, the row chosen from the request's thinking
  mode), the C kernels compile at `-O2`, the expert-cache ceiling is a third of
  physical memory, twelve decode switches that measured a wash or a loss are gone,
  and converting Qwen3.8 resumes and works through mirrors
  (`docs/release-notes-v5.10.md`). Gates: six lint gates clean (2,030 functions,
  20 scripts), **1,491 tests in 223 suites**, 7 goldens byte-identical, a
  warning-free scratch build, and a 4B speed record with every metric inside the
  gate — `gpu.routed_moe` read low on the first run under residual background load
  and both values are in the notes.
  `in_proj_a`/`in_proj_b` pair at the attention slot's own width loads and serves
  again (issue #16, a qwen38flash 4-bit install); the ten master prompts are
  runnable end to end, with a client's own summary as the baseline memory has to
  beat; and which judge runs the side-engine's tasks is a measurement. Beside
  those, one ordering fix: a search queues its T7 question before it answers
  (`docs/release-notes-v5.9.md`). Gates: six lint gates clean (2,035 functions,
  20 scripts), **1,484 tests in 222 suites**, 11 goldens byte-identical, a
  warning-free scratch build, and a 4B speed record with every generation metric
  inside the 10% gate — the two synthetic kernel metrics read low under system
  load and the notes carry both values and the reason.
- **5.8** (`4fc0726`) — the memory side-engine (a 4B on the CPU decides
  durability, duplication, contradiction and supersession, six questions a
  consolidation), the rule that holds a write back, IDF retrieval plus the
  background T7 caller, shared n-gram tables, per-tensor bit widths in the
  resident index, the DSH LAN manager, and one pinned harness release. Record:
  `docs/release-notes-v5.8.md`. Gates: six lint gates clean (2,035 functions, 19
  scripts), **1,482 tests in 222 suites**, 11 goldens byte-identical, a
  warning-free scratch build, and speeds inside the 10% gate against 5.7.
- **5.7** (`44e1ae9`) — one command installs a built engine, the app is gone, and
  every script runs on `/bin/bash` 3.2.57. `docs/release-notes-v5.7.md`.

## What is open

The [Project Tracker](https://github.com/Pummelchen/TinyTitan/wiki/Project-Tracker)
is the authority. As of 2026-10-17 it holds **one** row, and it cannot be closed by
an automated step on this machine. **TT-049** closed the same day: the wiki
Benchmarks page no longer publishes the 2026-08-30 6.82 tok/s for
Qwen3.8-Flash-Next 4-bit — that row is withdrawn, and the page carries the current
build's measured CLI levels as a protocol of their own (wiki `c2591e7`, head
`5e13783`, §5bn's controls and §5r). The gate0 obstacle is unchanged and is *why* the
row was not refreshed in its own table: this Mac's own desktop holds the GPU above the
benchmark protocol's 15 % preflight (§5v measured 61 % with the machine idle and
69.6 % mean through a decode run, §5bd found the reader's floor *is* the desktop's),
and a contended median is not a publishable level. A gate0-certified row needs a
preflight that measures TinyTitan's own occupancy rather than the desktop's.

1. **TT-048 — re-verify a fresh KAT-Coder-V2.5-Dev install.** Issue #19: the
   converter's per-expert duplicate guard was keyed on the source tensor name
   alone, while `convert_shard` adds one tensor once per requested width and
   `install_models.sh` always converts this family with `--bits 4 8`, so every
   fresh `katcoder` install stopped at the first routed expert with
   `duplicate source tensor …`. Fixed in `025dacb` (guard keyed by
   `(width, name)`), along with two shutdown defects the verification exposed
   (`14cc6e7`, `f1c70b7`). Verified on the real checkpoint: shard
   `model-00000`'s 2,075 per-expert routed tensors convert at both widths and
   write fused `switch_mlp.*` stacks with a 256-expert axis. What remains is the
   rest of the 13-shard fetch, both repacks, and the `docs/adding-a-model.md` §4
   checklist before the model is called supported again; the installs already
   under `models/` are unaffected. TT-038 (an embeddable engine) closed on
   2026-10-02 — the facade shipped as `TinyTitanLib`.
2. **Carried forward, not tracked as tasks:** the Qwen 3.8 port items — QSA
   indexer selections to the GPU, a higher expert slot budget, the n-gram gather a
   token ahead. TT-021–TT-023 were closed on 2026-09-19 (no other machines; no disk
   for the ~360 GB bf16 reference), so the M1–M6 claim stays a design intent and
   Qwen 3.8 long-context stays verified only at a lowered budget.
3. **Three audit follow-ups that measurement could not close.**
   (a) `try? await flush()` at `ContinuityEngine.swift:128` and `:147`, and the
   deferred barrier at `Journal.swift:145`, swallow a failure of the durability
   barrier without recording it in `journalFailure` — so a workspace whose `fsync`
   fails goes on reporting itself durable, which is the one promise the memory
   subsystem is written to keep honestly. It was not filed because it could not be
   reproduced here rather than because it looks fine: `flush()` casts its journal to
   `FileJournal` and returns for anything else, so no injected journal can fail a
   barrier through any seam, and nothing on this Mac makes `fsync` fail on a file
   whose `write` calls all succeeded. **Measured 2026-10-08, and it still cannot be
   produced** — see the audit ledger's AUD-204 row for the run and
   its numbers. A descriptor opened on a mounted 16 MB APFS disk image, then the
   image force-ejected underneath it: `fcntl(fd, F_FULLFSYNC)` returns -1 with
   `EBADF`, `fsync(fd)` returns **0**, and the next `write(2)` returns -1 with
   `EIO`. So the barrier's only failure signal is an errno the kernel does not set
   for a dead volume, while the append does fail — through `writeFully`, into the
   observer, onto `journalFailure`, where `isDurable` already reads it. The two
   `try?` remain an error with no caller and no trace in the source, and no machine
   here turns them into a lost record. Do not spend another session trying `hdiutil`
   variants; the next thing that could answer it is a volume that answers `fsync`
   with `EIO`, which is a failing drive rather than an experiment.
   (b) AUD-189's fix repairs a dangling journal line when the file is next *opened*;
   an append that fails partway inside a running process leaves the same line until
   then, and its next record would fuse onto it. The writer there already has an
   error in hand and records the durability failure, so the loss is reported — what
   is not pinned is the recovery. It has no test seam: a partial `write(2)` needs a
   disk that fills mid-call, and no gate here may fill one. Closing it needs either a
   volume that can refuse a write at a chosen byte count or a decision to accept the
   next open as the repair point.
   (c) AUD-190's sibling sweep is closed. It was filed as AUD-215 and fixed in
   `7bd3346` on 2026-10-08, and the reason it had been left unfilled — that no
   volume refuses the entry directory while still accepting the writes into it —
   was false. See the `chflags` trap below before accepting any "no test seam"
   claim about an unlink.
   (d) The Phase D sweep over the remaining `benchmark/` drivers is closed, and it
   is closed on measurements rather than on the sweep's summary: AUD-216, AUD-217,
   AUD-219 and AUD-220 were true and are fixed, and the two sites the sweep set
   aside as "a different and smaller defect" were then re-measured instead of being
   accepted as clear. `benchmark/ornith_four_program_matrix.py` is clear because
   every pass count there already ANDs the error away (`:768`, `:787`, `:875`) and
   `main()` returns 1 unless `passed == total`. `benchmark/ornith_concise_tool_ab.py`
   is clear for a reason worth keeping: its `artifact_passed` (`:640`) does count a
   row whose request errored — driven here with a synthesized record, an errored row
   scores `artifact_passed 2 of total 2` against `workflow_completed 1` — but that is
   what its name says it measures, the artifact on disk, and the two metrics sit side
   by side so the reader is not told a fiction; the run's verdict is
   `quality.passed and not error` over every row, which returns 1 on that same record.
   The rule the sweep leaves behind: file when the *passing verdict* or the *number
   the reader cannot see* is corrupted, not when a second named metric is merely
   optimistic. Do not re-litigate these two.
   (e) Phase D then swept the drivers one layer earlier — how a driver decides what
   to do with the word it was given, and whether its own docstring matches its
   dispatch — and that sweep is **not** exhausted: it produced AUD-221 (nine
   `benchmark/memory_*.py` drivers answering an unknown command with the report, one
   of them falling through to *another* driver's live arm) and AUD-222
   (`memory_book.py` documenting three arms while `ARMS` holds four, and
   `memory_value.py` shipping an uncalled `compiles()` that answers `False` when
   `swiftc` is simply missing). Both are closed with a general seam in
   `benchmark/test_driver_documented_commands.py`. The same seam then swept the
   non-memory drivers and produced two more, both closed: **AUD-223**
   (`capital_of_paris_report.py` printed `{combinations} x {prompts} x {repeats} =
   **N measured runs**` with `N` counted from the rows independently, so a truncated
   matrix read as `2 x 2 x 3 = **9**` and the line under it certified "every request
   was served" -- a row exists only for a run that *completed* -- while naming a
   fixed September archive whatever file was passed and reading `/tmp` with no
   argument), and **AUD-224** (`tinytitan_determinism_ab.py` printed its
   `deterministic across fresh processes` conclusion outside the loop that compared
   the streams, so two servers that carried no content -- `extract_deltas` answers
   `[]` for a JSON error body and raises nothing -- hashed equal and passed, with the
   token length hardcoded in the sentence while `max_tokens` is the argument, and
   `int(sys.argv[1])` at module level made every importer run the cast against its
   own argv), and **AUD-225** (`tinytitan_overlap_measure.py` `Popen`ed
   `TinyTitanServer` *at import*, which is both the model run nobody asked for and
   the reason the file had no test, cast `int(sys.argv[1])` against the importer's
   argv at :27, and printed three `=== ... ===` section headers over an empty
   capture and exited 0 -- with no line count in the header, so a one-of-two
   partial run read as a complete one). Its fix moved the length onto the
   `parse_max_tokens()` now shared in `tinytitan_profile.py`, which the determinism
   probe had written for itself, so both probes refuse a non-integer, a zero and a
   negative identically), and **AUD-226**
   (`tinytitan_quant_fidelity.py` printed `VERDICT: quantization is faithful` and
   exited 0 whenever `bad == 0`, where `bad` counts tensors that were compared and
   disagreed -- measured on the pre-fix blob with an empty `off_index.json`: six
   `missing` rows, "faithful", exit 0, with no `checked` counter anywhere in the
   file -- while reading its three index files at module scope, so
   `python3 -c "import tinytitan_quant_fidelity"` ran the whole probe and died on
   `FileNotFoundError`, and indexing `mlx_idx[stem + ".scales"]` beside a guarded
   weight-name check turned an absent affine entry into a `KeyError`), and **AUD-227**
   (`tinytitan_rdadvise_ab.py` `Popen`ed a fresh `TinyTitanServer` per arm *at module scope* --
   `run()` opened its log at :35 and spawned at :36-38, and `for mode in ("default", "off"):
   run(mode)` sat at :102-103 with no `__main__` guard, so `python3 -c "import
   tinytitan_rdadvise_ab"` loaded a model, sent two 512-token streamed requests, killed the
   server and did it again, which the guarded child proved without running it (`RuntimeError:
   opened a real benchmark log at import`, rc 1). Its `--- {mode} ---` header printed over
   whatever `gen`/`runner`/`gpu` had captured, with no line count and no refusal, so an arm whose
   server never printed `cb1_ms=` read as a clean measurement and exited 0; the only `sys.exit`
   in the file was :43, for a server that died before `/health`, and it killed the whole A/B
   instead of reporting the arm that failed and carrying on to the other. Three more came out of
   reading the arms rather than the plumbing: `MODEL = str(DEFAULT_MODEL_PATH)` at :23 ignored
   `TINYTITAN_BENCH_MODEL`, which the three sibling sweeps honour; the `default` arm only ever
   *added* env vars, so an inherited `TINYTITAN_RDADVISE_POLICY` became the control arm's setting
   while the header still said `default` (`ServerModelSession+Loading.swift:198` reads that var
   ahead of the loaded config, which is what makes the inheritance effective rather than
   cosmetic); and the docstring promised "Interleaved fresh servers" while the file ran each arm
   once in a fixed order. The fix moved the health wait and the two-request loop onto
   `wait_for_health`/`request_twice` in `tinytitan_profile.py` -- `tinytitan_overlap_measure.py`'s
   private copies were deleted and its call sites point at the shared ones, so AUD-225's driver
   keeps its behaviour and loses its duplication -- and put arm composition in
   `arm_environment()`, which *clears* the policy var for the `default` arm.), and **AUD-228**
   (`tinytitan_gap_bisect.py` -- `run(128, "len128")` and `run(1024, "len1024")` at :105-106 with
   no `__main__` guard, so importing it opened a log and spawned a server **twice**, which the
   guarded child proved (`RuntimeError: GUARD: opened a log at import: .../tinytitan_gap_len128
   .log`); on the pre-fix blob, with `Popen`, `http.client` and `server_command` faked so the
   server answers `/health` and writes a log with no counters, it printed
   `--- len128 (max_tokens=128) ---` / `--- len1024 (max_tokens=1024) ---` over two empty bodies
   and exited **0**; its only exit was `sys.exit(1)` inside `run()`, so one server that died before
   `/health` ended the sweep before the second length was tried. Two more fell out of reading the
   arms to find out which control varied: `MODEL` at :23 was handed to `server_command` directly,
   so `TINYTITAN_BENCH_MODEL` was ignored, and the docstring advertised "a length sweep **and an
   rdadvise-off comparison**" while `run()`'s `extra_env` parameter was never passed by either
   call and `grep -c RDADVISE` over the file is **0** -- a page pointing its reader at an arm the
   driver cannot run. The fix moved `bench_model()` and the arm verdict (`channel_verdict()`) into
   `tinytitan_profile.py` and made the rdadvise A/B call them too, so the two drivers cannot
   disagree about what an empty arm means.), and **AUD-229**
   (`capital_of_paris_smartness.py` -- `main()` at :177 with every run variable bound at module
   scope (:31-:39), so importing the file *ran the matrix*: collecting its suite with `urlopen`
   guarded appended the row to `$RESULTS`, and the driver's own `except Exception` at :156
   re-labelled the harness's guard as a server fault (`"error": "RuntimeError: a request ran during
   collection"`) and carried on. Worse than that one row: the globals froze at import, so **17 rows
   landed in that single file** from the later tests, each of which had set its own `RESULTS`,
   `PORT` and `PROMPTS` in the environment and none of which could reach them. `RUNS` unset died
   with `KeyError: 'RUNS'`, `MAXTOK=abc` with `ValueError` -- an importer cannot configure a
   refusal. The verdict was the finding: `main()` returned nothing and nothing counted the rows, so
   a 1 x 2 x 2 matrix against `PORT=8399` printed four `URLError: ... Connection refused` rows and
   exited **0**, `RUNS='[]'` printed nothing and wrote nothing and exited **0**, and a failed row
   read `cold=None load=Nones ttft=Nones content=Nonech`, the keys :147-148 set only on the success
   path being printed anyway by :167. The fix parses inside `main()` behind a `ConfigError` that
   names its variable (status 2), plans the matrix before the first request (`NOT MEASURED`, status
   1, for a plan of zero), counts as it writes (`{ok} ok, {failed} failed -- {written} of {planned}
   rows written`, status 1 unless every planned row came back ok), and prints `load=n/a` where the
   run never reached a value. Its sibling on the read side was checked, not assumed:
   `capital_of_paris_report.py` (AUD-223) filters to `status == "ok"` at :121 before it reads
   `load_s`/`cold` at :131, and `stat()` drops non-numbers, so the explicit `null` a failed row now
   carries cannot reach a table cell.), **AUD-230**
   (`tinytitan_maxthroughput.py` -- the peak-throughput ladder sliced its scraped rate list by
   prompt index (`r = rates[idx : idx + 2]` at :167) while the footers it scraped came from a
   request loop whose count it never checked, so two prompts and three footers printed
   `8bit essay: measured=44.00 (warmup 50.00) ct=[]` and `8bit digits: missing rate footer
   (request failed?)`: the 44.00 and the 50.00 were both *digits*' rates, `essay`'s two were never
   printed, and the row that read as a measurement measured the wrong prompt. A log with no
   footers at all printed two `missing rate footer` rows and `main()` returned `None`, so the run
   exited **0** having measured nothing. `MAXTPUT_PROMPTS` was read and filtered at module scope
   (:45-50, ending in `raise SystemExit`), so a typo killed the process during *collection* rather
   than refusing it; `--mtp` passed the one draft head to the baseline arm too (:83), which is the
   comparison the flag exists to make run twice against the same server; `TINYTITAN_BENCH_MODEL`
   was ignored while the dead `MODEL` global at :25 was read by nothing; and a path naming no
   width fell through the ternary at :80 and was reported as `4bit`. The fix counts footers and
   completion records against `REQUESTS_PER_PROMPT * len(prompts)` before it prints a row, carries
   the arm's status out of `run_quant()`, parses all four run variables inside `main()` behind a
   `ConfigError` (status 2), gives the two `--mtp` arms distinct commands, and names the install
   directory when its path carries no width.), and **AUD-231**
   (`tinytitan_slots_ab.py` -- `run()` printed its `--- slots=32 pin=None boot_s=0.0 ---` header
   unconditionally at :64 and then three uncounted `for line in ...: print(line)` loops (:104-109),
   so two servers that answered `/health` and emitted no counter line produced two headers, three
   empty bodies and exit **0**; its only exit was `sys.exit(1)` *inside* that per-arm function at
   :52, so the first arm whose server died ended the sweep and the second was never tried. The
   runner channel keyed on `cb1_ms=` (:100) while the line the runner prints carries `io_ms=`, so a
   line with the wrong field was collected as the decisive number; `pin=False` and a plain
   `pin=None` both wrote `tinytitan_slots_32.log` (:42), so a nopin arm and an unmoded run
   overwrote each other's log; and `int()` ran at :116/:120 with no handler, so `slots_ab 32 128x`
   and `TINYTITAN_AB_TOKENS=512x` were tracebacks at import rather than refusals. The fix puts the
   three channels in one `CHANNELS` table keyed on the field each prints, has `capture()` return
   them and `verdict()` emit `NOT MEASURED: {tag} -- no log line contained {marker!r}` for an empty
   one (status 1) and `ARM FAILED: {tag}` for an arm whose server never loaded *while still running
   every other arm*, moves the token count into `max_tokens(env)` and the slot list into
   `parse_args()` behind `ConfigError` (status 2), and makes `tag_for()` carry the pin mode so the
   tag and the log name cannot collide.), and **AUD-232**
   (`tinytitan_longgen.py` -- the docstring promises `1 warmup + 3 measured runs` and the driver
   kept that protocol for only one of its two columns: `ct={cts[1:] …}` at :130 dropped the warm-up
   while `mean = sum(rates) / len(rates) if rates else 0` at :127 averaged all four footers, so a
   server whose footers were 50.00 then 40.00 three times printed `mean=42.50` for runs that ran at
   40.00, and nothing anywhere counted footers against requests -- three for four averaged three,
   one printed the warm-up's own rate as the measurement, and a log with no footer printed
   `rates=[] mean=0.00 tok/s ct=[]` under a `main()` that returns nothing, so the whole sweep exited
   **0**. The `/health` wait was inlined at :45 and its 120 s budget expired *into* the run: the
   poll breaks only on a healthy reply, so a server still loading fell through to `request()` anyway
   and `resolve_api_model()` answers a fallback id when it cannot reach a server -- measured, 4 POSTs
   to a server that never answered. Configuration was frozen at import, so `TINYTITAN_BENCH_MODEL`
   was ignored (:114 reads `DEFAULT_MODEL_PATH`), a path naming no width was labelled `4bit` (:117),
   and the log name carried only that label (:39), so two 4-bit installs on one command line -- the
   documented usage -- wrote the same `longgen_4bit.log` and the second truncated the first. The fix
   prints the warm-up as its own column and the mean over `rates[WARMUP_RUNS:]`, refuses to print a
   row unless both channels hold exactly `REQUESTS_PER_MODEL` lines (`NOT MEASURED: …`, status 1),
   reports an arm whose server never loaded as `ARM FAILED: …` (status 1) with `main()` returning the
   worst status through `sys.exit(main())`, uses the profile's `wait_for_health()` with a
   load-sized `SERVER_LOAD_TIMEOUT` and sends nothing unless the server is served, reads the default
   install from `bench_model()`, keeps a width-less install's own name, and tags rows and logs with
   the install as well as the width so two same-width installs cannot share a capture.), and
   **AUD-233** (`tinytitan_benchmark.py` -- every count the summary line prints is derived
   from what answered rather than what was owed: `Answers verified: {verified}/{len(results)}` at :504
   and `Cache: {hits}/{len(results)}` at :502 divide by the rows that survived, and `expected_sends`
   at :426 is `len(results) * (2 if warm_sends else 1)` computed *after* the loop that dropped every
   failed prompt, so a cell whose twelve prompts all died printed `Answers verified: 0/0` with no
   footer warning at all -- zero footers compared against zero sends and matched -- under
   `PRODUCTION PROFILE COMPLETE` and `main() -> None`, exit **0**; `wait_ready()` at :576 polled
   `/health` ten times and never called `proc.poll()`, so the measured output of a server that had
   already exited and of one that was merely still loading was byte-identical (`FAILED to become
   ready!`, the cell skipped, `COMPLETE`, exit 0) and a cold 8-bit MoE past 100 s is declared dead;
   configuration was frozen at import -- `DEFAULT_MODEL_PATH` as the default so
   `TINYTITAN_BENCH_MODEL` was ignored, `quant_label = "4bit"` at :617 for any install whose path
   names no width, and `--matrix` launching `--mtp-model models/ornith-1.5_35B_A3B_MTP_4Bit` at :634
   for a Qwen target it had never seen; and the server's decode footers reached the rows by
   `zip(results, rates)` at :432, which on a 24-footer log and 12 rows labels rows 2-12 with prompts
   1-6's rates and leaves the other half unattached. The fix makes the cell report against what it
   owed (`cell_report()` prints `NOT MEASURED: {label} -- {n} of {len(PROMPTS)} prompts produced a
   row` and `-- {n} decode footers for {m} sends`, both status 1, and only a clean cell prints the
   `FOOTER DECODE` headline), returns the worst status out of `main()` through `sys.exit(main())` and
   banners `STOPPED -- {n} of {m} cells not measured` instead of `COMPLETE`, replaces `wait_ready()`
   with the profile's `wait_for_health(proc, port, timeout=SERVER_LOAD_TIMEOUT)` and prints
   `NOT RUN -- the server exited before /health answered` with no request sent, takes the install
   from `bench_model()`, derives the width with `quant_label()` and falls back to the install's own
   name, requires `--mtp-model` for `--matrix` (`REFUSED: ... Nothing launched.`, status 2, verified
   live with `Popen` and `HTTPConnection` both guarded), joins footers by
   `rates[k*step + step - 1]` under an exact-count guard, closes the log handle `launch_server`
   leaked, and writes `avg_decode_tok_s: null` for a cell that did not measure instead of the
   client-side average under the server's key.), and
   **AUD-234** (`tinytitan_longctx.py` -- the drift headline measured the wrong quarter and named it
   in a unit the file guessed. `rate_at(times[: max(2, len(times) // 2)], 0.5)` at :191 takes the last
   half *of the first half*, which is the second quarter of the generation, and prints it as
   `first-half`; and both rates are characters per second divided by the hardcoded
   `4.0 # ~4 chars/token` at :162 and printed as `tok/s`, while the same request's `usage` chunk carries
   the `completion_tokens` that make the conversion measured. Re-running the pre-fix function on a
   constructed 40 -> 10 tok/s decay prints `first-half=15.17 last-half=10.00`, a 5.17 tok/s drift where
   the two halves differ by 30.00. Below the headline, nothing reported: `if len(times) >= 4:` at :190
   has no else, so a short capture printed no drift line and no failure; `ttft` is `None` until the first
   content delta, so a stream that produced none raised TypeError in the row print at :139 instead of
   naming itself, and `rss_mb` returning `None` did the same to the RSS print at :185; `main()` at :165
   returns no status anywhere and the guard calls bare `main()` at :212, so the exit was 0 whatever was
   measured; the decode footers printed at :205 are never counted against the three requests sent;
   `proc.terminate()` at :199 sits after the requests, so any raise left a model server holding port
   8091; the load wait is an inline `for _ in range(240)` at :56 that never calls `proc.poll()`, so an
   exited server and a cold-but-loading one both ended in `RuntimeError("server not ready")` at :68 after
   120 s; `MODEL = str(DEFAULT_MODEL_PATH)` at :30 ignores `TINYTITAN_BENCH_MODEL`; and a 9,795-character
   prompt is called `~10k tokens` in the docstring and in the cell labels (`A:10k+128`), with no header
   line naming what the run is about to measure. The fix gives each kind of report its own function
   returning `(lines, status)` -- `row_report`, `drift_report`, `rss_report`, `footer_report` -- so a
   stream with no content delta is `NOT MEASURED: ... no first token to time` rather than a traceback, a
   capture under `MIN_DRIFT_SAMPLES` is a `NOT MEASURED` line rather than silence, a failed `ps` sample
   prints `unknown` and a failed baseline prints `NOT MEASURED`; `segment_rate(times, lo, hi, ct)`
   brackets the halves of the run's own character axis and converts with the request's
   `completion_tokens`; `run()` wraps the requests in `try/finally: stop_server(proc)` and returns the
   worst status, `main()` returns 2 for a refused configuration and the guard is `sys.exit(main())`; the
   load wait is `wait_for_health(proc, port, timeout=SERVER_LOAD_TIMEOUT)` at 2400 s and nothing is sent
   unless it answers; the install comes from `bench_model()`; the header and cell labels speak characters
   and take the token counts from the server's `usage`; and the docstring's character count is a number
   the suite reads back out and compares against `len(LONG_PROMPT)` -- which is how a prose defect I wrote
   into the fix itself got caught. 36 tests, 16/16 mutants killed, the refusals verified live with
   `Popen` and `HTTPConnection` both guarded; no live-model run, because the operator's server holds 8080.), and
   **AUD-235** (`memory_smoke.py` -- three requests go out and one request's status is read. `status` is bound
   from code-1 and novel-2 and never looked at in either, and the checks those two carry are negative ones
   (`if "ashgrove" in reply.lower():` -- the placement-isolation probe; `if "ashgrove" not in reply.lower():`
   -- the recall probe), so an unanswered request passes them by definition: driven with the store and recall
   answering 200 with the fact and the isolation request raising `HTTPError(500)`, the pre-fix file prints
   `[code-1] HTTP 500 ... ERROR: {"error": "boom"}` and then `SMOKE OK: placement by declared directory, fact
   carried within the project and not across it.` at exit 0. Below that, three sections could report nothing
   while looking measured: `if MEMDIR.exists():` at :133 wraps the whole journal check with no else, and
   `MEMDIR = Path(os.environ.get("TINYTITAN_MEMVAL_MEMDIR", ""))` at :22 is `PosixPath('.')` when the variable
   is unset -- measured in the interpreter, `.exists()` is True and `.rglob('*.ndjson')` walks the directory
   the run was launched from, so the check can pass on files that have nothing to do with it; `consolidation_lines()`
   at :27-30 returns `[]` for an unset *and* for a missing log, which is the same value the caller reads as
   "the engine consolidated nothing", so an unconfigured run sleeps its 90 x 2 s at :102-109 and then blames
   the model; and `model_id()` at :41 has no error path, so a refused connection was an uncaught `URLError`
   traceback rather than a verdict. The fix is `benchmark/test_memory_smoke.py` (46 tests) first, then the
   driver around one report per section, each returning `(lines, status)`: `why_not(row)` names why a row
   cannot carry its check (it raised, it answered other than 200, it held no content) and the floor, isolation
   and recall reports consult it before judging, so an errored request is `NOT MEASURED: code-1 -- HTTP 503
   answered instead of a completion, so the placement isolation check did not run`; `port()`/`memdir()`/`log_path()`
   read the environment when the run starts, return `None` for a blank value so `.` is never a memory
   directory, and a bad port is `REFUSED` at status 2 before any request; `journal_report` prints `NOT CHECKED`
   for an unset variable (naming it) and for a directory that is not there; `consolidation_report` tells an
   unset log, a missing log, a log with no consolidation line and a line that recorded `facts=0` apart, and the
   180 s wait only runs when the file exists; `ask()` catches `OSError` into the row and a non-JSON body into
   `error`; `run()` returns the worst status, the guard is `sys.exit(main())`, and a dead `/models` is
   `NOT RUN ... No request was sent.` The 120-token floor is now `PROMPT_FLOOR_TOKENS` with the measured 149 in
   the comment and a test pinning the floor between 100 and 149. 46 tests, 24/24 mutants killed -- four of them
   survivors of the first run, each becoming a test: a non-200 row carrying a body and a row that raised (both
   branches were masked by the empty-content branch ahead of them), and both `port()` guards (no test refused a
   bad port at all). Verified live with nothing listening on 8096: the two `REFUSED` lines at exit 2 and the
   `NOT RUN` line at exit 1 where the pre-fix driver raised. NOT evidenced: no live-model run -- this check
   needs a memory-enabled server on 8096 and the operator's server holds 8080, so the end-to-end verdict is
   reported as not checked rather than fetched for.)
   **AUD-236** (`tinytitan_3bit_probe.py` -- the headline number is a division by an error that is zero by
   construction. `e4 = rel_error(w, requantize(w, 4))` where `w` *is* the stored 4-bit tensor, so requantizing
   it to 4 bits is exact, `e4` is `0.0` on every input the file can reach, and `ratio = e3 / max(e4, 1e-12)` is
   `e3 x 1e12`: run on a synthetic install the pre-fix file prints a `4-bit` column of `0.0000`, per-tensor
   ratios of 193,958,893,418.31x to 211,191,222,071.65x, `median 3-bit/4-bit error ratio: 205600894987.58x`
   and exits 0, so the docstring's decision rule ("an error several times 4-bit's is a strong signal") can
   never fire against anything. Below that, the sample contained none of the weights the decision is about: the
   docstring says it quantizes "the real routed-expert and attention tensors" and the comment at :132-134 says
   "Attention + shared-expert tensors ... plus the router", while the six-name list holds neither a routed
   expert nor a router -- measured on an installed 125B-A6B MTP build, `model_weights.bin` is 51,681,440 bytes
   with 30 entries whose only `expert` names are the four `mlp.shared_expert*` projections, and 512 routed
   experts per layer sit in `packed_experts/layer_00.bin` at 1,417,674,752 bytes. `dequant4` at :68 assumed
   two nibbles per byte for every entry while the manifest declares the router 8-bit; `crosses_word` at :114
   was computed and then used only to print a yes/NO column, so the file's own second question ("how much extra
   unpack work each weight costs") was never measured; `MODEL` was read at import (:35) so
   `TINYTITAN_BENCH_MODEL` was a no-op; and `def main() -> int` at :128 with `return 0` at :191 as its only
   return, discarded by the bare `main()` at :195. The fix is `benchmark/test_tinytitan_3bit_probe.py`
   (34 tests, model-free) first, then the driver: `ratio_against()` returns `None` when the reference grid is
   no finer than the tensor's own, so the ratio is `NOT MEASURED` naming `TINYTITAN_BENCH_REFERENCE` instead of
   a division by the floor; the sample is built from role patterns against the resident index plus the routed
   experts read through `PackedExperts`/`layout.json`; widths come from the repository's own reader; the
   crossing count and added loads per weight are printed; `bench_model()`/`reference_dir()` read the environment
   at call time; `open_weights(directory, env)` names the variable that named the install it is refusing -- a
   sibling of the AUD-193 wrong-cause class, and it was a mutation-check survivor; `open_experts` returns `None`
   for a dense checkpoint and refuses when a layout describes a `layer_NN.bin` that is not there; a declared
   role the sample did not reach costs the run its status rather than printing a dash row; and the guard is
   `sys.exit(main())`. 34 tests, 22/22 mutants killed -- five survived the first run and became tests (a
   same-width twin reported as "no twin at all", the unasserted `PROBE INCOMPLETE`, the dense assertion
   satisfied by the always-declared router row, no test for a zero-tensor sample, and the wrong variable named
   in a reference refusal), and two survived the second run because the tests asserted only the word "finer",
   which the fixed trailing sentence also contains. Verified against the real installed 4-bit build: no
   reference prints `PROBE INCOMPLETE` and exits 1; the same-width build as the reference reports 2 tensors
   with no twin and 14 stored no finer than their own 4-bit grid, and yields exactly one legitimate ratio --
   the router at 2.15x, the only sampled role stored at 8 bits -- and still exits 1. NOT evidenced: a ratio
   against a wider reference, because there is no 8-bit install under `models/` and none may be fetched for a
   gate, so that path is proven only on synthetic installs; and no model run, the operator's server holding 8080.), and
   **AUD-237** (`tinytitan_ane_attention_probe.py` -- the go/no-go the file exists for was never computed. The
   docstring at :10-13 presents `4.21 s per layer-chunk` as the bar the decision is read against, and no line in the
   file divides by it: `--configs 512:0 --repeats 2` printed `CPU_AND_NE 7.55 ms/chunk-layer` directly beneath
   `GPU reference ... 4,215 ms per layer-chunk` and exited 0. That 4,215 is itself an average over 20 layer-chunks of
   two *unequal* shapes (4096:0 then 2007:4096), so it is not the cost of any chunk the probe measures. The fix makes
   the reference's own arithmetic explicit (`GPU_LAYER_SECONDS`/`GPU_PREFILL_TOKENS`/`FULL_ATTENTION_LAYERS`/
   `GPU_CHUNK_SHAPES`), puts it in the rows' unit through `ms_per_1k_layer_tokens()` (1,381.3) and prints
   `speedup_over_gpu()` per row. Below that: the numerics check had no criterion -- `rel = |got-ref|.mean() /
   max(denom, 1e-9)` at :313 was printed and never compared, so the default list exited 0 at a mean relative error of
   0.0998, and because `max(float('nan'), 1e-9)` is `nan` a non-finite gap propagated through that guard, printed
   `nan`, and was written into the artifact as a literal `NaN`, which RFC 8259 does not allow, at exit 0;
   `MAX_MEAN_REL_ERROR = 0.15` sits above the measured 0.0141-0.0998 band and over what a broken block gives, and
   `error_within_ceiling()` refuses a non-finite gap whatever the ceiling. The CPU_ONLY arm was wrapped in
   `if t <= 2048:` with no else (:317), so the 4096 row printed no wall-clock cross-check and its artifact read
   `"cpu_only_ms": null` -- the same value an errored arm leaves -- while `round(cpu_ms, 2) if cpu_ms else None` at
   :336 turned a measured 0 into that same null; `CPU_ONLY_MAX_CHUNK` is now a named constant, a skipped arm prints
   its reason, and the test is `is not None`. `--repeats 0` built the Core ML program and then died with `IndexError`
   at :309, and a malformed `--configs` was a bare `ValueError` traceback; `parse_configs()`/`parse_repeats()` raise
   `ConfigError` naming the flag and the entry, and a wrong field count is refused as a shape rather than as a
   non-number (the AUD-193 wrong-cause class). One config raising lost the whole artifact, and the artifact path was a
   CWD-relative literal (:352) with no mkdir, so a run from /tmp printed every row and then died with
   `FileNotFoundError` having written nothing; `measure_config()` returns one row per config carrying its own error so
   the sweep finishes, `ROOT` anchors `DEFAULT_OUT` at the repository, `out_path()` honours
   `TINYTITAN_ANE_PROBE_OUT` as a file or a directory, and `write_artifact()` mkdirs the parent, writes a non-finite
   measurement as `null` plus a note, and costs the run its status if it cannot write. `def main() -> int` at :263
   with `return 0` at :355 as its only return, discarded by the bare `main()` at :359, is now `sys.exit(main())` at
   :638 -- 2 for a refused configuration, 1 when a config errored, missed the ceiling, errored its cross-check or
   could not be written, and 0 only when everything measured and passed. 61 tests, model-free (the Core ML modules
   are registered as stubs before the driver is imported), 20/20 mutants killed in a scratch copy so no program was
   converted and no port opened -- the one survivor of the first run became a test, and it was a lesson about suites
   rather than about this file (see the method note below). Verified live with the fixed driver: `--configs
   512:0,512:1024 --repeats 2` exits 0 printing 95.9x and 62.6x against the normalized reference, and the default
   list exits 0 with `PROBE COMPLETE: 4 config(s)` and per-row ratios 87.3x/54.5x/27.6x/20.5x, the 4096 row now
   carrying `CPU_ONLY not run: chunk 4096 is over CPU_ONLY_MAX_CHUNK=2048`, and the *same* mean relative errors as the
   pre-fix run (0.0141/0.0204/0.0292/0.0998) -- so only the accounting moved, not the numerics. The 27.6x at 4096
   corroborates the 26.7x recorded in `docs/v4.4-decode-width-plan.md`. NOT evidenced: no model run, the operator's
   server (PID 17514, port 8080) holding the preflight `pgrep` check, and the GPU half of the comparison is the
   already-measured 84.3 s rather than a fresh number on this host.)
   **This seam -- the entry points that throw their status away -- has drained: the last of the
   original ten closed as AUD-239, and the nine non-`main()` guards it left behind are the report
   seam AUD-240 then measured and closed -- and the one-member asymmetry that seam exposed is AUD-241: `memory_mini.py` was the only one of the
   eight that never read `TINYTITAN_MEMVAL_RESULTS`, so naming a tree from the environment reported a
   tree nobody named, and its `report()` globbed every `*.json` in the directory it did read and took
   each one as its own record, which a sibling's arm file answered with `TypeError: list indices must
   be integers or slices, not str`. The next seam along was measured and is **not filed**: the `/health`
   poll. Six drivers still inline their own (`memory_mini.py:112`, `tinytitan_determinism_ab.py:95`,
   `tinytitan_gate0_profile.py:278`, `tinytitan_hit_fixup_ab.py:89`, `tinytitan_profile.py:233`,
   `tinytitan_ane_prefill_ab.py:132`), every one of them accepting the verdict on `"ok" in body` rather
   than the decoded `status`, and they disagree with the shared helper's documented contract --
   `wait_for_health` returns True on a timeout because "slow is not dead" (`tinytitan_profile.py:215-221`),
   while `hit_fixup_ab.py:94` raises `RuntimeError("server health check timed out")` for the same event.
   Not filed, because following each of those paths to the printed number found no corrupted verdict:
   `determinism_ab` falls through its timeout into `verdict()`, which already refuses a stream that
   carried no content (the AUD-216 family), and `gate0_profile` parses a log that a stale server on the
   port cannot fill with kernel-stat footers, so it dies at :415 naming the run. A substring that would
   be a false positive -- a body whose only `ok` is inside `token` -- was looked for in this repository's
   own health and error bodies and is not there: `/health` answers `{"status":"ok"}` and the 405 answers
   `method_not_allowed`. That is the AUD-182 predicate, unfixed on the python side and reachable only
   through a body this server does not write, so it is reported as hygiene, not filed as a bug. The next
   measured item, also not filed: `tinytitan_m5_sweep.py:281` builds its prompt file with
   `NamedTemporaryFile(delete=False)` and keeps no handle to close it (the `ResourceWarning` a CI run
   prints), unlinking at :334 -- nothing else reads that directory, so the cost is a leaked descriptor
   and a leftover file per interrupted sweep, not a wrong number. The same sweep, recorded so the next re-sweep reads a result rather than
   starting over: an AST walk of the 86 production python files under `benchmark/` and `tools/` found 32
   handlers whose body is only `pass`/`continue` and no bare `except:` at all. Two of the 32 needed
   following. One is AUD-242 (`resolve_api_model` answered the model identity from its own default
   whenever asking failed). The other, `tools/internal-speeds.py:435`, swallows an `Exception` around the
   `hw.memsize` sysctl and is not filed, because it degrades a record's environment block rather than a
   compared metric, and AUD-210 already gates the comparison that reads it. The 27 `json.JSONDecodeError:
   continue` handlers are candidate scanners, not verdicts: `extract_quiz` (`memory_book.py:254`,
   `memory_correct.py:341`, `memory_projects.py:378`, `memory_value.py:247`, `memory_volume.py:556`) does
   conflate a model that answered with no JSON and one that answered with JSON it cannot read, and that
   conflation is the instrument's documented contract -- `test_memory_mini_report.py`'s `answered()`
   fixture says so in its own docstring: a JSON-less answer *is* the measured zero the fidelity score is
   about -- and every caller keeps the full text anyway (`memory_book.py:323` writes each session's answer
   to its own `.md`, `memory_mini.py:242` keeps a 400-char `raw`). Not filed: the reader can see the
   answer, and the score is honest about what it scored. **The tree that holds the records is next along, and it is AUD-243:** `Path(os.environ.get(ENV, default))` accepts the empty string, so `TINYTITAN_MEMVAL_RESULTS=` bound every driver's results tree to `Path('')` -- the directory the command happened to run in. Measured from `/tmp`: the report printed `NOT MEASURED: no results in .` and exited 1, naming a directory that identifies nothing, while `memory_small_model.py`'s `os.environ.setdefault` left the blank in place and then `mkdir(parents=True, exist_ok=True)` succeeded against `.`, so a run wrote its records where no report looks. It is fixed as one function in `benchmark/memval_env.py` read by all nine sites, so a driver cannot omit the binding the way AUD-241 found `memory_mini` omitting it. **Its sibling was AUD-244, held pending the operator's decision and now closed:** `memval_master.sh` ran `report-all` as a statement, so the fixture's clean run printed `NOT MEASURED: no run of any scenario under ...` and then `##### MASTER DONE ... all 10 scenarios exited 0` with exit 0 -- the status AUD-240 added was unreachable from the script that calls it. The fix is two-branch: a scenario that failed, or every scenario exiting 0 over a tree the report cannot pool, both exit 1, with the summary naming `report-all exited N` and "no result record to pool". It needed the fixture to stop encoding the defect: the stub `memval_run.sh` wrote no record at all, so `test_a_clean_run_exits_zero` was pinning a run-that-measured-nothing as expected behaviour; it now leaves one record per arm (`RECORDS=0` takes them away) and the assertion is unchanged. The docstring's "`report-all` returns None, so its exit status is 0" had been false since AUD-240 and is corrected -- correcting a false sentence about the subject under test is a different act from weakening a test to pass, and the distinction is what the operator asked to keep visible. The `report_all` header placement above its refusal is **still the operator's call**. One correction to a closed row came out of this close's own gate run: `test_memory_mini_report.py` (AUD-241) was committed unformatted, because that close verified with a glob that does not match `test_*`; ruff 0.16.7 now formats it and `lint.sh python` is clean over the whole tree. The lesson is recorded here so the next close runs the gate, not a glob. **AUD-245 is the same tree one line further along, and it came from sweeping the environment rather than the code:** `TINYTITAN_MEMVAL_RUN=` collapsed three launches into one counted run, because the run token is part of every result file's name while the report's `*-r*.json` glob still matches it -- measured at 3 rows against 1 row for the same three launches, exit 0 either way. The sweep that found it classified every `TINYTITAN_*` read by what a blank value does, and the classes that were *not* filed are recorded so the next re-sweep does not reopen them: `float("")` dies loudly (`TINYTITAN_MEMVAL_TEMPERATURE`), and `TINYTITAN_MEMVAL_SERVER_LOG`, `TINYTITAN_MEMVAL_MEMDIR` and `tinytitan_3bit_probe.py:109` all read blank as unset and say so. One candidate whose first probe was withdrawn as unsound came back as **AUD-246, and the withdrawn instrument turned out to be right about the number and wrong about the cause**: rebuilt to fail observably -- the script's own arithmetic and `/usr/bin/seq` run in isolation, with stdout, stderr and status captured separately and the loop counted -- it shows `RUNS=0` giving `seq 1 0`, which is not an empty list but a *countdown* emitting "1" then "0", so a day's repeats become two and the second lands in `{arm}-r0.json`; `RUNS=-1` counts down to -1; `FIRST_RUN=0` makes `seq 0 2`, renumbering the interleaved repeats the documented recipe exists to keep apart and overwriting the earlier launch's records, all with the script exiting 0. That is fixed at the two reads in `memval_run.sh`, before the script makes a directory or looks for a binary, and `benchmark/test_memval_run_repeats.py` drives a copy of it in a temp checkout whose stub launcher prints and dies, so a guard that slipped could not start a model -- it would fail on `no release binary` instead. The falsified half is recorded as *not filed*: `RUNS=\"${TINYTITAN_MEMVAL_RUNS:-3}\"` uses `:-`, which substitutes for null as well as unset, so a **blank** count never reaches the arithmetic and runs the documented three repeats, which is what the report then shows -- so that suite asserts the blank takes its default rather than refusing it, and the first probe's `iterations=2` was a real countdown reached by a path the script does not have. **AUD-246's sibling sweep found no sibling**: every other `seq` loop in a tracked shell script is bounded by a literal (90, 50, 60, 30, 100, 240 -- all readiness waits that read no environment), and the one arithmetic site that turns an operator's number into a value, tools/server_launcher.sh:221's `ram_tier`, already refuses non-digits, the empty string and anything under its floor -- so the guard shape existed here and `memval_run.sh` was the site that had not used it. **AUD-247 is the proof that step 5 of the issue workflow pays: `memval_master.sh`'s `TINYTITAN_MASTER_SCENARIO=photograph` looked like shell noise, and it was the scope of a whole table.** `report_all()` reads all ten scenarios but calls `score_run()`, which takes no spec and uses the module globals bound once at import. `aggregate()` had always saved and swapped them per scenario; `report_all()` had not. Measured over one tree with a full-mark record for each arm of every scenario, run twice with only the environment differing: named `photograph`, nine of ten summary rows printed a zero numerator over **photograph's** key counts scaled by their own sessions (`ledger 0/35 0/63`, `cohort 0/35 0/63`, `filing 0/30 0/54`), and named `filing`, `photograph` printed `0/36 0/9` -- exit 0 either way, which is exactly what AUD-244's verdict cannot distinguish. The shape of those numbers is why this is a finding and not a cosmetic one: they do not read as `0/0` "not measured", they read as scenarios that ran and scored nothing. Fixed with the same save/swap/restore `aggregate()` uses; three new tests pin the denominators as arithmetic rather than as a golden, pin that the table is identical whichever scenario the shell names, and pin the restore -- the fix itself mutates module state, so the restore is part of it and is guarded. Sibling sweep came back clean, and the sweep is recorded so a re-sweep does not redo it: `memory_scenarios.py:331` iterates into a local `world`, `t6_prose_cases.py:429` names `photograph` in the record paths it reads so its hardcoded lookup is its declared scope, and the other five `memory_*` drivers bind their own quiz constants with no scenario table to walk. **AUD-248 was task-list debt, paid by re-measuring instead of recalling.** The note carried from the L5/L6 pass read `two run stamps read local time, five read UTC`; the count was stale (it is two against **eight**), and the assumption underneath it was worse -- the earlier note treated the zone as a fixed offset, and this Mac runs Europe/Berlin, which repeats 02:00-02:59:59 on 2026-10-25. So the inconsistency is not stylistic: `tinytitan_benchmark.py:604` and `tinytitan_longctx.py:443` build a *results name* from that clock, and a name that two runs share is a record the later run overwrites. The probe replays both write sites and prints `records lost: True`. Fixed with one UTC helper in the profile module both drivers already import, plus `date -u` for `dsh_route.sh`'s settings backup, and guarded by a test that walks every name-shaped timestamp in `benchmark/` and `tools/` in both languages -- because the fix at three sites is only durable if a fourth cannot appear. Two process notes from this close: the first fixture collected **zero** tests because a shared base class did not derive from `unittest.TestCase` -- a suite that collects nothing is a failure of the instrument, and it is reported here rather than smoothed over -- and `lint.sh docs` refused the CI registration until the new suite was *tracked*, so a suite goes into git before it goes into the workflow. One of the seven tests does not discriminate against the mutant; it is recorded as a shape test rather than claimed as a guard. **AUD-249 came out of the next seam along the same tree: which matches a report *reads*.** The result leaf is named after the install (`memval_run.sh:33 LABEL="${TINYTITAN_MEMVAL_LABEL:-$MODEL-${QUANT}bit}"`), so a tree that saw two installs holds two leaves for every scenario — and `aggregate()` took `next(sorted(root.glob(f"memory-{name}-*")))`, the alphabetically-first one, while printing the heading "pooled (every scored key-instance, all runs)". Measured over one tree with photograph answered right in `memory-photograph-aaa-model-4bit` and unanswered in `memory-photograph-zzz-model-8bit`: aggregate exited 0 with `worlds: 1` and pooled `carryable 45/45 100.0% | invalid 0`, while `report-all()` over the *same* tree read both leaves and printed `runs 2 ... invalid 10 ... cost 34`. The sweep that exists to compare installs therefore reported one install as the world, the second install contributed nothing to any figure, and no line named a directory or said how many it had read — the two commands disagreeing over one directory is the proof of the intent. It now pools every matching leaf into one world and prints `{leaves} leaf`/`{leaves} leaves` on the world line, so the heading's promise is checkable; a world stays a scenario rather than a leaf, because the unweighted mean is over worlds and treats them equally, so promoting each install to a world would silently weight a scenario by how many installs ran it. Two of the four new tests pass before the fix and are labelled as such in the class docstring, since what they pin is the shape the repair must keep (one world per scenario, and the deliberate `-firstpass` exclusion) and not the defect; the mutant `directories[:1]` is killed by exactly the two that discriminate. Two things this close did not smooth over: the group run in an order I made up myself (`sort -u`, not the workflow's order) failed two `test_memval_master_exit` tests, and instead of concluding my fix had broken them I re-ran the same order against HEAD's copy of `memory_master.py` — it fails there too, so it is a separate, pre-existing, order-dependent defect: `memory_small_model.py:40` publishes `TINYTITAN_MEMVAL_RESULTS` into `os.environ` at import, and `run_driver` hands `dict(os.environ, ...)` to the child, so the child's report reads another test's tree instead of the temp one its own stub wrote. That is **AUD-250**. The second is `aggregate()`'s `-firstpass` filter with no `-firstpass` writer anywhere in the tree, which is **now the operator's call** because `report_all()` has no such filter and picking one side changes what a future first-pass run counts. **Its two halves, measured.** Isolating the two failures suite-by-suite named `test_driver_documented_commands` as the polluter and a per-case walk named the test and the leaked value: `{'TINYTITAN_MEMVAL_RESULTS': (None, '.../memory-small-qwen2b'), 'TINYTITAN_PORT': (None, '8098')}` -- two knobs, not the one the note predicted. The production half is `memval_master.sh:40`: it ran `report-all` on whatever tree the environment named, and `TINYTITAN_MEMVAL_RESULTS` is a documented knob of these drivers, so the sweep's verdict could be taken from records it never wrote -- measured, a stubbed ten-scenario run that wrote nothing printed the *repository's own recorded runs* (`photograph summary 3 116/135 ... 2743`) under `MASTER DONE ... all 10 scenarios exited 0` at exit 0. It now names the tree it aggregates, the same default `memory_master.py:51` binds when the variable is unset. The suite half is that `memory_small_model.py:40` publishes at import -- right for a run, because the harness it then imports must write where it does, wrong for an inspection -- so those two fixes are one door: driver imports reload under `mock.patch.dict`, and a test asserts that reading a driver leaves the environment as it found it. Two things the instrument got wrong first, both recorded rather than smoothed: the hygiene test's first draft **passed vacuously**, because unittest's own ordering ran it after the drivers had already leaked, so it now clears the two knobs before snapshotting; and the fixture pin I tried first made the decoy test pass *while the script was still broken*, which is why the pin came out and the fix went into the driver -- a test that answers for a workaround proves nothing. Sibling sweep: all 27 model-free python suites were run in their own process and diffed for environment changes (`python3 .build/aud250/siblings.py`), and every one reports `leaks=none` now, so `memory_small_model.py` was the only import-time publisher; `memval_matrix.sh` reads no results tree so it has no victim side, and `memval_run.sh:244,252` already names its tree on both run and report. The recorded CI order (line 493 before line 498) is why 727 tests passed over this defect for a week -- a suite list that happens to run the victim before the polluter is not a guarantee, and the group is 729 now. Decisions still sitting with the operator: `report_all`'s column header above its refusal, ratification of the AUD-246 blank-premise test conversion and the AUD-244 stub/docstring correction (both disclosed with their diffs, neither a weakened assertion), whether a `-firstpass` leaf should be excluded from `report_all()` as it is from `aggregate()` (nothing writes one today, so the answer only shapes a future run), whether `memory_small_model.py`'s results-tree publish should move out of module scope into its `__main__` guard -- it would remove AUD-250's mechanism for every future in-process importer, and `test_memory_reports.py:572` asserts it happens *at import*, so that is a change to a test this audit wrote rather than a repair, and whether `test_driver_documented_commands.py:45`'s `sys.path.insert(REPO / "benchmark")` should be corrected (it resolves to `benchmark/benchmark`, inert while CI runs `cd benchmark` first), and the push -- nothing has been pushed, `main` is 236 ahead, and `git push` needs an explicit go, which this row re-measured rather than incremented (a publish is two pushes: this repository's `main` and the wiki's `master`).** **AUD-251 is the seam that came next, and it is measured and NOT FILED.** After a verdict taken from the wrong tree, the question was which tree each path constant names at all: `python3 .build/aud251/bases3.py` reads every `ROOT`/`REPO` binding in `benchmark/*.py` and `tools/*.py`, resolves each `ROOT / "…"` use against it, and reports a use whose first two segments repeat or whose path leaves the repository. Exactly one is structural: the `benchmark/benchmark` in `test_driver_documented_commands.py:45` already named above. The other eight apparent faults were my instrument, not the tree -- three files take `ROOT` from a parenthesized `from tinytitan_profile import (ROOT, ...)`, which a body-only scan does not see, and five are dotted attribute reads (`benchmark.ROOT` through an `import x as benchmark` alias, and `prepare.REPO`) that the path regex matched through the dot. Each was then resolved by importing the module rather than by reading its text, and `tinytitan_profile`, `ornith_concise_tool_ab`, `tinytitan_hit_fixup_ab` and `ornith_four_program_matrix` all bind `ROOT` to the repository root -- so no driver or suite references a path constant it does not bind, and `prepare_qwen38.REPO` is the Hub id `Qwen/Qwen3.8-Flash-Next`, not a directory. The paths that do not exist on disk are two kinds and both are expected: model installs under `models/`, which this checkout deliberately does not carry, and output trees a run creates (`.build/benchmark-logs/…`, `.build/benchmark-results/…`); neither is a mis-resolution. The probe's own blind spot was the second convention it could not see: ten files resolve a path with `os.path.dirname(os.path.abspath(__file__))` (eleven sites, and two more in `tools/`), and the one that matters is `tinytitan_benchmark.py:156`, which lands `bench-<cell>-<stamp>/aggregate.json` in `benchmark/benchmark-results/` while the probes beside it write `.build/benchmark-results/`. That is not the AUD-250 shape, and the check was made rather than assumed: the directory is git-ignored (`.gitignore:55`), it holds only real sweeps (`bench-cache_on_mtp_off_4bit-2026…`, `capital-of-paris-20260911T1935`), and `test_tinytitan_benchmark.py:384` patches `results_directory` out before `main()` runs, so the matrix suite writes to a temp tree and never into the operator's. The hazard of writing synthetic rows there is exactly what the m5 sweep's output override (item 5, closed under AUD-239) exists to prevent, and the matrix driver does not need it because its tests do not reach it. So: measured, not filed, recorded so the next re-sweep does not reopen it.** **AUD-252 came out of the list that sweep left behind, not out of the code.** `tools/lint.sh docs` prints `59/63 python suites registered in CI`, so the four unregistered ones are the exemption file `tools/python-suites-unrun.txt`, and an exemption is a claim. Three were re-measured by running them: 5, 22 and 6 tests skipped, and `benchmark/requirements.txt` still carries only numpy, safetensors and ml_dtypes, so CI really has no coremltools. The fourth could not be run -- it starts a server per model -- so its file was read, and the reading found the defect: `test_concurrent_sessions.py` promises that "three controls keep a clean result meaningful, rather than the silence of a detector that never fires", and its launched path gates the verdict on exactly that (`:632`), while its `BASE=` branch -- the one the same docstring tells an operator to run against a server they already have -- returned `0 if foreign == 0 and http_error == 0` (`:695`), dropping the canary and the cancellation controls and printing no summary either. Measured against a loopback stub answering every completion with empty text: `detector flagged: NOTHING -- the detector is blind`, `survivors intact: False; server answers afterwards: False`, **exit 0**. An empty answer carries no foreign marker precisely because it carries no text at all, so the dropped predicate was the silence wearing the shape of a clean run. One predicate and one summary now serve both entry points (`reset_state()`, `separation_state()`, `print_verdict()`), and `BASE=` resets too -- which is what lets the two stub modes share a process without inheriting each other's totals. The scope the original comment defended is kept and pinned: answer quality stays reported beside the verdict and out of it, and an `echo` stub that answers each session from its own prompt still exits 0. That third test passes against HEAD as well and is labelled in the class docstring as the shape to keep, not claimed as a guard; the two that discriminate kill the mutant that restores the old predicate. Sibling sweep: this is the only harness in the tree with an external-server mode (`grep -l 'environ.get("BASE")'` over `benchmark/` and `tools/`), and the launched path's `if not present: raise SystemExit("no installed model to test")` was checked rather than trusted, because `all(results.values())` over an empty dict is the AUD-212 shape and is unreachable only thanks to that line. `benchmark/test_concurrent_sessions_verdict.py` is tracked before its workflow line, so the list is 60/64 now, and the subject stays exempt with its reason unchanged. The process note is the one the operator asked to keep visible: the first RED run I reproduced was restored from a copy taken before the docstring and `ruff format` pass, which silently reverted committed work -- the restore is now from the commit, and `git status` clean after it is the check. Verification: new suite 3 OK, CI model-free group 732 OK in recorded order (was 729), `lint.sh python` and `docs` ok. No model run -- the operator's server (PID 17514) was untouched and the stub is mine -- **AUD-253 is the same verdict family on the shell side, and it came from sweeping the scripts rather than the drivers.** `tools/install_models.sh --all-4bit|--all-8bit` each looped `CATALOGUE` with a body whose last statement was the row filter's own test (`[[ "$w" == 8 ]] && install_one "$n"`), and the dispatcher `case` is the file's last statement, so the script's status was the *last catalogue row's width test* -- and that row is `qwen35-9b-8bit`. Measured on `/bin/bash 3.2.57` by replaying the dispatcher's own branch text with `install_one` stubbed (`python3 .build/aud253/replay.py`): `--all-8bit` with `qwen38flash-8bit` refusing installed 7 of 8 and exited **0**; `--all-4bit` with nothing refusing installed 11 of 11 and exited **1**. The two modes fail in opposite directions, and the 8-bit one is the silent half: a gated checkpoint that refuses without an `HF_TOKEN` is a per-row failure, so one missing model left the verdict reading as success. Both spellings are in the user-facing installation guide (`Installation-and-Configuration.md:65`), where the exit status is all a reader has. Fixed as `install_width`, which runs every row of the wanted width, collects what it missed, prints `installed N of M <width>-bit model(s)` and returns 1 when N is not M; `benchmark/test_install_models_batch_status.py` drives the branch text extracted from the dispatcher, so the instrument cannot drift away from what the script does and the extraction itself refuses if the dispatcher is rewritten -- and it uses the stub idiom `test_install_models_menu.py` already establishes, so nothing here can download, build or start a model. Two of its five tests fail before the repair; the other three pin the shape it had to keep (the filter selects exactly the 11 four-bit and 8 eight-bit rows, counted from the CATALOGUE the test parses rather than from a golden, and a clean sweep stays clean), and the `return 1` -> `return 0` mutant is killed by exactly the two that discriminate. Sibling sweep across all 26 tracked shell scripts: the other 40 `[[ ... ]] && cmd` sites assign, echo or `continue` rather than performing side-effecting work whose verdict leaves the scope, and the five scripts that act on many things at once each collect and test their own failures -- `memval_master.sh` and `memval_matrix.sh` (`failed`/`failed_count`, and their FAILED lines print `N of M`), `ane_sidecars.sh` (`exit 1` before the final `exit 0`), `assert-arch.sh` (dies on the first wrong artifact and refuses an empty directory rather than passing by vacuity), `verify_cpu_models.sh` (`status=0` ... `exit "$status"`, and a crashed benchmark still fails the per-model `grep -q`); `tsan-storm.sh`'s same-family defect was AUD-218. **Not filed, and recorded here so a re-sweep does not redo the reading:** `tools/repack_dense.sh:91-96` `continue`s past a model whose snapshot is gone and its closing line prints `all checks passed` over only the pairs it verified, so a partial run says `all` about an incomplete set -- each skip is named on stdout, an all-skipped run does exit 1 through the `nothing to verify` guard, the script is a one-time migration runbook whose own header reports the migration as done, and no branch of it is reachable without a model install, so there is no hermetic instrument to pin it with; whether the closing line should name the skips is **the operator's call**. One process note kept visible because it is the class this audit was founded on: the mutation restore was run as `git checkout -- tools/install_models.sh` *before* the fix had been committed, which reverted the fix itself. It was recovered from the pre-mutant copy and checked with `git diff` (33 insertions, 4 deletions, the intended change only) before committing -- the lesson being that "restore from the commit" only works once the change is in a commit, so a mutation check against uncommitted work must restore from a copy taken after its last edit. Verification: new suite 5 OK, the Installer gates CI step as recorded runs 192 OK (187 measured with the new suite removed), the model-free group 732 OK unchanged, and the gates re-run clean -- shell-portability over 27 scripts on system bash 3.2.57, shellcheck 0.11.0 with no warnings, ruff 0.16.7, and the docs count at 61/65 suites registered. (Their mode names are left unbackticked here on purpose: a paragraph that names three of them turns every other hyphenated token in it into a claimed mode, which is how this close's own measurement of `qwen38flash-8bit` got read as a lint gate.) No model run -- the operator's server (PID 17514) untouched -- and nothing fetched.**  Measured at the time by
   non-test driver's `__main__` body in `benchmark/`: nine called `main()` and discarded
   what it returned
   -- `memory_smoke.py:153` (a line AUD-235 rewrote; the guard is `sys.exit(main())` at :408 now),
   `tinytitan_3bit_probe.py:195` (a line AUD-236 rewrote; the guard is `sys.exit(main())` at :476 now),
   `tinytitan_ane_attention_probe.py:359` (a line AUD-237 rewrote; `sys.exit(main())` is at :638 now),
   `tinytitan_ane_realweight_rehearsal.py:201` (a line AUD-238 rewrote; `sys.exit(main())` is at :393 now),
   `tinytitan_longctx.py:212` (a line AUD-234 rewrote; the guard is at :466 now),
   `tinytitan_longgen.py:136` (AUD-232; :248), `tinytitan_maxthroughput.py:176` (AUD-230; :256),
   `tinytitan_slots_ab.py:125` (AUD-231; :190) (and
   `tinytitan_benchmark.py:657` until AUD-233; the guard is at :784 now) -- while the rest of the guarded files use
   `raise SystemExit(main())`, which is the same thing done right. (Those line numbers are the
   `main()` call itself; the paragraph that first listed them pointed one line higher, at the
   `if __name__ == "__main__":` guard above each, and was corrected by re-running the grep rather
   than by remembering.) Nine were closed then, and the sentence said "Eight" until AUD-239
   re-read it: AUD-230 found `tinytitan_maxthroughput.py` returning no
   status anywhere in the file, AUD-231 `tinytitan_slots_ab.py` keeping its only `sys.exit` inside the
   per-arm function, AUD-232 `tinytitan_longgen.py` averaging its warm-up into the headline mean and
   printing `mean=0.00 tok/s` for an arm whose log held no footer, AUD-233 `tinytitan_benchmark.py`
   measuring its own denominator out of its survivors, AUD-234 `tinytitan_longctx.py` printing the
   second quarter of a generation as its `first-half` in a guessed unit, AUD-235 `memory_smoke.py`
   reading one status out of three requests and printing `SMOKE OK` over an isolation probe that answered
   500, and AUD-236 `tinytitan_3bit_probe.py` dividing its headline ratio by an error that is 0.0 by
   construction while sampling none of the routed experts the verdict is about, and AUD-237
   `tinytitan_ane_attention_probe.py` printing a GPU reference of 4,215 ms per layer-chunk under every
   row and never dividing anything by it -- the go/no-go the file exists for was never computed, and a
   512-token row read `7.55 ms` beneath it; and AUD-238 `tinytitan_ane_realweight_rehearsal.py`, which had
   no verdict to fail at all -- `return 0` its only return, the per-layer error compared to nothing, and a
   layer list, a tensor stem and a per-tensor width each hardcoded against an install that contradicts all
   three. The one line number still open was re-run for AUD-238 and still held, and the same re-run of the
   whole `__main__`-body scan then named only `tinytitan_m5_sweep.py:280`. Re-running that grep for AUD-233 turned up a
   **tenth** site the nine never named: `tinytitan_m5_sweep.py:280` ended in a bare `main()` too (a line
   AUD-239 rewrote; `sys.exit(main())` is at :380 now). It
   was in the original list for no reason recorded anywhere -- `memory_smoke.py` and
   `tinytitan_longctx.py` were named there while returning nothing at all, so the omission was not a
   rule about what the list counted, and a remembered list is no substitute for re-running the grep.
   The scan was re-run for AUD-239 and it now names **zero** status-discarding `main()` sites: of the
   files with a `__main__` guard, every one that calls `main()` calls it through `sys.exit(main())`
   (`grep` over the ten closed drivers returns :256, :190, :248, :466, :784, :408, :476, :638, :393 and
   :380 -- maxthroughput, slots_ab, longgen, longctx, benchmark, memory_smoke, the 3-bit probe, the
   attention probe, the rehearsal and the sweep). The ten guard bodies that still do not match are not
   the same shape: `launcher_fixture.py:179` is `unittest.main()` and is a fixture, and the nine
   `memory_*` dispatchers (`memory_book.py:431`, `memory_correct.py:606`, `memory_master.py:523`,
   `memory_mini.py:467`, `memory_projects.py:587`, `memory_sim.py:680`, `memory_small_model.py:65`,
   `memory_value.py:328`, `memory_volume.py:783` -- re-measured after AUD-240, which added lines
   above every guard but `memory_sim`'s and `memory_small_model`'s) dispatch a `sys.argv[1]` command
   and never define or call a `main()` at all.
   Six of the nine are the `run_arm(command)`/`report()` shape, a seventh call
   site is `memory_small_model.py:83` running `bench.run_arm(arm)` against an imported sibling, and the
   odd two are `memory_mini` (`serve`/`run`/`verify`/`report`) and `memory_sim`
   (`validate`/`tail`/`detail`/`compare`). Measured while listing them: `run_arm` has no `return` in any
   of the six files that define it, so there is no status there for a guard to discard -- the shape is
   the opposite one, an arm that can only fail by raising -- and re-checked per file for the swallow
   that would make that wrong: the only `raise`/`except` lines inside any `run_arm` body are
   `memory_projects.py`'s two `raise SystemExit("ABORT: the projects share a workspace id: ...")`
   guards, which abort loudly, so `memory_small_model.py:83`'s discarded `bench.run_arm(arm)` cannot
   exit 0 over an arm that failed. This row used to say the `report()`s in `memory_correct`,
   `memory_master` and `memory_volume` "return a formatted percentage string, not a status", and that
   was read off the wrong `return`: those strings come from the `percent()` and `_pct()` helpers
   defined *inside or beside* `report()` (`memory_correct.py:569`, `memory_master.py:328` nested in the
   run loop and `:412` at module scope, `memory_volume.py:737`), while the report functions themselves
   returned nothing -- `memory_correct`'s pre-fix refusal ended on a bare `return`, which is `None`.
   Re-read for AUD-240, all three returned `None`.
   `memory_mini.py:480` was already the one dispatcher that propagated what it got (`raise SystemExit(report())`,
   and that `report()` returns `1 if total_refused else 0`). Whether those nine can print a clean verdict
   over arms that all failed **was** the open question, and AUD-240 measured it: seven of the nine
   dispatch a `report`, and every one of those seven printed its column header and exited 0 over an
   empty `TINYTITAN_MEMVAL_RESULTS` -- `memory_projects` answering its `Leaks, one line each:` section
   with the single word `none`, and `memory_mini` printing the 89%/100% fidelity preamble and "the 35B
   spends 45-55 s" over rows that had never been computed, while
   `memory_master.aggregate()` printed both arms' `carryable 0/0 n/a` pooled rows and then died in
   `statistics.mean([])`. The two that were *not* in that set are `memory_sim`, which has no `report`
   command at all, and `memory_small_model`, which runs an arm and never reports. All seven report
   guards are now `raise SystemExit(report())`, and `memval_run.sh` -- whose last command is `report` --
   exits on the report's verdict.
   Read what each of the ten was by kind, because they were not one shape. `tinytitan_ane_realweight_rehearsal.py` (:105)
   declared `def main() -> int` and ended on `return 0` (:197), so the handover's question there -- does a
   failed probe return anything *other* than 0 -- was answerable by grep and was answered for it: `return 0` was
   the only `return` inside its `main()`, and it contained no `sys.exit` or `raise SystemExit` anywhere, so it
   returned 0 whatever it measured. The three seam questions then found that the discarded status was the
   smallest part, and it is closed as AUD-238: what it returned 0 for was a rehearsal of ten layers where the
   manifest declares twelve, read through a tensor stem the installed family does not use, at a width 98 of
   those 1,079 entries are not, with no ceiling anywhere between the measured error and the verdict.
   Its sibling `tinytitan_ane_attention_probe.py` had that exact shape (:263
   and :355) and, like `tinytitan_3bit_probe.py` before AUD-236 (:128 and :191), its `return 0` was the least
   of it: the verdict it returned 0 for was a comparison the file never made, and its numerics "sanity check"
   had no ceiling, so the default list exited 0 at a mean relative error of 0.0998 and a non-finite gap would
   have printed `nan`, written a literal `NaN` into the JSON artifact and still exited 0. `tinytitan_m5_sweep.py:137`
   returned no status anywhere, which is the AUD-234 shape `tinytitan_longctx.py:165` had until it was closed,
   and its one `sys.exit` at :156 was the missing-CLI message, which exited 1 -- the same status an errored
   run would have carried -- through `sys.exit(str)`. It is measured and closed as AUD-239: the driver now
   returns 2 for every refusal (missing CLI, a `--slots`/`--chunks` list that reaches `int()` with a name for
   the flag, a context too small for the request), 1 when any run errored, was interrupted, or the results
   file could not be written, and 0 only when every planned run was measured and written; the footer counts
   against the plan rather than the survivors, the results honour `TINYTITAN_M5_SWEEP_OUT` instead of always
   landing in `benchmark/benchmark-results/`, and a clean exit with no timing footer is an error row rather
   than a measurement. The characterization this paragraph carried for
   `memory_smoke.py` -- that its only `raise SystemExit(1)` at :145 was "the AUD-231 shape where one failure
   ends the run rather than the sweep" -- is false, and reading it for the three seam questions is what found
   that: the raise is the *last* statement in `main()`, after all four checks, so no run ever ended early,
   and the real defect was the two statuses it bound and never read. It is measured and closed as AUD-235.
   What is left along a seam needs the same three questions read against it -- what does the file run at
   import, does its prose match what runs, and can the verdict it prints be passed by a run that measured
   nothing. On this seam the entry-point half is drained, so those questions now point at the
   `memory_*` dispatchers and the module-scope `os.environ` reads named below; neither has been read for
   them yet. The
   suspect this paragraph carried for `tinytitan_benchmark.py` is measured and closed as AUD-233 (it
   printed `Port {port} FAILED to become ready!` at :641, `continue`d, and reached the unconditional
   `{label} COMPLETE` at :653 with exit 0); the second suspect,
   `tinytitan_longgen.py:127`'s `mean = sum(rates) / len(rates) if rates else 0`, was closed as
   AUD-232. The unguarded-`__main__` list itself has drained: `for f
   in *.py; do grep -q '__main__' $f || echo $f; done` now names only `tinytitan_profile.py`, which
   is a library, and `test_qwen38_resume_e2e.py`, which is a test. Module-scope `os.environ` reads
   do survive in `tinytitan_knob_sweep.py:52` (the model id; its token count moved into
   `measured_tokens()` under AUD-277 so a bad `SWEEP_TOKENS` is a refusal rather than an
   import crash),
   `tinytitan_overlap_measure.py:40`, `capital_of_paris_report.py:37-43`,
   `tinytitan_vs_competitors.py:91-100` and every `memory_*` driver
   (`memory_book.py:45-53`, `memory_correct.py:58-65`, `memory_master.py:47-61`,
   `memory_mini.py:47`, and their siblings), but on their own they only freeze a value an importer
   cannot re-set, so they are worth
   reading while in the same file rather than a finding by themselves. `memory_volume.py:428` calls
   `check_scenario()` at module scope too, and it is a different kind: it audits the `QUIZ_PLAN`
   fixture and raises `SystemExit` on a duplicated key, so it is an import-time guard rather than
   an import-time benchmark.
   The argv half of the seam is drained, and re-run for AUD-239: `grep -rn "sys\.argv" benchmark/*.py`
   leaves no hit at column 0 in a non-test driver -- the only column-0 hit anywhere is inside a test's
   child-process heredoc -- but the wording this row carried, that every remaining read "sits inside a
   function", is imprecise and is corrected here: the nine `memory_*` reads sit at module scope *inside
   their `__main__` guard*, which is indented but is not a function. The rest do sit inside functions.
   The other half of the claim also holds, now by AST rather than by grep: no call to `open`,
   `subprocess.run`/`Popen`/`check_output`, `urlopen`, `HTTPConnection` or `os.system` sits outside a
   `def`/`class` and outside a `__main__` guard in any non-test driver -- 0 hits over the 62 non-test
   drivers in a 122-file directory. That second pass matters: the first version of the walk descended
   into function bodies and reported 160 hits, which was the scanner being wrong and not the tree, and
   the number only became trustworthy when it was read against the code the scan claimed to have
   skipped.
   A method note from AUD-228's mutant run, and the kind worth keeping: one mutant **survived**
   the first ten, and it was a finding about the *suite*, not the mutant. The suite asserted that
   `bench_model()` reads `TINYTITAN_BENCH_MODEL`, and the mutant left that function untouched but
   passed a hardcoded path at the `server_command` call -- so the driver would have printed the
   operator's model and launched somebody else's, and every test still passed. A fix that moves a
   value is not pinned until an assertion follows the value to the place it is used; add the
   call-site assertion (`test_the_named_model_is_the_one_handed_to_the_launcher`) rather than
   trusting the unit test of the reader. The same blind spot existed in AUD-227's suite and got the
   same assertion.
   `claude_openai_adapter.py`, which an
   earlier note here listed as carrying the same shape, does not: it reads no argv at
   all (`grep -n argv` over it is empty), so it is off the list on measurement.
   A method note the second finding cost: a mutation harness that greps for `FAIL:`
   and `ERROR:` lines reads a *crashed* run, which prints neither, as a surviving
   mutant. Check the runner's own `Ran N tests`/exit status, not just its failure
   lines. Its corollary, learned on AUD-225: when the mutant is the import-time
   server start, *do not run it* -- guard `subprocess.Popen` in the child and give
   the mutant a stub argv, so the regression trips the guard (`RuntimeError: Popen
   ran at import`, rc 1) instead of loading a model, and prove the guard is not
   hollow with the same child against the current file (`imported clean`, rc 0).
   A second corollary from AUD-226, where the side effect was a file read rather
   than a process: the same child works with `builtins.open` guarded for the names
   the driver wants, and it is cheap enough to use on the control run, so "the
   guard is hollow" cannot be the excuse for leaving an import-time read unfixed.
   A method note from AUD-237's mutant run, and the sharpest one of the batch: switching
   `if len(parts) != 2:` out of `parse_configs()` **survived** the suite, because the generator unpack on the very
   next line raised its own `ValueError`, which the adjacent clause caught and re-raised as "holds a token count that
   is not a number". The run still refused, still named `--configs`, still exited 2 -- for the wrong reason -- and
   every test passed, because each asserted only *that* a shape was refused. A refusal test that asserts the exit
   code and the flag and not the *reason* is a test of the nearest `except`, not of the guard above it; the same
   hole AUD-193 filed in the server's error messages exists inside a mutation harness.
   A second from the same fix, and it is about prose rather than code: the footer sentence I wrote for the quadratic
   caveat said a row with history "flatters the ANE by the quadratic term". Measurement says the opposite -- 95.9x at
   512:0 against 62.6x at 512:1024, a *smaller* gain once history is carried -- so a history-bearing row is the
   conservative one. The sentence was rewritten and the suite now asserts the word "flatters" is absent, because a
   sentence about which direction a number moves is a claim, and it needs both measurements beside it before it
   lands. This is the same class as the AUD-234 docstring that a test read back against `len(LONG_PROMPT)`: prose
   written during a fix is unverified until something compares it.
   One more, about finding them: a bare-call grep (`^[a-zA-Z_][a-zA-Z_0-9]*\(`) over
   `benchmark/*.py` also matches source snippets embedded in strings --
   `ornith_concise_tool_ab.py:106` and `ornith_four_program_matrix.py:120,131,159,188`
   are Swift, Dart, PyTorch and JAX samples inside a dict literal, not code this
   repository runs. Read every hit in context before filing it.
   (f) The coverage pass — the tracked scripts no test file names, which is how
   AUD-263 through AUD-268 were each found — is **not** exhausted, and this row
   re-measured its window rather than reusing the old one. The predicate is stated
   exactly because the number is a function of it: a tracked `tools/` or
   `benchmark/` `.py`/`.sh` file, `test_*.py` itself excluded, whose file name
   appears nowhere in `benchmark/test_*.py` or `tests/**/*.swift`. Run that over
   `git ls-tree` and `git show`, so the working tree cannot help it: **33** by
   file name and **26** by neither spelling at
   `7a3578b`, which is the commit that wrote the old numbers, so those numbers were
   correct on the tree each measured; **31** and **24** at `a682c53`; **30** and
   **23** there. Re-measured at `2e13553`, the commit AUD-278's suite landed in, the
   same predicate gives **25** by name and **18** by neither spelling, and the
   prose-mention caveat below still applies to both figures, so read the by-name
   count as an upper bound. Two more have left since that row was written, both true
   positives: `benchmark/tinytitan_mtp_phases.py`, whose module
   `test_mtp_phases_verdict.py` imports (AUD-273), and
   `benchmark/guard_source_rate.py`, which `test_guard_source_rate_verdict.py`
   imports (AUD-278). Three left between `7a3578b` and that row —
   `tools/reconcile_snapshot.py`, which AUD-270's suite drives through
   `importlib`, `tools/lint.sh`, which AUD-271's suite copies and executes, and
   `tools/patch_snapshot_precision.py`, which neither suite runs. That third
   departure is a false credit: the predicate counts a file name wherever it
   appears, and AUD-270's suite names that file only in its own prose, so the
   honest by-name count is **31**, not 30 — a mention is not a test and the
   predicate cannot tell them apart, which is the next thing to fix about the
   method (count only a file a test imports, spawns or `subprocess`es).
   `tools/lint.sh` leaving on that distinction is a true positive, and it is the
   first member of this queue to leave because a suite actually ran it.
   Re-measured at `b872581`, where AUD-279's suite landed, the same predicate
   gives **24** by file name and **17** by neither: `benchmark/composite_split.py`,
   the 209-line entry in the list below, leaves because
   `test_composite_split_verdict.py` imports the module and drives its real
   `main()`. The predicate's *scope* turned out to be as load-bearing as its
   wording: the first attempt at this close scanned every `.swift` file in the
   tree rather than `tests/**/*.swift`, which credited four `tools/` scripts to
   prose mentions inside engine sources
   (`sources/TinyTitan/CPUEngine/AffineSnapshot.swift` names `prepare_qwen35.py`,
   `sources/TinyTitan/Runtime/Inference/Model.swift` names `precision_probe.py`)
   and returned 21 and 14 where the recorded history needs 25 and 18; scoped to
   `tests/` it reproduces the recorded list entry for entry.
   Narrowed to scripts named by
   neither spelling the set is **23** where it was 24 and, before that, 26, and the gap between the two
   numbers is the method rather than the tree — `prepare_agentworld.py` and
   `verify_ane_sidecar.py` are named by a test by stem and not by file name.
   Measured again at `2e13553` by that script, largest first, and printed as the whole
   by-neither set rather than a curated head of it: `tools/prepare_qwen35.py` (884),
   `benchmark/claude_openai_adapter.py` (348), `benchmark/tinytitan_hit_fixup_ab.py`
   (269), `tools/prepare_ornith_mtp.py` (264), `benchmark/ane_prefill_layer_diff.py`
   (228), `tools/compare_qwen35_precision.py` (225), `tools/precision_plan_qwen35.py`
   (217), `benchmark/composite_split.py` (209), `tools/prepare_qwen38_mtp.py` (202),
   `tools/ane_sidecars.sh` (168), `tools/ssdai_diff_snapshot.py` (141),
   `tools/precision_probe.py` (120), `benchmark/tinytitan_v42_control_plane_ab.py`
   (109), `tools/embedded-dependency-check.sh` (82), `tools/library-facade-rules.py`
   (60), `benchmark/tinytitan_io_backend_ab.py` (60), `tools/prepare_qwen35_2b.py`
   (23), `benchmark/combos.sh` (15). `benchmark/guard_source_rate.py`, which this row
   listed at 340 lines and which is 478 now, is off it because AUD-278's suite imports
   the module. `tools/lint.sh` (1609 — the gate harness itself, invoked by CI and by other
   scripts but by no suite until AUD-271) is off the top of that list, and
   `benchmark/ornith_four_program_matrix.py` and `tools/qwen38_reference.py` stay off
   it because a stem scan clears both, as the previous row said when it put them in.
   The list this one replaced was short one row: it printed six names and, having dropped
   `tinytitan_hit_fixup_ab.py` (269) entirely, ended on `prepare_ornith_mtp.py` (264)
   where the command run at the commit the row measured puts `hit_fixup_ab` fifth and
   `prepare_ornith_mtp` sixth —
   a hand-curated list, which is the fifth time these numbers were written down
   rather than re-derived, and the reason the two numbers now carry their command.
   The list above is that command's whole output, so the count and the names cannot
   disagree again.
   The converter drivers stay the class with the most room: `prepare_qwen35.py`
   (measured at AUD-270's sibling sweep and deliberately not filed) and
   `qwen38_reference.py` decide whether a repack and a parity run are faithful.
   `benchmark/composite_split.py` (209) is the one this queue has now acted on: it was
   filed as AUD-279 out of AUD-278's sibling sweep, because it takes the same
   domain-free `--threshold`, `continue`s over the rows whose grounding it cannot
   compute and answers the shrunken file with a verdict about the repair, and its
   `facts()` skips a `--labels` entry whose journal is missing, prints the job count
   and exits 0. That row is closed at `b872581`, and the file is off the queue for the
   reason the queue exists: `test_composite_split_verdict.py` imports the module.
   `benchmark/side_engine_tasks.py` was filed in the same sweep as its sibling and
   carries the same label skip and a case's ground label derived from the scorer
   AUD-278 just put under a control; it is fixed in the same commit and was never on
   this list, because a suite already named it.

TT-018 (the plugin's delivery) and TT-020 (reaching the LAN manager from another
machine) were both closed on 2026-10-01: the `awesome-dsh-plugin` fork is gone,
and the LAN capability is verified across hosts with the direct interface bind
left as an upstream ask in `docs/dsh-upstream-asks.md`.

## Traps worth carrying forward

- **A driver that writes a refusal into a record is half the fix; the reader of that
  record is the other half.** AUD-216 moved `side_engine_judges.py`'s exception out of
  `completion` and into an `error` field, which is exactly what made
  `t6_prose_score.py` score the refusal as a wrong answer — it had never looked at
  either field (AUD-220). Judge files already on disk carry the *old* `<error …>`
  shape, so a reader has to learn both, and grepping the writer is not enough: grep
  the consumers of the file it emits. Two more traps from the same closes.
  `test_tsan_storm_exit.py` must set `TMPDIR` to its own temporary directory, because
  AUD-218's fix deliberately keeps the log directory on a failure and the first drafts
  leaked 49 `tt-tsan-storm.*` directories into `/tmp`; and a stubbed ThreadSanitizer
  runtime belongs at `Contents/Frameworks` relative to the `.xctest` bundle — that is
  where `tools/tsan-storm.sh` looks, and one level off makes every test die on
  "no ThreadSanitizer runtime at …".
- **`chflags uchg` on a file *inside* a directory builds the volume an unlink-refusal
  test needs.** The flag makes `FileManager.removeItem` on the containing directory
  answer EPERM while writes into that directory keep succeeding, so a store can
  create and populate a record it cannot evict. That seam cost a deferral to find:
  the sweep behind AUD-190 left the refused-eviction accounting bug unfilled because
  `writeDisk` uses `try` and rethrows, and it was reasoned that any volume refusing
  the directory must refuse the file first. It does not — the flag is on the file,
  the refusal is on the directory. AUD-215 is what it looked like once reachable.
  Before accepting "no test seam" for an unlink, try the flag.
- **A hand-set `baseURL` can 404 every model call.** `dsh-llm-deepseek` defaults to
  `protocol: messages`, whose root is `https://api.deepseek.com/anthropic`;
  `https://api.deepseek.com/v1` is the chat-completions root, and every request then
  goes to a path DeepSeek does not serve. All four fleet nodes were failing every
  turn this way until 2026-09-18. The generated route (`tools/dsh_route.sh`) does
  not make this mistake; config typed by hand does. Full entry under *Traps that
  have already cost time* in the engineering notes, which now live in the wiki
  repository's history (`git -C .qwen/wiki show 6acaa8f:Engineering-Notes.md`)
  rather than in the user wiki.
- **A version gate must be visible, not merely correct.** The harness collects a
  plugin's log records and prints them **only when the boot itself fails**, so a
  refusal reported through the host logger is invisible on a healthy boot. Both
  plugins write to stderr for that reason. Found by booting a throwaway harness in a
  temporary `DSH_HOME`, not by reading — do that again for any boot-time claim.
- **Renaming the checkout invalidates every install receipt and `.build`'s debug
  half.** Receipts bind absolute paths; re-issue with
  `swift run -c release TinyTitanRepack --verify-install --input-ssdai <dir>`
  and never hand-edit one. The debug tree is compiled against absolute paths too
  — after the rename, 7,312 files named the old path and `swift test` died with
  `precompiled file …_Builtin_stdbool….pcm was compiled with module cache path
  '/Users/andreborchert/Downloads/NVMAI/…'` **before a single test ran**;
  `release.sh` reports that as `swift test did not report a passing run`, which
  reads like a failing test. Remove `.build/debug` and let it rebuild.
- **A release tag that is not yet published may be force-moved.** The dry run's
  numbers must be in the notes, so the sequence is: commit prep → tag → dry run →
  fill in `### Verification` → commit → `git tag -f` → `git push --force origin
  vX.Y` → `--publish`. `--publish` re-runs every gate and rebuilds the archive,
  so **the published digest and size are never the dry run's** (5.11: 15,370,277
  bytes dry, 15,370,253 published; 5.10: 15,367,456 dry, 15,367,433 published;
  5.9: 15,437,771 dry, 15,437,857 published; 5.8: 15,436,743 dry, 15,436,730
  published; 5.5: 26,093,424 dry, 26,094,346 published) — that is what the
  placeholders are for.
- **A fire-and-forget registration can race the observer that awaits it.** The
  T7 schedule closure in `MemoryService` handed the question to an unstructured
  `Task { await hinter.register(…) }` and returned, so `waitForRetrievalHints()`
  could return before the question was queued; the plain suite passed and only
  `--sanitize=thread` failed, on the 5.9 release commit. Fixed by awaiting the
  queueing hop, which never runs the engine. The general form: when something is
  described as background, the seam that observes it must wait for the
  *hand-over*, not merely for already-started work.
- **`swift build -c release --build-tests` cannot pass on this tree, and it is not
  a manifest defect.** It fails `unable to resolve Swift module dependency to a
  compatible module` for the libraries the tests `@testable import` — measured 2 of
  2 on 2026-10-06, seven modules named once and one the next time, because the build
  stops at the first module it cannot bind. `swift test -c release` on the same tree
  builds and runs (`Build complete! (148.88 sec)`), and adding `-Xswiftc
  -enable-testing` to the failing command makes it succeed too (`Build complete!
  (147.46 sec)`), which is the mechanism: a release `--build-tests` compiles the
  product libraries without `-enable-testing`, so a `@testable` import has no
  compatible module to bind. Ledger AUD-152, filed first as an unreproduced flake.
  So: run release-config tests with `swift test -c release`, and do not purge
  `.build` or edit `Package.swift` when this error appears.
- **A test that awaits a parked task cannot time out, so it hangs instead of
  failing.** Two probe harnesses written to catch a lost wake produced no output at
  all and had to be killed after 7 and 10 minutes: a task group joins *every* child,
  so the child blocked on `await parkedTask.value` outlives the timeout child and
  `group.cancelAll()`, and cancelling the waiter does not cancel the unstructured
  task it is awaiting. The deadline has to be a polling loop over a counter the work
  writes on its way out — which is how `concurrentSwitchesWakeEveryQueuedCaller` and
  `RoutingGateTests` are written, and both then fail in 30.0 s and 10.1 s instead of
  hanging. A second defect was hiding behind the first: the fixture gate stored one
  continuation for any number of waiters, so the second concurrent waiter overwrote
  the first.
- **A pinned tool can be shadowed before the gate's remedy can reach it.** This host
  has Homebrew `ruff` 0.16.10 in `/opt/homebrew/bin`, which precedes pipx's
  `~/.local/bin` on `PATH`, so `pipx install --force ruff==0.16.7` — the remedy the
  gate printed — installs the pin where nothing will use it. The gate now names the
  binary it resolved. Until that PATH entry is removed the `python` gate is *not
  checked* on this host; the check itself passes with the pinned ruff first on
  `PATH` (`ok (ruff 0.16.7, check and format clean, parses under 3.13)`), so the
  repository's python is clean and only the environment is not.
- **The synthetic kernel metrics swing with the machine, not the code.** QKV GEMV
  and GDN in-projection have read 55.4–78.6 and 66.8–77.4 GB/s across the
  v5.5–v5.8 records on this machine, and 5.9 measured 63.5/67.9 while macOS's
  `dasd` held a core at ~95% and Chrome was active. 5.10 hit the same on
  `gpu.routed_moe`: **37.1 GB/s against 43.6** on the first run with Chrome
  helpers and `mediaanalysisd` still active, **41.4** on the re-run — and that
  counter has ranged 36.8–56.1 across the stored records. One re-run, with both
  values and the reason in `### Verification`, is the treatment; a search for the
  best of many is not. The speed gate compares
  against the previous release's record, which can be the series' high-water
  mark. Cross-check the generation metrics and the greedy response hash (an
  unchanged hash means no arithmetic moved), then record both values and the
  reason in `### Verification` rather than re-rolling the number into a
  flattering record.
- **A `file:` plugin install is a copy.** After editing
  `plugins/dsh-tinytitan/`, re-install it
  (`dsh plugin --profile web remove dsh-tinytitan`, then `add
  file:<checkout>/plugins/dsh-tinytitan`) or the harness keeps running the copy.
  This bit on 2026-09-18: both installed copies were a week of edits behind, and
  only a re-install plus a harness restart put the current code in service.
- **The wiki is a second repository with its own history.** Pull `.qwen/wiki`
  before editing it; the tracker and the Changelog are separate commits; the
  fine-grained PAT can read it but was rejected for push, so use the `gh`
  credential helper (`gh auth setup-git`). A push means **both** repositories.
- **Verification is what is installed.** `models/` is pruned for disk on purpose;
  a target with no install is reported *not checked* and named in the notes, and
  nothing is fetched to change that. A stored baseline is never deleted because
  its model is currently absent.
- **`release.sh` needs `HEAD` to *be* the tag** and a release build at
  `.build/release/TinyTitanCLI` to exist **before** it starts.
  The golden phase refuses to run beside any model process.
- **The `xcode-27` runner is an x64 Actions agent on arm64 hardware.** Inside its
  jobs `uname -m` prints `x86_64` while `swift --version` targets arm64, and
  `/usr/bin/sandbox-exec` ships only arm64e slices. Anything that injects a native
  arm64 dylib into spawned processes therefore dies on that binary: CodeQL's tracer
  failed with `posix_spawn error: Bad CPU type in executable (86)` before compiling
  a single file, for months, and the workaround is `swift build --disable-sandbox`
  in `.github/workflows/codeql.yml`. The durable fix is an `osx-arm64` Actions
  runner; after that the flag can go and the manifest sandbox return. Do not
  re-diagnose it as a tool bug.
- **Installing a toolchain can change what a scanner scans.** The first `npm ci` in
  CI made the Markdown link check read `plugins/*/node_modules/**` and report 461
  broken links from vendored READMEs. Build output, vendored dependencies and
  model stores are not project source: exclude them explicitly, and re-run any
  whole-tree scanner after adding an install step.
- **Opening a journal for writing now truncates its last incomplete line.**
  `FileJournal` was append-only, except at compaction, until 2026-10-08; AUD-189 put
  `dropDanglingTail` in the initializer, because the unterminated tail a kill leaves is
  also the line `O_APPEND` writes the next record onto, and the fused line decodes as
  neither. The bytes removed are ones `replay` already dropped, so no readable record
  is lost — but a journal copied out of a workspace after a crash is shorter than the
  same file was a minute before the server opened it, and that is the repair, not a
  second defect.
- **Say what a guard actually reads, not what it intends**, and **test a claim
  rather than trusting it** — both defects that reached a release in this project
  were claims broader or more specific than the code.
