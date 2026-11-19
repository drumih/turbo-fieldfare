# Ornith ANE Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified Ornith 1.5 35B-A3B 4-bit ANE prefill sidecar, measure it against GPU-only prefill, and retain only measured, quality-preserving optimizations.

**Architecture:** Use the existing `qwen36` exporter and Ornith runtime bridge unchanged for the baseline. The ANE handles full-attention blocks for 10 layers while the GPU retains GDN, MoE, KV-cache, residual, and decode work; fixed 4,096-token chunks select history-specific Core ML functions through 16,384 prompt tokens.

**Tech Stack:** Python 3.11 coremltools environment, NumPy, Core ML ML Program, Swift 6.4, Metal, existing Python benchmark harnesses.

**Spec:** `docs/superpowers/specs/2026-11-18-ornith-ane-prefill-design.md`

## Global Constraints

- Use only `models/ornith-1.5_35B_A3B_4Bit`; do not download, duplicate, move, or reinstall a model.
- Export exactly chunk 4,096 and maximum history 12,288; prompts below one full chunk stay on GPU.
- Do not install missing dependencies to make a gate pass; report the blocked check.
- No performance measurement while the host is busy; use alternating warm paired arms.
- Require NumPy mean relative error ≤2%, identical 32-token greedy output on every checked prompt, and `pos0/prefill_logits` mean relative error ≤2%.
- Never quote a speedup from a run containing `ane-prefill fallback`.
- Decode throughput is a regression guard, not the optimization target.
- If the final ANE configuration does not beat GPU, rename the generated sidecar to `ane_prefill-rejected` so it cannot be selected by default.
- Run package tests serially and only one model-using process at a time; follow every model-run precondition in `AGENTS.md`.
- After pushing `main`, synchronize applicable README/wiki documentation and the wiki Project Tracker on `master`.

## Review Focus

- Missing or broken coremltools interpreter must block export explicitly, not produce an empty success (Task 1, interpreter preflight).
- Prompts below 4,096 tokens must remain GPU-only (Task 2, short-prompt routing check).
- Prompts beyond 16,384 tokens must fall back safely for uncovered chunks without corrupting output (Task 2, coverage-boundary check).
- A fallback or partial ANE run must not be recorded as an ANE speedup (Tasks 2 and 3, fallback/engagement assertions).
- A correct but slower sidecar must not remain enabled by default (Task 5, negative-result cleanup check).

---

### Task 1: Export and statically verify the Ornith baseline sidecar

**Files:**
- Create locally, gitignored: `models/ornith-1.5_35B_A3B_4Bit/ane_prefill/`
- Read: `tools/ane_sidecars.sh`, `tools/export_ane_prefill.py`, `tools/verify_ane_sidecar.py`
- Test: exporter's built-in compile, load, compute-plan, and NumPy verification

**Interfaces:**
- Consumes: existing Ornith manifest, weights, and documented coremltools interpreter.
- Produces: `ane_prefill/ane_prefill.json` with chunk 4,096, histories `[0,4096,8192,12288]`, 10 covered layers, `aneCompileVerified: true`, and matching model geometry.

- [ ] **Step 1: Check repository and model-run preconditions**

Run:

```bash
git status --short --branch
sw_vers
swift --version
df -h .
memory_pressure -Q
test -f models/ornith-1.5_35B_A3B_4Bit/manifest.json
test -f models/ornith-1.5_35B_A3B_4Bit/verified-install.json
test -x "$HOME/.venvs/coreml-py311/bin/python"
pgrep -fl 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

Expected: macOS 26+, Swift 6.4, sufficient disk, acceptable memory pressure, complete existing install, executable coremltools interpreter, and no model/test process. Stop and report any failed condition; do not terminate processes or install anything.

- [ ] **Step 2: Export and verify the baseline**

Run:

```bash
tools/ane_sidecars.sh ornith-1.5_35B_A3B_4Bit
```

Expected: exit 0, one exported sidecar, and `verified: 1`. Preserve the complete command result and any compiler diagnostics. A Core ML refusal or verification failure stops implementation for diagnosis.

- [ ] **Step 3: Pin the generated contract**

Read `ane_prefill/ane_prefill.json` and assert chunk 4,096, histories `[0,4096,8192,12288]`, Ornith's `qwen36` geometry, exactly the manifest's 10 full-attention layers, `aneCompileVerified: true`, and the install's weights SHA. Confirm every advertised package/function exists and NumPy relative error did not exceed 2%.

### Task 2: Verify live routing and model quality

**Files:**
- Read: `benchmark/ane_prefill_correctness.py`, `benchmark/ane_prefill_layer_diff.py`
- Create: `benchmark/ane-prefill/ane-correctness-ornith15-4bit-baseline.json`
- Create: `benchmark/ane-prefill/ane-layer-diff-ornith15-4bit-baseline.json`
- Test: real Ornith GPU/ANE generation and activation comparison

**Interfaces:**
- Consumes: Task 1 sidecar and portable release CLI.
- Produces: evidence of real ANE engagement, exact 32-token output agreement, and logits within 2%.

- [ ] **Step 1: Build the portable release executable**

Run: `swift build -c release`

Expected: exit 0 with the pinned Swift 6.4 toolchain and no native CPU override.

- [ ] **Step 2: Verify short prompts remain on GPU**

Run the CLI with a five-to-twenty-token prompt, `--max-new 1 --temperature 0 --prefill-chunk 4096`, and `TINYTITAN_PREFILL_ANE=on`.

Expected: exit 0 and an `ane-prefill fallback` eligibility message; no ANE program executes for a partial chunk.

- [ ] **Step 3: Compare long-prompt greedy output**

Run:

```bash
python3 benchmark/ane_prefill_correctness.py \
  --models ornith-1.5_35B_A3B_4Bit --max-new 32 \
  --label ane-correctness-ornith15-4bit-baseline --record
```

Expected: exit 0, both arms reach at least one full ANE chunk, the ANE arm has no fallback, and the recorded `on.identical` is `true`. Non-identical output fails acceptance even if coherent.

- [ ] **Step 4: Compare live activations and logits**

Run:

```bash
python3 benchmark/ane_prefill_layer_diff.py \
  --models ornith-1.5_35B_A3B_4Bit \
  --attention-layer 3 --label ornith15-4bit-baseline --record
```

Expected: exit 0, real ANE engagement, linear-layer controls consistent with unchanged layers, and `pos0/prefill_logits.mean_relative_percent <= 2.0`. Record maximum absolute error as diagnostic evidence.

- [ ] **Step 5: Check the advertised coverage boundary**

Run one greedy prompt exceeding 16,384 tokens with ANE on, one new token, and chunk 4,096.

Expected: exit 0 with explicit fallback for the first uncovered chunk and the rest of that request on GPU; no crash, stale shadow-KV use, or false full-ANE claim.

### Task 3: Measure the paired prefill baseline

**Files:**
- Create: `benchmark/ane-prefill/ane-matrix-ornith15-4bit-baseline.json`
- Test: `benchmark/ane_prefill_ab_matrix.py` paired warm measurement

**Interfaces:**
- Consumes: Task 2's accepted sidecar and release build.
- Produces: GPU-only and ANE prefill medians, warm-up/load observations, and a baseline go/no-go decision.

- [ ] **Step 1: Recheck that the host is quiet**

Run the model-process and memory-pressure preconditions and inspect current CPU load. Expected: no competing model process and no workload that would invalidate a paired timing. Defer measurement rather than benchmarking a busy host.

- [ ] **Step 2: Run the fixed approximately 4,300-token baseline**

Run:

```bash
python3 benchmark/ane_prefill_ab_matrix.py \
  --models ornith-1.5_35B_A3B_4Bit --repeats 2 \
  --label ane-matrix-ornith15-4bit-baseline --record
```

Expected: exit 0, two measured runs per arm after warm-up, `on.used_ane: true`, and a recorded median for each arm. Preserve all timing footers and digests. Do not infer quality from the one-token digests.

- [ ] **Step 3: Measure the OpenCode-sized band if needed for scaling**

If the fixed prompt leaves history loading or the long-prompt benefit unresolved, run the same harness with a longer fixed prompt that crosses at least one additional history variant. Repeat Task 2's 32-token and logit checks for that prompt before accepting its timing.

- [ ] **Step 4: Decide the optimization target from evidence**

Record ANE/GPU medians, absolute seconds saved or lost, first-run load cost, and the dominant measured ANE-side cost. If no candidate remains, proceed to Task 5 with the verified baseline; if ANE loses, only a specifically measured bottleneck may proceed to Task 4.

### Task 4: Optimize one evidenced bottleneck at a time

**Files:**
- Modify only the exporter, Ornith runtime, or focused tests implicated by Task 3.
- Add an exact per-candidate addendum to this plan before each code change.

**Interfaces:**
- Consumes: a named Task 3 bottleneck and its measured cost.
- Produces: one independently reviewable candidate with a paired timing and quality result.

- [ ] **Step 1: Write the candidate-specific failing test**

Add a focused test for the behavior being changed: residency bounds and reuse for loading, handoff/shadow-KV ordering for synchronization, or graph geometry for an exporter change. Run it against the current implementation and confirm it fails for the intended reason. If no model-free test can pin the behavior, define the model-gated probe before implementation.

- [ ] **Step 2: Implement the minimal candidate**

Change only the named mechanism. Do not alter chunk defaults, decode, model weights, quality thresholds, or unrelated `qwen36` engines. Keep files within 500 physical lines and functions within repository limits.

- [ ] **Step 3: Run focused and static verification**

Run the new test, relevant existing test suites, and NumPy verification after any export change. Expected: all pass with no fallback or silent skip.

- [ ] **Step 4: Repeat live quality and paired performance checks**

Repeat Task 2 Steps 3–4 and Task 3's paired benchmark against the immediately preceding accepted default. Accept only a repeatable win outside run-to-run noise with identical 32-token output and logits ≤2%; otherwise revert the candidate and record the negative result.

- [ ] **Step 5: Repeat or stop on evidence**

Return to Step 1 only for another separately measured bottleneck. Stop when no tested candidate produces a repeatable quality-preserving improvement; do not continue speculative optimization.

### Task 5: Final verification, documentation, and publication

**Files:**
- Modify: `benchmark/ane-prefill/README.md`
- Modify as applicable: affected engineering records, README, wiki pages, wiki Project Tracker
- Create or modify only tests required by accepted Task 4 changes

**Interfaces:**
- Consumes: the final accepted configuration and all measurement records.
- Produces: reproducible documentation, clean gates, and synchronized `main` and wiki `master` publications.

- [ ] **Step 1: Apply the final routing decision**

If the accepted ANE configuration wins, leave `ane_prefill` active at chunk 4,096. Otherwise rename it to `ane_prefill-rejected` and verify a default long-prompt run uses GPU without attempting to load the rejected sidecar.

- [ ] **Step 2: Run final real-model verification**

Run the Ornith golden check and repeat final live ANE correctness after any runtime/model-load change. Expected: golden text identity and the Task 2 quality gates pass. Record exact commands, exit codes, complete footers/errors, and deviations.

- [ ] **Step 3: Run model-free regression gates**

Run:

```bash
swift test --no-parallel
tools/lint.sh
```

Expected: all tests and all twenty lint gates pass. Do not claim unrun model gates as passes.

- [ ] **Step 4: Audit and document the result**

Audit sibling `qwen36` paths for unintended changes and record findings. Update the ANE guide with Ornith's geometry, actual engagement, quality results, warm/cold costs, paired medians, coverage ceiling, and negative candidates. Keep measurements tied to host/build/model/prompt; do not generalize them into promises.

- [ ] **Step 5: Commit and publish both repositories**

Review `git status` and remove only temporary outputs created by this work that are not deliverables. Commit the design, plan, accepted source/tests, and measurement documentation with an evidence-based message; push `main`. Then inspect README/wiki/Project Tracker, commit and push needed wiki updates to `master`, and confirm both worktrees are clean and synchronized.

---

## 2026-11-19 result — the fp32 core does not move the divergence; ANE rejected for Ornith

The staged gates ran on a full re-export of the fp32-core exporter (`ba76b292`):
export exit 0 after 32m14s, all ten layers and four histories compiled with
`aneCompileVerified: true` and the same weights SHA. Every gate reproduced the
pre-fix failure, and the fp32 core made the graph slower:

| Gate | Result |
| --- | --- |
| 1 — sidecar verify | synthetic **0.555%** (unchanged), real captured normed **4.245%** (unchanged), exit 1 |
| 2 — live layer diff | `on` logits **5.55%**, linear control 0.000% — numerically identical to the fp16 baseline |
| 3 — identical greedy | `on.identical` **false**, common prefix **0**; the ANE arm repeats `S` where GPU answers; ANE prefill **148.03 s** vs GPU **93.56 s** |
| probe — `compute_precision=FLOAT32` | 59/173 ops on the ANE (fp16: 74), **382 ms** vs 244 ms per layer, real-input error **4.248%**, max output diff only 0.00415 |

Reading: `compute_precision=FLOAT16` was not the lever either. A whole-program
FLOAT32 build changed the output by ~4e-3 and left the real-input error at
4.2%, so the difference is not intermediate precision inside the core. The
remaining hypothesis is the graph's fp16 output boundary and/or a semantic
difference that only shows at the real input's output scale (mean |out| 0.0101
against 0.0934 on the synthetic input); it is **not** established. No further
export without naming it.

Decision, per Task 5 Step 1: the local sidecar is renamed
`ane_prefill-rejected`, so the runtime cannot select it, and `d750fa2c` plus
`ba76b292` are reverted — both were numerically inert on the live model and the
fp32 core costs 1.56x per layer (59 ANE-assigned operations instead of 74). The
negative result is tracked as TT-099; the evidence records are committed under
`benchmark/ane-prefill/`.
