# Ornith ANE prefill: baseline-first optimization

## Intent

Build and validate an Ornith 1.5 35B-A3B 4-bit ANE prefill sidecar for the
OpenCode-style long-prompt workload, then optimize only the costs the baseline
shows. The target is lower prefill latency and time to first token, not higher
decode throughput. The model must remain correct: a faster result that changes
model quality beyond the established ANE/GPU numerical boundary is rejected.

## Current state

Ornith uses the `qwen36` full-attention geometry. The production runtime bridge
already exists in `sources/Engines/Ornith35B4Bit/`; the generic exporter
supports that family and reads its geometry from the installed model. The local
install currently has no `ane_prefill` sidecar, so its prefill is GPU-only even
though ANE routing is enabled by default.

The sidecar runs full attention for 10 of 40 layers. GDN/linear attention, MoE
and expert streaming, KV-cache management, the residual tail, and all decode
remain on the existing GPU path. The exporter emits fp16, fixed-shape Core ML
programs, one function per covered layer and history position. It checks ANE
compilation, ANE operation assignment when the installed coremltools exposes a
compute plan, and successful loading of every advertised function before
committing the sidecar.

## Goals

- Export `models/ornith-1.5_35B_A3B_4Bit/ane_prefill` for chunk 4,096 and
  histories 0, 4,096, 8,192, and 12,288, covering prompts through 16,384 tokens.
- Prove graph geometry against the independent NumPy reference and prove live
  requests actually use the ANE rather than silently falling back to GPU.
- Establish a paired GPU-only versus ANE prefill baseline at a representative
  long prompt, including warm-up/load behavior and response correctness.
- Measure and, where worthwhile, optimize the dominant Ornith-specific costs:
  per-variant Core ML loading, staging/synchronization, memory residency, and
  graph execution. Preserve the current asynchronous next-layer preload unless
  measurements justify changing it.
- Record exactly what improved, what did not, the remaining costs, and the
  supported prompt band in engineering and user-facing documentation.

## Non-goals and anti-goals

- No decode-kernel, expert-cache, sampling, or model-weight optimization.
- No Qwen 3.8 sparse-attention changes and no attempt to reuse its compiled
  sidecar; Ornith has its own weights and dense full-attention semantics.
- No 1,024-token sidecar in the first pass. It serves a different prompt band
  and increases the number of model variants loaded on long prompts.
- No automatic per-request chunk switching: chunk width is fixed when the
  runner starts. Short or partial chunks deliberately remain on the GPU.
- No new dependency, model download, model duplication, or cache purge to make
  a check pass. No benchmark while the host is too busy for a trustworthy
  paired measurement.
- No claim of success from compilation alone, and no speedup quoted from an
  arm that fell back to GPU.

## Constraints

Use the existing Python/coremltools exporter, Swift runtime bridge, and
benchmark harness. Use the documented coremltools interpreter if present; if it
is absent or unusable, report export as blocked rather than silently installing
packages. Use only the existing Ornith install. Model runs require the
repository's documented OS, toolchain, memory-pressure, install, and process
preconditions. Keep production source files within the repository's size and
lint rules, and run real inference for any runtime numerics or prefill-path
change.

ANE attention uses fp16 and a different reduction order from the GPU, so byte
identity between ANE and GPU activations is not the oracle. Graph correctness
requires the NumPy check at its existing maximum 2% mean relative error. Live
acceptance requires **identical 32-token greedy output** (therefore zero top-1
divergence) between GPU and ANE on every checked prompt, plus
`pos0/prefill_logits` mean relative error of at most **2%**, using the existing
layer-diff metric `mean(abs(gpu - ane)) / mean(abs(gpu))`. Maximum absolute
logit error and the linear-layer controls are recorded as diagnostics. The
initial fixed benchmark prompt and any additional OpenCode-sized performance
prompt each get this quality check. Exceeding either threshold or diverging in
generated text rejects the change; coherent divergence is not sufficient.

## Approaches considered

### Baseline first — selected

Export the existing Ornith graph unchanged, verify it, and measure ANE off/on
with the existing paired harness. Use the resulting timings and diagnostics to
select optimization work. This is the smallest path to a working sidecar and
prevents unmeasured graph or runtime changes from being mistaken for progress.

### Optimize before baseline — rejected

Change model residency, graph structure, or scheduling before measuring Ornith.
This might attack a real cost, but there would be no trustworthy way to assign
the effect or detect a regression. Qwen 3.8 already demonstrates that a correct
ANE graph can still lose to the model's GPU path.

## Design and execution sequence

### 1. Baseline export and static verification

Run `tools/ane_sidecars.sh ornith-1.5_35B_A3B_4Bit` with its defaults: chunk
4,096, maximum history 12,288. The script exports into a temporary location and
leaves any previous sidecar untouched on failure, then runs
`tools/verify_ane_sidecar.py` against NumPy. Record metadata, geometry, covered
layers, prompt ceiling, export/load failures, ANE operation assignment evidence,
and relative error. A missing interpreter or refused variant is a reported
blocker, not a pass.

### 2. Live correctness and engagement

Build the portable release binary once. Run Ornith with
`--prefill-chunk 4096`, ANE explicitly off and on, greedy sampling, and a prompt
long enough to fill more than one chunk. Confirm the ANE arm has no fallback
marker and exercises every expected layer/history variant. Use the existing
correctness and layer-diff instruments rather than treating a one-token digest
as proof. Check short-prompt GPU routing and behavior beyond 16,384 tokens so
coverage limits remain explicit.

### 3. Paired baseline measurement

On a quiet host, use `benchmark/ane_prefill_ab_matrix.py` for alternating
GPU-only and ANE arms after one discarded warm-up per arm. Record prefill
seconds, prompt tokens, response digests, fallback status, and load/warm-up
costs. Start with the harness's approximately 4,300-token prompt; add a longer
OpenCode-sized prompt only if needed to expose scaling or history-variant load
costs. Decode speed is recorded only as a guard against collateral regression,
not as the objective.

### 4. Evidence-led optimization

Profile the baseline before changing code. Candidate work is conditional:

- If first-use or per-history model loading dominates, measure preload overlap
  and residency before changing cache lifetime or eager loading. Bound any new
  residency by actual package memory and the 10 GB-class Ornith deployment.
- If staging or command-buffer waits dominate, measure those intervals before
  changing synchronization. Preserve the GPU/ANE handoff and shadow-KV contract.
- If graph execution dominates, test one graph change at a time with NumPy and
  live correctness after each. Do not adopt fused SDPA or another numerically
  unsafe substitution merely because it compiles.

Every candidate requires a paired benchmark against the immediately preceding
accepted default. Keep only measured wins with no quality loss. A candidate that
ties within the noise floor or loses is recorded and rejected. If the verified
baseline itself ties with or loses to GPU and no candidate wins, move the
sidecar to the non-runtime name `ane_prefill-rejected` so the default installation
remains GPU-only, retain the negative measurements, and do not present the
result as an enabled optimization. The stopping point is an explained cost
profile with no remaining tested candidate producing a repeatable win, not an
arbitrary token/s or latency promise.

### 5. Regression, documentation, and publication

Run targeted exporter/runtime tests and the full serial package suite after
source changes, plus all lint gates and the Ornith golden-quality check required
for a runtime/model-load change. Re-run live ANE correctness after the final
optimization. Audit sibling `qwen36` integrations for accidental behavior
changes, but do not re-export or benchmark unrelated models.

Update `benchmark/ane-prefill/README.md` with Ornith's measured row and coverage
limits, and update any affected engineering records. After committing and
pushing `main`, inspect the README and wiki for stale statements and update the
wiki Project Tracker on `master` with the result or any remaining tasks. Report
commit, host/RAM, macOS, Swift, exact commands, exit codes, complete timing
footers or errors, and every protocol deviation.

## Acceptance criteria

- The 4,096/12,288 Ornith sidecar exists locally, advertises only variants that
  compiled and loaded, and passes NumPy geometry verification.
- Live long-prompt runs demonstrate ANE engagement without fallback and satisfy
  the defined output/activation quality checks; short prompts remain on GPU.
- A paired baseline records both prefill arms and warm-up/load behavior on a
  quiet host. No performance ratio is reported if these conditions are unmet.
- Every retained optimization has its own paired measurement and quality pass.
  If the unchanged baseline already wins and no candidate improves it, shipping
  that verified baseline is the correct optimized result.
- Tests, lint, golden quality, documentation, tracker state, and both pushes are
  complete for any committed source change. Any check that cannot run is named
  as not checked rather than omitted or forced green.

## Builder decisions

The builder may choose the exact profiling instrument and which candidate to
try first from baseline evidence. The builder may add focused tests around any
new runtime behavior using the repository's existing test conventions. The
builder must not broaden the work to other model families, change the default
chunk width, add automatic chunk selection, or relax correctness thresholds
without a new design decision.
