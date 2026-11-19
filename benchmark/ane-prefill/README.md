# ANE prefill: off vs on

One record per sweep, measured with `benchmark/ane_prefill_ab_matrix.py`. The
question it answers is narrow and worth asking separately from
[internal speeds](../internal-speeds/README.md): **for each installed model, how
much does routing the full-attention prefill block to the Neural Engine save?**

```bash
python3 benchmark/ane_prefill_ab_matrix.py \
  --models qwen3.5_2B_4Bit qwen3.5_4B_4Bit qwen-agentworld_35B_A3B_4Bit \
  --repeats 2 --label v5.6 --record
```

The record is written **after every model**, so a run that is held or
interrupted keeps the rows it already earned. Resume it with the same command
plus `--skip-done`, which skips the models already stored (a model whose run
*failed* is not counted as done — a refusal stays re-attemptable).

## What it measures

One variable: `TINYTITAN_PREFILL_ANE`. Each model runs a discarded warm-up per
arm — the first ANE run pays the Core ML compile, which on AgentWorld 4-bit was
~68 s against an 86 s prefill — and then `--repeats` measured runs per arm
(default 2), alternating so drift lands on both. The metric is **prefill seconds
from the CLI's own footer**, on a fixed prompt, `--prefill-chunk 4096`, greedy,
one new token.

**The prompt is deliberately just over one chunk** (~4,300 tokens, 23,000
characters of the fixed paragraph): the ANE serves only a full 4,096-token
chunk, so a shorter prompt measures two GPU arms and would report "1.0×" as if
it were a finding about the ANE. The sweep says `prompt_too_short` instead. The
prompt is the quadratic term, so shortening it is the main lever on run time —
but ~21,750 characters is the floor.

| Field | What it is |
| --- | --- |
| `off.prefill_seconds_median` | GPU prefill, median of the off arms |
| `on.prefill_seconds_median` | ANE prefill, median of the on arms |
| `on.used_ane` | every on arm ran without the runtime's GPU-fallback line |
| `speedup` | off / on, present **only** when `used_ane` is true |
| `digests` | the greedy response hashes per arm |
| `ane_unavailable` | the ANE cannot serve this model at all, with the reason |

**A speedup is never reported from an arm that fell back.** The runtime prints
`ane-prefill fallback … using the GPU path` when a chunk is not eligible; that
arm's seconds are a GPU time, and quoting a ratio from it is how a model that
never touched the ANE comes to look "tested".

## Is it the same model? (correctness)

Speed is the easy half. **Whatever the digests say, this sweep does not answer
whether the ANE computes the model's attention** — it generates one greedy
token, and a single token usually agrees even when the attention behind it is
wrong, so the digests are a determinism check and nothing more.

`benchmark/ane_prefill_correctness.py` is the check that answers it: the same
long prompt, greedy, `--max-new` tokens (default 32) per arm, compared as text
— how many leading characters agree and where the first divergence is.

```bash
python3 benchmark/ane_prefill_correctness.py \
  --models qwen3.8-flash-next_125B_A6B_4Bit --max-new 32 \
  --label v5.6-ane-38 --record
```

A model whose full-attention layers select keys with a QSA indexer gets a third
arm: `causal`, which runs the ANE with the causal-only mask the fold exists to
replace (`TINYTITAN_ANE_MASK=causal`, a verification control that prints a
warning). It is the negative control. `on` is expected to track the GPU arm and
`causal` to diverge from it; an arm that fell back is reported as an error,
never as a comparison.

## What the ANE covers, and when it engages

The ANE runs the **full-attention block** of a chunk, for the covered layers
only. GDN/linear-attention layers, the MoE and its expert streaming, the KV
cache, and all of decode stay on the GPU. Two consequences worth holding onto:

- the win scales with the **full-attention share** of the model. The 35B-A3B
  offloads 10 of 40 layers; the dense 2B offloads 6 of 24, so its ratio is
  smaller for the same attention speed-up;
- it grows with **prompt length**, because the offloaded share is the quadratic
  one.

Three conditions must hold or the chunk silently stays on the GPU:

1. a sidecar exists for the model and its recorded geometry matches (see below);
2. the configured prefill chunk **equals the sidecar's chunk** — the graph's
   shapes are fixed by it, so a nearer width cannot be fed. The dense Qwen 3.5
   family is on 4,096 by default; a family left on 128 can never reach the ANE
   at all. The harness reports `prompt_too_short` rather than a misleading
   "1.0×" when the prompt does not reach one full chunk;
3. the chunk is **full**, or a continuation of a long prompt. A prompt under one
   chunk has no shape to run and deliberately stays on the GPU.

## The decision rule (plan item A)

Two questions, in order: **which family** is the model, and **how long** are the
prompts this deployment serves. The second is a *deployment* choice, not a
per-request one — the prefill chunk is fixed when the runner starts, and the
sidecar it selects must be built for exactly that width — so "match the chunk to
the prompt" means match it to the workload.

| The prompts this deployment serves | Export and configure | Measured |
| --- | --- | --- |
| under 1,024 tokens | nothing — leave the ANE to decline | "hello" (5 tokens): GPU 0.11 s, against a padded chunk costing tens of seconds |
| 1,024 – 4,095 | `--chunk 1024` | dense 2B at ~2,500 tokens: 23.33 s → 17.88 s, **1.30×** |
| 4,096 – 16,384 | `--chunk 4096` (the default) | per-model matrix below; AgentWorld **1.68×** at 4,333 tokens |
| beyond 16,384 | `--chunk 4096 --max-history 32768` | coverage becomes 36,864 tokens; the ANE accepted that shape here |

**Why not one width for everything.** A chunk-1,024 sidecar reaches the mid band
that 4,096 cannot serve at all, but it costs more on long prompts: the runtime
loads one Core ML model per (layer, history) — the variant is named by the
chunk's start position — so a 16,384-token prefill is four loads per covered
layer at chunk 4,096 and sixteen at chunk 1,024, at roughly 0.5 s each. 1,024
wins where 4,096 cannot run; 4,096 wins above it. 2,048 buys nothing: measured
against 1,024 at the same prompt it tied (22.73 s → 17.52 s, 1.30×), so the
narrower width is the one to ship — it wins the same and covers strictly more.
Both records are kept (`ane-chunk1024-2b-4bit.json`,
`ane-chunk2048-2b-4bit.json`) so the claim can be re-run.

Per family, with the bands the same:

| Family | < 1,024 | 1,024 – 4,095 | 4,096 – 16,384 | > 16,384 |
| --- | --- | --- | --- | --- |
| `qwen36` — Qwen 3.6, Ornith 1.5, KAT-Coder, AgentWorld | GPU | ANE, chunk 1,024 | ANE, chunk 4,096 | widen `--max-history` |
| `qwen3_5_dense` — Qwen 3.5 2B/4B/9B | GPU | ANE, chunk 1,024 | ANE, chunk 4,096 | widen `--max-history` |
| `qwen38flash` — Qwen 3.8 125B-A6B | GPU | **GPU** | **GPU** | **GPU** |
| `qwen38flash_mtp` — its one-layer MTP draft | GPU | GPU | GPU | GPU |
| the CPU engine, any model | CPU | CPU | CPU | CPU |

## Qwen 3.8: the fold is wired and correct, and the ANE still loses

Qwen 3.8's full-attention layers do not compute a different attention: they
**choose keys** with a QSA indexer, and the GPU path already computes that choice
as a compacted keep list (`keepIndices`/`keepCounts`, which is why that path
forces the causal-tiled kernel). Dense attention matches that selection only
through `keptBlocks × compressRatio + (compressRatio − 1)` = **2,051 visible
keys**, and the smallest chunk the ANE accepts is a full 4,096 — so a sidecar fed
the causal mask would attend to keys the model drops, silently.

That is now wired rather than refused. The sidecar's mask input is an *arbitrary*
additive mask, so the runtime folds the same selection in — `-30000` on every
dropped key, which `exp()` underflows to zero exactly as an omitted key
contributes nothing — and **the graph does not change at all**. What was added is
the fold itself (a per-layer mask, because each layer selects its own keys), the
`selectionFolded` contract the runtime refuses to run without, and the checks
below. `TINYTITAN_ANE_MASK=causal` is a verification control that feeds the
causal-only mask instead, and prints a warning when it does.

**Then it was measured, and it does not pay.** Both 3.8 widths, prompt ~4,333
tokens, chunk 4,096, two measured runs per arm after a warm-up each:

| Model | ANE off | ANE on | Ratio |
| --- | ---: | ---: | ---: |
| Qwen 3.8 125B-A6B 4-bit | 197.5 s | 273.5 s | **0.72×** |
| Qwen 3.8 125B-A6B 8-bit | 428.3 s | 491.8 s | **0.87×** |

The mechanism, all measured rather than inferred:

- **The GPU path is already sparse.** It gathers the indexer's ~2,051 selected
  keys per query; the ANE graph computes **dense attention over the whole
  context** and masks the dropped keys. For the 4,333-token prompt that is a
  4,096 × 8,192 score matrix per covered layer-chunk against the GPU's
  4,096 × 2,051 gather — about 4× the arithmetic — and this graph's ANE
  advantage is small to begin with: one `h0` prediction measured **0.43 s on the
  ANE against 0.65 s on the CPU alone**, where the 35B's blocks were 26.7× the
  GPU's cost.
- **The per-variant Core ML load is now the dominant term.** One
  `MLModel(contentsOf:configuration:)` of this sidecar measured **6.9 s for
  `h0`, 13.6 s for `h4096`** on the same machine whose 35B sidecar documents
  ~0.5 s. A 4,333-token prompt visits 24 (layer, history) variants, so model
  setup alone is minutes — against a GPU prefill of ~185 s, and the asynchronous
  preload can only hide what the MoE stage lasts.

So `tools/ane_sidecars.sh` does **not** export one for `qwen38flash`, and the
family's row above is GPU everywhere. The exporter still supports the family —
the geometry is the same block, and the fold is the thing any future attempt
would need — so a sidecar can be built explicitly to re-measure:

```bash
~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit --chunk 4096 --max-history 8192
```

**A gather graph — the one variant that would remove the fold's extra arithmetic
— was sized and rejected.** The ANE accepts the `gather`, but a gathered key has
to be materialised once for every query that selects it: 64× the dense score
matrix, 103 GB at the real chunk. Measured, it is 5.7× slower where it runs and
fails outright one chunk size up, and it does not help the load either — that
scales with the score arena, not with package composition. The numbers, the probe
and the reproduction commands are in
[`docs/ane-gather-graph-sizing.md`](../../docs/ane-gather-graph-sizing.md) and
`benchmark/ane_gather_probe.py`.

**Do not raise that past 8,192 on 3.8.** Its `h12288` variant is accepted by the
converter and then fails to *load* (`functionName` must be nil unless the model
type is ML Program) on both widths — 24 heads of 4,096 × 16,384 fp16 scores is a
3.2 GB arena, against the 35B's, which loads. The exporter now compiles *and*
loads every function it is about to record and fails the export if one does not,
so a sidecar cannot advertise coverage it cannot serve.

Three checks cover the correctness of what was wired:

- `tools/verify_ane_sidecar.py` runs automatically on a sidecar that records
  `selectionFolded`: the graph is fed a QSA-shaped mask (1,538 of 4,096 keys kept
  on average) and checked against the NumPy reference under that same mask. On
  **both widths** the graph tracks the reference to **0.47 %**, while the
  causal-only mask moves the reference by **7.6 %** and is off by 7.9 % — 16× the
  fp16 noise, which is what makes the check non-vacuous. A selection that barely
  moves the reference fails as vacuous (`--min-selection-effect`).
- the runtime refuses to attend densely: a chunk past the dense-exact window with
  no selection throws rather than quietly dropping the fold.
- `benchmark/ane_prefill_correctness.py --max-new 32` compares a 32-token greedy
  continuation between the arms, with the causal-only mask as the negative
  control. Run on the 4-bit install (the runtime path is the same on both), the
  folded arm is **textually identical to the GPU arm** — and so is the causal
  control, which is exactly why text alone is not enough here. A **digest at
  `--max-new 1` cannot do this** at all.
- `benchmark/ane_prefill_layer_diff.py` compares the runtime's own activation
  dumps per arm, and it is the check that separates them: the causal control's
  `prefill_logits` differ from the folded arm's by 13.6–17.8 %, so the mask does
  reach the graph. Two related numbers were measured while pinning that down: on
  the dense 2B the ANE arm's `L3_after` differs from the GPU arm's by 0.94 %, and
  lowering the 3.8 QSA budget from 2,048 to 8 moves `L3_after` by 10.2 % and the
  logits by 69 %. The folded and causal arms are by contrast *bit-identical* at
  `L3_after`, because at the first full-attention layer the blocks this indexer
  drops contribute below fp16 resolution; the divergence appears at later
  attention layers, which the dump's layer limit does not cover.

**The sidecar covers prompts up to `max(histories) + chunk` = 16,384 tokens by
default.** Past that the chunk at `startPosition` 16,384 has no `h16384` variant
and falls back — and because a chunk may only run on the ANE when every prior
chunk's shadow rows exist, the *rest of that request* stays on the GPU too. For
long-context work:

```bash
~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
  --model models/<install> --chunk 4096 --max-history 32768
```

That raises the ceiling to 36,864 tokens. The export is not free and not
guaranteed: it compiles one more variant per covered layer, and the ANE can
refuse a shape that large — which is what issue #7 was. The exporter now fails
loudly and writes nothing in that case, so the attempt is safe to make; a
refusal means the smaller `--max-history` stays. On this machine (M3) it does
not refuse: a one-layer `--max-history 32768` export compiled all nine variants
and recorded `maxPromptTokens: 36864`. The refusal in issue #7 was an M5.

`v5.5-ane-matrix-3` — 23,000-character prompt (4,333 tokens), chunk 4,096,
two measured runs per arm after a discarded warm-up each, on an M3 24 GB:

| Model | ANE off | ANE on | Saved | Ratio |
| --- | ---: | ---: | ---: | ---: |
| qwen3.5 2B 4-bit | 49.8 s | 36.0 s | 13.8 s | 1.38× |
| qwen3.5 2B 8-bit | 104.1 s | 91.6 s | 12.4 s | 1.14× |
| qwen3.5 4B 4-bit | 140.6 s | 102.2 s | 38.5 s | 1.38× |
| qwen3.5 4B 8-bit | 275.6 s | 218.1 s | 57.5 s | 1.26× |
| qwen3.5 9B 4-bit | 210.7 s | 168.2 s | 42.5 s | 1.25× |
| qwen3.5 9B 8-bit | 442.4 s | 392.9 s | 49.5 s | 1.13× |
| AgentWorld 35B-A3B 4-bit | 106.5 s | 63.3 s | 43.2 s | **1.68×** |
| Qwen 3.8 125B-A6B 4-bit | 175.1 s | — | — | no sidecar (sparse indexer) |

Longer prompts move these ratios up: the same dense 2B at 6,027 tokens measured
1.61×, and AgentWorld at 6,027 tokens ~2.05×, because the offloaded share is the
quadratic one. Shorter prompts move them down, and the 8-bit rows are lower than
their 4-bit siblings at the same length because the non-attention prefill the
ANE never touches is heavier there.

The ANE's saving is a roughly **fixed amount of attention work**: on the dense
2B it was 30.0 s at 4-bit and 30.2 s at 8-bit, so the *ratio* falls as the
non-attention prefill (which the ANE never touches) grows.

None of this is a switch. `TINYTITAN_PREFILL_ANE` is on by default for every
GPU-path model; what varies is whether there is a sidecar the chunk can match.

## Ornith 1.5 35B-A3B: the fp32 attempts were inert, and the sidecar is rejected (2026-11-19)

Chunk 4,096, 4,333-token prompt, M3 24 GB, release CLI. The pre-fix fp16
sidecar already failed the identical-greedy gate; two exporter changes were
made to fix it and neither moved the live numbers:

| Gate | fp16 baseline | fp32 softmax (`d750fa2c`) | fp32 attention core (`ba76b292`) |
| --- | ---: | ---: | ---: |
| verify, synthetic N(0,1) | 0.555% | 0.555% | 0.555% |
| verify, real captured normed | 4.245% | — | 4.245% |
| live logits, `on` | 5.55% | 5.55% | 5.55% |
| identical greedy | false (prefix 0) | — | false (prefix 0) |
| ANE prefill, 4,333 tokens | 162.66 s | 160.42 s | 148.03 s |
| GPU prefill, same prompt | 105.85 s | 95.14 s | 93.56 s |

A whole-program `compute_precision=FLOAT32` probe was run as well: 59 of 173
operations stayed on the ANE (fp16: 74), predict took 382 ms against 244 ms per
layer, and the real-input error was 4.248% — the graph changed by at most
0.00415 in output while the error stayed where it was. Intermediate precision in the core is therefore not the lever, and the
probes now explain the number. `k_new`/`v_new` match the reference at 0.059%,
rounding q/k/v/gate to fp16 after rms+rope moves the block by 0.052%, and the
block's **absolute** error is ~5e-4 on both inputs (0.0934 x 0.555% = 5.2e-4
synthetic, 0.0101 x 4.245% = 4.3e-4 real) — the 7.6x larger *relative* error on
the real input is mostly its smaller output scale, not an amplitude-dependent
defect. The live ANE-vs-GPU difference is the full-layer 0.34-0.6% per attention
layer, which compounds over ten layers to 5.55% logits and flips the greedy
first token. This is numerical non-equivalence, not a wiring, input or KV
bug. The decisive comparison runs the **same package** on three compute units
and scores each against the fp32 reference on the real input:

| Compute unit | Relative error | Peak |
| --- | ---: | ---: |
| CPU + Neural Engine | **4.245%** | 56.4% |
| CPU only | **2.419%** | 96.9% |
| CPU + GPU | **0.075%** | 6.4% |

The graph is right — the GPU executes it to 0.075%. The error is not in the
attention core: a standalone core fed the same q/k/v scores **0.64%**. It is
in the block's last two stages, measured separately on the same package:

| Stage (ANE) | Relative error |
| --- | ---: |
| attention core (scores/softmax/probs·v) | 0.64% |
| output gate (`sigmoid(gate) * attn`) | 1.74% |
| `wo` projection | 3.96% |
| gate + `wo` | 4.25% |

Both are fp16 **accumulation/cancellation** losses: `gated` has mean 0.008
while the `wo` partial products are ~0.1-0.7 and cancel to a 0.0101 output,
so 10-bit partial sums lose the result. Splitting `wo` into 4 or 16 fp32
partials changes nothing (4.252% / 4.259%) — CoreML fuses the decomposition
back, and each partial still cancels internally. A cast or a different softmax
cannot repair it (`compute_precision=FLOAT32` leaves 4.248%).

The actionable split is therefore structural: let the sidecar return the
attention core plus the pre-sigmoid gate, and do `sigmoid`/`wo` on the GPU
where the accumulation is fp32 — the core alone is 0.64%.

This is a **regression from the v4.5 qualification**, not an inherent limit of
the block: at v4.5 the same Ornith 1.5 35B-A3B 4-bit measured **2.31x** at a
6,103-token prompt (132.90 s to 57.52 s) and its greedy smoke output matched the
GPU path token-for-token (`docs/v4.5-ane-prefill.md`). The cache hand-off is not
the regression: on the fp32-core sidecar, `k_new`/`v_new` match the reference's
post-rope K/V in token-major layout to **0.059%** (a standalone probe, not the
verifier, which checks only `out`). What changed between v4.5 and this
install/runtime is the open question for TT-099.

The fix is the **tail split**: the exporter now returns the attention core
(`attn`) and the pre-sigmoid gate (`gate`), and the runtime applies
`sigmoid`/`wo` on the GPU in fp32. Measured on the re-exported split sidecar
(2026-11-19, M3 24 GB, chunk 4,096):

| Gate | Split sidecar |
| --- | --- |
| verify, synthetic | **0.147%** |
| verify, real captured normed | **0.619%** (fused: 4.245%) |
| live logits, `on` | **4.34%** (fused: 5.55%) |
| identical greedy | false, but **coherent** and common prefix **35** (fused: 0, repeated `S`) |
| prefill, 4,333 tokens (single run, default budget) | ANE **78.14 s** vs GPU **101.34 s** |
| paired prefill @ `--ram-budget 10G`, 2 runs/arm | GPU **140.10 s**, ANE **118.00 s**, **1.19×** (22.10 s saved) |

The residual 4.34% comes from the attention core’s own 0.62% block error, so
the identical-greedy bar is still open; the answer is valid text rather than
the fused graph’s degenerate loop. A sidecar records `"tail": "split"`; a
sidecar without the field keeps the historical fused path.

Decision: the fp32 changes were reverted (they were numerically inert and the
core was 1.56x slower per layer), the local sidecar is renamed
`ane_prefill-rejected`, and `tools/ane_sidecars.sh` refuses to export one for
this install until tracker TT-099 is resolved. The records are
`ane-correctness-ane-correctness-ornith15-4bit-{baseline,fp32core}.json` and
`ane-layer-diff-ornith15-4bit-{baseline,fp32softmax,fp32core}.json`.

## Which families can carry a sidecar

`tools/export_ane_prefill.py` reads the attention geometry from the model's own
manifest, so support is a property of the graph, not a list:

| Family | Installs | Sidecar | Why |
| --- | --- | --- | --- |
| `qwen36` | AgentWorld 35B-A3B, Qwen 3.6 35B | yes | the geometry the graph was written for |
| `qwen36` | Ornith 1.5 35B-A3B | **no** | identical-greedy gate fails and the ANE is slower (above; TT-099) |
| `qwen3_5_dense` | Qwen 3.5 2B / 4B / 9B | yes | same attention block, read from the manifest |
| `qwen38flash` | Qwen 3.8 125B-A6B | exportable, **not installed** | the block and the fold are right, but the ANE loses on this model (above) |
| `qwen38flash_mtp` | Qwen 3.8's one-layer MTP draft | **no** | verified, not prefilled, on the ANE |

The 3.8 build names its per-head norms `q_norm`/`k_norm` rather than
`q_norm.weight`, which the loader resolves from the index rather than assuming.

A sidecar also records the geometry it was built for, and the runtime refuses
one that disagrees with the model (family, hidden width, head split, chunk,
covered layers). The exporter resolves each tensor's **per-tensor** width rather
than its slot width: the dense installs declare a 4-bit attention slot over
8-bit `k_proj`/`v_proj`. It also loads every function it records, so a
specialization the ANE refuses cannot be advertised as coverage.

## Reading a record

A record is valid for one (machine, build, model, prompt) combination. The two
arms are **not** bit-identical by construction — the ANE computes attention in
fp16 with a different reduction order — so the digests are expected to differ;
what must hold is that each arm is internally stable, and that `on.used_ane` is
true. Compare a run only against another run of the same model and prompt.

A differing digest is therefore not evidence of a bug, and a matching one is not
evidence of correctness. Whether the two arms generate the *same text* is the
separate measurement above (the `ane-correctness-*.json` records; the Ornith ones are
committed here), and even *that* is not
enough on a prompt whose continuation is nearly deterministic: on 3.8 the
causal-only control agreed with the GPU for all 32 tokens too. The per-arm
activation comparison (the `ane-layer-diff-*.json` record, likewise removed)
is what separates them, via
`prefill_logits`.