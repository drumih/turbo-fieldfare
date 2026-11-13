# Qwen3.8-Flash-Next port — verified design record

Requested in [issue #2](https://github.com/Pummelchen/TinyTitan/issues/2). Every
number below was read from the official checkpoint's `config.json`, the
`transformers` `qwen4_exp` modeling source, or the quantized checkpoints'
tensor indexes — none is assumed. This document is the contract for the port;
later sessions implement against it.

## Sources — and the one thing the request cannot have

**No official MLX quantization exists.** `Qwen/Qwen3.8-Flash-Next` publishes
bf16 safetensors only (132 shards, ~360 GB — cannot be quantized locally on
this disk). Every MLX quantization is community-published. The uniform 4-bit
and 8-bit checkpoints are:

| pin | quant | total | notes |
| --- | --- | ---: | --- |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit` @ `de597762aa61` | affine 4-bit, **group 32** | 111.6 GB | no MTP, vision tower present in bf16 |
| `Vontra/Qwen3.8-Flash-Next-MLX-8bit-MTP` @ `545f9a172c6f` | affine 8-bit, **group 32** | 203.0 GB | MTP bundled |

**Decision 2026-08-27: wait for an mlx-community release** rather than pin
the Vontra checkpoints — the request was for official quantizations, and the
community table above is recorded so the pins can be wired quickly if the
decision changes. Re-check the group size when the mlx-community repos
appear; mlx_lm's default is group 64, which would make the group-32
parameterization below unnecessary.

Were community pins ever adopted, the trust model would match Ornith's
(repo + revision + source-index SHA) and must be stated in user-facing docs. There is **no
group-64 8-bit anywhere**, so the port must parameterize TinyTitan's group size
(the manifest already records `groupSize` per slot; kernels and tools
hardcode 64 today). Affine group 32 costs ~6% more bytes per weight than
group 64 (5.0 vs 4.75 effective bits at 4-bit).

**Disk**: 284 GB free; 4-bit (111.6) + 8-bit (203.0) = 314.6 GB. Both cannot
be installed simultaneously without the user freeing ~35 GB or removing an
existing model (their call, never ours). Plan: 4-bit first.

## Architecture (`model_type: qwen4_exp`, text config)

The checkpoint is **multimodal** (`Qwen4ExpForConditionalGeneration`, vision
tower, interleaved mrope). This port is **text-only**, like the Ornith port;
with equal text positions the mrope collapses exactly to standard partial
RoPE, and `rotate_half` over the first 64 dims matches TinyTitan's existing
NeoX-subdim convention.

Familiar (config-level deltas from Qwen3.5-MoE):

| | Qwen3.5-MoE (shipped) | Qwen3.8-Flash-Next |
| --- | --- | --- |
| layers | 40, (3×GDN → full)×10 | **48**, (3×GDN → QSA)×12 |
| hidden | 2048 | **2560** |
| experts | 256, top-8, ffn 512 | **512, top-10, ffn 640** |
| router | softmax→topk, no norm | softmax→topk, **norm_topk_prob=True** |
| attention | 16Q/2KV, dim 256, packed q+gate, per-head norms, sigmoid gate | **24Q**/2KV, dim 256, same structure |
| GDN | 16 QK / 32 V, dim 128, conv 4 | 16 QK / **48 V**, dim 128, conv 4 |
| RoPE | NeoX-subdim, factor 0.25, θ 1e7, 262K native | identical |
| vocab | 248,320 | identical |
| shared expert | 1, sigmoid-gated | identical (ffn 640) |
| MTP | 1 layer | 1 layer (hybrid full-attention; deferred) |

Genuinely new subsystems, in dependency order:

### 1. Hyper-connections (`hc_count=4`, `hc_lowrank=320`)

The residual stream is **4×2560 = 10,240 wide for the whole stack**. Entry:
`hidden = embeds.repeat(4)`. Every sublayer (attention and MLP separately)
runs a `GatedResidual`:

```
normed  = grouped_rmsnorm(streams)                    # per-2560 groups, one weight [10240]
mix     = sigmoid(W_up @ silu(W_down @ normed / 4))   # 10240 -> 320 -> 10240
input   = mean_over_streams(mix * normed_streams)     # -> 2560, feeds the block
inject  = 2 * sigmoid(W_inj @ normed / 4)             # -> 4 per-stream weights
streams = streams + block_output ⊗ inject             # outer product back into 10240
```

Final collapse is a model-level mixer (same, without inject) — **there is no
`model.norm`**; lm_head applies to the mixed 2560. KV cache, GDN state, and
expert flow are all per-2560-block; only the residual plumbing is 4-wide.
Cost: ~40 KB of extra weights per sublayer, elementwise + two skinny GEMVs.

### 2. Qwen Sparse Attention indexer (`self_attn.indexer.*`)

Per full-attention layer: `index_qk_proj` 2560 → (4+1)×128; q per-head
RMS-normed then RoPE'd; the single key stream is cached **raw** (pre-norm,
pre-RoPE). Selection per query:

- visible keys grouped in blocks of `compress_ratio=4`, block key = fp32 mean
  of 4 raw keys → RMS norm → RoPE at the block's first position;
- score = Σ over the 4 q-heads of relu(q·k) / √128;
- top `2048/4 = 512` blocks kept, plus the incomplete tail block always;
- result is a boolean mask ANDed into the causal mask of the main attention.

**Exactness window: whenever a query sees ≤ 2048+tail keys, everything is
selected and dense attention is bit-exact.** Beyond that the indexer is
mandatory for faithfulness. Runtime plan: dense path first (with a hard
context gate at the budget), indexer second. The indexer needs its own tiny
"raw key" cache (128 fp16/token/QSA-layer ≈ 3 MB per 12 layers at 4K).

### 3. PLE / n-gram embeddings (layer index 1 only, `ple_layer_ids=[2]` is 1-based)

16 hashed lookups per token (2 n-gram orders × 8 heads) into a ~320M-row
table (16 prime-sized head vocabularies just above 20M each; rows are
`2560/16 = 160` wide; the table is quantized and **51.2 B params ≈ 32 GB at
4-bit**). Hashing is exact integer math, CPU-side:

- multipliers: `splitmix64`-derived odd 63-bit constants per (layer, position)
  from `seed=1234` (shipped as a tensor — read it, do not re-derive);
- id = XOR of `token_shift_s * multiplier_s` for s in n-gram, mod the head's
  prime, plus the head's offset; token shifts reset at EOS boundaries
  (`_shift_right_ignore_eos` — a 2-token rolling context, cacheable like a
  conv state);
- gathered 16×160 → concat 2560 → `value_proj` (→2560) and `key_proj`
  (→10240); gate per stream = signed-sqrt of (key·query_normed)/√2560 through
  a sigmoid; plus a **dilated depthwise conv** (k=4, dilation=3, per-channel,
  silu) over the gated value, state length 9; result added to the hyper
  streams before the layer body.

Streaming: one token touches 16 rows × ~100 B (4-bit g32) — ideal SSD reads.
New `.ssdai` section: `ngram_table.bin` with a row-addressable layout, plus
resident buffers for multipliers/offsets/vocab-sizes.

### 4. Top-10 MoE and 3D expert storage

`MoE.maxStreamedExperts = 8` is load-bearing across argument buffers, the
K8-specialized phase-2 PSO, and the cache planner; top-10 needs those
parameterized. MLX stores experts stacked (`mlp.switch_mlp.gate_proj` etc.,
3D `[512, ...]` with per-row g32 scales); the repacker slices to per-expert
blobs exactly like today. Expert blob at 4-bit/g32 ≈ 3.07 MiB (vs 1.688
today), 512 per layer × 48 layers ≈ 75.5 GB of experts.

MLX tensor-name mapping (verified against the Vontra index):

| MLX (`language_model.model.*`) | HF (`model.language_model.*`) |
| --- | --- |
| `layers.N.mlp.switch_mlp.{gate,up,down}_proj` | `layers.N.mlp.experts.{gate_up,down}_proj` |
| `vision_tower.*` (bf16, unquantized) | `model.visual.*` — **skipped, text-only** |
| everything else | same suffixes; `A_log`/`dt_bias`/PLE buffers present ✓ |

## Performance expectations (projections, to be measured)

6 B active × 0.625 B/param (4-bit g32) ≈ 3.75 GB/token GPU reads → ~59 ms
floor on this M3 → **≤17 tok/s ceiling**; expert SSD traffic ~480
activations/token at a 3.07 MiB stride with an unknown hit rate against 512
experts/layer — the measured SSD saturation (~3.4 GB/s) makes **exposed
expert I/O the expected wall**, plausibly single-digit tok/s on 24 GB.
16–32 GB machines are exactly the audience; the memory story fits TinyTitan's
thesis better than any model yet (only 6 B active, table lookups tiny).

## Phasing

- **P0 (foundations)**: this document; `qwen38flash` family + ArchConfig;
  group-size parameterization end-to-end; manifest v2 fields (ngram section,
  indexer/PLE/hyper geometry); repacker source pins + mapping + `ngram_table`
  writer. Installable artifact at the end of P0.
- **P1 (runtime)**: group-32 kernel support; hyper-connection kernels and the
  reworked layer loop; GDN at 16/48; dense QSA with the ≤2048 exactness gate;
  PLE path; top-10 MoE. CLI generation at the end of P1.
- **P2 (fidelity + scale)**: indexer for >2048; parity harness against
  `transformers` on short prefixes (the golden analog — greedy logit match on
  the exactness window); benchmarks; server enablement; 8-bit install once
  the disk decision is made; MTP sidecar last (MTP is measured-off anyway).

Deferred by decision: vision, MTP, YaRN >262K (native first).

## Cross-check against the official announcement (2026-08-27)

The launch blog (https://qwen.ai/blog?id=qwen3.8-flash-next) confirms the
verified geometry above and adds facts the config/source audit could not show:

- **This is the Qwen4 preview architecture.** Qwen states it plays the role
  Qwen3-Next played for Qwen3.5–3.8: the GDN+QSA / Gated-Residual / n-gram
  design will carry the whole Qwen4 family. The family-schema and ArchConfig
  work is therefore an investment in every upcoming Qwen model, not one port.
- **The n-gram offload strategy is official.** Qwen ships the 51 B-parameter
  table expecting it to live outside accelerator memory: "lookup locations
  can be determined in advance, these parameters can be stored in Host Memory
  and asynchronously prefetched in parallel with model computation." Our
  planned row-addressable `ngram_table.bin` on SSD is the same idea one tier
  down, with a stronger property than expert streaming: lookups depend only
  on token ids, never on router output, so decode can prefetch the next
  token's rows the moment it is sampled — the read is fully off the critical
  path. Prefill can batch all rows for a chunk up front.
- **"Gated Residual" is the official name** for what the config calls
  hyper-connections: 4 residual branches with element-wise, data-dependent
  read/write gates (the low-rank 320 projections). Qwen explicitly *removed*
  Hyper-Connection's branch-mixing matrices as unnecessary — the simplified
  form we mapped is the intended one. The residual state "supports FP8
  storage"; we start fp16 in Metal and note fp8-as-bytes as a later
  bandwidth lever (4 branches × 2560 quadruples residual traffic).
- **QSA indexing is per-layer by design** (independent sequence compression
  per layer, contrasted with cross-layer index reuse), so no shared index
  cache is needed. Claimed kernel speedups are 7.6× prefill / 4.9× decode at
  1M tokens — irrelevant below the 2048-token exactness gate, which supports
  dense-first phasing.
- **MTP is multi-step trained** for higher real acceptance, and its
  full-attention layers use QSA too. Still last in phasing; the 35 B verify
  economics don't transfer, so it would need fresh qualification.
- **Reasoning effort is not binary — verified and implemented.** The pinned
  upstream `chat_template.jinja` (fetched 2026-08-27) defines
  `reasoning_effort` `low|medium|xhigh` (default `xhigh`) while
  `enable_thinking` is on: `xhigh` and `low` inject an instruction sentence
  at the head of the system block, `medium` injects nothing, and invalid
  values raise. Thinking on also pre-opens `<think>\n` in the generation
  prompt (the off branch emits the closed block), and upstream
  `generation_config.json` defaults to temperature 1.0 / Top-K 20 /
  Top-P 0.95. **Implemented 2026-09-21 as two rows** — thinking
  `1.0 / 0.95 / 20`, instruct `0.7 / 0.80 / 20 / presence 1.5`, both min-p 0.0 —
  selected by the request's thinking mode
  (`GenerationDefaults.qwen38Thinking` / `qwen38Instruct`, applied in
  `OpenAIRequestValidator`). The presence penalty had to be implemented first:
  the sampler refused every non-zero value until then, and min-p is still not
  implemented (0.0 in both rows, and a non-zero is refused rather than ignored).
  TinyTitan now has
  the per-family control: `ModelFamily.reasoningControl` gates
  `--reasoning-effort` / `TINYTITAN_REASONING_EFFORT` / the API's
  `reasoning_effort` field (binary families reject them; effort is a
  load-time control validated against the manifest family), the tokenizer
  passes the effort into the bundled template and mirrors its system-block
  injection on the manual ChatML path, and the real template renders
  correctly under the Swift Jinja engine (fixture-tested, including the
  effort sentences). Remaining P1 wiring: `ManifestReader.peekIdentity`
  must learn to report `qwen38flash` (it derives qwen36/qwen36MTP from
  layer shape today), and the Mac app picker stays binary until the model
  is installable.
- **Totals as marketed**: 125 B backbone + 51 B n-gram + 6 B active/token;
  native 262,144 context, YaRN to 1M — matching the ArchConfig. Weights are
  live on HF/ModelScope (bf16 official); still no official MLX artifact, so
  the wait-for-mlx-community decision stands.

## Source found and verified against a real artifact (2026-08-28)

`RockTalk/Qwen3.8-Flash-Next-MLX-4bit` at revision
`478474da92599ad0cf9f8bd447e658b29cb8480a`
(`model.safetensors.index.json` sha256
`d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b`).
Not an mlx-community release — the wait-for-mlx-community decision is
superseded because this artifact answers every open question and mlx-community
has not shipped.

### The group-size question is answered: 64

`quantization: {group_size: 64, bits: 4, mode: affine}` — the format TinyTitan
already uses. **The planned P1 group-32 kernel work is dropped.** Per-path
overrides are 8-bit g64 on `embed_tokens`, `lm_head`, `mlp.gate` (router) and
`shared_expert_gate`; everything else including all 512 routed experts is
4-bit g64. `manifest.quant` already carries independent `weightBits` per slot,
so this is expressible today.

### The design record is confirmed, not merely plausible

Every geometry figure matches: 48 layers, `full_attention_interval` 4 (36
`linear_attn` + 12 `self_attn` tensor sets, exactly the 3-GDN-to-1-attention
pattern), 512 experts top-10, ffn 640, hidden 2560, vocab 248,320, head_dim
256 with 24 Q / 2 KV, GDN 16 K-heads / 48 V-heads at dim 128, `hc_count` 4,
`hc_lowrank` 320, `ple_layer_ids` [2].

`ple_constants.json` confirms the PLE reverse-engineering independently:
`ple_n_heads` 16, `ple_head_dim` 160, 16 prime head vocabularies just above
20M, `ngram_size` 3, `heads_per_ngram` 8, eos 248044, and three layer
multipliers shipped as data (read them, as planned). Row count computes to
320,001,446 x 160 x fp16 = 102,400,491,520 bytes, matching `ngram_table.bin`
byte for byte.

QSA indexer tensors are present and separate per layer
(`self_attn.indexer.index_{q,k}_proj`, `{q,k}_layernorm`), and the upstream
card independently reports the mask is O(n^2) above 2051 tokens — the same
dense-exactness boundary the <=2048 gate was designed around.

There are **no layer norms**: `hc_norm` inside each hyper-connection block
replaces `input_layernorm` / `post_attention_layernorm` entirely.

### Sizes and the disk constraint

| part | bytes | |
| --- | ---: | ---: |
| backbone shards (49) | 71,425,417,784 | 71.4 GB |
| `mtp-weights.safetensors` | 1,467,252,412 | 1.5 GB |
| `ngram_table.bin` (fp16) | 102,400,491,520 | 102.4 GB |
| **total** | **175,293,161,716** | **175.3 GB** |

The n-gram table ships **fp16, not quantized** — 3.2x the ~32 GB the plan
assumed at 4-bit. With 288 GB free the full install fits **only because the
installer streams into `.ssdai` without staging a second checkpoint**:
downloading the repo first and then repacking would need ~352 GB and fail.
Quantizing the table during repack is possible but is a lossy change to a
component whose quality contribution is unmeasured — do not do it silently.

### What the repacker already handles, and what it does not

Qwen 3.6 shares the GDN+MoE lineage, so `RepackPlanner` already maps
`.mlp.switch_mlp.*` 3D fused experts, the whole `linear_attn` family
(`in_proj_qkv/z/a/b`, `conv1d`, `A_log`, `dt_bias`, `norm`, `out_proj`),
`shared_expert*` and `self_attn.{q,k,v,o}_proj`. New work:

- **tensor prefix differs**: `model.language_model.layers.N.*` here against
  qwen36's `language_model.model.layers.N.*`, and `lm_head.*` sits at the top
  level rather than under `language_model.`;
- **hyper-connections**: `attn_hyper_connection.*`, `mlp_hyper_connection.*`
  (`block_inject_weight`, `input_mix_weight_{up,down}`, `hc_norm`) plus a
  model-level `hyper_connection_mixer.*`;
- **QSA indexer** tensors per full-attention layer;
- **PLE**: `ple.conv1d` plus `ngram_table.bin` and `ple_constants.json` as
  passthrough sidecars, not safetensors tensors;
- **MTP**: `mtp.layers.0.*` is a complete layer with its own `self_attn`,
  indexer and **its own 512-expert `switch_mlp`** (hence 1.5 GB) — it should
  become an SSD-streamed sidecar like the existing Ornith/Qwen MTP path, not
  resident weights;
- **manifest v2**: ngram section, per-slot quant bits, indexer/PLE/hyper
  geometry.

### Trust

Created 2026-08-28, **0 downloads, 0 likes**, unknown author, no community
validation. The code is open (MIT, `Rocktalk-Holdings/mlx-qwen4exp`) and the
card reports measured M3 Ultra figures (26.4-26.6 tok/s greedy), so it reads
as serious work rather than a drive-by upload — but the quantization quality
is unverified by anyone. Pinning the revision plus the receipt's SHA256 covers
integrity, not quality.

### Quantization fidelity: VERIFIED SOUND (2026-08-28)

A greedy logit parity against `transformers` is not runnable here -- the bf16
checkpoint is ~360 GB against 288 GB free -- but the question underneath it is
answerable directly and cheaply. `benchmark/tinytitan_quant_fidelity.py` pulls the
same tensors from both repos by HTTP range (no bulk download), dequantizes the
MLX affine blocks, and compares against the official bf16 values.

The meaningful test is not an absolute error threshold but whether the
quantizer reaches the floor the format allows, so each tensor is also compared
against an ideal affine round-trip of the official weights:

| tensor | bits | MAE ratio | ideal floor | excess |
| --- | ---: | ---: | ---: | ---: |
| L3 `self_attn.q_proj` | 4 | 0.1096 | 0.1074 | **1.02x** |
| L3 `self_attn.o_proj` | 4 | 0.1068 | 0.1047 | **1.02x** |
| L0 `linear_attn.out_proj` | 4 | 0.1158 | 0.1136 | **1.02x** |
| L3 `mlp.shared_expert.gate_proj` | 4 | 0.1120 | 0.1097 | **1.02x** |
| L19 `self_attn.q_proj` | 4 | 0.1058 | 0.1037 | **1.02x** |
| L3 `mlp.gate` (router) | 8 | 0.0096 | — | — |

**1.02x the floor everywhere.** The quantizer is as accurate as 4-bit affine
g64 permits; the residual 2% is consistent with quantizing from bf16. The
8-bit router lands at 0.96%, matching theory for 8-bit. Shapes, packing order
and the scale/bias convention all decode correctly against the official
tensors, which also validates the dequantization contract the repacker will
need. **Green light to build on this artifact.**

Note this measures weight fidelity, not end-to-end behaviour; a short greedy
logit comparison against the reference implementation is still worth running
once the runtime can execute the model.

## P0 implementation status (2026-08-28)

Landed, each gated by lint + the full serial suite and, where the runtime was
touched, both goldens byte-identical:

| piece | commit | what it does |
| --- | --- | --- |
| ArchInfo loader | `f6cd041` | reads the whole `qwen4_exp` config: layer mask, GDN, hyper-connection, indexer and PLE geometry; converts `ple_layer_ids` from 1-based and rejects out-of-range; reads the source group size instead of defaulting; contract-checks the production shape |
| tensor classification | `a11e17f` | the `model.language_model.*` prefix, top-level `lm_head.*`, and the MTP draft held out for its own sidecar; verified against all 3,164 real tensor names with zero unknowns |
| manifest geometry | `969225b` | writes and validates the new geometry, so a checkpoint whose hyper-connection rank, indexer budget, PLE layer set or group size differs is refused instead of silently mis-run |
| passthrough planning | `423c85d` | `ple_constants.json` (required) and `ngram_table.bin` (optional, 102 GB) as resumable chunked range copies |

Also corrected along the way: `quantGroupSize` was 32 in the runtime
`ArchConfig` from the Vontra-era assumption; the pinned artifact is 64.

### Still needed before an install runs

1. **Resolve passthrough sizes remotely and pass them to the planner.** The
   plumbing exists; the remote repacker does not call it yet. These are
   standalone files, not index entries, so their size comes from a
   `resolveFileInfo` before planning.
2. **`SupportedModelSource` pin.** Ready to write:
   repo `RockTalk/Qwen3.8-Flash-Next-MLX-4bit`, revision
   `478474da92599ad0cf9f8bd447e658b29cb8480a`, index sha256
   `d7fe03ad2d1365e24ae2e305c829f600b80fac845d128383421a7be4adbdda1b`,
   download 175,293,161,716 bytes.
3. **MTP sidecar install.** Its 81 tensors sit in their own
   `mtp-weights.safetensors`, so it wants a second install invocation
   producing its own directory, like the Ornith MTP path. It carries its own
   512-expert set, so those experts must be SSD-streamed too, not resident.

None of that makes the model *execute* -- that is P1 (hyper-connection
residual plumbing, QSA dense path with the <=2048 exactness gate, the PLE
block, top-10 MoE) and is the larger half of the work.

## P1 status (2026-08-29)

The blockers listed here previously are cleared. What follows is what exists,
what it was verified against, and what genuinely remains.

### Cleared

| piece | commit | verified against |
| --- | --- | --- |
| top-10 routing, 512 experts | `f705873`, `836074f` | both goldens byte-identical at k=8 |
| Gated Residual primitives | `94ef43f`, `80c37b0` | CPU references per kernel |
| Gated Residual composition | `9081afd` | CPU reference of the whole formula |
| PLE row hashing | `9c7d640` | golden vectors from the reference implementation |
| n-gram table gather | `a7267f1` | synthetic table encoding its own row indices |
| QSA dense-exactness window | `ba3e5f9` | derived, and equal to the card's 2,051 |

Notes worth keeping:

- **`kMaxStreamedExperts` is 16**, one bound for every family, because Metal
  array extents must be compile-time. Kernels read only `[0, top_k)`, so the
  eight unused pointers cost 64 bytes per argument buffer and nothing else.
- **`_kn` kernel variants** exist for router select and both phase-2 reduces,
  dispatched only when k != 8 so the shipping models stay on kernels whose
  output is byte-pinned. The launch width is the contract: exactly k
  simdgroups, so the kernels carry no `sg >= k` guard -- an early return there
  would skip a `threadgroup_barrier` every thread must reach.
- **The router already implements `norm_topk_prob=true`**: it softmaxes over
  the selected top-k, which equals softmax-over-all then renormalize. Wiring
  `ArchConfig.routerNormTopK` should preserve this, not change it.
- **Grouped RMSNorm is not the per-head kernel.** That one shares a single
  weight across groups; the hyper connection needs a distinct 2560 slice per
  stream out of its [10240] weight.
- **The write gate is `2 * sigmoid`.** Plain sigmoid caps every stream at 1.0
  and makes the residual strictly contractive -- plausible, and wrong.
- **PLE's context cut latches**, and a token that is itself EOS does not cut
  its own context. One mixed value serves all 8 heads of an n-gram order.

### What remains

1. **Layer-loop integration.** The structural change and where regression risk
   concentrates: the residual becomes 4 x 2560 = 10,240 wide through the whole
   stack, which touches buffer allocation, the prefill and decode paths, and
   the KV/GDN state plumbing in `RealForwardRunner`. The primitives above are
   built to make this two calls bracketing each block, but nothing calls them
   yet. **The model cannot execute until this lands.**
2. **PLE block body.** The gather exists; the value/key projections, the
   signed-sqrt gate and the dilated depthwise conv (k=4, dilation=3, state 9)
   do not.
3. **QSA indexer**, for contexts past 2,051 keys. Below that the gate says
   dense is exact; above it the indexer is mandatory for faithfulness.
4. **MTP sidecar install**, which carries its own 512-expert set and wants its
   own directory like the Ornith MTP path.

### How to verify, when it runs

Weight fidelity is settled (`benchmark/tinytitan_quant_fidelity.py`); behaviour is
not. The reference implementation is public
(`Rocktalk-Holdings/mlx-qwen4exp`), so the honest check is a greedy logit
comparison against it on a short prefix -- the analogue of the golden baseline
the shipping models use. Nothing here should be called correct on the strength
of unit tests alone.

## PLE block: verified against the reference (2026-08-29)

Read from `mlx_qwen4exp/ple.py`, not inferred. The design record above is
wrong in two places, both of which would have produced plausible output with
no error:

1. **The block adds two terms, not one.** The result is
   `hidden_wide + gated + conv_out`. The record described only the convolution
   being added; dropping `gated` would silently remove the whole gated-value
   contribution.
2. **The gate's signed square root has a floor.**
   `sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))` -- the record omitted the 1e-6,
   which matters exactly where `s` is near zero and the derivative blows up.

Also corrected: the convolution's **dilation is `ngram_size` (3)**, not an
independent constant, and its history is `(K-1) * dilation = 9`.

The full block, with shapes at production geometry:

```
emb        = table[rows]                      # [T,16,160] -> flat [T,2560]
                                              #   head axis is the slow one
key        = key_proj(emb)                    # [T,10240]
value      = value_proj(emb)                  # [T,2560]
key_w      = groupedNorm(key,  norm_key,  hc=4)
query      = groupedNorm(hidden_wide, norm_query, hc=4)
s          = sum over D of key_w * query / sqrt(2560)      # [T,4] per stream
gate       = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))       # [T,4]
gated      = value broadcast over streams * gate           # [T,4,2560]
normalized = groupedNorm(gated, norm_conv, hc=4)           # [T,10240]
conv[t,c]  = sum over k of w[c,k] * xpad[t + k*3, c]       # K=4, dilation=3
                                              #   tap k reads (K-1-k)*3 back
conv_out   = silu(conv)
result     = hidden_wide + gated + conv_out
state      = last 9 rows of xpad
```

Three norm weights, all `[hc_dim]`: `norm_key`, `norm_query`, `norm_conv`.
The two projections are ordinary INT4 GEMVs and the norms are the grouped
RMSNorm already built, so the kernels still missing are the per-stream dot,
the signed-sqrt gate, the broadcast scale, and the dilated depthwise causal
convolution with carried state.

The convolution's tap indexing is the part most worth pinning in a test: tap
`k` reads `(K-1-k) * dilation` positions back, so an off-by-one in either
direction still produces smooth, plausible output.

## It answers (2026-08-30)

```
$ TinyTitanCLI --model qwen3.8-flash-next_125B_A6B_4Bit \
    --prompt "The capital of France is" --max-new 20 --temperature 0
 Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain
```

### How it was found

Six defects stood between "runs" and "answers" (the sixth is below). Not one of them threw, and
not one was visible in the output as anything but fluent nonsense --- every
single one produced a smooth, confident, wrong distribution. They were found
by numerical parity against the reference implementation, stage by stage, and
they would not have been found any other way.

The harness is in `tools/`:

- `ssdai_reader.py` reads tensors straight out of the installed
  `model_weights.bin` and `packed_experts/`, dequantizing INT4 and INT8 affine
  groups. Both sides therefore run on the *same* weights, so quantization is
  common-mode and any disagreement is in the forward pass.
- `qwen38_reference.py` is the model in numpy, one token at a time, carrying
  all four states: the KV cache, the delta-rule state, the Gated DeltaNet
  convolution tail and the PLE convolution history.
- `qwen38_parity.py`, `qwen38_full_forward.py` and
  `qwen38_sequence_parity.py` compare it against a run's activation dumps.

The runtime side is `TINYTITAN_ACT_DUMP=<dir>`, with `TINYTITAN_ACT_DUMP_POSITIONS`
for how many positions to capture. It is off unless the variable is set.

The dumps hang off the decode path, so checking a *prompt's* positions needs
`TINYTITAN_SEQUENTIAL_HC_PREFILL=1` as well --- batched prefill never calls
`produceToken`, and without the oracle the reference's caches start at the
first generated token and every comparison after it is shifted.

Position 0 is the workhorse: every carried state is empty there, so the whole
stack reduces to closed-form algebra and a divergence is unambiguously one
layer's math. Once position 0 was exact end-to-end, running further positions
separated "wrong block" from "wrong hand-off between tokens".

### The five defects

1. **Attention split-KV scratch was sized for 16 query heads**; this model has
   24. The only one of the five that failed loudly (a precondition trap).
   `Attention.maxQHeads`.

2. **The Gated DeltaNet output gate was silu, and this family uses sigmoid**
   (`output_gate_type` in its config). Inherited from Qwen 3.6, where silu is
   right. Both are smooth and positive-ish, so the model kept generating.
   Fixed with a function constant (`FC_GDN_SIGMOID_GATE`) selected from
   `LinearAttentionConfig.outputGate`, so the choice is declared per family
   rather than defaulted from the other one.

3. **The PLE convolution applied silu twice** --- once in the kernel, once in
   the composing Swift --- and **read its BF16 tap weights as FP16**. The
   dtype error is the instructive one: reinterpreting BF16 bytes as FP16
   yields plausible small numbers, never a NaN or a crash.

4. **The MoE computed 8 of 10 experts.** The specialized decode pipelines bake
   D, F and k in as function constants; D and F came from the model but k was
   a hardcoded 8. This family shares Qwen 3.6's hidden dimensions and routes
   to ten experts, so it took the specialized path and phase 1 simply never
   wrote the last two activation slots, which the reduce then summed as zeros.
   The residual was the two lowest-weighted experts --- about 1% of the MLP
   output, which reads exactly like quantization noise. Pinned by
   `MoEFusedFFNTests.specializedPipelineComputesEveryExpertSlot`.

5. **The vocabulary head used the attention slot's bit width.** It is 8-bit
   here while attention is 4-bit --- a combination no earlier family had --- so
   the head read a packed-INT8 tensor as INT4. This one was total: the logit
   vector had cosine -0.18 against the reference while every one of the 48
   layers beneath it was exact. `RealForwardRunner.affineHead`.

The fused greedy head was also disabled for hyper-connection families: it
folds a plain RMSNorm into the vocabulary GEMV, and this stack ends in the
gated mixer that collapses the residual streams, not in an RMSNorm.

### The sixth defect: zero-centred RMSNorm (2026-09-01)

A build converted from Qwen's own bf16 release generated fluent nonsense while
the MLX-sourced build answered. The cause was one line of convention.

`Qwen3_5RMSNorm` initialises its parameter to **zeros** and computes
`normalized * (1.0 + weight)`. The checkpoint therefore stores the *offset from
one*, not the gain. TinyTitan's runtime multiplies by the stored value, so the +1
has to be folded in at conversion -- which is exactly what MLX's converter
does. `prepare_qwen38.py` copied the raw value through, so gamma was wrong by
one on **148 tensors**: every hyper-connection norm (97 of them -- both norms
in all 48 layers plus the final mixer), the three PLE norms, and QSA's
`q_norm`/`k_norm` and indexer layernorms.

`linear_attn.norm` is untouched and must stay that way: it is
`Qwen3NextRMSNormGated`, which uses `weight` directly. So must `ple.conv1d`,
which is a convolution kernel and not a norm at all.

**Why every check passed.** This is the instructive part, and it is a stronger
version of the common-mode warning above.

- Comparing the install against the checkpoint passed at cosine 1.000000,
  because the checkpoint genuinely stores that form. The weights were never
  wrong; the *interpretation* was.
- Comparing per-tensor bit widths, names, the `gate_up` split, the 95.4 GiB
  n-gram table and the PLE constants against the MLX build all passed, because
  none of them is the affected quantity.
- Running `transformers`' own `Qwen3_5GatedDeltaNet` on our weights matched at
  cosine 1.000000 across positions -- but the hyper-connection, PLE and QSA
  ground truths were built from *transcribed* norm classes, and the
  transcription carried the same assumption. A ground truth is only as
  independent as its weakest borrowed line.

What finally isolated it was running the reference on the MLX build's weights
over HTTP range requests: identical code, different weights, ` Paris`. That
reduced the search to "which weights differ", and a direct install-vs-MLX
comparison -- never run before, because both had already been compared against
the checkpoint -- showed `MLX == mine + 1` on exactly the norms above, with
`linear_attn.norm` identical.

**The lesson to carry:** comparing two artifacts against a third does not
establish that they agree with each other, and a convention is invisible to
every check that reads the value the same way at both ends.

### Prefill

Prefill started as a one-token-at-a-time loop through the decode path,
because the Gated Residual's chain is written per token and batching it means
batched variants of all of it. That was correct and unusable: every token paid
a full pass over the routed experts, so a 1,761-token prompt took nine
minutes.

It is batched now --- the grouped norm, the three gate projections, the stream
reduce, the inject, and the whole PLE block all take a row count --- and the
sequential path is kept behind `TINYTITAN_SEQUENTIAL_HC_PREFILL=1` as the oracle
it is checked against. The two produce byte-identical output.

What actually buys the time is chunk size, because routed experts are what
prefill spends itself on and a longer chunk amortizes them. Measured on a
1,761-token prompt, interleaved A/B/B/A: 129.9 s at the old 128-token default
against 75.2 s at 2,048, with the repeats agreeing to 0.6%. Against the
sequential path it is roughly 7x. The family's default is 2,048, which is also
the largest chunk the sparse-attention gate can ever admit.

### The sparse-attention indexer

`QSAExactness` was written to name the boundary past which dense attention
stops matching this model's sparse selection, and then never called --- so
every long context ran dense with nothing reporting it, which is the exact
failure the type exists to describe. It is enforced now, and past the window
the decode path runs the real indexer instead of refusing.

The indexer keeps two caches per full-attention layer: the **raw** keys,
cached before their norm and rope because a block's pooled vector is the mean
of raw members, and the **pooled** block vectors, post-norm and post-rope at
the block's own position (`blockIndex * compressRatio`, not the query's). A
decode step repools one block; a prefill chunk repools the range it touched.

Selection happens on the host. The ordering is score-descending,
index-ascending over *cells*, every cell in a block carries its block's score,
and the cell budget can cut a block in half --- clear in twenty lines of Swift
and fiddly on the GPU. It costs one barrier on the twelve full-attention
layers of a decode step that is bound by expert I/O. Moving it to the GPU is
the optimization, and this is the reference to make it against.

Prefill selects too, per query, so a long prompt is no longer refused. The
chunk's block scores are one kernel over (query, block); the selection is the
same host computation, one row per query, and costs about a second on a
2,048-token chunk against the ~75 s that chunk already takes.

Verified against the reference the same way everything else was, with
`TINYTITAN_QSA_BUDGET` lowering the budget so the sparse path engages after 67
tokens instead of 2,051: across 80 positions of a real generation, layer 3's
attention output matches to cosine 0.99997 or better on both sides of the
boundary. Without that knob every check of this code would be a
multi-thousand-token run.

The batched prefill selection is checked against the decode one, which is
what was checked against the reference. At the **first** full-attention layer,
where both paths still see the same input, their block scores agree to cosine
1.0 and their selections are byte-identical.

That comparison has to be made at the first layer, and the reason is worth
recording: at a deliberately punishing budget (67 of 290 keys, a quarter of
the context) the two paths' *final* logits diverge to cosine 0.994. Top-k over
scores is chaotic --- 0.2% of fp16 drift moves one block across the cut, and a
different block out of sixteen changes the answer. At a production-like ratio
(74% kept) the same A/B produces identical text. So the divergence measures
the budget, not the code, and only the first layer isolates the question being
asked.

### What parity says now

At position 0, every layer entry and the stack output match the reference to
cosine 1.00000. Across a prompt the residual drifts to about 0.99 by position
five --- FP16 activations through 48 layers, entering the carried state and
compounding --- and the argmax tracks the reference through the tokens
checked. The per-block checks (`GDN`, `QSA`, `MoE`, `PLE`) hold at 1.00000 at
every position, which is what says the drift is precision rather than a bug.

Both shipping goldens (`tools/golden-baseline.sh --check 4|8`) are
byte-identical after all of this. The families differ from this one in every
place that was touched: silu output gate, top-k of 8, and a head that shares
the attention slot's width.

### Parity of the build converted from Qwen's bf16 release

The numbers above were measured on an install repacked from a third-party MLX
4-bit release. Converting from `Qwen/Qwen3.8-Flash-Next` directly
(`tools/prepare_qwen38.py`) produces a model that loads and runs but generates
incoherent text, and position-0 parity does not reach the standard above:

    layer 0 GDN   cos 0.99995   |runtime| / |reference| 0.993
    layer 1 GDN   cos 0.97932                           0.835
    layer 2 GDN   cos 0.99939                           0.941

Every layer is short of 1.00000, not just the worst one, and the runtime's
output is smaller than the reference at all three. The disagreement is spread
evenly across channels rather than localised to any head, which is the shape
of a scale or a gate rather than a corrupted tensor.

What that rules out, each verified rather than assumed:

- the expert `gate_up` split (MoE block exact at 1.00000)
- the indexer query/key split (QSA exact; matches the reference's
  `torch.split([n_heads * dim, kv_heads * dim])`)
- the n-gram gather: row ids recomputed from `ple_constants.json` and read
  straight out of `ngram_table.bin` match the runtime's dumped embedding at
  cos 1.00000, max difference 0.0
- the PLE hash constants, which reproduce the int64 buffers Qwen ships
- the weights themselves: their ~10% deviation from bf16 is 4-bit
  quantisation error and is common-mode, since the harness reads the same
  install the runtime does

What it points at is the quantiser. This build's affine group-64 packing is
ours (`quantize_affine`); the previous one was MLX's. Both produce the same
layout and both are read by the same dequantisation, so a systematic
magnitude difference concentrated in the linear-attention path is the next
thing to measure -- starting with whether the GPU and the numpy harness agree
on dequantising one of these tensors, since everything above assumes they do.

### Still open

- **Decode is 3-5 tok/s**, and it is bound by expert I/O, not compute: a
  token touches 10 experts across 48 layers. A route trace simulated against
  an LRU cache once put the hit rate at 65% for 64 slots and 78% at 128,
  "where it saturates". **Measured on the runtime 2026-09-18 (TT-011), it does
  not saturate there:**

  | slots | cache | hit | tok/s | MiB/token |
  | ---: | ---: | ---: | ---: | ---: |
  | 64 | 7.92 GiB | 67.7% | 3.44 | 408.0 |
  | 96 (shipped) | 11.88 GiB | 76.5% | 3.66 | 296.7 |
  | 128 | 15.84 GiB | 82.6% | **1.21** | 220.0 |

  7,550 prompt tokens, 256 new, greedy, interleaved two runs per point
  (`benchmark/expert_cache_slots.py`). The hit rate keeps climbing (+6.1 points
  from 96 to 128) and the reads drop 26%, but at 15.8 GiB this 24 GB machine
  falls 3x on decode — the cache no longer fits beside the weights — so the
  shipped 96 slots stay the default and 128 is reachable with
  `--ram-budget 16G` on a machine that can hold it. The 512-experts-at-top-10
  working set is far wider than Qwen 3.6's 256-at-top-8 (93.6% hit), which is
  why this family buys cache where the others have flattened.
- The n-gram gather still sits inline on the decode path, where it could be
  issued a token ahead. It is 5 KiB a token against the experts' hundreds of
  megabytes, so it is not where the time is.
- The **QSA indexer** runs on both paths now, so neither long prompts nor
  long generations are refused. What remains is that both selections are host
  computations behind a barrier; moving them to the GPU is the optimization,
  and the current code is the reference to make it against.
- The **MTP draft head** is installed and works, and is a measured loss. See
  below.
- Long-context behaviour is verified only at a lowered budget, where the
  sparse path engages after 67 tokens. The arithmetic is the same at 2,051,
  but nothing has been diffed against a reference at that length --- the
  reference cannot hold this model.

## MTP: measured, and closed (2026-08-30)

### Re-verified against the converted install (2026-09-01)

The 2026-08-30 measurement paired the draft with the MLX-sourced target. After
the target was rebuilt from Qwen's own release and the norm fold landed
(`1cc393a`), the pairing was re-checked: the sidecar loads (`mtp=on:384MiB`),
the two installs no longer share a `sourceSnapshotHash` and nothing objects,
and the output is coherent.

It is still a large loss. Same server settings, same prompt, 128 tokens,
prompt cache off, warm:

| | wall | approx tok/s |
| --- | ---: | ---: |
| MTP on | 65.6 s / 67.8 s | ~2.0 |
| MTP off | 23.2 s / 22.5 s | ~6.4 |

About **3x slower**, which is the same verdict the acceptance arithmetic gave
and slightly worse than the 2.8x it predicted. The recommendation is unchanged:
MTP stays off unless asked for. Note the draft head installed here still comes
from `TinyTitanRepack --model qwen38flash-mtp` against the MLX repack; the
converter's own MTP path is fixed but has never produced a build.


The draft head installs from its own 1.37 GB shard inside the target's
repository (`TinyTitanRepack --model qwen38flash-mtp`) and runs through the
existing speculative loop (`TinyTitanServer --mtp-model`). It is **off unless
asked for**, and should stay that way.

**Measured, 150 tokens, same prompt:**

| | | |
|---|---|---|
| Acceptance | 71.3% (62 of 87 passes) | count, sound |
| Emitted per pass | 1.72 | count, sound |
| Width-2 union | 1.65x weight reads | geometry, sound |
| Ceiling at 100% acceptance | 2.00x | geometry, sound |
| Width-2 verify pass | ~2.75x a scalar token | **timing, contended** |
| End to end | ~0.71x scalar | **timing, contended** |

**The two timings were taken on a contended machine** and should be re-run
before being quoted: a VM at 299% CPU and 77% GPU utilization from another
process, which showed up afterwards as a 44% spread within a single config.
The counts and the geometry are unaffected, and they are what closes this:
1.72 tokens per pass against a 2.00x ceiling cannot win once a pass costs more
than 1.72x, and the union says it costs 1.65x in weight reads alone before any
per-pass overhead. 71.3% acceptance is *good* and changes nothing.
This reproduces the Qwen 3.6 verdict (`3abd32b`) on a different model with
different geometry -- and the geometry predicted it: adjacent tokens share
3.49 of 10 experts here against 3.3 of 8 there, so the width-2 union is 1.65x
against 1.59x.

### The fusion is inferred, not verified

Every other part of this port was checked against a reference. This one could
not be. Upstream `transformers` drops the `mtp.*` namespace outright
(`_keys_to_ignore_on_load_unexpected`), and the MLX port infers the fusion
from tensor shapes, says so in its own comments, and applies the gamma twice
while doing it.

What the shapes fix: `pre_fc_norm_hidden` is `[hc_dim]` while `fc_hidden` is
`[D, D]`, so the wide residual is normed and then collapsed before the
projection sees it, and a plain mean is the collapse that inverts `hc_init`.
Two `[D, D]` matrices rather than one `[D, 2D]` is what says the branches are
summed rather than concatenated.

Being uncertain here is safe in a way it was not elsewhere: speculative
decoding emits the target's own argmax, so a wrong fusion costs throughput and
never output. 71.3% acceptance says the reading is at least close to right.

One check the reference's prose would have got wrong: it states the two
`pre_fc_norm_*` gammas are stored zero-centred and need `+1` folded in. They
are not, in this checkpoint -- `pre_fc_norm_embedding` has mean +0.236, which
is exactly the reference's *post*-fold value. The MLX conversion already
applied it, as it did for every other gamma here. Folding again would have
produced a draft that was simply never accepted, with nothing to say why.

### Not byte-exact against the scalar path

Qwen 3.6's MTP is byte-identical to scalar decoding because its verify path
was deliberately built to reproduce the decode numerics row by row. This one
uses the generic batched prefill, so row 0's reassociation differs and the two
diverge at near-ties: on the measured prompt they agree for 205 characters and
then split between two equally correct continuations. Both are valid greedy
decodes of the same model. Given the throughput verdict this was not worth
tuning, but it means MTP here can change output where the Qwen 3.6 path
cannot.

### Three bugs it took to get there

All three were silent, and one was the third instance of a single mistake.

1. The width-2 verify's **own MoE tail** did a plain residual add into the
   wide residual, bypassing the hyper-connection write gate. It is a separate
   code path from the batched prefill's tail, so fixing one did not fix it.
2. The **fused pair head** applied a plain RMSNorm instead of the mixer
   collapse. That is the same defect as the decode head's, in a third place:
   *any* fused head that folds in an RMSNorm is wrong for this family.
3. The **PLE convolution window** advanced two rows per verify and never
   rolled back on rejection. The KV row and the pooled indexer blocks are both
   recomputed by whatever replaces the rejected row, so they self-heal; a
   rolling window does not, and stays desynchronized for the rest of the
   generation.

`TINYTITAN_MTP_TRACE=1` prints the four numbers per step (boundary, draft,
prediction after each row, accepted). Speculative decoding is meant to be
exact, so when its output differs from scalar the question is always which of
those is wrong, and the text cannot answer it.

## Evaluation builds: precision of the non-expert weights (2026-11-13)

Two builds sit beside the 4-bit install to measure what the non-expert
weights ("the rest": attention, GDN, hyper-connection, PLE projections,
indexer, shared expert, router, embedding, LM head) buy at higher precision.
Neither changes the installer default. Routed experts stay 4-bit and the
n-gram table stays fp16 in all three; the table is the same file, hardlinked.

| Build | Converter flag | Rest | Resident floor |
| --- | --- | --- | --- |
| `qwen3.8-flash-next_125B_A6B_4Bit` | (default) | 4-bit; embedding, head, router 8-bit | 3.72 GiB |
| `..._4Bit_8BitRest` | `--rest-8bit` | every quantised tensor 8-bit | ~6.1 GiB (computed) |
| `..._4Bit_BF16Rest` | `--rest-bf16` | bf16 | 9.72 GiB |

`tools/build_qwen38_8bitrest.sh` builds the 8-bit-rest install. By default it
needs no download: `tools/requantize_qwen38_rest.py` quantises BF16Rest's bf16
rest to 8 bits and hardlinks the experts, n-gram table and tokenizer from the
4-bit install, after checking that every bf16 tensor re-quantised at the 4-bit
install's width reproduces that install's bytes. `--download` instead runs
`prepare_qwen38.py --rest-8bit` from Qwen's checkpoint. With every attention-slot tensor at 8 bits the repacker derives an
8-bit attention slot, so the GDN a/b pair is at the slot's own width, which
`Model.validateRoleUniformity` accepts.

Decode, 200 tokens greedy, expert profile off, M3 24 GB
(`tools/ram-matrix.sh`; mutex / Swift-rename prompts, tok/s):

| `--ram` | 4-bit | BF16Rest |
| --- | --- | --- |
| 8 | 3.40 / 3.36 | fails (8 slots < top-10) |
| 10 | 3.86 / 3.57 | fails |
| 12 | 4.18 / 3.99 | 2.50 / 2.23 |
| 14 | 4.26 / 4.11 | 2.80 / 2.60 |
| 16 | 4.85 / 4.50 | 3.01 / 2.81 |
| 18 | 5.19 / 4.78 | 3.24 / 2.91 |
| default | 5.16 / 4.75 | 1.18 / 0.98 (oversubscribed) |

At equal slots and hit rate (64, 0.73) BF16Rest costs ~70 ms/token, which is
the extra ~8 GB of dense reads at the M3's bandwidth. The default budget does
not count the bf16 floor and swaps. The 8-bit-rest row and a quality
comparison (KL and top-1 agreement against BF16Rest) are open.
