# Unsloth Dynamic v3.1 on Qwen3.8-Flash-Next — what actually changed, and what TinyTitan should copy

Status: analysis, 2026-11-08. Sources: Unsloth's
[Qwen3.8-Flash-Next guide](https://unsloth.ai/docs/models/qwen3.8-next),
[Dynamic 3.1 methodology page](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/dynamic-3.1-ggufs),
the [GGUF repository](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), and
direct inspection of the published GGUF headers at the revisions before and
after the `Better UD 3.1 quants` commits (2026-11-04).

## Executive summary

Unsloth does **not** publish the v3.1 algorithm; their own docs answer "the
available release notes do not specify the exact v3.1 changes". The artifacts
do show what changed:

1. **v3.1 is a bit-allocation release, not a new model or kernel.** The total
   file footprint is essentially unchanged. At `UD-Q4_K_XL` the tensor type map
   and sampled tensor bytes are identical before and after v3.1; only sharding
   moved. At `UD-Q3_K_XL` the same byte budget is reallocated: attention and
   shared experts move from Q8_0 to Q6_K, and the freed bits buy per-layer mixed
   expert quants (IQ4/IQ3/IQ2/Q5/Q2).
2. **The invariant is a quality floor for small, always-on tensors.** Across
   every quant, including `UD-IQ1_S`: attention stays Q5_K/Q6_K/Q8_0, shared
   experts Q8_0, hyper-connection mixers Q8_0, the QSA indexer BF16, norms F32,
   embeddings Q8_0, and the PLE/ngram table IQ4_NL (4-bit minimum). Only the
   routed experts are compressed hard, and their precision varies by layer and
   by projection (down is often a rank above gate/up).
3. **Calibration did not change.** The imatrix file has the same SHA-256,
   926 entries and 45 chunks before and after v3.1. The improvement is in how
   the budget is allocated under that imatrix, not in new calibration data.
4. **The chat template and MTP files are not v3.1 changes.** Both revisions ship
   the same patched template and the same MTP heads. They are still two real
   quality/performance levers TinyTitan does not fully match.
5. **TinyTitan currently spends its bytes in the opposite place.** A 4-bit
   Qwen3.8 install is ~174 GB, of which ~102 GB is the fp16 PLE table and only
   ~72 GB is the model. The attention/GDN/shared-expert weights are 4-bit. A
   v3.1-style rebalance — 4-bit PLE table, 8-bit always-on weights, 4-bit
   routed experts — is ~100 GB and should score better, not worse.

## Evidence

### Type maps, before and after v3.1

Counts over all 1,224 tensors, parsed from the GGUF headers.

| tensor role | `UD-Q3_K_XL` before v3.1 | `UD-Q3_K_XL` after v3.1 |
| --- | --- | --- |
| attention qkv/gate/q/k/v/o | Q8_0 | Q6_K (one layer Q8_0) |
| shared expert gate/up/down | Q8_0 | Q6_K (one layer Q8_0) |
| expert gate/up | IQ3_XXS (47), IQ4_XS (1) | IQ3_XXS (38), IQ4_XS (6), IQ2_S (4) |
| expert down | IQ4_NL (43), Q8_0 (5) | IQ4_NL (26), Q5_0 (20), Q2_0 (2) |
| hyper mixers / inject | Q8_0 / F32 | Q8_0 / F32 |
| indexer | BF16 | BF16 |
| PLE table | IQ4_NL | IQ4_NL |

At `UD-Q4_K_XL` both revisions are the same map: 503 x Q8_0, 94 x Q4_K,
43 x Q5_1, 2 x Q5_K, 1 x IQ4_NL (the PLE table), plus F32/BF16 small tensors.
A 64 KiB sample of `blk.0.attn_qkv`, `blk.0.ffn_gate_exps` and
`blk.11.ffn_down_exps` hashes identically before and after. Q4 did not need the
new allocator; Q3 and below did.

### What the allocation policy appears to optimize

The per-layer expert tables are not monotonic or uniform:

- `UD-Q3_K_XL` gate/up: IQ2_S on layers 0, 8-10; IQ4_XS on 15-18, 32, 47;
  IQ3_XXS everywhere else.
- `UD-Q3_K_XL` down: Q5_0 on 20 layers, IQ4_NL on 26, Q2_0 on layers 8 and 10.
- `UD-Q2_K_XL` mixes IQ2_XXS/IQ2_S/IQ3_XXS per layer, with down at IQ4_NL and
  Q2_0.
- `UD-IQ1_S` pins gate/up at IQ2_XXS/IQ2_S and down at Q2_0, but still keeps
  attention at Q5_K/Q6_K and shared experts at Q8_0.

That is the signature of a **budgeted search over per-tensor precision**, not a
fixed recipe: a floor on the tensors whose error breaks routing/attention, and
the remaining bytes spent where the measured loss per byte is lowest. The same
imatrix is the error model for the search.

### Runtime consequence

Total bytes are constant, so the speed effect comes from *which* bytes are read
per token. In `UD-Q3_K_XL` the always-on attention/shared path drops Q8_0 to
Q6_K (25% fewer hot bytes) while the streamed expert path is re-tuned. The
7% faster claim is consistent with that plus fewer generated tokens; it is not
a new kernel. At Q4 the bytes are unchanged, so a Q4 user should not expect a
v3.1 speedup from the weights alone.

## Template fixes (separate from v3.1, still worth copying)

Unsloth's GGUF chat template differs from the official Qwen template in ways
that matter to an agent harness. The template is identical in both revisions,
so this is an Unsloth packaging fix, not v3.1:

- **Merges all leading `system`/`developer` messages into one system block.**
  The official template renders only `messages[0]` and raises if a later system
  message appears. A harness that sends global + project instructions as two
  system messages either loses one or errors.
- **Maps `reasoning_effort: high` to `xhigh`.** The official template raises on
  `high`; clients commonly send low/medium/high.
- **Preserves `developer` role and validates tool-call arguments.** It rejects a
  function call with no name and an arguments value that is a JSON string
  instead of an object, rather than silently rendering no parameters.
- **`preserve_thinking` defaults to true**, as the official template does.

TinyTitan's manual ChatML renderer currently throws on a second system message
(`Tokenizer+ChatMLTemplate.swift:67`) and strips all historical reasoning
(`:82`). Those are direct behavior differences from the reference prompt.

## TinyTitan's current position

| lever | TinyTitan today | Unsloth v3.1 / Qwen reference |
| --- | --- | --- |
| 4-bit attention/GDN/shared | 4-bit | Q6_K–Q8_0 floor |
| small sensitive tensors | 4-bit unless building 8-bit | BF16/Q8 floor |
| routed experts | uniform 4-bit | per-layer mixed, down often higher |
| embedding / lm_head | 8-bit | Q8_0 |
| PLE table | fp16, 102.4 GB | IQ4_NL, ~28.8 GB |
| quantizer | min/max affine group 64, no imatrix | imatrix-weighted PTQ |
| group size | 64 (fixed in kernels) | K-quant superblocks |
| quality metric | per-tensor round-trip / top-k agreement | KLD, top-1, agentic pass rate |
| template | one system message, strips thinking | merged system, high→xhigh, preserve |
| MTP | opt-in, pure-greedy, measured −43% | default, 1.3–1.7x claim |

Two size facts drive the plan:

- `quantize_ngram_table.py` writes 4-bit group-32 rows at 100 B, 32.0 GB for the
  production table; a group-64 variant is 90 B, 28.8 GB. Either replaces
  102.4 GB.
- The non-table part of the current 4-bit install is ~72 GB. Promoting every
  always-on projection from 4 to 8 bits costs roughly 1.8–2 GB. A 4-bit table
  plus 8-bit always-on weights is therefore ~100–103 GB, comfortably under the
  old 174 GB and inside a 128 GB machine; with group-64 and no other change it
  approaches 96 GB.

## Would "bf16 everywhere except 4-bit experts" beat v3.1?

This is the natural TinyTitan-shaped alternative: leave every non-expert tensor
and the PLE table at the checkpoint's 16-bit dtype, and quantize only the routed
experts, which are ~121B of the 125B backbone parameters. TinyTitan's 16-bit
path is **bf16**, the checkpoint dtype; "fp16" here means unquantized 16-bit,
and converting bf16 weights to fp16 is not a quality improvement. It is feasible — the
bf16-dense Ornith build is the precedent — but the quality claim depends on
which Unsloth quant it is compared with.

Size first (decimal GB, TinyTitan's affine 4-bit group-64 is 4.5 bpw including
scale/bias):

| component | parameters | 16-bit | 4-bit |
| --- | ---: | ---: | ---: |
| routed experts | 120.8B | 241.6 GB | 68.0 GB |
| non-expert backbone | 4.2B | 8.4 GB | 2.4 GB |
| PLE table | 51.2B values | 102.4 GB | 28.8 GB (g64) |
| **total, bf16 + 4-bit experts** | | | **178.8 GB** |
| **total, all 4-bit** | | | **~99–103 GB** |

Unsloth's published sizes for comparison: UD-Q4_K_XL 111.3 GB (the current
file sum is ~122 GB), UD-Q5_K_XL 158.3 GB, UD-Q6_K_XL 169.2 GB, Q8_0 188.2 GB.
The proposed build is therefore at or above **Q6_K_XL**'s footprint. At that
size the fair comparison is Q6, not Q4.

The error budget explains why size is not the whole story:

- Experts are **96.6%** of the backbone parameters. Their quantization error
  dominates the logit error; non-expert and table precision are second-order.
- Unsloth Q4 does not use uniform 4-bit experts. Gate/up are Q4_K, down is
  Q5_1 on 43 layers and Q8_0 on five sensitive layers, all fitted with a
  45-chunk imatrix. That is roughly 4.7–5.0 effective bits on the expert path,
  with the largest errors weighted down.
- TinyTitan's uniform 4-bit affine min/max experts are 4.5 bpw, with no
  imatrix and no per-layer down elevation. Per bit, that is weaker than
  Unsloth's Q4 expert path, not stronger.
- The bf16 non-expert gain is real but small: Unsloth already keeps those
  tensors at Q8/Q6, whose error is already ~1/16 of a 4-bit step. Perfecting
  3.4% of the parameters does not recover the error in the other 96.6%.
- The fp16 PLE table is genuinely better than IQ4_NL, but Unsloth keeps the
  table at 4 bits in every quant and still reaches 94.1% top-1 at Q8, so the
  table is not the dominant error term.

**Verdict.** Against **UD-Q4_K_XL**, the proposal is plausible and worth
measuring: it has a 56–68 GB larger budget and removes all non-expert error, so
it could land at or slightly above Q4 on some benchmarks. Against **Q5/Q6** at the
same ~160–180 GB footprint it should lose, because those quants spend the bytes
on the expert path where the error actually is, while this build spends them on
a table and on non-experts that are already nearly lossless. It is not "on par
or higher than the Unsloth 3.1 models" as a class.

The way to make the claim true is not to leave everything at 16 bits; it is to
copy the allocation: imatrix-weighted 4-bit experts, down projections one rank
higher, sensitive layers elevated, and a 4-bit table to free the budget. That
is the P1 recipe above. The all-16-bit-except-experts build is a useful
**quality upper bound** for the expert path, and the right first experiment to
separate expert error from table/non-expert error.

## Implementation plan

### P0 — measure first

Add a model-level quality harness before changing the recipe. Per-tensor
round-trip fidelity cannot see the errors that matter.

1. `TinyTitanBench kld` (or a `tools/quant_quality.py` driver around
   `RealForwardRunner.produce`): load a BF16 teacher and a candidate install,
   run the same 128–512-token held-out prompts, and report mean KLD, 99.9% KLD,
   top-1 agreement, and top-1 agreement at 32 greedy tokens (the
   Divergence@32 metric Unsloth uses).
2. A token-use metric: tokens to completion on a fixed agentic prompt set, not
   just tok/s. Unsloth's 50% claim is a token-count claim.
3. Use agentic/chat/multilingual calibration and evaluation text. Unsloth's own
   docs warn that calibrating and evaluating on Wikipedia overstates quality.

### P0 — cheap, high-confidence changes

4. **Make the 4-bit PLE table the default**, and add a group-64 row format
   (90 B/row, 28.8 GB) to `NgramTableReader`/`quantize_ngram_table.py`.
   This is a 70–74 GB footprint reduction for a table read 16 rows at a time;
   Unsloth keeps the table at 4 bits in every quant, so the quality direction is
   supported even though the formats differ.
   Files: `tools/quantize_ngram_table.py`, `tools/prepare_qwen38.py` (`:175`),
   `tools/install_models.sh`, `Infrastructure/Streaming/NgramTableReader.swift`.
5. **Promote the small sensitive tensors at 4-bit, not only at 8-bit.**
   `PROMOTE_TO_BF16_AT_8BIT` (`prepare_qwen38.py:328`) already names the
   measured set: router, shared-expert scalar gate, hyper-connection inject,
   GDN a/b, PLE key, QSA indexer q/k. At 4-bit the indexer currently agrees on
   the same keys 0.0% of the time against 49.5% at 8 bits, per the converter's
   own measurement. This costs ~100 MB and is the best quality-per-byte change
   available. Extend `quant_bits` to apply the list at both widths; the runtime
   already supports per-tensor overrides.
6. **Template parity.** Merge leading system/developer messages into one block;
   accept `high` as an alias for `xhigh`; make historical thinking preservation
   configurable and default it to the template's true; validate tool-call
   arguments at the API boundary. Files:
   `Tokenization/Tokenizer+ChatMLTemplate.swift`,
   `Runtime/Family/ReasoningControl.swift`,
   `Lib/OpenAIRequestValidator.swift`; update the reasoning/template tests.

### P1 — the actual v3.1 idea

7. **Imatrix-weighted affine quantization.** Replace min/max group fitting in
   `quantize_affine` with a weighted least-squares fit of `(scale, bias)` per
   group, using per-input-channel importance `in_sum2 / counts`. Two or three
   fixed-point iterations are enough: initialize from min/max, quantize, solve
   the 2x2 weighted normal equations for scale/bias, repeat. This changes no
   runtime format: same packed nibbles, same bf16 scale/bias, same kernels.
   Generate the imatrix from TinyTitan forward passes over the calibration set,
   or map Unsloth's public `imatrix_unsloth.gguf` tensor names onto the TinyTitan
   source names (it is 580 MB, 926 entries, 45 chunks, and unchanged across
   v3.1). Prefer generating our own for licensing and name fidelity.
8. **A per-tensor allocation recipe file.** Extend `quant_bits(name, width)` to
   accept a layer index and a recipe of `{role, layer ranges, bits}`. Start from
   the Unsloth pattern — always-on floor, expert down one rank above gate/up,
   early/late layers elevated — then let the KLD harness prune. Keep the recipe
   in the converter, not in the runtime; the manifest already records the
   result as per-tensor overrides.
9. **Ship a mixed 4/8 build**: 4-bit routed experts, 8-bit attention/GDN/shared
   expert/embedding/head, 4-bit table. This is the closest TinyTitan can get to
   `UD-Q4_K_XL` with its 4/8-only kernels, and is ~100 GB. Measure decode cost:
   doubling the hot attention bytes may cost more than the quality is worth on
   a 24 GB machine, so keep the current all-4 build as the speed row.
10. **Tune group size only if the kernels can follow.** Group 64 is fixed in the
    Metal/C kernels today; a 32-group scale/bias grid would improve quality at
    ~+0.5 bit/value. Do not promise it without kernel work.

### P2 — performance items Unsloth has and TinyTitan does not

11. **MTP is not a drop-in win.** TinyTitan's measured MTP result is −43.2% at
    85.7% acceptance (`docs/qwen38-mtp-and-overlap-verdict.md:51`), and it only
    engages for pure-greedy requests (`RawCompletion.isPureGreedy`). Unsloth's
    1.3–1.7x is on a decode-native draft path with sampling. Fix the verify
    path and add sampling-correct acceptance before enabling MTP by default;
    do not auto-attach the sidecar merely because Unsloth does.
12. **Hot/cold byte-aware allocation.** The general form of v3.1: maximize
    quality subject to a footprint cap, where each tensor's cost includes
    whether it is resident and read every token, or streamed per top-k expert.
    This explains both the Q3 reallocation and the Q4 no-op. Implement it as an
    offline search over the recipe, using the KLD harness as the objective.
13. **PLE gather / mmap.** TinyTitan already preads rows and prefetches off the
    decode path. Re-measure with the 4-bit and group-64 tables; the table is
    3.2x smaller, so the latency/bandwidth balance changes.

## Acceptance matrix

Build these five installs from the same checkpoint and run the same suite:

| arm | table | attention/GDN/shared | experts | quantizer |
| --- | --- | --- | --- | --- |
| A (today) | fp16 | 4-bit | 4-bit | min/max |
| B | q4 g32 | 4-bit | 4-bit | min/max |
| C | q4 g64 | 8-bit | 4-bit | min/max |
| D | q4 g64 | 8-bit | 4-bit | imatrix WLS |
| E | q4 g64 | 8-bit | per-layer recipe | imatrix WLS |

Accept E only if it beats A on mean KLD and token-to-completion at a footprint
below 105 GB, without losing more than ~5% decode tok/s on the reference
machine. If C already matches the quality target, stop before D/E.

## Caveats

- Unsloth's v3.1 mechanics are unpublished; the allocation story is inferred
  from headers, byte samples and the unchanged imatrix. Treat "budgeted search"
  as the best-supported model, not a claim about their code.
- The `7% faster` and `50% fewer tokens` figures are Unsloth's, on their
  hardware and harness. Neither is a TinyTitan measurement.
- GGUF Q4_K/Q5_1/IQ types are not portable to TinyTitan's affine format. Copy
  the *allocation policy*, not the block formats.
- 6-bit was deliberately withdrawn from TinyTitan because its packing and
  bandwidth measured worse; this plan stays inside 4/8/bf16.
- A per-tensor width must still satisfy `validateRoleUniformity` and the fused
  kernel policy; q/o, k/v and the three GDN projections must move together.
