# GLM-5.3-Flash port — evaluation record

**Status: evaluated and closed on 2026-11-12. Nothing here is implemented, and
nothing is supported.** The owner decided not to proceed after the decode
projection in §5: about 1 tok/s on the 24 GB M3, against 5.5 tok/s for
Qwen3.8-Flash-Next on the same machine. This document keeps the facts and the
plan so the question does not have to be researched again; §7 names what would
have to change for it to be worth reopening.

Requested: plan how to run `zai-org/GLM-5.3-Flash` through the normal converter
path (bf16 source, affine group-64 snapshot, `TinyTitanRepack`, `.ssdai`), with
the routed experts at 4-bit and everything else at 8-bit.

The conclusion, stated up front: **this is a runtime port, not a wiring job.**
The converter path fits the checkpoint, but four of its subsystems have no
kernel in this tree, and the model's bytes per token put it at the SSD's wall
on a 24 GB machine.

Everything in §1 was read on 2026-11-11 from the checkpoint's `config.json`,
`generation_config.json`, `chat_template.jinja` and `model.safetensors.index.json`
through the Hugging Face API. Sizes and speeds in §2 and §5 are calculated, not
measured, and are labelled so. No shard was downloaded. The repository was read
at `e83df77`.

## 1. The checkpoint

| | Value | Read from |
| --- | --- | --- |
| Conversion source | `zai-org/GLM-5.3-Flash-BF16`, 120 shards, `total_size` 642,646,653,816 bytes | index + tree |
| Serving release | `zai-org/GLM-5.3-Flash`, 62 shards, routed experts in FP8 (e4m3), the rest BF16 | `quantization_config` + tree |
| Pinned sha | **not recorded** — pin it before any conversion | — |
| `gated` / license | no / MIT | HF API, card |
| `model_type` | `glm5_next` (`Glm5NextForConditionalGeneration`), text config `glm5_next_text` | `config.json` |
| Declared parameters | 321.3B (card: 320B total, 18B active) | HF API |
| Context | 1,048,576 | `config.json` |
| `tie_word_embeddings` | `false` — `lm_head.weight` is a real tensor at the archive root | `config.json` + index |
| Chat template | present (10.8 KB) | tree |
| Sampling | temperature 1.0, top-p 0.95 | `generation_config.json` |
| EOS ids | 154820, 154827, 154829 (pad 154820) | `generation_config.json` |

### Text geometry

| Field | Value |
| --- | --- |
| `num_hidden_layers` / `hidden_size` / `vocab_size` | 45 / 4096 / 154,880 |
| `layer_types` | (3 × `linear_attention` → 1 × `deepseek_sparse_attention`) × 11, then one more linear layer: 34 linear, 11 sparse |
| Linear attention (KDA) | 64 heads, head dim 128, short conv kernel 4, `gate_lower_bound` −5.0 |
| Sparse attention (MLA) | 64 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `qk_rope_head_dim` **0**, `v_head_dim` 256 |
| Indexer (DSA) | 32 heads × 128, `index_topk` 2048, `index_kpool` 4 with a learned compress gate and APE, tail always selected, interleaved RoPE on the indexer only |
| `mlp_layer_types` | layers 0–2 `dense` (`intermediate_size` 12,288), layers 3–44 `sparse` |
| MoE | 288 routed experts, top-8, `moe_intermediate_size` 2048, 1 shared expert |
| Router | `scoring_func` sigmoid, `topk_method` `noaux_tc` with `e_score_correction_bias`, `norm_topk_prob` true, `routed_scaling_factor` 2.5, router dtype float32 |
| Hyper-connections | `mhc` true, `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 |
| Activation / norm | silu with `swiglu_limit` 10.0, `rms_norm_eps` 1e-5 |
| Draft head | `num_nextn_predict_layers` 1, with its own 288 experts |
| Vision | 24-layer tower, hidden 1024, projecting to 4096 — excluded from a text-only port |

### Tensor layout that decides the converter

- Text tensors sit under `model.language_model.*`; the head is `lm_head.weight`.
- **Routed experts are one tensor triple per expert**:
  `layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`, 12,096 experts
  over the 42 sparse layers. That is the Laguna shape, not the stacked Qwen3.8
  shape.
- Dense layers carry `mlp.{gate_proj,up_proj,down_proj}.weight`.
- A KDA layer carries `q_proj`, `k_proj`, `v_proj`, `o_proj`, `b_proj`,
  `f_a_proj`/`f_b_proj`, `g_a_proj`/`g_b_proj`, `q/k/v_conv1d`, `A_log`,
  `dt_bias` and `o_norm`. The FP8 release additionally lists serving fusions
  (`fused_qkvbfg_a_proj`, `qkv_proj`) that the BF16 index does not need.
- Each layer carries `hc_attn_{base,fn,scale}` and `hc_ffn_{base,fn,scale}`.

### Prompt side

The template is GLM's own, not ChatML: it opens with `[gMASK]<sop>`, uses
`<|system|>`-style role markers, and emits tool calls as
`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`.
`reasoning_effort` takes `low`, `high` or `max` and defaults to `max`;
`clear_thinking` defaults to false, and the card asks chat use to pass true.
None of the tool-call parsers in this tree reads that dialect.

## 2. Size under the requested split (calculated)

At 4.5 and 8.5 effective bits per weight, which is what the Laguna installs
work out to (66.5 GB and 125.0 GB for 118B parameters).

| Part | Parameters | Width | Size |
| --- | ---: | --- | ---: |
| Routed experts (42 × 288 × 3 × 4096 × 2048) | 304.4B | 4-bit | ≈171 GB |
| Everything else, text only | ≈8.9B | 8-bit | ≈9.5 GB |
| **Install** | | | **≈181 GB** |
| Draft-head sidecar, optional | ≈7.3B | same split | ≈4.5 GB |

One expert is 25.17M parameters, ≈14.2 MB at 4-bit. The 304.4B / 8.9B split is
cross-checked against two community GGUF cards and against the geometry above.

Disk: the source is 642.6 GB streamed. With the converter's fetchers capped at
four and `TinyTitanRepack --consume-snapshot`, as the `ornith122` rows do, the
peak is ≈210 GB; without them it is over 360 GB.

Source choice, had the port gone ahead: the BF16 repository, because quantizing
once from the original is this project's rule. The FP8 release is about half
the download but its experts would be quantized twice.

## 3. What the engine is missing

1. **KDA linear attention.** `gdn.metal` is the nearest relative; KDA adds
   low-rank forget and output gates, a decay term and 64 heads.
2. **MLA attention with no positional encoding.** Nothing like it is in the tree.
3. **The DSA indexer.** `qsa.metal` is a starting point; key pooling with a
   learned compress gate is new.
4. **mHC.** It is the Sinkhorn form, not Qwen3.8's gated residual.
   `deepseek-v41-flash-port.md` describes the algorithm; it was never built.
5. **The MoE tile.** `kMoEXMaxD` and `Model.maximumThreadgroupTileWidth` are
   3072 and would have to be 4096, with the load guard. The GDN tile is already
   4096.
6. **Serving for the sigmoid router and dense leading layers.** Both exist only
   in the Laguna decode probe, which has no serving runner.
7. **The prompt side** of §1: template, tool-call parser, reasoning-effort
   mapping, three stop tokens.

## 4. The plan that was not started

Each phase ends at a gate; nothing would have been called supported before
phase 6.

1. **Design record, no downloads.** Pin the sha, read the `glm5_next` modeling
   source in transformers, confirm the tokenizer type and that every last
   dimension is group-64 aligned, and confirm the runtime accepts 8-bit
   resident tensors beside 4-bit experts.
2. **Reference and surrogate.** A layer-streaming numpy forward
   (`tools/glm53_reference.py`) and a tiny synthetic checkpoint with the real
   tensor names, so parity and converter tests run without the source.
3. **Converter.** `tools/prepare_glm53.py`, modelled on
   `prepare_laguna_s21.py`: experts 4-bit; other matrices 8-bit; router gate,
   mHC tensors, norms and 1-D tensors at source precision; vision and the draft
   head skipped; `--plan`, and a python suite beside the existing converter
   suites.
4. **Repacker and format.** A `glm53flash` family, config reader, geometry
   cross-check, tensor schema and classification tests.
5. **Runtime.** The kernels of §3 in that order, each with a validation
   reference and parity against the phase-2 forward; then the serving runner on
   the library's single generation path, prefill on the GPU only.
6. **Real conversion and verification**, on the operator's explicit request:
   the bar in `adding-a-model.md` §4.
7. **Wiring and documentation**: the eight wiring points, README, wiki, tracker.
8. **Later, each measured on its own**: draft-head sidecar, ANE prefill
   sidecar, decode tuning.

## 5. Decode projection on the M3 24 GB (calculated)

Calibrated against Qwen3.8-Flash-Next 4-bit on the same machine, from
`qwen38-decode-12gb-measurements.md`:

| Quantity | Measured on Qwen3.8 4-bit |
| --- | --- |
| Decode | 4.6–5.7 tok/s (175–217 ms per token) |
| GPU busy per token | 97–113 ms for ≈3.4 GB of active weights, ≈32 GB/s effective |
| Expert-read ceiling | 3.92 GB/s, four-thread reader, at its wall |
| Hit rate at 96 slots per layer (9.6 × top-k) | 0.85–0.87 |
| Hit rate at 24 slots per layer (2.4 × top-k) | ≈0.53, back-calculated from the 3.4 tok/s row |

Applied to GLM-5.3-Flash, per token:

| | Uniform 4-bit | Experts 4-bit, rest 8-bit |
| --- | ---: | ---: |
| Routed expert bytes (336 × 14.2 MB) | 4.76 GB | 4.76 GB |
| Resident bytes touched (≈8.5B parameters) | 4.8 GB | 9.0 GB |
| GPU compute at 32–46 GB/s | 210–305 ms | 300–430 ms |
| Resident RAM | ≈5 GB | ≈9.5 GB |
| Expert cache that fits | ≈10.5 GiB, 19 slots per layer | ≈6 GiB, 11 slots per layer |
| Assumed hit rate | ≈0.50 | ≈0.35 |
| Expert bytes read per token | 2.4 GB | 3.1 GB |
| SSD time at 3.9 GB/s | 610 ms | 790 ms |
| Token time, reads fully hidden to not hidden | 610–960 ms | 790–1,265 ms |
| **Decode** | **1.0–1.6 tok/s** | **0.8–1.2 tok/s** |

What drives it:

- **The SSD is the wall.** A token needs 2.4–3.1 GB from disk, against
  ≈0.17 GB for Qwen3.8, and the cache holds 4–7% of the experts because each
  one is five times larger.
- **Compute alone caps it under 5 tok/s.** With every expert in RAM, 17.3B
  active parameters give ≈3.3–4.8 tok/s at uniform 4-bit on this GPU.
- **The 8-bit remainder costs twice**: ≈100 ms of compute, and 4.5 GB taken
  from the expert cache.

Assumptions, weakest first: the hit rate carries Qwen3.8's slots-to-top-k curve
over to a model whose expert skew is unknown (at 0.2 the mixed build falls to
≈0.7–1.0 tok/s); the new kernels reach the existing ones' effective GB/s; the
process budget is 16–17 GiB; draft-head speculation is not counted. As a sanity
check, Laguna-S-2.1 (8B active, 5.3 MB experts) measures 1.5 tok/s at 4-bit in
its decode probe, and GLM has twice the active weights and 2.7× larger experts.

## 6. Decision

Closed by the owner on 2026-11-12: no GLM-5.3-Flash work proceeds. Several
weeks of kernel work for ≈1 tok/s does not serve this project's target, which
is decode speed on low-RAM Macs.

## 7. What would reopen it

- A machine whose RAM holds a useful fraction of the 171 GB of experts — the
  projection's compute-only ceiling is the number to beat there.
- A smaller checkpoint of the same architecture: the kernels of §3 would then
  be the whole job, and this record is the starting point.
- A measured expert-usage trace showing a far higher hit rate than §5 assumes.

*(The wiki tracker archive entry was pushed on 2026-11-13 as **TT-062** — the
draft's TT-061 had been taken by the wide-verify task before this section could
be cleared.)*
