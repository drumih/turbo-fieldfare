# Kimi K3 — Exact Decode Dataflow (engine contract)

Normative sources: `modeling_kimi_linear.py` and `config.json` from
[huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(revision pinned by the repacker). This file condenses them into the exact
per-token recipe the Metal/Swift engine must reproduce. Where the HF code is
ambiguous for single-token decode, `kimi-k3-in-c` (verified against the `fla`
library) is the tie-breaker.

All layer indices below are **0-based** (the HF config lists are 1-based;
subtract 1). 93 layers total.

## Layer schedule

- MLA (full attention) 0-based layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39,
  43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, **92** — the last two
  layers are both MLA (1-based list ends …, 92, 93).
- KDA layers: the other 69 (0-based 0,1,2,4,5,6,…,90).
- Layer 0 is KDA attention **and** the only dense-MLP layer
  (`first_k_dense_replace = 1`, intermediate 33792).
- AttnRes block boundaries at 0-based layers 0, 12, 24, 36, 48, 60, 72, 84
  (`layer_idx % 12 == 0`).

## Primitives

- `KimiRMSNorm(x, w, eps=1e-5)`: fp32; `x * rsqrt(mean(x²) + eps) * w`.
- SiTU-GLU (`SituAndMul`, β1=4 gate, β2=25 up): input is `cat(gate, up)`;
  `out = (β1·tanh(g/β1)·sigmoid(g)) ⊙ (β2·tanh(u/β2))`, computed in fp32.
  Note sigmoid reads the **uncapped** gate.
- AttnRes (blocks B (m vectors), prefix p, proj w∈R^H, norm n):
  `v = [B; p]`; `k = v_fp32 ⊙ rsqrt(mean(v_fp32², -1) + eps)` (no norm weight
  yet); `score_i = Σ (k_i ⊙ (n.weight ⊙ w))`; `probs = softmax(scores)`;
  `out = probs · v_fp32` cast back. Output **replaces** the stream (no extra
  residual add).

## Per-layer flow (layer i, stream x, block list B, prefix carried in x)

1. Preserve `incoming = x`. If B non-empty, set
   `x = AttnRes(B, incoming, self_attention_res_proj, self_attention_res_norm)`.
2. If i % 12 == 0: `B.append(incoming); prefix = None` else `prefix = incoming`.
3. `h = input_layernorm(x)` → attention (KDA or MLA, below) → `attn_out`.
4. `prefix = (prefix == None) ? attn_out : prefix + attn_out`.
5. `h = AttnRes(B, prefix, mlp_res_proj, mlp_res_norm)` (uses post-append B).
6. `h = post_attention_layernorm(h)` → MoE/dense (below) → `mlp_out`.
7. `prefix = (prefix == None) ? mlp_out : prefix + mlp_out`; stream = prefix.

Model head: after layer 92, `x = AttnRes(B, x, output_attn_res_proj, output_attn_res_norm)`
(final blocks list has 8 entries + prefix = 9 vectors), then `norm(x)`,
then lm_head. Embedding lookup at the start (no scale factor).

## KDA layer (69 layers; 96 heads, d=128, conv width 4)

1. Projections: `q,k,v = W_q x, W_k x, W_v x` (7168→12288 each, no bias).
2. Depthwise causal conv1d width 4 + SiLU on q, k, v (conv state: last 3
   inputs/channel; weights (and bias — verify presence against the checkpoint
   index; fla `ShortConvolution` defaults to bias=True)).
3. `z = W_fb (W_fa x) + dt_bias` (7168→128→12288); per head h:
   `A = exp(A_log[h])`; per channel: `g = −5 · sigmoid(A · z)`;
   `α = exp(g)` ∈ (e⁻⁵, 1). `β_h = sigmoid(W_b x)` (per-head scalar, fp32).
4. L2-normalize q, k per head (in-kernel `use_qk_l2norm_in_kernel`).
5. Scale q by `128^(−1/2)` (fla default `scale = head_k_dim ** -0.5`).
6. Delta rule per head (state S: 128×128 fp32):
   `S ← Diag(α) S + β · k (v − (Diag(α) S)ᵀ k)ᵀ` ; `o = Sᵀ q`.
   (Equivalent to `S ← (I − β k kᵀ) Diag(α) S + β k vᵀ`.)
7. Output per head: `FusedRMSNormGated` — `o_h = RMSNorm(o_h, o_norm.weight) ⊙ sigmoid(gate_h)`
   where `gate = W_g x` (7168→12288, full-rank gate), then flatten 96×128,
   `o_proj` (12288→7168).

## MLA layer (24 layers; 96 heads, NoPE)

1. `q_lat = RMSNorm(W_qa x)` (7168→1536, `q_a_layernorm`); `q = W_qb q_lat`
   (1536→96×192); per head split `[128 nope | 64 rope-part]` (never rotated).
2. `kv = W_kva x` (7168→576) → `[512 latent | 64 rope-part]`; cache
   **`[kv_a_layernorm(latent512), rope64]`** per token (fp16; 576 values).
3. Per head, absorbed decode: `q̃_h = W_kvb_k,h^T q_nope_h` (512-dim;
   `W_kvb` 512→96×256 split `[128 k_nope | 128 v]` per head);
   `score_h(s) = (q̃_h · latents + q_rope_h · ropes) × 192^(−1/2)`;
   softmax fp32; `o_h = W_kvb_v,h (Σ_s p_s · latents)` (128-dim).
   (Exact: kv_b is linear, so absorbing into q is identical to expanding k/v.)
4. Flatten 96×128, multiply by `sigmoid(W_g x)` (output gate, 7168→12288),
   `o_proj` (12288→7168).

## MoE block (92 layers; layer 0 uses dense MLP instead)

1. Router (fp32): `logits = W_gate x` (896×7168); `s = sigmoid(logits)`;
   select top-16 by `s + e_score_correction_bias`; weights =
   `gather(s, idx)` renormalized by `sum + 1e-20`; `routed_scaling_factor = 1`.
2. `x_lat = W_down x` (7168→3584).
3. Per selected expert: SiTU-GLU on `[w1 x_lat | w3 x_lat]` (3584→3072 each),
   then `w2` (3072→3584). Experts are MXFP4 on disk (E2M1 nibbles, low nibble =
   even element; E8M0 scale byte per 32 weights; scale byte 255 ⇒ group = 0).
4. `y_lat = Σ_k w_k · expert_k(x_lat)` (fp32 accumulate).
5. `y = W_up RMSNorm(y_lat, routed_expert_norm)` (3584→7168, eps 1e-5).
6. `y += shared_experts(x)`: KimiMLP, SiTU-GLU, intermediate 6144, full width
   7168 (trunk-precision weights).

## Checkpoint tensor names (language_model.model.*)

`embed_tokens.weight`, `norm.weight`, `output_attn_res_{proj,norm}.weight`,
`language_model.lm_head.weight`; per layer N (0-based):
`layers.N.{input_layernorm,post_attention_layernorm}.weight`,
`layers.N.{self_attention_res,mlp_res}_{proj,norm}.weight`;
KDA: `layers.N.self_attn.{q,k,v,g}_proj.weight`,
`layers.N.self_attn.{q,k,v}_conv1d.{weight,bias}`,
`layers.N.self_attn.A_log`, `layers.N.self_attn.{f_a_proj,f_b_proj}.weight`,
`layers.N.self_attn.{dt_bias,b_proj.weight,o_norm.weight,o_proj.weight}`;
MLA: `layers.N.self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,g_proj,o_proj}.weight`;
MoE: `layers.N.block_sparse_moe.gate.{weight,e_score_correction_bias}`,
`layers.N.block_sparse_moe.experts.E.{w1,w2,w3}.weight_{packed,scale}`,
`layers.N.block_sparse_moe.routed_expert_{down,up}_proj.weight`,
`layers.N.block_sparse_moe.routed_expert_norm.weight`,
`layers.N.block_sparse_moe.shared_experts.{gate,up,down}_proj.weight`;
dense layer 0: `layers.N.mlp.{gate,up,down}_proj.weight`.
Vision tower / `mm_projector` are excluded from the text bundle.
