# Kimi K3 on TurboFieldfare — Evaluation and Design Basis

Status: design basis for the K3 port. Audience: anyone reviewing why the K3
engine is built the way it is. Every external claim carries its source.

Target host: MacBook Pro, Apple M5 Max, 128 GB unified memory, macOS 26.6,
Swift 6.3. Budget: ~64 GB for model residency, the remainder for context,
system, and the OS page cache.

## 1. The model we are porting

Kimi K3 (Moonshot AI, weights released 2026-07-27) — verified against the
official `config.json`, `modeling_kimi_linear.py`, and the tech report:

- 2.78T total parameters, ~104.2B active per token. 93 layers.
- **69 KDA layers** (Kimi Delta Attention): gated delta-rule linear attention,
  `S_t = (I − βk kᵀ) Diag(α) S_{t−1} + βk vᵀ`, out `Sᵀq`. 96 heads × 128,
  causal depthwise conv width 4 + SiLU on q/k/v, L2-normed q/k, low-rank decay
  `g = −5·sigmoid(e^A·z)` (α ∈ (e⁻⁵, 1)), per-head scalar β = sigmoid(Wβx),
  head RMSNorm then **full-rank** sigmoid output gate. Fixed recurrent state
  (96×128×128 per layer, ~216 MB total in bf16, 432 MB in fp32) — O(1) in
  context length.
- **24 Gated MLA layers** (3 KDA : 1 MLA interleaved, plus terminal layer 93):
  DeepSeek-style MLA with `q_lora_rank 1536`, `kv_lora_rank 512`,
  `qk_nope 128`, `qk_rope 64` (cached but **never rotated — the model is fully
  NoPE**, no RoPE anywhere, no YaRN), `v_head 128`, 96 heads, output gate,
  scale = 192^(−1/2). KV cache = one 576-element latent per token
  (27.6 KB/token across 24 layers in fp16; ~3.6 GB at 128 K context).
- **AttnRes (Attention Residuals)**: depth-axis attention replacing plain
  residual adds. Layers are partitioned into blocks of 12 (`attn_res_block_size`),
  giving ≤9 block representations (8 blocks + embedding). Each layer applies
  AttnRes twice (pre-attention, pre-MoE): a learned pseudo-query scores
  RMSNorm-ed block vectors through softmax; output is the weighted sum plus a
  running prefix sum within the block.
- **Stable LatentMoE**: 896 routed experts, top-16, tokens are projected
  7168→3584 (`routed_expert_down_proj`) before dispatch, experts run at width
  3584 with `moe_intermediate_size 3072` (SiTU-GLU, β1=4 gate / β2=25 up,
  bounded |β1·β2| ≤ 100), the aggregate is RMSNorm-ed then projected
  3584→7168. Two shared experts (one MLP, intermediate 6144) added unweighted.
  Router: FP32 sigmoid scores, top-16 by `score + e_score_correction_bias`
  (Quantile-Balancing bias, frozen at inference), combining weights are the
  **unbiased** sigmoid scores renormalized to sum 1. Grouped top-k is a no-op
  (1 group).
- Layer 0 is a dense MLP (intermediate 33792, SiTU-GLU). All other 92 layers MoE.
- Vocab 163840, untied embedding and lm_head, RMSNorm eps 1e-5, tiktoken BPE
  tokenizer, XTML chat wire format, generation stop token 163586
  (`<|end_of_msg|>`), thinking channels always on.

Sources: [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3),
[config.json](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json),
[modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/raw/main/modeling_kimi_linear.py),
[kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3).

## 2. MLX vs GGUF on Metal — the requested evaluation

The claim "MLX is faster than GGUF on Metal" is **conditionally true, and
irrelevant at this scale**:

- MLX leads prefill/TTFT decisively on M5 (Neural-Accelerator GEMM path since
  MLX v0.30.0, [release notes](https://github.com/ml-explore/mlx/releases/tag/v0.30.0);
  Apple-measured TTFT 3.3–4.06× on M5 vs M4,
  [machinelearning.apple.com](https://machinelearning.apple.com/research/exploring-llms-mlx-m5))
  and usually leads small-model decode
  ([arXiv 2511.05502](https://arxiv.org/abs/2511.05502)).
- llama.cpp has caught up: it beats MLX by 10–24 % on large-MoE decode on an
  M5 Max 128 GB ([stared/benching-local-llms-on-apple-silicon](https://github.com/stared/benching-local-llms-on-apple-silicon/blob/main/README.md),
  Jun 2026) and its Metal backend gained Metal-4 tensor support
  ([PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634),
  [PR #20962](https://github.com/ggml-org/llama.cpp/pull/20962)). Once decode
  saturates memory bandwidth both runtimes converge
  ([ax-engine PERFORMANCE-RESULTS](https://github.com/defai-digital/ax-engine/blob/main/docs/PERFORMANCE-RESULTS.md)).
- **Neither runtime can load Kimi K3 on 128 GB.** Smallest artifacts as of
  2026-08-06: Unsloth GGUF UD-IQ1_S 594 GB; MLX REAP-pruned 2-bit 181 GB with
  documented degraded factual recall
  ([mlx-community/Kimi-K3-mlx-reap160-2bit](https://huggingface.co/mlx-community/Kimi-K3-mlx-reap160-2bit)).
  llama.cpp does not support K3 yet
  ([PR #26185](https://github.com/ggml-org/llama.cpp/pull/26185), open).
  A custom streaming engine is the only path — which is what this repo is.

**Format decision** (the part of the question that does matter here):

- Routed experts stay in their **native MXFP4** (E2M1 nibble + E8M0 exponent
  per 32 weights, 0.53125 bytes/weight). The 2.78T model was
  quantization-aware-trained in exactly this format; copying the bytes verbatim
  avoids a lossy re-quant of 1.447 TB and matches what the QAT optimized for.
  One expert is exactly 17,547,264 bytes — already a multiple of the 16 KB
  format alignment.
- The BF16 trunk (~113.5 GB: KDA/MLA projections, shared experts, latent
  projections, router, embeddings, lm_head, norms) is quantized during repack
  to the **MLX-style group-64 affine int4/int8** this repo already implements —
  one uniform block layout, one kernel family, trivially page-aligned for
  streaming, and directly eligible for the Metal 4 `mpp::tensor_ops` int4/int8
  path on macOS 26.
- GGUF k-quants were rejected: a dozen block formats with irregular per-tensor
  strides (bad for paged streaming), no NAX int4 path on macOS 26, and no
  measured quality edge that changes outcomes at ≥4 bpw.

## 3. Feasibility math (be honest about the envelope)

Per-token work at decode:

| Component | Bytes touched | Source |
|---|---|---|
| Routed experts (16 × 92 layers × 17,547,264 B) | **25.83 GB / token** | SSD |
| Resident trunk (int4 profile, ~30 GB working set) | ~30 GB / token | RAM @ ~577 GB/s → ~55 ms |
| lm_head int8 (163840 × 7168) | 1.2 GB / token | RAM → ~2 ms |
| KDA state + MLA KV + AttnRes | < 0.1 GB / token | RAM |

SSD is the wall. M5 Max NVMe sustained reads for cold expert data bound decode
to roughly **0.3–1 tok/s** (25.8 GB ÷ 7–17 GB/s depending on page-cache
warmth and read amplification), plus ~60–80 ms of RAM-side compute that
overlaps with the I/O. For calibration, the CPU-only reference
implementation reports 10.7 s/token on a 124-core EPYC with 3.2 GB/s NVMe at
the same 128 GB budget
([kimi-k3-in-c PERFORMANCE.md](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/main/docs/PERFORMANCE.md)).

**Positioning: this is a batch/research runtime, not an interactive chat
engine.** Prefill is compute-bound (GPU/NAX), decode is SSD-bound.

## 4. Memory budget (64 GB model residency)

| Resident item | Precision | Size |
|---|---|---|
| KDA projections (69 layers), MLA projections (24), shared experts, latent down/up, dense layer | int4 g64 | ~27 GB |
| Embedding + lm_head | int8 | ~2.4 GB |
| Router weights (selection accuracy is quality-critical) | FP32 | ~2.4 GB |
| Norms, AttnRes params, biases | BF16/FP32 | ~0.1 GB |
| **Resident total** | | **~32 GB** |
| KDA recurrent state (69 layers, fp32) | fixed | 0.43 GB |
| MLA latent KV (fp16) @ 128 K context | grows 27.6 KB/tok | ~3.6 GB |
| Expert slots (16 active + 16 prefetch) × 17.5 MB + scratch | | ~1.5 GB |

≈ 38 GB wired, inside the 64 GB budget; the rest of RAM is left to the OS
page cache, which absorbs expert reuse for free (see §5 — trusting it beat
every custom cache in flash-moe). A `--trunk-quant int8` repack profile
(~60 GB resident) is provided for quality A/B; int4 is the default until
real-weight A/B says otherwise, because post-hoc trunk quantization was
measured at ~17 % worst-row error by kimi-k3-in-c (the QAT covered experts
only).

## 5. Techniques adopted from the two reference projects

From [danveloper/flash-moe](https://github.com/danveloper/flash-moe) and its
M5-Max fork ([gorroai](https://github.com/gorroai/flash-moe),
[Anemll](https://github.com/Anemll/flash-moe) — "Beyond the DRAM Wall:
20.34 tok/s on M5 Max, 4.67× over baseline"), ordered by their measured wins:

1. **Temporal expert prediction / prefetch** (+31–55 %, their biggest single
   win): store this token's per-layer routing, prefetch next token's experts
   into a second buffer set while attention runs. Adopted as
   `K3ExpertStreaming`'s default policy, behind a runtime toggle — K3's
   Quantile Balancing flattens expert usage, so the hit rate must be measured
   on real weights before the default is final.
2. **Bounded pread fanout**: decode and chunked prefill share one scheduler;
   the M5 K3 profile issues one contiguous 17.5 MB read per expert and tunes
   its worker limit online among 1/2/4. Page-aligned split reads remain an
   explicit `--expert-io-splits 2|4|8` A/B control, and fixed workers remain
   available through `--expert-io-workers 1...32`.
3. **Cache policy is model-specific.** Gemma keeps buffered I/O, but Colibri's
   K3 cold-stream measurements favor direct reads. Production-sized K3 expert
   descriptors therefore resolve `--expert-io-cache auto` to Darwin
   `F_NOCACHE`; small fixtures stay buffered, and both `buffered` and
   `uncached` are exposed for real-model A/B. This does not require multiple
   disks and does not add a custom expert cache.
4. **NAX for prefill only** (their LM-head GEMM 4.5×, but batch-1 decode −8 %).
   Adopted as the split in §6.

From [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c):

1. **Multiply MXFP4 directly from packed nibbles** — never materialize a
   dequantized expert (194 GB/token of format conversion avoided there).
   Adopted in the decode GEMV kernels.
2. **LRU is structurally useless for K3 experts** (Quantile Balancing flattens
   usage; measured 0.0 % hit rate below a ~36 GB arena). Adopted: no large
   cross-layer expert LRU/LFU; keep-last + prediction-driven prefetch + OS
   page cache instead. This replaces the Gemma path's 16-slot LFU.
3. **Config reader that refuses defaults** — a missing `full_attn_layers`
   silently turns 24 MLA layers into KDA. Adopted: the v2 manifest validator
   rejects absent arch fields rather than defaulting them.
4. Trunk quantization warning (§4) — adopted as the int4/int8 profile split.

## 6. M5 Neural Accelerator (NAX) plan

Facts: NAX is reachable only via Metal 4 `MTLTensor` + MSL 4.0
`mpp::tensor_ops` (no raw intrinsic;
[tract #2275](https://github.com/sonos/tract/discussions/2275)); int4/int8
quantized tensor ops are supported on macOS 26, native fp4/fp8 with scale
planes lands in macOS 27 / Metal 4.1
([WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/));
batch-1 GEMV decode gains nothing (bandwidth-bound; flash-moe measured −8 %,
[BaseRT](https://arxiv.org/html/2607.19438v1) keeps decode on GEMV kernels);
MLX's NAX GEMM path measures ~56–61 TFLOP/s on M5 Max
([ax-engine](https://github.com/defai-digital/ax-engine/blob/main/docs/PERFORMANCE-RESULTS.md)).

Therefore:

- **Decode**: existing-style simd-group GEMV kernels (MXFP4 packed, int4/int8
  affine). No NAX. Documented, not re-litigated.
- **Prefill** (the compute-bound phase, and what NAX is for):
  1. Trunk projections: int4/int8 `mpp::tensor_ops::matmul2d` QMM — the same
     pattern as this repo's existing `Metal/TensorCore/tensorops.metal`.
  2. MXFP4 expert GEMM: dequant-to-fp16 tile + fp16 `matmul2d` (what MLX does
     on macOS 26; flip to native fp4 scale-planes when macOS 27 ships).
  3. KDA chunked prefill: K3's bounded log-decay makes all 16-token diagonal
     tiles dense matmuls by design — NAX-friendly.
  4. Grouped MoE prefill (token×expert pairs) on `matmul2d`.
- A runtime check compiles the tensor-ops library only when
  `__HAVE_TENSOR__` is available; GEMV/MPP-less fallback stays for pre-M5
  silicon (the same code runs portably, slower).

## 7. What is explicitly out of scope

- MoonViT-V2 vision tower (text-only engine).
- The real 1.56 TB download and on-device real-weight run — the repacker
  supports it; validation here is a synthetic tiny-K3 with the official
  tensor-name layout plus CPU-reference cross-checks.
- MTP/EAGLE-3 speculative decoding (a 0.40 B draft exists at
  `inference-optimization/Kimi-K3-0.40B`). Noted as the most promising
  Phase-2 lever: verifying K tokens per pass amortizes the 25.8 GB expert
  read over K positions, which is the only mechanism that attacks the SSD
  wall directly.
- macOS 27 native fp4 tensor-ops path (toggle when shipped).

## 8. Expected result

A `.gturbo` v2 bundle of ~1.5 TB on disk (1.447 TB MXFP4 experts verbatim +
~30–60 GB quantized trunk), ~38 GB wired at runtime, decode at an estimated
0.3–1 tok/s cold / better warm, NAX-accelerated prefill, validated end-to-end
on synthetic weights with byte-exact format fixtures and CPU-reference kernel
tests — ready for the real checkpoint with a single repack command.
