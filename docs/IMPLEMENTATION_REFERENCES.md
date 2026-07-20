# Implementation references

TurboFieldfare builds on work by other developers and researchers. These are
the sources that materially shaped its Gemma 4 implementation, Metal kernels,
out-of-core runtime, and experiments under the 8 GB memory constraint.

Upstream code references whose exact behavior mattered are pinned to commits
checked on 2026-07-16. Project home pages remain branch-level when they are
included for broader design context rather than a line-level claim.

## Model and weights

- The official [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
  defines the model family, the 26B A4B mixture-of-experts shape, hybrid
  attention, and intended capabilities.
- Hugging Face Transformers provides executable decoder cross-checks for
  [Gemma 4 math](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/models/gemma4/modular_gemma4.py),
  [configuration](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/models/gemma4/configuration_gemma4.py),
  and [RoPE utilities](https://github.com/huggingface/transformers/blob/a8609bed2ad1593e7d756006525a10053d4d5bc6/src/transformers/modeling_rope_utils.py).
  TurboFieldfare used them to verify normalization, K/V derivation, routing,
  MoE combination, layer scaling, and final logits.
- [MLX-LM Gemma 4](https://github.com/ml-explore/mlx-lm/blob/15b522f593b7ca5fbc0cac6f7572d40859d2d8fe/mlx_lm/models/gemma4_text.py)
  and [MLX-VLM Gemma 4](https://github.com/Blaizzy/mlx-vlm/blob/84f43753380355c0455a2bafb291d4b7cbcf81d1/mlx_vlm/models/gemma4/language.py)
  supplied independent implementation checks. MLX-LM also served as the
  bounded logit and quality reference.
- [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit),
  pinned at
  [`0d77464e`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/tree/0d77464eeb233a2da68ebf9d7dc4edaac7db956d),
  is the source of the weights, configuration, tokenizer, and chat-template
  sidecars. The repacker preserves its group-64 MLX affine values rather than
  requantizing them.
- [Hugging Face swift-transformers](https://github.com/huggingface/swift-transformers)
  is the direct tokenizer dependency. TurboFieldfare adds bounded streaming
  detokenization around it.

## Metal and kernels

- Pinned [MLX Metal kernels](https://github.com/ml-explore/mlx/tree/4367c73b60541ddd5a266ce4644fd93d20223b6e/mlx/backend/metal/kernels)
  were the main reference for quantized QMV/QMM, RMSNorm, RoPE, and attention
  geometry. The tagged [v0.32.0 vector SDPA](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/kernels/sdpa_vector.h)
  inspired the D512 one-pass attention variant.
- Pinned [llama.cpp/ggml Metal](https://github.com/ggml-org/llama.cpp/tree/79bba02a6741de194912d370015866414faa83ad/ggml/src/ggml-metal)
  informed row-SIMD quantized matvec, register-resident decode,
  capability-gated kernels, memory mappings, and resource hazards.
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) supplied transferable
  patterns for SIMD-per-row GEMV, packed loads, reductions, online softmax, and
  split-KV attention.

<a id="apple-metal"></a>
### Apple platform contracts

- Apple's [Metal Performance Primitives guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
  and [Metal tensor session](https://developer.apple.com/videos/play/wwdc2026/330/)
  define the platform contract for `MTLTensor`, cooperative inputs,
  `mpp::tensor_ops::matmul2d`, execution scopes, data types, and alignment.
  They guided the staged affine MPP prefill path.

### Apple shader operations

- Apple's [inline Metal 4 operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)
  documents shader-local tensors and inline cooperative operations considered
  in the Metal 4 experiments.

## Out-of-core inference

- [`danveloper/flash-moe`](https://github.com/danveloper/flash-moe),
  [`Anemll/flash-moe`](https://github.com/Anemll/flash-moe), and
  [SwiftLM](https://github.com/SharpAI/SwiftLM) informed Apple-Silicon
  SSD-backed MoE, positional reads, reusable expert buffers, I/O workers, and
  GPU synchronization.
- Apple's [LLM in a Flash](https://arxiv.org/abs/2312.11514) framed the
  out-of-core problem around transferred bytes, useful read size, and flash
  scheduling.
- Apple's Darwin [`pread(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/pread.2.html),
  [`mmap(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html),
  and [`fcntl(2)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
  document concurrent offset reads, page alignment, and file-advice APIs such
  as `F_RDADVISE`.

## KV-cache and attention research

- [FlashAttention](https://arxiv.org/abs/2205.14135),
  [online normalizer calculation](https://arxiv.org/abs/1805.02867), and
  [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
  supplied the tiled-attention and associative online-softmax principles behind
  TurboFieldfare's split-KV kernels.
- Pinned [vLLM attention operators](https://github.com/vllm-project/vllm/tree/530852f9591a822ff4065908778a58fa015f0e69/vllm/v1/attention/ops)
  and [vLLM Metal kernels v2](https://github.com/vllm-project/vllm-metal/tree/11f1b453b74c60d113d67f9a5e7fda41500fd5b5/vllm_metal/metal/kernels_v2)
  were implementation references for transformed and quantized KV caches,
  fused dequantization, and online-softmax reduction.
- [TurboQuant](https://arxiv.org/abs/2504.19874) and
  [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) informed the rejected K4/V4
  KV-cache experiments.
