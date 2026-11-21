# Inkling 2 Mini in TinyTitan 7.x — port plan and quality contract

**Status: Phase 0 complete; text-only scope chosen; Swift engine components are
landing. The vLLM reference was dropped on 2026-11-21.** The Python and
Swift contract readers validate the real metadata (commit `5ffc1515`,
`6e9c988e`), but the engine is not implemented. This is a new engine family, a
new mixed-quantization runtime and a multimodal surface — not a model-profile
entry. The phases below are ordered so the quantized weights are never
requantized.

### Decisions and current blockers (2026-11-21)

Decided: **text-only**, **preserve NVFP4/MXFP8**, and the checkpoint install is
**approved**. The ~20 GB download is in progress under `models/inkling-2-mini`
(operator-approved; not a gate side effect).

- **No external reference:** the vLLM path was dropped on 2026-11-21. The
  engine is being built in Swift; without an independent host there is no
  logits oracle, so quality claims must come from the architecture and
  deterministic tests until one is available.
- **Host limit:** 24 GiB RAM vs the model card's 32 GB recommendation. Text-only
  expert streaming may fit; 1M context is a larger-host target.

Phase 0's metadata reader is `benchmark/inkling2_quant_contract.py` (with
`test_inkling2_quant_contract` in CI): it validates the real HF metadata and
proves the 20-layer NVFP4 / 4-layer MXFP8 split before any weight is loaded.
The Swift `TinyTitanInkling2` target now carries the real tokenizer (validated
against HF IDs), a safetensors range reader, the exact NVFP4/MXFP8 dequantizer,
the text architecture config parser, a streaming expert loader that reads one
expert's byte range and applies its per-projection scales, and a dense BF16/F32
tensor accessor. Embedding-row reads, embedding/final norms and RMSNorm are in
place, a real embedding row was validated against the partial checkpoint, and
the sparse router plus softcapped SwiGLU/MoE combine are implemented, and the
attention block has its FP32 primitives (L2 clip, softmax, short convolution
with carry state) and relative/sliding band math. A real layer-0 expert's
NVFP4 values are pinned against an independent Python dequant (first five
values and prefix sums). A real layer-0 sliding attention block and a real layer-5 global attention
block now run on a single embedding row; both return finite output plus conv
state. The complete 20 GB checkpoint is downloaded and the real NVFP4 and
MXFP8 expert paths are tested. A complete layer-0 decoder block (norm, sliding
attention, routed MLP, both short convolutions and the block residual) now runs
on one token, and a real layer-5 global block now exercises the three-input
attention path, loads the checkpoint's real attention-residual queries, and has
the loop message controllers, and a streamed LM head (all 201,024 logits are
finite on a real hidden row). The full 24-layer prelude/loop/coda forward now
runs on one token: 201,024 finite logits in 25.5 s of forward time (145-185 s model load) after moving the routed-expert
GEMV (NVFP4/MXFP8 Metal kernels), the LM head and the attention projections
(BF16 GEMV with FP32 activation/output) to the GPU, with the loop repeating the
whole 6..17 core. The relative-attention score/softmax/sink/gate core now runs
on the GPU and matches the CPU path. A GPU short-convolution kernel matches the
CPU path, but the one-token path keeps the small convolution on CPU because the
dispatch cost made it slower (33 s vs 25.5 s); norms and embedding are still CPU.
Attention KV state and the decoder/text caches are now threaded through the 48
layer visits. A real continuation run produced two greedy tokens in one cached
session (`conveni`, `ég`); this is not coherent text and is not a quality claim.
The sampling loop and CLI/server are still absent. Two reference-parity fixes
landed: the router correction bias now affects selection only (not the sigmoid
scores), and the fused expert gate/up rows are read interleaved. Three more parity fixes landed: the loop now repeats the whole 6..17 block,
GQA maps query head h to KV head h//3, and K/V normalization happens before the
attention copy. Using the checkpoint's exact chat template, a "The capital of France is" prompt
now emits `<|message_model|><|content_thinking|>The user has given...` — a
coherent, on-topic reasoning continuation. The earlier echo was caused by a
wrong manual assistant prefix. A greedy `Inkling2Generator` now drives the cached loop. With the exact chat
template and 8 tokens it produces
`<|message_model|><|content_thinking|>The user has given me a`; with the
system effort prompt it produces `The user asks: "The`. Both are coherent
reasoning, still short of the final answer. A 32-token run was killed by
memory pressure from per-token Metal allocations; per-token autorelease pools and reusable GPU weight/scratch buffers were added;
16 tokens now complete safely. A direct-answer prefix produces correct responses to six factual prompts:
Paris, H₂O, four, Jupiter, cold and nine. The thinking path also produces
coherent reasoning. This is a 6/6 short factual check, not a 100% quality
verdict. The one-shot CLI now
detects an Inkling directory and runs it directly:
`TinyTitanCLI --model models/inkling-2-mini --prompt 'The capital of France is' --max-new 8`
prints `The capital of France is Paris.<|end_message|>`. The loopback server,
sampling controls and a broader quality suite are still absent.
The first greedy token is id 184,972 (`البيانات`). Sampling,
per-token caches, CLI/server integration and Metal kernels remain; this is a
correctness reference, not a usable runtime.

## The checkpoint

[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
is a single ~20 GB mixed-precision checkpoint (26.4 B parameters):

- 24-layer decoder with the middle 12 layers run three times (48 applications),
  128 routed experts at top-8, hidden 2304, 1M-token context, hybrid
  sliding-window (1024) and global attention, relative attention
  (`d_rel = 16`, extent 1024), short convolutions and per-block attention
  residuals.
- A 23-layer vision transformer for image inputs, plus a 3-layer MTP sidecar.
- Tokenizer: `tiktoken/tokenizer.model`, not the Qwen `GFTokenizer` format.
- Quantization: ModelOpt `inkling2` recipe —
  - **NVFP4** (`nvfp4_e2m1`, 16-element blocks, FP8 E4M3 scales) for routed
    experts in layers 0–19;
  - **MXFP8** (`mxfp8_e4m3`, 32-element blocks, UE8M0 scales) for routed experts
    in layers 20–23;
  - BF16 for attention, norms, embeddings, vision, MTP and unembed; a small
    FP32 attention-residual group.

## What TinyTitan has today

- `ModelFamily` carries Qwen3.5/3.6/3.8/4, K2-Horizon and Laguna. Inkling is
  none of them; its graph, tensor contract and tokenizer are new.
- Runtime quantization is **INT4/INT8 affine** (plus BF16). GGUF's `nvfp4` /
  `f8E4M3` type names are recognised in the format reader, but there is no
  FP4/FP8 GEMM or block-scale path in the runtime.
- No vision tower, no looped layers, no relative attention, no MTP for this
  shape, no tiktoken tokenizer, no image input surface.
- The 1M context is beyond the existing Qwen path's native 262K and would need
  the same thresholded/YaRN treatment; Inkling's own config already declares
  1M, so its positional scheme is checkpoint-specific.

## The quality contract

“Keep its quants original” means:

1. **No requantization.** The NVFP4 and MXFP8 payloads and their scales are
   preserved bit-for-bit in the install.
2. **Dequantize only at the kernel boundary.** Metal has no NVFP4/MXFP8 tensor
   cores, so weights are expanded to BF16/FP32 for GEMM; the values fed to the
   arithmetic are exactly the dequantized checkpoint values, not a new
   quantization.
3. **Reference parity.** Every stage is compared against the PyTorch/vLLM
   reference on the same checkpoint (logits, top-1 tokens, and the model's own
   greedy output). “100% quality” is a measured parity claim, not a promise a
   port can make before those tests exist.
4. **No quality shortcut under a speed flag.** A faster kernel that changes
   top-1 divergence or logits beyond the measured noise floor is rejected.

## Phases

| # | Phase | Deliverable | Gate |
| --- | --- | --- | --- |
| 0 | Contract | `ModelFamily.inkling2`, manifest schema for mixed quant groups and scales, checkpoint inventory reader | Rejects unknown tensors; names every group |
| 1 | Text architecture | tiktoken tokenizer, embedding, 24-layer looped graph, hybrid attention, sconv, relative attention, 1M positional scheme, text-only | Greedy text parity against the reference on a fixed prompt set |
| 2 | Quant runtime | NVFP4 block-16 + FP8-scale decode, MXFP8 block-32 + UE8M0 decode, expert streaming, BF16 GEMM | Dequant output bit-identical to the reference; logits within the noise floor |
| 3 | Vision (if in scope) | Vision tower, projector, image input in CLI/server | Image/text parity on the reference image set |
| 4 | MTP and qualification | 3-layer MTP sidecar, golden baselines, 1M-context memory/KV guard | Qualified record; no unmeasured claim |
| 5 | Surface | Catalog, installer, manifest, server/CLI routing, docs | `catalog --check`, release gate |

Phase 1 is the first runnable artifact; phases 2–4 are what make it the
*quality-preserving* port the request asks for. Phase 0 can start without a
20 GB download by reading the remote config/quant metadata; phase 1 needs the
install under `models/`.

## Constraints and decisions

- **Memory:** ~20 GB of weights; the model card recommends 32 GB Apple silicon.
  A 24 GB Mac is tight even before KV/expert cache, and the 1M context is not a
  24 GB target.
- **Scope:** text-only first vs full multimodal. Vision and MTP roughly double
  the work; image input also reaches the server/CLI API.
- **Quant path:** preserving NVFP4/MXFP8 is the only route that satisfies
  “original quants”; converting to BF16 preserves the dequantized values but
  expands the install to ~53 GB, and affine 4/8-bit requantization does **not**
  meet the 100% quality condition.
- **Reference:** the comparison needs a PyTorch/vLLM run of the same checkpoint
  on a machine that can execute it. This host has no NVIDIA GPU; a CPU reference
  is possible for small stages but not the full 26B model at 1M context.

## Decision needed

1. Text-only first, or full multimodal?
2. Preserve NVFP4/MXFP8 (phases 1–2) or allow a BF16 port (larger install)?
3. Install the ~20 GB checkpoint under `models/` for phase 1? Installing is an
   operator action, never a gate side effect.
