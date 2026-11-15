import Foundation

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape, and family-specific kernel behavior. Stored in
/// `manifest.json -> arch.family`; absent means the compatible Qwen3.5-MoE
/// 35B-A3B baseline used by Qwen 3.6 and Ornith 1.5.
public enum ModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    /// Qwen3.8-Flash-Next (`qwen4_exp` text config), 125B-A6B: 48 layers of
    /// (3x gated-DeltaNet -> 1x sparse-indexed attention), 512 experts top-10,
    /// hyper-connection residual streams, and hashed n-gram embeddings at
    /// layer 1. Text-only; the checkpoint's vision tower is never repacked.
    /// See docs/qwen38-flash-next-port.md for the verified design record.
    case qwen38flash = "qwen38flash"
    /// The Qwen3.8-Flash-Next draft head: one full-attention layer with its
    /// own 512 experts, hyper-connection gates and indexer, plus two fusion
    /// projections that combine the target's wide residual with the next
    /// token's embedding. Shares the target's embedding and head.
    case qwen38flashMTP = "qwen38flash_mtp"
    /// Qwen 4.0 Flash (`Qwen4-Omni-120B-Think`), 120B-A?B: the Qwen4-generation
    /// successor of Qwen3.8-Flash-Next. Its text contract is the same shape the
    /// 3.8 preview uses -- 48 layers of (3x gated-DeltaNet -> 1x
    /// sparse-indexed attention), 512 experts top-10, hyper-connection residual
    /// streams, and a hashed n-gram block -- but it is a separate family so the
    /// two ports can diverge. Text-only; the checkpoint's vision and audio
    /// towers are never repacked.
    ///
    /// The family value is written into `manifest.arch.family`; it is TinyTitan's
    /// own name for the path, not the checkpoint's `model_type`.
    case qwen4flash = "qwen4flash"
    /// The Qwen 4.0 Flash draft head: one full-attention layer with its own
    /// experts, hyper-connection gates and indexer, plus the two fusion
    /// projections that combine the target's wide residual with the next
    /// token's embedding. Shares the target's embedding and head.
    case qwen4flashMTP = "qwen4flash_mtp"
    /// Qwen 3.5's dense text models (2B, 4B, 9B), served by the CPU engine.
    ///
    /// Its own value rather than a reuse of `qwen36`: `ModelFamily` is what the
    /// GPU loader dispatches on, so a dense model wearing the MoE family's name
    /// would reach `Model.load` and the `qwen36` schema checks -- which demand
    /// affine MoE-shaped tensors -- before the CPU engine saw it. The raw value
    /// matches the `model_type` the converter writes.
    case qwen35Dense = "qwen3_5_dense"
    /// IFM K2-Horizon-MoVA-36B-A4B (`model_type: k2_horizon`).
    ///
    /// 48 layers, two expert systems per layer -- 100 routed SwiGLU experts at
    /// top-8 and 64 MoVA value experts at top-4 inside attention -- a two-group
    /// RMS norm, a softplus attention gate and a sigmoid router with a
    /// selection bias and a 2.5 scale. Layers 0-2 are dense.
    ///
    /// The install format and the loader's identity reader know this family;
    /// the GPU graph and kernels do not exist yet (see
    /// `docs/plan-k2-horizon.md`), so `Model.load` refuses it by name at schema
    /// validation rather than running a K2 payload through the Qwen graph.
    case k2Horizon = "k2_horizon"
    /// Poolside Laguna-S-2.1 (`model_type: laguna`).
    ///
    /// 48 layers, 256 routed experts at top-10 plus one shared expert, GQA
    /// with 8 KV heads and a *different* query-head count on the 12 global
    /// layers (48) and the 36 sliding-window layers (72), per-head softplus
    /// attention gating, per-head QK RMS norms, a sigmoid router with an
    /// aux-loss-free selection bias and a 2.5 routed scale, and two RoPE
    /// configurations (YaRN on global, plain on sliding). Layer 0 is dense.
    ///
    /// The probe graph lives in `LagunaGraphProbe`; the general
    /// `RealForwardRunner` refuses the family, exactly as it does for K2.
    case laguna = "laguna"
}
