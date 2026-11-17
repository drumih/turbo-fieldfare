import Foundation
import TinyTitanCore

extension ArchConfig {
    /// The typed family view of the raw `family` string. The generic Core
    /// `ArchConfig` carries the family as a string (no model knowledge in Core,
    /// spec §1.3); the runtime resolves it to the typed enum for dispatch.
    public var modelFamily: ModelFamily { ModelFamily(rawValue: family) ?? .qwen36 }
    public static let qwen36_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: "qwen36",
        attnOutputGate: true,
        attentionScale: 0.0625,  // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    /// Native one-layer Qwen3.5-MoE MTP draft for compatible Qwen/Ornith
    /// targets. Its 65,536-token KV cache is at most
    /// 128 MiB in FP16 and smaller with compressed storage; truncating draft
    /// context can only lower acceptance because every emitted token is still
    /// verified by the full target.
    public static let qwen36MTP = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 65_536,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 1,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [1],
        hiddenActivation: "silu",
        family: "qwen36_mtp",
        attnOutputGate: true,
        attentionScale: 0.0625,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true)

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Qwen3.8-Flash-Next 125B-A6B (`qwen4_exp` text config), text-only.
    /// Geometry read from the official checkpoint's config.json and the
    /// transformers modeling source; see docs/qwen38-flash-next-port.md.
    /// The full-attention slots describe the QSA layers (every 4th layer);
    /// the sliding-window mirrors mask value 1 exactly as qwen36 does.
    public static let qwen38FlashNext = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 48,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen38LayerMask(),
        hiddenActivation: "silu",
        family: "qwen38flash",
        attnOutputGate: true,
        attentionScale: 0.0625,  // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 48,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4,
            outputGate: .sigmoid),
        hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
        sparseIndexer: SparseIndexerConfig(
            numHeads: 4, numKVHeads: 1, headDim: 128,
            budget: 2048, compressRatio: 4),
        ple: PLEConfig(
            layerIndices: [1],  // config ple_layer_ids [2] is 1-based
            embedDim: 2560,
            convKernelSize: 4,
            ngramSize: 3,
            vocabSizeBase: 20_000_000,
            headsPerNgram: 8,
            vocabDivisor: 128,
            seed: 1234),
        routerNormTopK: true,
        // Verified against the pinned artifact's `quantization.group_size`
        // (benchmark/tinytitan_quant_fidelity.py decodes it correctly at 64).
        // The design record had assumed 32 from the Vontra checkpoints.
        quantGroupSize: 64)

    /// The Qwen3.8-Flash-Next MTP draft. One full-attention layer, no
    /// DeltaNet, no n-gram block; everything else is the target's geometry
    /// because the draft has to speak the same residual.
    public static let qwen38FlashNextMTP = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 1,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [1],
        hiddenActivation: "silu",
        family: "qwen38flash_mtp",
        attnOutputGate: true,
        attentionScale: 0.0625,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: .none,
        hyperConnections: HyperConnectionConfig(count: 4, lowRank: 320),
        sparseIndexer: SparseIndexerConfig(
            numHeads: 4, numKVHeads: 1, headDim: 128,
            budget: 2048, compressRatio: 4),
        ple: .none,
        routerNormTopK: true,
        quantGroupSize: 64)

    private static func qwen38LayerMask() -> [UInt8] {
        // Same kinds as qwen36: 2 = gated-DeltaNet linear, 1 = full (QSA)
        // attention on every 4th layer ((i + 1) % 4 == 0), 48 layers.
        var mask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Registry keyed by `manifest.arch.family` for auto-detection at load.
    public static let knownArchitectures: [ModelFamily: ArchConfig] = [
        .qwen36: .qwen36_35B_A3B,
        .qwen36MTP: .qwen36MTP,
        .qwen38flash: .qwen38FlashNext,
        .qwen38flashMTP: .qwen38FlashNextMTP,
        .qwen4flash: .qwen4Flash,
        .qwen4flashMTP: .qwen4FlashMTP,
        .laguna: .lagunaS21,
    ]

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
        shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        shapes.append((m: intermediateSize, n: hiddenSize))
        shapes.append((m: hiddenSize, n: intermediateSize))
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = [(m: numExperts, n: hiddenSize)]
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
}
