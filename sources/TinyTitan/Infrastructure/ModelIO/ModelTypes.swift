import Foundation
import Metal

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int  // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int  // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE baseline so earlier manifests validate unchanged.
    public let family: ModelFamily
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention (Qwen 3.6 uses 0.0625 = 256^-0.5).
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize). False for Qwen 3.6.
    public let embeddingScaledBySqrtHidden: Bool
    /// Router carries `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors. False (Qwen 3.6): plain quantized linear router with
    /// renormalized top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Dual-branch FFN sandwich: pre/post feedforward norms plus a per-layer
    /// residual scalar. False (Qwen 3.6) = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. True (Qwen/NeoX sub-dim): rotation confined to
    /// the first `rotaryDim` elements, pairing (i, rotaryDim/2 + i), frequency
    /// divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig
    public let hyperConnections: HyperConnectionConfig
    public let sparseIndexer: SparseIndexerConfig
    public let ple: PLEConfig
    /// Softmax-then-top-k with renormalization of the kept probabilities
    /// (`norm_topk_prob`). False for Qwen3.5-MoE; true for Qwen3.8-Flash-Next.
    public let routerNormTopK: Bool
    /// Affine quantization group size this architecture's checkpoints use.
    /// 64 for every Qwen3.5-MoE-era install; 32 for the available
    /// Qwen3.8-Flash-Next community quantizations. The manifest records the
    /// per-slot value; this is the architecture-level default for planning.
    public let quantGroupSize: Int

    // K2-Horizon (MoVA) extensions. The defaults describe every Qwen family,
    // so adding them changes no existing architecture contract.
    /// RMS-norm groups inside one hidden width: each group normalizes over
    /// `hiddenSize / normGroups` elements and scales with its own weight slice.
    /// 1 is the ordinary whole-vector RMS norm.
    public let normGroups: Int
    /// The function applied to the attention output gate, when
    /// `attnOutputGate` is true.
    public let attentionGate: AttentionGateFunction
    /// The routed-expert router carries a bias used only for selection.
    public let routerHasBias: Bool
    /// Multiplier on the renormalized top-k routing weights.
    public let routerScaling: Double
    /// Layers whose MLP is dense (no routed experts). Empty for a uniform MoE.
    public let denseLayerIndices: [Int]
    /// MoVA value-expert attention. `.none` for every non-K2 family.
    public let mova: MoVAConfig
    /// Laguna's per-layer head counts, QK norms and RoPE dialects. `.none`
    /// for every other family.
    public let laguna: LagunaGeometry

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        family: ModelFamily = .qwen36,
        attnOutputGate: Bool = true,
        attentionScale: Double = 0.0625,
        embeddingScaledBySqrtHidden: Bool = false,
        routerScaled: Bool = false,
        ffnSandwichNorms: Bool = false,
        sharedExpertGated: Bool = true,
        ropeNeoxSubdim: Bool = true,
        linearAttention: LinearAttentionConfig = .none,
        hyperConnections: HyperConnectionConfig = .none,
        sparseIndexer: SparseIndexerConfig = .none,
        ple: PLEConfig = .none,
        routerNormTopK: Bool = false,
        quantGroupSize: Int = 64,
        normGroups: Int = 1,
        attentionGate: AttentionGateFunction = .sigmoid,
        routerHasBias: Bool = false,
        routerScaling: Double = 1,
        denseLayerIndices: [Int] = [],
        mova: MoVAConfig = .none,
        laguna: LagunaGeometry = .none
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.family = family
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.linearAttention = linearAttention
        self.hyperConnections = hyperConnections
        self.sparseIndexer = sparseIndexer
        self.ple = ple
        self.routerNormTopK = routerNormTopK
        self.quantGroupSize = quantGroupSize
        self.normGroups = normGroups
        self.attentionGate = attentionGate
        self.routerHasBias = routerHasBias
        self.routerScaling = routerScaling
        self.denseLayerIndices = denseLayerIndices
        self.mova = mova
        self.laguna = laguna
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
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
        family: .qwen36,
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
        family: .qwen36MTP,
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
        family: .qwen38flash,
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
        family: .qwen38flashMTP,
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

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
/// unchecked-invariant: every stored property is a `let`; only MTLBuffer's
/// lack of Sendable forces @unchecked. The view describes where a tensor sits
/// in the resident buffer and never mutates it.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(
        buffer: MTLBuffer,
        offset: UInt64, length: UInt64,
        scaleOffset: UInt64, scaleLength: UInt64,
        biasOffset: UInt64, biasLength: UInt64,
        shape: (UInt32, UInt32, UInt32, UInt32),
        dtype: UInt8
    ) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
