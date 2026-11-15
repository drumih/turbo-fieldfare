import Foundation

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
    public let family: String
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
    /// Per-layer head counts, QK norms and RoPE dialects for the families that need them. `.none`
    /// for every other family.
    public let perLayerHeadGeometry: PerLayerHeadGeometry

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
        family: String = "",
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
        perLayerHeadGeometry: PerLayerHeadGeometry = .none
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
        self.perLayerHeadGeometry = perLayerHeadGeometry
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
}
