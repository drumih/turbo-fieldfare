import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`.
/// Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    case qwen38flash = "qwen38flash"
    case qwen38flashMTP = "qwen38flash_mtp"
    /// Qwen 4.0 Flash (`qwen4-omni-120b-think`): the Qwen4-generation successor
    /// of Qwen3.8-Flash-Next. A separate family on purpose -- the loader and
    /// cross-check are copied so a 4.0 checkpoint can change shape without
    /// touching the 3.8 port.
    case qwen4flash = "qwen4flash"
    case qwen4flashMTP = "qwen4flash_mtp"
    /// Qwen 3.5's dense text models (2B, 4B, 9B). A whole model, not a draft
    /// head, and the only family here whose weights carry no routed experts.
    case qwen35Dense = "qwen3_5_dense"
    /// K2-Horizon's sparse family: 100 routed experts, a shared expert, a
    /// grouped RMS norm, a softplus attention gate and MoVA value experts.
    case k2Horizon = "k2_horizon"
    /// Laguna-S-2.1: per-head softplus gating, QK norms, two RoPE dialects
    /// and a 1:3 global-to-sliding layer mask.
    case laguna = "laguna"
    /// A draft sidecar is loaded beside a target and prompted through the
    /// target's tokenizer, so a local import of one carries no tokenizer of
    /// its own.
    var isDraftHead: Bool {
        switch self {
        case .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP: return true
        default: return false
        }
    }
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int  // shared expert FFN
    let moeIntermediateSize: Int  // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE text architecture used by Qwen 3.6 and Ornith 1.5.
    let family: RepackModelFamily
    let attnOutputGate: Bool
    let attentionScale: Double
    let embeddingScaledBySqrtHidden: Bool
    let routerScaled: Bool
    let ffnSandwichNorms: Bool
    let sharedExpertGated: Bool
    let ropeNeoxSubdim: Bool
    let linearNumKHeads: Int
    let linearNumVHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelSize: Int

    // Qwen3.8-Flash-Next extensions. Zeroed for architectures that do not
    // have them, which keeps the qwen36 contract and its cross-check exact.
    let hcCount: Int
    let hcLowRank: Int
    let indexerNumHeads: Int
    let indexerNumKVHeads: Int
    let indexerHeadDim: Int
    let indexerBudget: Int
    let indexerCompressRatio: Int
    /// 0-based layer indices carrying the PLE block (config is 1-based).
    let pleLayerIndices: [Int]
    let pleEmbedDim: Int
    let pleConvKernelSize: Int
    let pleNgramSize: Int
    let pleVocabSizeBase: Int
    let pleHeadsPerNgram: Int
    let pleVocabDivisor: Int
    /// Router renormalizes the top-k probabilities (`norm_topk_prob`).
    let routerNormTopK: Bool
    /// Affine quantization group size of the source checkpoint.
    let quantGroupSize: Int

    // K2-Horizon (MoVA) extensions. Defaults describe the Qwen families, so
    // adding them here changes no existing contract.
    let movaNumExperts: Int
    let movaTopKExperts: Int
    let normGroups: Int
    let attentionGate: String
    let routerScaling: Double
    let routerHasBias: Bool
    let movaRouterScaling: Double
    let movaRouterHasBias: Bool
    let movaActivation: String
    let denseLayerIndices: [Int]

    init(
        hiddenSize: Int, intermediateSize: Int, moeIntermediateSize: Int,
        numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
        headDim: Int, fullHeadDim: Int, vocabSize: Int, slidingWindow: Int,
        finalLogitSoftcap: Double, ropeTheta: Double, fullRopeTheta: Double,
        partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
        topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8], hiddenActivation: String,
        family: RepackModelFamily, attnOutputGate: Bool,
        attentionScale: Double, embeddingScaledBySqrtHidden: Bool,
        routerScaled: Bool, ffnSandwichNorms: Bool, sharedExpertGated: Bool,
        ropeNeoxSubdim: Bool, linearNumKHeads: Int, linearNumVHeads: Int,
        linearKeyHeadDim: Int, linearValueHeadDim: Int,
        linearConvKernelSize: Int,
        hcCount: Int = 0, hcLowRank: Int = 0,
        indexerNumHeads: Int = 0, indexerNumKVHeads: Int = 0,
        indexerHeadDim: Int = 0, indexerBudget: Int = 0,
        indexerCompressRatio: Int = 0,
        pleLayerIndices: [Int] = [], pleEmbedDim: Int = 0,
        pleConvKernelSize: Int = 0, pleNgramSize: Int = 0,
        pleVocabSizeBase: Int = 0, pleHeadsPerNgram: Int = 0,
        pleVocabDivisor: Int = 0,
        routerNormTopK: Bool = false, quantGroupSize: Int = 64,
        movaNumExperts: Int = 0, movaTopKExperts: Int = 0,
        normGroups: Int = 1, attentionGate: String = "sigmoid",
        routerScaling: Double = 1, routerHasBias: Bool = false,
        movaRouterScaling: Double = 1, movaRouterHasBias: Bool = false,
        movaActivation: String = "silu", denseLayerIndices: [Int] = []
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
        self.linearNumKHeads = linearNumKHeads
        self.linearNumVHeads = linearNumVHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelSize = linearConvKernelSize
        self.hcCount = hcCount
        self.hcLowRank = hcLowRank
        self.indexerNumHeads = indexerNumHeads
        self.indexerNumKVHeads = indexerNumKVHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.pleLayerIndices = pleLayerIndices
        self.pleEmbedDim = pleEmbedDim
        self.pleConvKernelSize = pleConvKernelSize
        self.pleNgramSize = pleNgramSize
        self.pleVocabSizeBase = pleVocabSizeBase
        self.pleHeadsPerNgram = pleHeadsPerNgram
        self.pleVocabDivisor = pleVocabDivisor
        self.routerNormTopK = routerNormTopK
        self.quantGroupSize = quantGroupSize
        self.movaNumExperts = movaNumExperts
        self.movaTopKExperts = movaTopKExperts
        self.normGroups = normGroups
        self.attentionGate = attentionGate
        self.routerScaling = routerScaling
        self.routerHasBias = routerHasBias
        self.movaRouterScaling = movaRouterScaling
        self.movaRouterHasBias = movaRouterHasBias
        self.movaActivation = movaActivation
        self.denseLayerIndices = denseLayerIndices
    }

    /// Whether this architecture has the extension geometry the manifest's
    /// extension block carries, so the block is written and then required.
    ///
    /// A property of the *values*, not of the family name, and defined the same
    /// way the runtime defines its half of the comparison
    /// (`HyperConnectionConfig.enabled`, `SparseIndexerConfig.enabled`,
    /// `PLEConfig.enabled`): writer and reader then agree by construction
    /// instead of by two lists kept in step.
    var declaresExtensionGeometry: Bool {
        hcCount > 0 || indexerBudget > 0 || !pleLayerIndices.isEmpty
    }

    /// Ceiling for the checkpoint's `config.json`, stated here because two
    /// readers take that same file — this one and `IndexLoader` — and a document
    /// should have one bound, not two that can drift apart.
    ///
    /// 8 MiB is not borrowed from the manifest's 64 MiB: this is the converter's
    /// own trust boundary, where the operator chose the directory, so a cap here
    /// risks refusing a *legitimate* checkpoint rather than an attacker's. The
    /// installed tokenizer copy of the file measures 12,935 bytes, so the ceiling
    /// is ~650x the real document and the history in
    /// `IndexLoader.maximumIndexBytes` — a 4 MiB bound that turned out to be below
    /// a legitimate file — is the reason for the margin rather than for a tighter
    /// number. It is still a bound: `Posix.readBoundedData` checks it before
    /// allocating, so a corrupt or planted file cannot make the converter
    /// allocate without limit. To convert a checkpoint whose config really does
    /// exceed it, raise this one constant; the refusal names the size and the cap.
    static let maxConfigBytes: UInt64 = 8 * 1024 * 1024

    /// `model_type` values that mean the Qwen 4.0 Flash text contract.
    ///
    /// Kept explicit rather than prefix-matched: `qwen4_exp` is the 3.8
    /// preview's own contract, and a future `qwen4_*` variant with a different
    /// tensor set must be a deliberate addition here rather than a config that
    /// falls into this loader because its spelling looked close.
    static let qwen4FlashModelTypes: Set<String> = [
        "qwen4_omni_moe", "qwen4_omni", "qwen4_flash",
        // A text-only export may state the inner spelling at the root.
        "qwen4_omni_moe_text", "qwen4_omni_text", "qwen4_flash_text",
    ]

    static func load(configPath: String, maxBytes: UInt64 = maxConfigBytes) throws -> ArchInfo {
        let data: Data
        do {
            data = try Posix.readBoundedData(configPath, maximumBytes: maxBytes)
        } catch RepackError.installStateCorrupt(let path, let detail) {
            throw RepackError.configJsonInvalid(path: path, detail: detail)
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // A vision-language or MoE checkpoint nests the text model under
        // `text_config`. A dense Qwen 3.5 snapshot is already flat -- the
        // converter writes the text config at the root, because that is the
        // shape the CPU engine reads -- so the root *is* the text config
        // there. Accepting both keeps one reader for one architecture instead
        // of a second config shape nobody else can parse.
        let tc = (root["text_config"] as? [String: Any]) ?? root
        if (root["model_type"] as? String) == "qwen3_5_mtp" {
            return try loadQwen36MTP(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen3_5_moe" {
            return try loadQwen35MoE(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen3_5_dense" {
            return try loadQwen35Dense(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "qwen4_exp" {
            return try loadQwen4Exp(configPath: configPath, tc: tc, root: root)
        }
        // Qwen 4.0 Flash (`qwen4-omni-120b-think`). The spellings are the ones
        // the published family uses (`qwen4_omni_moe`, following Qwen3-Omni's
        // `qwen3_omni_moe`), plus the shorter forms a text-only export may
        // declare. `qwen4_exp` is deliberately *not* here: it is the 3.8
        // preview's model_type and stays on its own family.
        if let modelType = root["model_type"] as? String,
            Self.qwen4FlashModelTypes.contains(modelType)
        {
            return try loadQwen4Flash(configPath: configPath, tc: tc, root: root)
        }
        if (root["model_type"] as? String) == "k2_horizon" {
            return try loadK2Horizon(configPath: configPath, tc: tc)
        }
        if (root["model_type"] as? String) == "laguna" {
            return try loadLaguna(configPath: configPath, tc: tc)
        }
        throw RepackError.configJsonInvalid(
            path: configPath,
            detail: "unsupported model_type (expected qwen3_5_moe, "
                + "qwen3_5_mtp, qwen3_5_dense, qwen4_exp, qwen4_omni_moe, "
                + "qwen4_omni, qwen4_flash or k2_horizon)")
    }

}
