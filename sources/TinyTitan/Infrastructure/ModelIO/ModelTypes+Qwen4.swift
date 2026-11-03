import Foundation

/// Qwen 4.0 Flash (`qwen4-omni-120b-think`) runtime presets.
///
/// Split from `ModelTypes.swift` under the 500-line-per-file rule. The presets
/// are a separate family from `qwen38FlashNext` on purpose: the 4.0 checkpoint
/// can change a field without touching the 3.8 model, and the loader/cross-check
/// pair on the repack side is separate for the same reason.
extension ArchConfig {

    /// Qwen 4.0 Flash (`qwen4-omni-120b-think`), text-only.
    ///
    /// A separate preset from `qwen38FlashNext` on purpose. It starts as the
    /// 3.8 preview's verified geometry because the 4.0 checkpoint is not
    /// published yet; the loader reads the checkpoint's own config, so the
    /// numbers here are the contract the runtime validates that config against.
    /// When the weights land, reconcile this preset and
    /// `ArchInfo.crossCheckQwen4Flash` with the real `config.json` together.
    public static let qwen4Flash = ArchConfig(
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
        fullAttentionLayerMask: Self.qwen4LayerMask(),
        hiddenActivation: "silu",
        family: .qwen4flash,
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
        quantGroupSize: 64)

    /// The Qwen 4.0 Flash draft. One full-attention layer, no DeltaNet, no
    /// n-gram block; everything else is the target's geometry because the
    /// draft has to speak the same residual.
    public static let qwen4FlashMTP = ArchConfig(
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
        family: .qwen4flashMTP,
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

    private static func qwen4LayerMask() -> [UInt8] {
        // Copied from the 3.8 preview: 36 gated-DeltaNet layers and 12 QSA
        // layers (every 4th). Kept as its own function so a 4.0 checkpoint with
        // a different pattern changes only this family's mask.
        var mask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { mask[i] = 1 }
        return mask
    }
}
