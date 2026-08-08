import Foundation

// Canonical Kimi K3 (2.78T MoE) descriptor for tests and later repack
// stages. Numbers verified against the official config.json; see
// docs/KIMI_K3_EVALUATION.md section 1 for the sourced derivation.
package enum KimiK3FormatProfile {
    package static let modelID = "moonshotai/Kimi-K3"
    package static let numLayers = 93
    package static let expertsPerLayer = 896
    // One MXFP4 expert = 3 matrices x (params/2 packed + params/32 scales)
    // with w1/w3: 3584 -> 3072 and w2: 3072 -> 3584.
    package static let expertStride: UInt64 = 17_547_264

    // Explicit 1-based layer lists as recorded in the source config: 3 KDA :
    // 1 MLA interleaved (MLA at 4, 8, ..., 92) plus terminal MLA layer 93.
    // Layer 1 (0-based layer 0) is the dense-FFN layer; its attention is KDA.
    package static let denseLayers: [Int] = [1]
    package static let fullAttnLayers: [Int] = (1...23).map { $0 * 4 } + [93]
    package static let kdaLayers: [Int] = (1...93).filter { !fullAttnLayers.contains($0) }

    package static let flags: [String: Bool] = [
        "streamingPresent": true,
        "latentMoE": true,
        "kdaLayers": true,
        "attnRes": true,
    ]

    package static let arch = GTurboManifestArchV2(
        hiddenSize: 7_168,
        vocabSize: 163_840,
        numLayers: 93,
        denseMLPIntermediateSize: 33_792,
        rmsNormEpsilon: 1e-5,
        tieWordEmbeddings: false,
        hiddenActivation: "situ_glu",
        bosTokenID: 163_584,
        eosTokenID: 163_586,
        denseLayers: denseLayers,
        kdaLayers: kdaLayers,
        fullAttnLayers: fullAttnLayers,
        kda: GTurboManifestKDAV2(
            numHeads: 96, headDim: 128, convWidth: 4,
            decayLowRankSize: 128, decayProjectionSize: 12_288,
            gateLowerBound: -5.0, fullRankOutputGate: true),
        mla: GTurboManifestMLAV2(
            numHeads: 96, qLoraRank: 1_536, kvLoraRank: 512,
            qkNopeHeadDim: 128, qkRopeHeadDim: 64, vHeadDim: 128,
            outputGate: true),
        attnRes: GTurboManifestAttnResV2(blockSize: 12),
        moe: GTurboManifestMoEV2(
            numExperts: 896, topKExperts: 16,
            latentBottleneckSize: 3_584, expertIntermediateSize: 3_072,
            numSharedExperts: 2, sharedExpertIntermediateSize: 6_144,
            situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
            routerRenormalize: true, routerCorrectionBias: true))

    private static let trunkInt4 = GTurboManifestQuantSlotV2(
        weightBits: 4, scheme: GTurboFormatV2.quantSchemeAffine4G64,
        scaleType: "BF16", biasType: "BF16", groupSize: 64)

    private static let trunkInt8 = GTurboManifestQuantSlotV2(
        weightBits: 8, scheme: GTurboFormatV2.quantSchemeAffine8G64,
        scaleType: "BF16", biasType: "BF16", groupSize: 64)

    private static let embeddingInt8 = GTurboManifestQuantSlotV2(
        weightBits: 8, scheme: GTurboFormatV2.quantSchemeAffine8G64,
        scaleType: "BF16", biasType: "BF16", groupSize: 64)

    private static let routerFP32 = GTurboManifestQuantSlotV2(
        weightBits: 32, scheme: GTurboFormatV2.quantSchemeFP32,
        scaleType: "none", biasType: "none", groupSize: 1)

    private static let routedExpertMXFP4 = GTurboManifestQuantSlotV2(
        weightBits: 4, scheme: GTurboFormatV2.quantSchemeMxfp4E2M1G32E8M0,
        scaleType: "E8M0", biasType: "none", groupSize: 32)

    package static let quantInt4 = GTurboManifestQuantV2(
        embedding: embeddingInt8,
        attention: trunkInt4,
        router: routerFP32,
        sharedExpert: trunkInt4,
        latentProjection: trunkInt4,
        denseMLP: trunkInt4,
        routedExpert: routedExpertMXFP4)

    package static let quantInt8 = GTurboManifestQuantV2(
        embedding: embeddingInt8,
        attention: trunkInt8,
        router: routerFP32,
        sharedExpert: trunkInt8,
        latentProjection: trunkInt8,
        denseMLP: trunkInt8,
        routedExpert: routedExpertMXFP4)

    /// Backward-compatible name for the default int4 trunk profile.
    package static let quant = quantInt4
}
