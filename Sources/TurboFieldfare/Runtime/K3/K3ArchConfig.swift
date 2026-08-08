import Foundation
import TurboFieldfareFormat

/// Compile-time architecture baseline for Kimi K3, mirroring the house
/// `ArchConfig` pattern: `manifest.json -> arch` must match this field-by-field
/// at load time; mismatches throw `ModelError.archMismatch`.
///
/// The canonical `kimiK3` values mirror `KimiK3FormatProfile` (verified there
/// against the official config.json). The manifest records layer lists 1-based;
/// this type exposes both the recorded 1-based lists and derived 0-based sets.
public struct K3ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numLayers: Int
    public let denseMLPIntermediateSize: Int
    public let rmsNormEpsilon: Double
    public let tieWordEmbeddings: Bool
    public let hiddenActivation: String
    public let bosTokenID: Int
    /// Generation stop token (`<|end_of_msg|>`).
    public let eosTokenID: Int
    /// 1-based lists as recorded on the wire. `denseLayers` overlaps the
    /// attention lists; `kdaLayers`/`fullAttnLayers` are disjoint and cover
    /// every layer.
    public let denseLayers: [Int]
    public let kdaLayers: [Int]
    public let fullAttnLayers: [Int]

    // KDA (Kimi Delta Attention)
    public let kdaNumHeads: Int
    public let kdaHeadDim: Int
    public let kdaConvWidth: Int
    public let kdaDecayLowRankSize: Int
    public let kdaDecayProjectionSize: Int
    public let kdaGateLowerBound: Double
    public let kdaFullRankOutputGate: Bool

    // MLA (gated multi-head latent attention)
    public let mlaNumHeads: Int
    public let mlaQLoraRank: Int
    public let mlaKVLoraRank: Int
    public let mlaQKNopeHeadDim: Int
    public let mlaQKRopeHeadDim: Int
    public let mlaVHeadDim: Int
    public let mlaOutputGate: Bool

    // AttnRes
    public let attnResBlockSize: Int

    // LatentMoE
    public let moeNumExperts: Int
    public let moeTopKExperts: Int
    public let moeLatentBottleneckSize: Int
    public let moeExpertIntermediateSize: Int
    public let moeNumSharedExperts: Int
    public let moeSharedExpertIntermediateSize: Int
    public let situGLUGateBeta: Double
    public let situGLUUpBeta: Double
    public let routerRenormalize: Bool
    public let routerCorrectionBias: Bool

    // Streaming metadata
    public let expertsPerLayer: Int
    public let expertStride: UInt64

    public init(hiddenSize: Int, vocabSize: Int, numLayers: Int,
                denseMLPIntermediateSize: Int, rmsNormEpsilon: Double,
                tieWordEmbeddings: Bool, hiddenActivation: String,
                bosTokenID: Int, eosTokenID: Int,
                denseLayers: [Int], kdaLayers: [Int], fullAttnLayers: [Int],
                kdaNumHeads: Int, kdaHeadDim: Int, kdaConvWidth: Int,
                kdaDecayLowRankSize: Int, kdaDecayProjectionSize: Int,
                kdaGateLowerBound: Double, kdaFullRankOutputGate: Bool,
                mlaNumHeads: Int, mlaQLoraRank: Int, mlaKVLoraRank: Int,
                mlaQKNopeHeadDim: Int, mlaQKRopeHeadDim: Int, mlaVHeadDim: Int,
                mlaOutputGate: Bool,
                attnResBlockSize: Int,
                moeNumExperts: Int, moeTopKExperts: Int,
                moeLatentBottleneckSize: Int, moeExpertIntermediateSize: Int,
                moeNumSharedExperts: Int, moeSharedExpertIntermediateSize: Int,
                situGLUGateBeta: Double, situGLUUpBeta: Double,
                routerRenormalize: Bool, routerCorrectionBias: Bool,
                expertsPerLayer: Int, expertStride: UInt64) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.denseMLPIntermediateSize = denseMLPIntermediateSize
        self.rmsNormEpsilon = rmsNormEpsilon
        self.tieWordEmbeddings = tieWordEmbeddings
        self.hiddenActivation = hiddenActivation
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.denseLayers = denseLayers
        self.kdaLayers = kdaLayers
        self.fullAttnLayers = fullAttnLayers
        self.kdaNumHeads = kdaNumHeads
        self.kdaHeadDim = kdaHeadDim
        self.kdaConvWidth = kdaConvWidth
        self.kdaDecayLowRankSize = kdaDecayLowRankSize
        self.kdaDecayProjectionSize = kdaDecayProjectionSize
        self.kdaGateLowerBound = kdaGateLowerBound
        self.kdaFullRankOutputGate = kdaFullRankOutputGate
        self.mlaNumHeads = mlaNumHeads
        self.mlaQLoraRank = mlaQLoraRank
        self.mlaKVLoraRank = mlaKVLoraRank
        self.mlaQKNopeHeadDim = mlaQKNopeHeadDim
        self.mlaQKRopeHeadDim = mlaQKRopeHeadDim
        self.mlaVHeadDim = mlaVHeadDim
        self.mlaOutputGate = mlaOutputGate
        self.attnResBlockSize = attnResBlockSize
        self.moeNumExperts = moeNumExperts
        self.moeTopKExperts = moeTopKExperts
        self.moeLatentBottleneckSize = moeLatentBottleneckSize
        self.moeExpertIntermediateSize = moeExpertIntermediateSize
        self.moeNumSharedExperts = moeNumSharedExperts
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.situGLUGateBeta = situGLUGateBeta
        self.situGLUUpBeta = situGLUUpBeta
        self.routerRenormalize = routerRenormalize
        self.routerCorrectionBias = routerCorrectionBias
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
    }

    /// Canonical Kimi K3 (2.78T) baseline. Mirrors `KimiK3FormatProfile`.
    public static let kimiK3 = K3ArchConfig(
        hiddenSize: 7_168,
        vocabSize: 163_840,
        numLayers: 93,
        denseMLPIntermediateSize: 33_792,
        rmsNormEpsilon: 1e-5,
        tieWordEmbeddings: false,
        hiddenActivation: "situ_glu",
        bosTokenID: 163_584,
        eosTokenID: 163_586,
        denseLayers: KimiK3FormatProfile.denseLayers,
        kdaLayers: KimiK3FormatProfile.kdaLayers,
        fullAttnLayers: KimiK3FormatProfile.fullAttnLayers,
        kdaNumHeads: 96, kdaHeadDim: 128, kdaConvWidth: 4,
        kdaDecayLowRankSize: 128, kdaDecayProjectionSize: 12_288,
        kdaGateLowerBound: -5.0, kdaFullRankOutputGate: true,
        mlaNumHeads: 96, mlaQLoraRank: 1_536, mlaKVLoraRank: 512,
        mlaQKNopeHeadDim: 128, mlaQKRopeHeadDim: 64, mlaVHeadDim: 128,
        mlaOutputGate: true,
        attnResBlockSize: 12,
        moeNumExperts: 896, moeTopKExperts: 16,
        moeLatentBottleneckSize: 3_584, moeExpertIntermediateSize: 3_072,
        moeNumSharedExperts: 2, moeSharedExpertIntermediateSize: 6_144,
        situGLUGateBeta: 4.0, situGLUUpBeta: 25.0,
        routerRenormalize: true, routerCorrectionBias: true,
        expertsPerLayer: KimiK3FormatProfile.expertsPerLayer,
        expertStride: KimiK3FormatProfile.expertStride)

    // MARK: - Derived layer schedules (0-based)

    /// 0-based MLA (full attention) layers: {3, 7, 11, …, 87, 91, 92}.
    public var mlaLayers0: Set<Int> { Set(fullAttnLayers.map { $0 - 1 }) }
    /// 0-based KDA layers: every layer not in `mlaLayers0`.
    public var kdaLayers0: Set<Int> { Set(kdaLayers.map { $0 - 1 }) }
    /// 0-based dense-MLP layers (overlaps the attention schedule): {0}.
    public var denseLayers0: Set<Int> { Set(denseLayers.map { $0 - 1 }) }
    /// 0-based MoE layers: full depth minus the dense layers.
    public var moeLayers0: Set<Int> {
        Set(0..<numLayers).subtracting(denseLayers0)
    }

    public func isMLA(layer0: Int) -> Bool { mlaLayers0.contains(layer0) }
    public func isKDA(layer0: Int) -> Bool { kdaLayers0.contains(layer0) }
    public func isDense(layer0: Int) -> Bool { denseLayers0.contains(layer0) }
    public func isMoE(layer0: Int) -> Bool { moeLayers0.contains(layer0) }
    /// AttnRes block boundaries sit at 0-based layers 0, 12, 24, …, 84.
    public func isAttnResBoundary(layer0: Int) -> Bool {
        layer0 % attnResBlockSize == 0
    }

    /// Ordinal of `layer0` among KDA layers (0..<69), nil for MLA layers.
    public func kdaOrdinal(layer0: Int) -> Int? {
        guard isKDA(layer0: layer0) else { return nil }
        return kdaLayers0.filter { $0 < layer0 }.count
    }

    /// Ordinal of `layer0` among MLA layers (0..<23), nil for KDA layers.
    public func mlaOrdinal(layer0: Int) -> Int? {
        guard isMLA(layer0: layer0) else { return nil }
        return mlaLayers0.filter { $0 < layer0 }.count
    }

    /// Number of AttnRes boundary layers (block-list appends per token): 8.
    public var attnResBoundaryCount: Int {
        (0..<numLayers).filter { isAttnResBoundary(layer0: $0) }.count
    }

    // MARK: - Derived dimensions

    /// KDA projection width: 96 × 128 = 12,288.
    public var kdaChannels: Int { kdaNumHeads * kdaHeadDim }
    /// KDA recurrent-state floats per layer: 96 × 128 × 128.
    public var kdaStateElementsPerLayer: Int {
        kdaNumHeads * kdaHeadDim * kdaHeadDim
    }
    /// KDA conv-state floats per layer: 3 projections × channels × (width − 1).
    public var kdaConvStateElementsPerLayer: Int {
        3 * kdaChannels * (kdaConvWidth - 1)
    }
    /// MLA per-token cached row: [kv_lora latent | rope] = 512 + 64 = 576.
    public var mlaCacheRowElements: Int { mlaKVLoraRank + mlaQKRopeHeadDim }
    /// MLA q width per layer: heads × (nope + rope) = 96 × 192.
    public var mlaQElements: Int { mlaNumHeads * (mlaQKNopeHeadDim + mlaQKRopeHeadDim) }
    /// MLA kv_b expansion width: heads × (nope + v) = 96 × 256.
    public var mlaKVBRows: Int { mlaNumHeads * (mlaQKNopeHeadDim + mlaVHeadDim) }
    /// MLA attention output width: heads × v = 96 × 128.
    public var mlaOutputElements: Int { mlaNumHeads * mlaVHeadDim }
    /// MLA absorbed-decode scale: (nope + rope)^(−1/2) = 192^(−1/2).
    public var mlaDecodeScale: Float {
        Float(1.0 / (Double(mlaQKNopeHeadDim + mlaQKRopeHeadDim)).squareRoot())
    }
}

extension K3ArchConfig {
    /// Field-by-field match of the manifest against this baseline, mirroring
    /// `ManifestReader.validateArch`. Throws `ModelError.archMismatch` on the
    /// first divergence. Also pins the streaming metadata (layer count,
    /// experts per layer, expert stride) and the quant-slot contract the
    /// runtime kernels accept. The embedding/router/routed-expert slots stay
    /// fixed while the four trunk slots must consistently select int4 or
    /// int8 affine-g64.
    package static func validate(manifest: GTurboManifestV2,
                                 expecting e: K3ArchConfig) throws
        -> GTurboManifestQuantV2 {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        let a = manifest.arch
        try check("hiddenSize", a.hiddenSize, e.hiddenSize)
        try check("vocabSize", a.vocabSize, e.vocabSize)
        try check("numLayers", a.numLayers, e.numLayers)
        try check("denseMLPIntermediateSize", a.denseMLPIntermediateSize,
                  e.denseMLPIntermediateSize)
        try check("rmsNormEpsilon", a.rmsNormEpsilon, e.rmsNormEpsilon)
        try check("tieWordEmbeddings", a.tieWordEmbeddings, e.tieWordEmbeddings)
        try check("hiddenActivation", a.hiddenActivation, e.hiddenActivation)
        try check("bosTokenID", a.bosTokenID, e.bosTokenID)
        try check("eosTokenID", a.eosTokenID, e.eosTokenID)
        try check("denseLayers", a.denseLayers.description, e.denseLayers.description)
        try check("kdaLayers", a.kdaLayers.description, e.kdaLayers.description)
        try check("fullAttnLayers", a.fullAttnLayers.description,
                  e.fullAttnLayers.description)
        try check("kda.numHeads", a.kda.numHeads, e.kdaNumHeads)
        try check("kda.headDim", a.kda.headDim, e.kdaHeadDim)
        try check("kda.convWidth", a.kda.convWidth, e.kdaConvWidth)
        try check("kda.decayLowRankSize", a.kda.decayLowRankSize, e.kdaDecayLowRankSize)
        try check("kda.decayProjectionSize", a.kda.decayProjectionSize,
                  e.kdaDecayProjectionSize)
        try check("kda.gateLowerBound", a.kda.gateLowerBound, e.kdaGateLowerBound)
        try check("kda.fullRankOutputGate", a.kda.fullRankOutputGate,
                  e.kdaFullRankOutputGate)
        try check("mla.numHeads", a.mla.numHeads, e.mlaNumHeads)
        try check("mla.qLoraRank", a.mla.qLoraRank, e.mlaQLoraRank)
        try check("mla.kvLoraRank", a.mla.kvLoraRank, e.mlaKVLoraRank)
        try check("mla.qkNopeHeadDim", a.mla.qkNopeHeadDim, e.mlaQKNopeHeadDim)
        try check("mla.qkRopeHeadDim", a.mla.qkRopeHeadDim, e.mlaQKRopeHeadDim)
        try check("mla.vHeadDim", a.mla.vHeadDim, e.mlaVHeadDim)
        try check("mla.outputGate", a.mla.outputGate, e.mlaOutputGate)
        try check("attnRes.blockSize", a.attnRes.blockSize, e.attnResBlockSize)
        try check("moe.numExperts", a.moe.numExperts, e.moeNumExperts)
        try check("moe.topKExperts", a.moe.topKExperts, e.moeTopKExperts)
        try check("moe.latentBottleneckSize", a.moe.latentBottleneckSize,
                  e.moeLatentBottleneckSize)
        try check("moe.expertIntermediateSize", a.moe.expertIntermediateSize,
                  e.moeExpertIntermediateSize)
        try check("moe.numSharedExperts", a.moe.numSharedExperts, e.moeNumSharedExperts)
        try check("moe.sharedExpertIntermediateSize", a.moe.sharedExpertIntermediateSize,
                  e.moeSharedExpertIntermediateSize)
        try check("moe.situGLUGateBeta", a.moe.situGLUGateBeta, e.situGLUGateBeta)
        try check("moe.situGLUUpBeta", a.moe.situGLUUpBeta, e.situGLUUpBeta)
        try check("moe.routerRenormalize", a.moe.routerRenormalize, e.routerRenormalize)
        try check("moe.routerCorrectionBias", a.moe.routerCorrectionBias,
                  e.routerCorrectionBias)
        try check("numLayers", manifest.numLayers, e.numLayers)
        try check("expertsPerLayer", manifest.expertsPerLayer, e.expertsPerLayer)
        try check("expertStride", manifest.expertStride, e.expertStride)

        // The K3 runtime is built for exactly this flag set; a bundle missing
        // any of them is not a K3 streaming model.
        for flag in ["streamingPresent", "latentMoE", "kdaLayers", "attnRes"] {
            try check("flags.\(flag)", manifest.flags[flag] ?? false, true)
        }

        // Closed runtime quant contract. Reject mixed trunk widths as well as
        // otherwise-valid schemes which the K3 kernels do not implement.
        let q = manifest.quant
        guard q == KimiK3FormatProfile.quantInt4
                || q == KimiK3FormatProfile.quantInt8 else {
            throw ModelError.archMismatch(
                field: "quant",
                expected: "consistent K3 affine4-g64 or affine8-g64 trunk profile",
                actual: "attention=\(q.attention.scheme)/\(q.attention.weightBits), "
                    + "sharedExpert=\(q.sharedExpert.scheme)/\(q.sharedExpert.weightBits), "
                    + "latentProjection=\(q.latentProjection.scheme)/"
                    + "\(q.latentProjection.weightBits), denseMLP="
                    + "\(q.denseMLP.scheme)/\(q.denseMLP.weightBits)")
        }
        return q
    }
}
