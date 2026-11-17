import Foundation
import TinyTitanCore

extension ArchConfig {
    /// Poolside Laguna-S-2.1, 118B-A8B. Geometry from the checkpoint's
    /// `config.json` at `0f573140834b11cfac0c2af97a101a7a69a13e22`; see
    /// `docs/plan-laguna-s-2.1.md`.
    ///
    /// `numHeads` states the *global* layer count (48); the sliding layers'
    /// 72 is in `laguna.headsPerLayer`, because a single scalar cannot
    /// describe both and silently using 48 on a sliding layer would read the
    /// q_proj at the wrong width.
    public static let lagunaS21 = ArchConfig(
        hiddenSize: 3072,
        intermediateSize: 12_288,
        moeIntermediateSize: 1024,
        numHeads: 48,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 128,
        fullHeadDim: 128,
        vocabSize: 100_352,
        slidingWindow: 512,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 500_000.0,
        partialRotaryFactor: 0.5,
        numLayers: 48,
        numExperts: 256,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.lagunaLayerMask(),
        hiddenActivation: "silu",
        family: "laguna",
        attnOutputGate: true,
        attentionScale: 1.0 / Double(128).squareRoot(),
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        routerNormTopK: true,
        quantGroupSize: 64,
        normGroups: 1,
        attentionGate: .softplus,
        routerHasBias: true,
        routerScaling: 2.5,
        denseLayerIndices: [0],
        mova: .none,
        perLayerHeadGeometry: PerLayerHeadGeometry(
            headsPerLayer: Self.lagunaHeads(),
            qkNorm: true,
            gatePerHead: true,
            swaPartialRotaryFactor: 1.0,
            yarnFactor: 128,
            yarnOriginalMax: 8192,
            yarnBetaFast: 32,
            yarnBetaSlow: 1,
            yarnAttentionFactor: 1.4852030263919618))

    /// 12 global-attention layers, every fourth layer starting at 0: the
    /// checkpoint's 1:3 global-to-sliding ratio.
    private static func lagunaLayerMask() -> [UInt8] {
        (0..<48).map { $0 % 4 == 0 ? 1 : 0 }
    }

    private static func lagunaHeads() -> [Int] {
        (0..<48).map { $0 % 4 == 0 ? 48 : 72 }
    }

}
