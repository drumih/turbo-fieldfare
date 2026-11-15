import Foundation
import TinyTitanCore

/// The parts of Laguna-S-2.1 that are not expressible in the cross-family
/// `ArchConfig` fields: the per-layer head counts, the QK norms, the per-head
/// gate, and the two RoPE dialects.
///
/// It is a *checked preset* rather than a manifest block: the runtime supports
/// one Laguna checkpoint, so the geometry is stated once here and the loader
/// refuses a manifest whose generic fields (hidden, layers, heads, KV heads,
/// head dim, window, both thetas, the layer mask) differ. `.none` describes
/// every other family.
public struct LagunaGeometry: Sendable, Equatable {
    /// Query heads per layer. 48 on the 12 global layers, 72 on the 36
    /// sliding-window layers; the KV side stays at 8.
    public let headsPerLayer: [Int]
    /// A per-head RMS norm on q and k, before RoPE.
    public let qkNorm: Bool
    /// `g_proj` emits one value per head, broadcast across the head's 128
    /// channels, instead of one value per channel.
    public let gatePerHead: Bool
    /// Rotary fraction on sliding-window layers (global layers use
    /// `ArchConfig.partialRotaryFactor`).
    public let swaPartialRotaryFactor: Double
    /// YaRN on the global layers: the config gives the factor, the original
    /// context and the two beta corrections outright.
    public let yarnFactor: Double
    public let yarnOriginalMax: Int
    public let yarnBetaFast: Double
    public let yarnBetaSlow: Double
    public let yarnAttentionFactor: Double

    public init(
        headsPerLayer: [Int],
        qkNorm: Bool,
        gatePerHead: Bool,
        swaPartialRotaryFactor: Double,
        yarnFactor: Double,
        yarnOriginalMax: Int,
        yarnBetaFast: Double,
        yarnBetaSlow: Double,
        yarnAttentionFactor: Double
    ) {
        self.headsPerLayer = headsPerLayer
        self.qkNorm = qkNorm
        self.gatePerHead = gatePerHead
        self.swaPartialRotaryFactor = swaPartialRotaryFactor
        self.yarnFactor = yarnFactor
        self.yarnOriginalMax = yarnOriginalMax
        self.yarnBetaFast = yarnBetaFast
        self.yarnBetaSlow = yarnBetaSlow
        self.yarnAttentionFactor = yarnAttentionFactor
    }

    public static let none = LagunaGeometry(
        headsPerLayer: [], qkNorm: false, gatePerHead: false,
        swaPartialRotaryFactor: 1, yarnFactor: 1, yarnOriginalMax: 0,
        yarnBetaFast: 32, yarnBetaSlow: 1, yarnAttentionFactor: 1)

    public var enabled: Bool { !headsPerLayer.isEmpty }
}

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
        family: .laguna,
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
        laguna: LagunaGeometry(
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
