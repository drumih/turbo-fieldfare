import Foundation

/// The parts of the per-layer-head family that are not expressible in the cross-family
/// `ArchConfig` fields: the per-layer head counts, the QK norms, the per-head
/// gate, and the two RoPE dialects.
///
/// It is a *checked preset* rather than a manifest block: the runtime supports
/// one such checkpoint, so the geometry is stated once here and the loader
/// refuses a manifest whose generic fields (hidden, layers, heads, KV heads,
/// head dim, window, both thetas, the layer mask) differ. `.none` describes
/// every other family.
public struct PerLayerHeadGeometry: Sendable, Equatable {
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

    public static let none = PerLayerHeadGeometry(
        headsPerLayer: [], qkNorm: false, gatePerHead: false,
        swaPartialRotaryFactor: 1, yarnFactor: 1, yarnOriginalMax: 0,
        yarnBetaFast: 32, yarnBetaSlow: 1, yarnAttentionFactor: 1)

    public var enabled: Bool { !headsPerLayer.isEmpty }
}
