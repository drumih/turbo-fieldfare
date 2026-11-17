import Foundation

/// The dense CPU engine's geometry: every dimension the affine engine reads
/// from the snapshot's config, promoted from `AffineSnapshot.Configuration`
/// (byte-identical copies in the runtime and every dense variant) because the
/// TT-097 `CPUEngineRunner` contract exposes it. Pure value struct — model-
/// free by construction, which `core-purity` enforces.
public struct CPUEngineConfiguration: Sendable, Equatable {
    public let hiddenSize: Int
    public let layers: Int
    public let heads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let fullAttentionInterval: Int
    public let linearKeyHeads: Int
    public let linearValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let convKernel: Int
    public let intermediateSize: Int
    public let vocabulary: Int
    public let normEpsilon: Float
    public let ropeTheta: Float
    public let partialRotaryFactor: Double
    public let tiedEmbedding: Bool
    /// The context the checkpoint claims. A CPU engine will not want all
    /// of it -- attention here is a plain loop over the cache -- but the
    /// ceiling belongs to the model, not to the server.
    public let maxPositions: Int

    public init(
        hiddenSize: Int,
        layers: Int,
        heads: Int,
        keyValueHeads: Int,
        headDim: Int,
        fullAttentionInterval: Int,
        linearKeyHeads: Int,
        linearValueHeads: Int,
        linearKeyHeadDim: Int,
        linearValueHeadDim: Int,
        convKernel: Int,
        intermediateSize: Int,
        vocabulary: Int,
        normEpsilon: Float,
        ropeTheta: Float,
        partialRotaryFactor: Double,
        tiedEmbedding: Bool,
        maxPositions: Int
    ) {
        self.hiddenSize = hiddenSize
        self.layers = layers
        self.heads = heads
        self.keyValueHeads = keyValueHeads
        self.headDim = headDim
        self.fullAttentionInterval = fullAttentionInterval
        self.linearKeyHeads = linearKeyHeads
        self.linearValueHeads = linearValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.convKernel = convKernel
        self.intermediateSize = intermediateSize
        self.vocabulary = vocabulary
        self.normEpsilon = normEpsilon
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.tiedEmbedding = tiedEmbedding
        self.maxPositions = maxPositions
    }

    /// Dimensions of the rotation, which is partial here: 64 of 256.
    /// Rotating the whole head is the single most plausible way to get a
    /// model that runs and is quietly wrong.
    public var rotaryDim: Int { Int(Double(headDim) * partialRotaryFactor) }

    /// Whether this layer is full attention rather than Gated DeltaNet.
    /// Every fourth, counting from the end of each group.
    public func isAttention(_ layer: Int) -> Bool {
        (layer + 1) % fullAttentionInterval == 0
    }
}
