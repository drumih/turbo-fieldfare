import Foundation

public struct VisionConfig: Sendable, Equatable {
    public let hiddenSize = 1_152
    public let intermediateSize = 4_304
    public let numLayers = 27
    public let numHeads = 16
    public let headDimension = 72
    public let patchSize = 16
    public let patchDimension = 768
    public let maximumPatches = 2_520
    public let poolingKernel = 3
    public let textHiddenSize = 2_816
    public let positionEmbeddingSize = 10_240
    public let rmsEpsilon: Float = 1e-6
    public let ropeTheta: Float = 100

    public var maximumPooledTokens: Int {
        maximumPatches / (poolingKernel * poolingKernel)
    }

    public init() {}
}
