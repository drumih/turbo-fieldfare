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

/// Identifies the pipeline that turns an attached file into the tower's input.
///
/// A stored conversation keeps the post-resize pixels, so replay does not
/// re-run decode, orientation, compositing or resampling — but it does re-run
/// patchification and normalization. Bump this whenever a change to any stage
/// would give the same stored file different soft tokens; a conversation
/// recorded under an older version then fails identity and stays readable
/// instead of continuing on tokens this build would not have produced.
public enum VisionImageProcessing {
    public static let version = 1
}
