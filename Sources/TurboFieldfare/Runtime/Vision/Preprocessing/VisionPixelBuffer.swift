import Metal

public struct VisionImagePlan {
    public let metadata: VisionImageMetadata
    public let geometry: Gemma4ImageGeometry
    package let opened: OpenedVisionImage
    package let started: ContinuousClock.Instant
}

public struct VisionPixelBuffer {
    public let patchesBF16: MTLBuffer
    public let positionsInt32x2: MTLBuffer
    public let metadata: VisionImageMetadata
    public let geometry: Gemma4ImageGeometry
    public let wallNanoseconds: UInt64
    public let allocatedBytes: Int
}
