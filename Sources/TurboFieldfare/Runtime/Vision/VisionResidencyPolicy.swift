public enum VisionResidencyPolicy: String, Sendable, CaseIterable, Codable {
    case onDemand = "on-demand"
    case keepReady = "keep-ready"

    /// The default every entry point uses. It is one constant because the four
    /// entry points — runtime, conversation, session, CLI — each wrote their own
    /// default, and one of them drifted to `keepReady`, so a conversation held
    /// 1.1 GB of tower mappings open for the life of the process while the CLI
    /// released them. Measured, keeping them costs ~5 MB of footprint and saves
    /// ~105 ms of remapping, which is not worth pinning by default.
    public static let defaultPolicy: VisionResidencyPolicy = .onDemand
}

public struct VisionExpertResidencyTransition: Sendable, Equatable {
    public let policy: VisionResidencyPolicy
    public let gpuDrainNanoseconds: UInt64
    public let releasedLayerCount: Int
    public let releasedSlotScratchBytes: UInt64
    public let remainingOpenLayerCount: Int
    public let remainingSlotScratchBytes: UInt64
    public let wallNanoseconds: UInt64
}
