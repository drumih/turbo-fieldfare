import Foundation

public struct ManifestIdentity: Equatable, Sendable {
    public let modelID: String
    public let family: ModelFamily
    /// Routed-expert width, which is what "a 4-bit model" names: the experts
    /// are almost all of the payload.
    public let weightBits: Int
}
