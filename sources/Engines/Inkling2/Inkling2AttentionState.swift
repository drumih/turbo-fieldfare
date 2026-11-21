import Foundation

/// Per-layer attention continuation state: the post-convolution K/V for every
/// token already seen, the convolution carry state, and the absolute position.
public struct Inkling2AttentionState: Sendable {
    public var k: [Float]
    public var v: [Float]
    public var keys: [Float]
    public var values: [Float]
    public var position: Int
}
