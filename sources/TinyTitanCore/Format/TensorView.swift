import Foundation
import Metal

/// unchecked-invariant: every stored property is `let`; the view is an
/// immutable window (offset/length/shape) over an MTLBuffer whose bytes are
/// read-only for the view's lifetime. @unchecked only because MTLBuffer is a
/// reference type; the value itself carries no mutable state.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(
        buffer: MTLBuffer,
        offset: UInt64, length: UInt64,
        scaleOffset: UInt64, scaleLength: UInt64,
        biasOffset: UInt64, biasLength: UInt64,
        shape: (UInt32, UInt32, UInt32, UInt32),
        dtype: UInt8
    ) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
