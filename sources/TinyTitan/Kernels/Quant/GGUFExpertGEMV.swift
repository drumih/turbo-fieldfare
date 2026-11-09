import Foundation
import Metal

/// One entry point for the as-is expert math: the caller says which GGUF block
/// type a matrix is in, and the matching fused decode-and-dot kernel runs.
///
/// The five formats are the ones the checkpoint's routed experts use. A type
/// with no fused kernel is refused by name rather than routed to a kernel that
/// would read the bytes as the wrong layout - which is silent nonsense, not a
/// crash.
package final class GGUFExpertGEMV {
    private let iq2s: IQ2SGEMV
    private let iq3xxs: IQ3XXSGEMV
    private let iq4nl: IQ4NLGEMV
    private let other: IQ4XSQ2GEMV

    package init(context: MetalContext) throws {
        self.iq2s = try IQ2SGEMV(context: context)
        self.iq3xxs = try IQ3XXSGEMV(context: context)
        self.iq4nl = try IQ4NLGEMV(context: context)
        self.other = try IQ4XSQ2GEMV(context: context)
    }

    package func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int,
        type: GGUFType
    ) throws {
        switch type {
        case .iq2S:
            try iq2s.encode(
                commandBuffer: commandBuffer, weights: weights, weightsOffset: weightsOffset,
                x: x, y: y, rows: rows, columns: columns)
        case .iq3XXS:
            try iq3xxs.encode(
                commandBuffer: commandBuffer, weights: weights, weightsOffset: weightsOffset,
                x: x, y: y, rows: rows, columns: columns)
        case .iq4NL:
            try iq4nl.encode(
                commandBuffer: commandBuffer, weights: weights, weightsOffset: weightsOffset,
                x: x, y: y, rows: rows, columns: columns)
        case .iq4XS, .q2_0:
            try other.encode(
                commandBuffer: commandBuffer, weights: weights, weightsOffset: weightsOffset,
                x: x, y: y, rows: rows, columns: columns, type: type)
        default:
            throw MetalError.invalidState(
                "no fused GEMV for \(type.name); the expert path has no kernel that reads it")
        }
    }
}
