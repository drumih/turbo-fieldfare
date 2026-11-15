import Foundation
import Metal
import TinyTitanCore

/// Affine-quantized weights against a chunk of tokens, for prefill.
///
/// `AffineQuantGEMV` reads a row's weights for every token it serves; this
/// keeps the row in registers and loops over the chunk, so the weight bytes are
/// read once per chunk instead of once per token. The lane mapping matches the
/// coalesced GEMV (a 64-element group covered by `64/elementsPerWord` lanes,
/// one word each), which is what makes `tokens == 1` agree with the GEMV path.
///
/// `y` is `[tokens][rows]`, token-major, and `x` is `[tokens][columns]`.
final class AffineQuantGEMM {
    let weightBits: Int
    private let pipeline: MTLComputePipelineState

    /// The kernel holds the row in registers, so its length is bounded.
    static let maximumIterations = 32

    init(context: MetalContext, weightBits: Int) throws {
        precondition(
            weightBits == 4 || weightBits == 8,
            "the affine GEMM decodes 4- or 8-bit weights, not \(weightBits)")
        self.weightBits = weightBits
        self.pipeline = try context.pipeline(
            "affine_quant_gemm_simd",
            constants: [
                MetalFunctionConstant(
                    index: 100, value: .uint32(UInt32(weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: 256)
    }

    /// Whether this shape fits the register-resident row. A 64-element group is
    /// covered by `32 / wordsPerGroup` lanes, so the row's group count must
    /// divide by that; a caller with an unsupported shape uses the per-token
    /// GEMV instead. The kernel's own `iterations` guard would otherwise turn
    /// the dispatch into a silent no-op.
    static func canEncode(
        rows: Int, columns: Int, tokens: Int, weightBits: Int
    ) -> Bool {
        guard rows > 0, tokens > 0, columns > 0,
            columns.isMultiple(of: Quantization.groupSize),
            weightBits == 4 || weightBits == 8
        else { return false }
        let wordsPerGroup = 2 * weightBits
        let groupsPerIteration = 32 / wordsPerGroup
        let groups = columns / Quantization.groupSize
        guard groupsPerIteration > 0,
            groups.isMultiple(of: groupsPerIteration)
        else { return false }
        return groups / groupsPerIteration <= maximumIterations
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        rows: UInt32, columns: UInt32, tokens: UInt32
    ) throws {
        precondition(
            Self.canEncode(
                rows: Int(rows), columns: Int(columns), tokens: Int(tokens),
                weightBits: weightBits),
            "affine GEMM shape \(rows)x\(columns)x\(tokens) is outside the "
                + "register-resident row for \(weightBits)-bit weights")
        precondition(weightsOffset.isMultiple(of: MemoryLayout<UInt32>.alignment))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var rowCount = rows
        var columnCount = columns
        var tokenCount = tokens
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(rows) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
