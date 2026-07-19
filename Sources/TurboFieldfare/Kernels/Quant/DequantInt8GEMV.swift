import Foundation
import Metal

/// Swift wrapper for `dequant_int8_gemv_simd`.
///
///   y = W * x
///
/// where W (M rows, N cols) is MLX-affine 8-bit (group=64) with BF16 scale +
/// BF16 bias per group, x is FP16 [N], y is FP16 [M].
///
/// Same kernel, two call sites:
///   * router  — M=128,    N=2816 (one per layer)
///   * lm_head — M=262144, N=2816 (final classifier, tied with embed table)
///
/// One SIMD group (32 threads) runs per output row.
final class DequantInt8GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 128, n: 2816),     // router.proj
        Shape(m: 2112, n: 2816),    // shared expert gate/up
        Shape(m: 2816, n: 2112),    // shared expert down
    ]

    init(context: MetalContext) throws {
        let kernelName = "dequant_int8_gemv_simd"
        self.pso = try context.pipeline(kernelName)

        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                kernelName,
                constants: [
                    MetalFunctionConstant(index: 70, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 71, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 72, value: .bool(true)),
                ])
        }
        self.specializedPSOs = variants
    }

    /// Encodes the GEMV onto `commandBuffer`. Offsets allow passing the same
    /// mmap'd `MTLBuffer` for weights / scales / biases when they share a
    /// page-aligned blob.
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales:  MTLBuffer, scalesOffset:  Int = 0,
                       biases:  MTLBuffer, biasesOffset:  Int = 0,
                       x:       MTLBuffer,
                       xOffset: Int = 0,
                       y:       MTLBuffer,
                       yOffset: Int = 0,
                       m: UInt32,
                       n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(xOffset >= 0 && yOffset >= 0, "buffer offsets must be non-negative")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(specializedPSOs[Shape(m: m, n: n)] ?? pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales,  offset: scalesOffset,  index: 1)
        enc.setBuffer(biases,  offset: biasesOffset,  index: 2)
        enc.setBuffer(x,       offset: xOffset,       index: 3)
        enc.setBuffer(y,       offset: yOffset,       index: 4)
        var mVar = m
        var nVar = n
        enc.setBytes(&mVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)

        let rowsPerTG = 8
        let tgSize  = MTLSize(width: 32 * rowsPerTG, height: 1, depth: 1)
        let tgCount = MTLSize(width: (Int(m) + rowsPerTG - 1) / rowsPerTG,
                              height: 1, depth: 1)
        enc.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
