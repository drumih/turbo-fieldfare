import Foundation
import Metal

/// MXFP4 (E2M1 nibbles + E8M0 group-of-32 scales) matrix-vector products for
/// the K3 decode path. Dequant matches `QuantizationMXFP4` bit-for-bit; the
/// GEMVs accumulate in FP32, one SIMD per row, eight rows per threadgroup —
/// the `DequantInt4GEMV` shape.
final class K3MXFP4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    /// Canonical K3 LatentMoE expert shapes: w1/w3 are 3072x3584, w2 is
    /// 3584x3072.
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 3072, n: 3584),
        Shape(m: 3584, n: 3072),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]
    private let pipelineF32: MTLComputePipelineState
    private let specializedPipelinesF32: [Shape: MTLComputePipelineState]

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.pipeline = try library.pipeline(
            device: context.device, name: "k3_mxfp4_gemv",
            maxTotalThreadsPerThreadgroup: 512)
        self.pipelineF32 = try library.pipeline(
            device: context.device, name: "k3_mxfp4_gemv_f32",
            maxTotalThreadsPerThreadgroup: 512)

        var specialized: [Shape: MTLComputePipelineState] = [:]
        var specializedF32: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            let constants: [MetalFunctionConstant] = [
                MetalFunctionConstant(index: 50, value: .uint32(shape.m)),
                MetalFunctionConstant(index: 51, value: .uint32(shape.n)),
                MetalFunctionConstant(index: 52, value: .bool(true)),
            ]
            specialized[shape] = try library.pipeline(
                device: context.device, name: "k3_mxfp4_gemv",
                constants: constants, maxTotalThreadsPerThreadgroup: 512)
            specializedF32[shape] = try library.pipeline(
                device: context.device, name: "k3_mxfp4_gemv_f32",
                constants: constants, maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specialized
        self.specializedPipelinesF32 = specializedF32
    }

    /// FP16 in/out variant (trunk-style activations). `y[m] = W x` with FP32
    /// accumulation. `n` must be a multiple of 32 (one E8M0 scale per group).
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32) {
        Self.validate(m: m, n: n, weightsOffset: weightsOffset,
                      xOffset: xOffset, xStride: MemoryLayout<Float16>.stride,
                      yOffset: yOffset, yStride: MemoryLayout<Float16>.stride)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        Self.bind(encoder: encoder,
                  weights: weights, weightsOffset: weightsOffset,
                  scales: scales, scalesOffset: scalesOffset,
                  x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        Self.dispatch(encoder: encoder, m: m,
                      rowsPerThreadgroup: Self.rowsPerThreadgroup)
        encoder.endEncoding()
    }

    /// FP32 in/out variant (LatentMoE phase-2-style activations and exact
    /// parity tests). Same dequant and FP32 accumulation.
    func encodeF32(commandBuffer: MTLCommandBuffer,
                   weights: MTLBuffer, weightsOffset: Int = 0,
                   scales: MTLBuffer, scalesOffset: Int = 0,
                   x: MTLBuffer, xOffset: Int = 0,
                   y: MTLBuffer, yOffset: Int = 0,
                   m: UInt32, n: UInt32) {
        Self.validate(m: m, n: n, weightsOffset: weightsOffset,
                      xOffset: xOffset, xStride: MemoryLayout<Float>.stride,
                      yOffset: yOffset, yStride: MemoryLayout<Float>.stride)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelinesF32[Shape(m: m, n: n)] ?? pipelineF32)
        Self.bind(encoder: encoder,
                  weights: weights, weightsOffset: weightsOffset,
                  scales: scales, scalesOffset: scalesOffset,
                  x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: m, n: n)
        Self.dispatch(encoder: encoder, m: m,
                      rowsPerThreadgroup: Self.rowsPerThreadgroup)
        encoder.endEncoding()
    }

    private static func validate(m: UInt32, n: UInt32,
                                 weightsOffset: Int,
                                 xOffset: Int, xStride: Int,
                                 yOffset: Int, yStride: Int) {
        precondition(m > 0)
        precondition(n > 0 && n % 32 == 0,
                     "N must be a positive multiple of 32 (MXFP4 group)")
        // The kernels read packed weights through `ushort*`; the v2 format
        // guarantees even sub-tensor offsets inside an expert blob, but the
        // buffer base the caller binds at must stay 2-aligned too.
        precondition(weightsOffset % 2 == 0,
                     "MXFP4 GEMV needs a 2-aligned weightsOffset, got \(weightsOffset)")
        precondition(xOffset % xStride == 0 && yOffset % yStride == 0)
    }

    private static func bind(encoder: MTLComputeCommandEncoder,
                             weights: MTLBuffer, weightsOffset: Int,
                             scales: MTLBuffer, scalesOffset: Int,
                             x: MTLBuffer, xOffset: Int,
                             y: MTLBuffer, yOffset: Int,
                             m: UInt32, n: UInt32) {
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(y, offset: yOffset, index: 3)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 5)
    }

    private static func dispatch(encoder: MTLComputeCommandEncoder,
                                 m: UInt32, rowsPerThreadgroup: Int) {
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerThreadgroup,
                                           height: 1, depth: 1))
    }
}

/// Standalone MXFP4 dequant to FP32, one thread per element. Exists for the
/// bit-exact kernel-vs-`QuantizationMXFP4` tests and for debug tooling; never
/// on the decode hot path (experts are consumed packed).
final class K3MXFP4Dequant {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try K3MetalLibrary.shared.pipeline(
            device: context.device, name: "k3_mxfp4_dequant")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                m: UInt32, n: UInt32) {
        precondition(m > 0)
        precondition(n > 0 && n % 32 == 0,
                     "N must be a positive multiple of 32 (MXFP4 group)")
        precondition(outOffset % MemoryLayout<Float>.stride == 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 4)
        let total = Int(m) * Int(n)
        let threadsPerThreadgroup = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (total + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
