import Foundation
import Metal

/// Swift wrappers for `embed_k3.metal`: the int8 embedding row gather
/// (`k3_embed_gather_int8`, fp16 out) and the int8 lm_head GEMV
/// (`k3_lmhead_gemv_int8`, fp32 logits out). Both follow the house K3
/// dispatcher pattern: raw buffers + offsets, a generic pipeline plus a
/// specialized one for the canonical decode shape.
final class K3Embed {
    private static let realD: UInt32 = 7_168
    private static let realConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 93, value: .uint32(realD)),
        MetalFunctionConstant(index: 94, value: .bool(true)),
    ]

    private let pipeline: MTLComputePipelineState
    private let specializedPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.pipeline = try library.pipeline(
            device: context.device, name: "k3_embed_gather_int8")
        self.specializedPipeline = try library.pipeline(
            device: context.device, name: "k3_embed_gather_int8",
            constants: Self.realConstants)
    }

    /// `out[d] = dequant(table[tokenID][d])`, fp16 out, one thread per
    /// element. `table`/`scales`/`biases` are the affine8-g64 embedding view
    /// (pass the resident buffer three times with the view's offsets).
    func encodeGather(commandBuffer: MTLCommandBuffer,
                      table: MTLBuffer, tableOffset: Int = 0,
                      scales: MTLBuffer, scalesOffset: Int = 0,
                      biases: MTLBuffer, biasesOffset: Int = 0,
                      out: MTLBuffer, outOffset: Int = 0,
                      tokenID: UInt32, d: UInt32) {
        precondition(d > 0 && d % UInt32(Quantization.groupSize) == 0,
                     "D must be a positive multiple of \(Quantization.groupSize)")
        precondition(tableOffset >= 0 && scalesOffset % 2 == 0 && biasesOffset % 2 == 0)
        precondition(outOffset % MemoryLayout<Float16>.stride == 0)
        var token = tokenID
        var width = d
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            d == Self.realD ? specializedPipeline : pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 5)
        let threadsPerThreadgroup = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(d) + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}

/// int8 affine g64 lm_head GEMV: `logits[m] = W[m] . x`, fp32 out. Numerics
/// match the house `dequant_int8_gemv_simd` (per-group
/// `fma(s, dot_qx, acc)` / `fma(b, sum_x, acc)`, `simd_sum`), so the K3 head
/// agrees with the v1 path's quantization behavior; only the output dtype
/// differs (K3 keeps logits fp32 — no softcap, fp32 sampler input).
final class K3LMHeadGEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShape = Shape(m: 163_840, n: 7_168)

    private let pipeline: MTLComputePipelineState
    private let specializedPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.pipeline = try library.pipeline(
            device: context.device, name: "k3_lmhead_gemv_int8")
        self.specializedPipeline = try library.pipeline(
            device: context.device, name: "k3_lmhead_gemv_int8",
            constants: [
                MetalFunctionConstant(index: 90, value: .uint32(Self.realDecodeShape.m)),
                MetalFunctionConstant(index: 91, value: .uint32(Self.realDecodeShape.n)),
                MetalFunctionConstant(index: 92, value: .bool(true)),
            ])
    }

    /// Encodes the GEMV. `weights`/`scales`/`biases` are the affine8-g64 head
    /// view; `x` is the fp16 normed hidden state `[n]`; `y` receives fp32
    /// logits `[m]`.
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32) {
        precondition(m > 0)
        precondition(n > 0 && n % UInt32(Quantization.groupSize) == 0,
                     "N must be a positive multiple of \(Quantization.groupSize)")
        precondition(scalesOffset % 2 == 0 && biasesOffset % 2 == 0)
        precondition(xOffset % MemoryLayout<Float16>.stride == 0)
        precondition(yOffset % MemoryLayout<Float>.stride == 0)
        var rows = m
        var cols = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            Shape(m: m, n: n) == Self.realDecodeShape ? specializedPipeline : pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&cols, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + Self.rowsPerThreadgroup - 1)
                        / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
