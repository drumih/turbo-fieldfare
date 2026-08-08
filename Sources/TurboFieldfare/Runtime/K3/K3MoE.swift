import Foundation
import Metal

/// Byte offsets of the six MXFP4 sub-tensors inside one packed K3 expert
/// blob, in the v2 schema order
/// (`w1_packed, w1_scales, w2_packed, w2_scales, w3_packed, w3_scales`).
/// Layout must match `K3ExpertOffsets` in moe_k3.metal; passed via setBytes.
@frozen
public struct K3ExpertSubtensorOffsets {
    public var w1PackedOff: UInt32
    public var w1ScalesOff: UInt32
    public var w2PackedOff: UInt32
    public var w2ScalesOff: UInt32
    public var w3PackedOff: UInt32
    public var w3ScalesOff: UInt32

    public init(w1PackedOff: UInt32, w1ScalesOff: UInt32,
                w2PackedOff: UInt32, w2ScalesOff: UInt32,
                w3PackedOff: UInt32, w3ScalesOff: UInt32) {
        self.w1PackedOff = w1PackedOff
        self.w1ScalesOff = w1ScalesOff
        self.w2PackedOff = w2PackedOff
        self.w2ScalesOff = w2ScalesOff
        self.w3PackedOff = w3PackedOff
        self.w3ScalesOff = w3ScalesOff
    }

    /// Offsets for the canonical blob layout: w1 (F x D), w2 (D x F), w3
    /// (F x D), each `packed` (rows x cols/2 bytes) immediately followed by
    /// `scales` (rows x cols/32 bytes). `dLatent` and `intermediate` must be
    /// multiples of 32.
    public static func canonical(dLatent: UInt32, intermediate: UInt32)
        -> K3ExpertSubtensorOffsets
    {
        precondition(dLatent > 0 && dLatent % 32 == 0)
        precondition(intermediate > 0 && intermediate % 32 == 0)
        let w1Packed = UInt64(intermediate) * UInt64(dLatent) / 2
        let w1Scales = UInt64(intermediate) * UInt64(dLatent) / 32
        let w2Packed = UInt64(dLatent) * UInt64(intermediate) / 2
        let w2Scales = UInt64(dLatent) * UInt64(intermediate) / 32
        let w3Packed = w1Packed
        let w3Scales = w1Scales
        var off = UInt64(0)
        let w1p = off; off += w1Packed
        let w1s = off; off += w1Scales
        let w2p = off; off += w2Packed
        let w2s = off; off += w2Scales
        let w3p = off; off += w3Packed
        let w3s = off; off += w3Scales
        precondition(off <= UInt64(UInt32.max),
                     "expert blob \(off) bytes exceeds UInt32 sub-tensor offsets")
        return K3ExpertSubtensorOffsets(
            w1PackedOff: UInt32(w1p), w1ScalesOff: UInt32(w1s),
            w2PackedOff: UInt32(w2p), w2ScalesOff: UInt32(w2s),
            w3PackedOff: UInt32(w3p), w3ScalesOff: UInt32(w3s))
    }

    /// Total expert blob size implied by the canonical layout (the expert
    /// stride). 17,547,264 bytes for the canonical K3 shape.
    public static func canonicalBlobSize(dLatent: UInt32, intermediate: UInt32)
        -> UInt64
    {
        let elements = UInt64(dLatent) * UInt64(intermediate)
        // Three matrices; each costs (1/2 + 1/32) bytes per element.
        return 3 * (elements / 2 + elements / 32)
    }
}

/// Fused K3 LatentMoE decode for the top-k (canonical 16) routed experts.
///
/// Phase 1 computes, per selected expert, the fused w1/w3 MXFP4 GEMV against
/// the fp16 latent vector and applies SiTU in FP32, writing `h` (FP32,
/// `[topK, intermediate]`). Phase 2 runs the w2 MXFP4 GEMV per expert and
/// reduces the k expert outputs with the FP32 router weights into `y_lat`
/// (FP32, `[dLatent]`). There is no residual add in the K3 routed latent
/// path.
///
/// All selected expert blobs live in one buffer; `slotOffsets` is a device
/// buffer of `topK` UInt64 byte offsets into it (a full layer's expert
/// region exceeds 4 GB, hence 64-bit). One launch covers all k experts so
/// the SSD reads for the token share a single command buffer.
final class K3MoE {
    static let maxTopK = 16

    private static let rowsPerThreadgroup = 8
    private static let realDecodeDLatent: UInt32 = 3584
    private static let realDecodeIntermediate: UInt32 = 3072
    private static let realDecodeTopK: UInt32 = 16
    private static let realDecodeConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 53, value: .uint32(realDecodeDLatent)),
        MetalFunctionConstant(index: 54, value: .uint32(realDecodeIntermediate)),
        MetalFunctionConstant(index: 55, value: .uint32(realDecodeTopK)),
        MetalFunctionConstant(index: 56, value: .bool(true)),
    ]

    private let phase1PSO: MTLComputePipelineState
    private let phase1SpecializedPSO: MTLComputePipelineState
    private let phase2PSO: MTLComputePipelineState
    private let phase2SpecializedPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.phase1PSO = try library.pipeline(
            device: context.device, name: "k3_moe_phase1_situ",
            maxTotalThreadsPerThreadgroup: 512)
        self.phase1SpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_moe_phase1_situ",
            constants: Self.realDecodeConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.phase2PSO = try library.pipeline(
            device: context.device, name: "k3_moe_phase2_reduce",
            maxTotalThreadsPerThreadgroup: 512)
        self.phase2SpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_moe_phase2_reduce",
            constants: Self.realDecodeConstants,
            maxTotalThreadsPerThreadgroup: 512)
    }

    /// Phase 1: `h[k*F + f] = SiTU(w1_k[f] . x_lat, w3_k[f] . x_lat)`.
    /// `x_lat` is FP16 `[dLatent]`; `h` is FP32 `[topK * intermediate]`.
    func encodePhase1(commandBuffer: MTLCommandBuffer,
                      experts: MTLBuffer, expertsOffset: Int = 0,
                      slotOffsets: MTLBuffer, slotOffsetsOffset: Int = 0,
                      xLat: MTLBuffer, xLatOffset: Int = 0,
                      h: MTLBuffer, hOffset: Int = 0,
                      subtensorOffsets: K3ExpertSubtensorOffsets,
                      dLatent: UInt32, intermediate: UInt32, topK: UInt32) {
        Self.validate(dLatent: dLatent, intermediate: intermediate, topK: topK,
                      expertsOffset: expertsOffset)
        precondition(xLatOffset % MemoryLayout<Float16>.stride == 0)
        precondition(hOffset % MemoryLayout<Float>.stride == 0)
        var dimension = dLatent
        var inter = intermediate
        var expertCount = topK
        var offsets = subtensorOffsets
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(dLatent: dLatent, intermediate: intermediate,
                                   topK: topK)
                ? phase1SpecializedPSO : phase1PSO)
        encoder.setBuffer(experts, offset: expertsOffset, index: 0)
        encoder.setBuffer(slotOffsets, offset: slotOffsetsOffset, index: 1)
        encoder.setBuffer(xLat, offset: xLatOffset, index: 2)
        encoder.setBuffer(h, offset: hOffset, index: 3)
        encoder.setBytes(&offsets, length: MemoryLayout<K3ExpertSubtensorOffsets>.stride,
                         index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&inter, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 7)
        let rows = Int(topK * intermediate)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Phase 2: `y_lat[d] = sum_k routingWeights[k] * (w2_k[d] . h_k)`.
    /// `h` is the FP32 `[topK * intermediate]` phase-1 output; `y_lat` is
    /// FP32 `[dLatent]`. One threadgroup per output row, one SIMD per expert.
    func encodePhase2(commandBuffer: MTLCommandBuffer,
                      experts: MTLBuffer, expertsOffset: Int = 0,
                      slotOffsets: MTLBuffer, slotOffsetsOffset: Int = 0,
                      h: MTLBuffer, hOffset: Int = 0,
                      routingWeights: MTLBuffer, routingWeightsOffset: Int = 0,
                      yLat: MTLBuffer, yLatOffset: Int = 0,
                      subtensorOffsets: K3ExpertSubtensorOffsets,
                      dLatent: UInt32, intermediate: UInt32, topK: UInt32) {
        Self.validate(dLatent: dLatent, intermediate: intermediate, topK: topK,
                      expertsOffset: expertsOffset)
        precondition(hOffset % MemoryLayout<Float>.stride == 0)
        precondition(routingWeightsOffset % MemoryLayout<Float>.stride == 0)
        precondition(yLatOffset % MemoryLayout<Float>.stride == 0)
        var dimension = dLatent
        var inter = intermediate
        var expertCount = topK
        var offsets = subtensorOffsets
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(dLatent: dLatent, intermediate: intermediate,
                                   topK: topK)
                ? phase2SpecializedPSO : phase2PSO)
        encoder.setBuffer(experts, offset: expertsOffset, index: 0)
        encoder.setBuffer(slotOffsets, offset: slotOffsetsOffset, index: 1)
        encoder.setBuffer(h, offset: hOffset, index: 2)
        encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        encoder.setBuffer(yLat, offset: yLatOffset, index: 4)
        encoder.setBytes(&offsets, length: MemoryLayout<K3ExpertSubtensorOffsets>.stride,
                         index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&inter, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(dLatent), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Int(topK),
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func validate(dLatent: UInt32, intermediate: UInt32,
                                 topK: UInt32, expertsOffset: Int) {
        precondition(topK >= 1 && topK <= UInt32(maxTopK),
                     "topK \(topK) outside 1...\(maxTopK)")
        precondition(dLatent > 0 && dLatent % 32 == 0,
                     "dLatent must be a positive multiple of 32 (MXFP4 group)")
        precondition(intermediate > 0 && intermediate % 32 == 0,
                     "intermediate must be a positive multiple of 32 (MXFP4 group)")
        // Packed sub-tensors are read through `ushort*`; expert base +
        // sub-tensor offset must stay 2-aligned.
        precondition(expertsOffset % 2 == 0,
                     "K3MoE needs a 2-aligned expertsOffset, got \(expertsOffset)")
    }

    private func useRealDecodeConstants(dLatent: UInt32, intermediate: UInt32,
                                        topK: UInt32) -> Bool {
        dLatent == Self.realDecodeDLatent
            && intermediate == Self.realDecodeIntermediate
            && topK == Self.realDecodeTopK
    }
}
