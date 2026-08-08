import Foundation
import Metal

/// K3 MLA absorbed decode attention over the latent cache. Stages per
/// token: `encodeCacheAppend` (norm + append the current token), then
/// `encodeAbsorbQ` (q̃ per head), `encodeAttnDecode` (split-KV partial +
/// combine over the fp16 latent cache), and `encodeOutProject` (value-side
/// expansion + sigmoid output gate, fp16 out for the o_proj trunk GEMV).
/// `encodeKVBExpand` is a per-model-load layout conversion, not per token.
///
/// The split-KV partial scratch is owned here (mirrors `Attention`): the
/// decode passes run once per layer, serially, inside one command buffer.
/// The cache buffer itself is owned by the caller (Stage-C state manager).
final class K3MLA {
    static let maxNumHeads = 96
    static let maxLatent = 512
    static let maxRope = 64
    static let maxHeadDim = 128        // max(nope, vHead)
    static let maxChunks = 64
    private static let defaultChunks = 16

    private static let realNumHeads: UInt32 = 96
    private static let realLatent: UInt32 = 512
    private static let realRope: UInt32 = 64
    private static let realNope: UInt32 = 128
    private static let realVHead: UInt32 = 128
    private static let realConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 74, value: .uint32(realNumHeads)),
        MetalFunctionConstant(index: 75, value: .uint32(realLatent)),
        MetalFunctionConstant(index: 76, value: .uint32(realRope)),
        MetalFunctionConstant(index: 77, value: .uint32(realNope)),
        MetalFunctionConstant(index: 78, value: .uint32(realVHead)),
        MetalFunctionConstant(index: 79, value: .bool(true)),
    ]

    private let expandPSO: MTLComputePipelineState
    private let absorbQPSO: MTLComputePipelineState
    private let absorbQSpecializedPSO: MTLComputePipelineState
    private let partialPSO: MTLComputePipelineState
    private let partialSpecializedPSO: MTLComputePipelineState
    private let combinePSO: MTLComputePipelineState
    private let combineSpecializedPSO: MTLComputePipelineState
    private let outProjectPSO: MTLComputePipelineState
    private let outProjectSpecializedPSO: MTLComputePipelineState
    private let cacheAppendPSO: MTLComputePipelineState
    private let cacheAppendSpecializedPSO: MTLComputePipelineState

    private let mPartial: MTLBuffer
    private let dPartial: MTLBuffer
    private let oPartial: MTLBuffer

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.expandPSO = try library.pipeline(
            device: context.device, name: "k3_mla_kvb_expand")
        self.absorbQPSO = try library.pipeline(
            device: context.device, name: "k3_mla_absorb_q")
        self.absorbQSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_mla_absorb_q",
            constants: Self.realConstants)
        self.partialPSO = try library.pipeline(
            device: context.device, name: "k3_mla_attn_decode_partial",
            maxTotalThreadsPerThreadgroup: 512)
        self.partialSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_mla_attn_decode_partial",
            constants: Self.realConstants, maxTotalThreadsPerThreadgroup: 512)
        self.combinePSO = try library.pipeline(
            device: context.device, name: "k3_mla_attn_decode_combine")
        self.combineSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_mla_attn_decode_combine",
            constants: Self.realConstants)
        self.outProjectPSO = try library.pipeline(
            device: context.device, name: "k3_mla_out_project")
        self.outProjectSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_mla_out_project",
            constants: Self.realConstants)
        self.cacheAppendPSO = try library.pipeline(
            device: context.device, name: "k3_mla_cache_append")
        self.cacheAppendSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_mla_cache_append",
            constants: Self.realConstants)

        let md = Self.maxNumHeads * Self.maxChunks
        guard let m = context.device.makeBuffer(
                length: md * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let d = context.device.makeBuffer(
                length: md * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let o = context.device.makeBuffer(
                length: md * Self.maxLatent * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.mPartial = m
        self.dPartial = d
        self.oPartial = o
    }

    /// Chunked split geometry for the decode passes: up to `defaultChunks`
    /// chunks, one per (head, chunk) threadgroup.
    static func splitGeometry(seqLen: UInt32, numHeads: UInt32)
        -> (numChunks: Int, chunkLen: Int)
    {
        let eff = max(1, Int(seqLen))
        let numChunks = max(1, min(defaultChunks, min(maxChunks, eff)))
        let chunkLen = (eff + numChunks - 1) / numChunks
        return (numChunks, chunkLen)
    }

    /// One-time per-load conversion of the dequantized fp16 kv_b weight
    /// ([H*(N+V), L] row-major, per-head N k-rows then V v-rows) into the
    /// absorb-friendly kT ([H][L][N], transposed k part) and v
    /// ([H][V][L], row copy) planes.
    func encodeKVBExpand(commandBuffer: MTLCommandBuffer,
                         kvB: MTLBuffer, kvBOffset: Int = 0,
                         kT: MTLBuffer, kTOffset: Int = 0,
                         v: MTLBuffer, vOffset: Int = 0,
                         numHeads: UInt32, latent: UInt32,
                         nope: UInt32, vHead: UInt32) {
        Self.validateDims(numHeads: numHeads, latent: latent, rope: 0,
                          nope: nope, vHead: vHead)
        var h = numHeads, l = latent, n = nope, vh = vHead
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(expandPSO)
        encoder.setBuffer(kvB, offset: kvBOffset, index: 0)
        encoder.setBuffer(kT, offset: kTOffset, index: 1)
        encoder.setBuffer(v, offset: vOffset, index: 2)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&l, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&vh, length: MemoryLayout<UInt32>.stride, index: 6)
        let total = Int(numHeads) * Int(latent) * Int(nope)
            + Int(numHeads) * Int(vHead) * Int(latent)
        let threadsPerThreadgroup = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (total + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// q̃_h = W_kvb_k,h^T q_nope_h plus rope passthrough. `q` is fp16
    /// [H][N+R]; `qAbs` is fp32 [H][L+R].
    func encodeAbsorbQ(commandBuffer: MTLCommandBuffer,
                       kT: MTLBuffer, kTOffset: Int = 0,
                       q: MTLBuffer, qOffset: Int = 0,
                       qAbs: MTLBuffer, qAbsOffset: Int = 0,
                       numHeads: UInt32, latent: UInt32,
                       rope: UInt32, nope: UInt32) {
        Self.validateDims(numHeads: numHeads, latent: latent, rope: rope,
                          nope: nope, vHead: 0)
        var h = numHeads, l = latent, r = rope, n = nope
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealConstants(numHeads: numHeads, latent: latent, rope: rope,
                             nope: nope, vHead: 0)
                ? absorbQSpecializedPSO : absorbQPSO)
        encoder.setBuffer(kT, offset: kTOffset, index: 0)
        encoder.setBuffer(q, offset: qOffset, index: 1)
        encoder.setBuffer(qAbs, offset: qAbsOffset, index: 2)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&l, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 6)
        let rows = Int(numHeads) * Int(latent + rope)
        let rowsPerThreadgroup = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Split-KV absorbed decode over the latent cache: fp32 `qAbs`
    /// [H][L+R] against fp16 cache rows [0, seqLen), writing fp32
    /// `outLat` [H][L]. `scale` is 192^-0.5 for canonical K3.
    func encodeAttnDecode(commandBuffer: MTLCommandBuffer,
                          cache: MTLBuffer, cacheOffset: Int = 0,
                          qAbs: MTLBuffer, qAbsOffset: Int = 0,
                          outLat: MTLBuffer, outLatOffset: Int = 0,
                          seqLen: UInt32,
                          numHeads: UInt32, latent: UInt32, rope: UInt32,
                          scale: Float) {
        precondition(seqLen >= 1, "decode attends over at least the current token")
        Self.validateDims(numHeads: numHeads, latent: latent, rope: rope,
                          nope: 0, vHead: 0)
        let geometry = Self.splitGeometry(seqLen: seqLen, numHeads: numHeads)
        let useSpecialized = useRealConstants(
            numHeads: numHeads, latent: latent, rope: rope, nope: 0, vHead: 0)

        var h = numHeads, l = latent, r = rope, sl = seqLen
        var cl = UInt32(geometry.chunkLen), nc = UInt32(geometry.numChunks)
        var sc = scale
        guard let p1 = commandBuffer.makeComputeCommandEncoder() else { return }
        p1.setComputePipelineState(useSpecialized ? partialSpecializedPSO : partialPSO)
        p1.setBuffer(cache, offset: cacheOffset, index: 0)
        p1.setBuffer(qAbs, offset: qAbsOffset, index: 1)
        p1.setBuffer(mPartial, offset: 0, index: 2)
        p1.setBuffer(dPartial, offset: 0, index: 3)
        p1.setBuffer(oPartial, offset: 0, index: 4)
        p1.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 5)
        p1.setBytes(&l, length: MemoryLayout<UInt32>.stride, index: 6)
        p1.setBytes(&r, length: MemoryLayout<UInt32>.stride, index: 7)
        p1.setBytes(&sl, length: MemoryLayout<UInt32>.stride, index: 8)
        p1.setBytes(&cl, length: MemoryLayout<UInt32>.stride, index: 9)
        p1.setBytes(&nc, length: MemoryLayout<UInt32>.stride, index: 10)
        p1.setBytes(&sc, length: MemoryLayout<Float>.stride, index: 11)
        p1.dispatchThreadgroups(
            MTLSize(width: Int(numHeads) * geometry.numChunks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        p1.endEncoding()

        var l2 = latent, h2 = numHeads, nc2 = UInt32(geometry.numChunks)
        guard let p2 = commandBuffer.makeComputeCommandEncoder() else { return }
        p2.setComputePipelineState(useSpecialized ? combineSpecializedPSO : combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(outLat, offset: outLatOffset, index: 3)
        p2.setBytes(&l2, length: MemoryLayout<UInt32>.stride, index: 4)
        p2.setBytes(&h2, length: MemoryLayout<UInt32>.stride, index: 5)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.stride, index: 6)
        p2.dispatchThreadgroups(
            MTLSize(width: Int(numHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// o_h = W_kvb_v,h · out_lat_h, flattened and gated by sigmoid(gate).
    /// `v` is the fp16 [H][V][L] plane from `encodeKVBExpand`; `outLat` fp32
    /// [H][L]; `gate` fp16 [H*V]; `out` fp16 [H*V].
    func encodeOutProject(commandBuffer: MTLCommandBuffer,
                          v: MTLBuffer, vOffset: Int = 0,
                          outLat: MTLBuffer, outLatOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          numHeads: UInt32, latent: UInt32, vHead: UInt32) {
        Self.validateDims(numHeads: numHeads, latent: latent, rope: 0,
                          nope: 0, vHead: vHead)
        var h = numHeads, l = latent, vh = vHead
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealConstants(numHeads: numHeads, latent: latent, rope: 0,
                             nope: 0, vHead: vHead)
                ? outProjectSpecializedPSO : outProjectPSO)
        encoder.setBuffer(v, offset: vOffset, index: 0)
        encoder.setBuffer(outLat, offset: outLatOffset, index: 1)
        encoder.setBuffer(gate, offset: gateOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&l, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&vh, length: MemoryLayout<UInt32>.stride, index: 6)
        let rows = Int(numHeads) * Int(vHead)
        let rowsPerThreadgroup = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Appends [ kv_a_layernorm(latent) | rope ] as fp16 at `position`.
    /// `kvA` is the raw fp16 [L+R] kv_a projection; `normWeight` fp32 [L].
    func encodeCacheAppend(commandBuffer: MTLCommandBuffer,
                           cache: MTLBuffer, cacheOffset: Int = 0,
                           position: UInt32,
                           kvA: MTLBuffer, kvAOffset: Int = 0,
                           normWeight: MTLBuffer, normWeightOffset: Int = 0,
                           eps: Float,
                           latent: UInt32, rope: UInt32) {
        Self.validateDims(numHeads: 0, latent: latent, rope: rope,
                          nope: 0, vHead: 0)
        var pos = position, l = latent, r = rope, e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealConstants(numHeads: 0, latent: latent, rope: rope,
                             nope: 0, vHead: 0)
                ? cacheAppendSpecializedPSO : cacheAppendPSO)
        encoder.setBuffer(kvA, offset: kvAOffset, index: 0)
        encoder.setBuffer(normWeight, offset: normWeightOffset, index: 1)
        encoder.setBuffer(cache, offset: cacheOffset, index: 2)
        encoder.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&l, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&r, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&e, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func validateDims(numHeads: UInt32, latent: UInt32,
                                     rope: UInt32, nope: UInt32,
                                     vHead: UInt32) {
        precondition(numHeads <= UInt32(maxNumHeads))
        precondition(latent > 0 && latent <= UInt32(maxLatent),
                     "latent \(latent) exceeds kernel staging (\(maxLatent))")
        precondition(rope <= UInt32(maxRope))
        precondition(latent + rope <= 576,
                     "cache row \(latent + rope) exceeds kernel staging (576)")
        precondition(nope <= UInt32(maxHeadDim) && vHead <= UInt32(maxHeadDim))
    }

    private func useRealConstants(numHeads: UInt32, latent: UInt32,
                                  rope: UInt32, nope: UInt32,
                                  vHead: UInt32) -> Bool {
        (numHeads == 0 || numHeads == Self.realNumHeads)
            && latent == Self.realLatent
            && (rope == 0 || rope == Self.realRope)
            && (nope == 0 || nope == Self.realNope)
            && (vHead == 0 || vHead == Self.realVHead)
    }
}
