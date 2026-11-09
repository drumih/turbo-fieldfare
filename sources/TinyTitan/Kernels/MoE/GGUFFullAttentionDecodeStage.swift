import Foundation
import Metal

/// One full-attention layer's decode step from the arena: the packed
/// `[query ; gate]` q projection split per head, weighted per-head q/k norms,
/// partial NeoX RoPE, GQA attention against a contiguous KV cache, the sigmoid
/// output gate, and the o projection.
///
/// The checkpoint's q projection emits 2 x qDim: all queries first, then the
/// gate, which `encodeSplitQGate` reorders per head. The sparse indexer is not
/// consulted here: this attends every cached key, which is the exact result for
/// prompt lengths inside the indexer's top-k window and a correct (if denser)
/// one beyond it.
package final class GGUFFullAttentionDecodeStage {
    private let gemv: BF16GEMV
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let arena: GGUFResidentArena
    private let mapped: GGUFMappedShards
    private let fused: GGUFExpertGEMV
    private let arch: ArchConfig

    private let headDim: Int
    private let numHeads: Int
    private let numKVHeads: Int
    private let qDim: Int
    private let kvDim: Int
    private let rotaryDim: UInt32
    private let maxSeq: Int

    private var kCache: [Int: MTLBuffer] = [:]
    private var vCache: [Int: MTLBuffer] = [:]
    private let kScratch: MTLBuffer
    private let vScratch: MTLBuffer
    private let qPacked: MTLBuffer
    private let qScratch: MTLBuffer
    private let gateScratch: MTLBuffer
    private let attnOut: MTLBuffer

    package init(
        context: MetalContext, arch: ArchConfig, arena: GGUFResidentArena,
        mapped: GGUFMappedShards, maxSeq: Int
    ) throws {
        self.gemv = try BF16GEMV(context: context)
        self.fused = try GGUFExpertGEMV(context: context)
        self.mapped = mapped
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.arena = arena
        self.arch = arch
        self.headDim = arch.fullHeadDim
        self.numHeads = arch.numHeads
        self.numKVHeads = arch.numFullKVHeads
        self.qDim = arch.numHeads * arch.fullHeadDim
        self.kvDim = arch.numFullKVHeads * arch.fullHeadDim
        self.rotaryDim = UInt32(Double(arch.fullHeadDim) * arch.partialRotaryFactor)
        self.maxSeq = maxSeq

        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: bytes, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            return buffer
        }
        self.kScratch = try make(kvDim * 2, "full k scratch")
        self.vScratch = try make(kvDim * 2, "full v scratch")
        self.qPacked = try make(2 * qDim * 2, "full q packed")
        self.qScratch = try make(qDim * 2, "full q")
        self.gateScratch = try make(qDim * 2, "full gate")
        self.attnOut = try make(qDim * 2, "full attn out")
        for layer in 0..<arch.numLayers where arch.fullAttentionLayerMask[layer] == 1 {
            kCache[layer] = try make(maxSeq * kvDim * 2, "full k cache")
            vCache[layer] = try make(maxSeq * kvDim * 2, "full v cache")
        }
    }

    package func encode(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        position: Int,
        x: MTLBuffer,
        xOffset: Int = 0,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        guard let kBuffer = kCache[layer], let vBuffer = vCache[layer] else {
            throw MetalError.invalidState("layer \(layer) is not a full-attention layer")
        }
        let prefix = "blk.\(layer)."
        func offset(_ suffix: String) throws -> Int {
            guard let entry = arena.entries[prefix + suffix] else {
                throw MetalError.invalidState("missing \(prefix + suffix) in the arena")
            }
            return Int(entry.offset)
        }
        let kvRowOffset = position * kvDim * 2
        // The fused GEMV writes a whole buffer, so k and v land in scratch,
        // take their norm and RoPE there, and are blitted into the cache row.
        precondition(xOffset == 0 && outOffset == 0, "fused GEMV takes whole buffers")
        func fusedProjection(_ name: String, into y: MTLBuffer, rows: Int) throws {
            guard let weight = try mapped.weight(named: prefix + name) else {
                throw MetalError.invalidState("missing \(prefix + name) in the shards")
            }
            try fused.encode(
                commandBuffer: commandBuffer, weights: weight.buffer,
                weightsOffset: weight.offset, x: x, y: y,
                rows: rows, columns: arch.hiddenSize, type: weight.type)
        }
        try fusedProjection("attn_q.weight", into: qPacked, rows: 2 * qDim)
        try fusedProjection("attn_k.weight", into: kScratch, rows: kvDim)
        try fusedProjection("attn_v.weight", into: vScratch, rows: kvDim)
        try elementwise.encodeSplitQGate(
            commandBuffer: commandBuffer, packed: qPacked,
            q: qScratch, gate: gateScratch, heads: numHeads, dim: headDim)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer, x: qScratch,
            weight: arena.buffer, weightOffset: try offset("attn_q_norm.weight"),
            out: qScratch, headDim: UInt32(headDim), numHeads: numHeads, eps: 1e-6)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer, x: kScratch,
            weight: arena.buffer, weightOffset: try offset("attn_k_norm.weight"),
            out: kScratch,
            headDim: UInt32(headDim), numHeads: numKVHeads, eps: 1e-6)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: qScratch,
            position: UInt32(position), headDim: UInt32(headDim),
            numHeads: UInt32(numHeads), rotaryDim: rotaryDim,
            theta: Float(arch.fullRopeTheta))
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: kScratch,
            position: UInt32(position), headDim: UInt32(headDim),
            numHeads: UInt32(numKVHeads), rotaryDim: rotaryDim,
            theta: Float(arch.fullRopeTheta))
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: kScratch, sourceOffset: 0, to: kBuffer,
            destinationOffset: kvRowOffset, size: kvDim * 2)
        blit.copy(
            from: vScratch, sourceOffset: 0, to: vBuffer,
            destinationOffset: kvRowOffset, size: kvDim * 2)
        blit.endEncoding()
        try attention.encodeFull(
            commandBuffer: commandBuffer, q: qScratch,
            k: kBuffer, v: vBuffer, out: attnOut,
            headDim: UInt32(headDim), numQHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads), seqLen: UInt32(position + 1),
            scale: Float(arch.attentionScale))
        try elementwise.encodeSigmoidGateMul(
            commandBuffer: commandBuffer, out: attnOut, gate: gateScratch, count: qDim)
        guard let outputWeight = try mapped.weight(named: prefix + "attn_output.weight") else {
            throw MetalError.invalidState("missing \(prefix)attn_output.weight in the shards")
        }
        try fused.encode(
            commandBuffer: commandBuffer, weights: outputWeight.buffer,
            weightsOffset: outputWeight.offset, x: attnOut, y: out,
            rows: arch.hiddenSize, columns: qDim, type: outputWeight.type)
    }

    package func resetCache() {
        for buffer in kCache.values { memset(buffer.contents(), 0, buffer.length) }
        for buffer in vCache.values { memset(buffer.contents(), 0, buffer.length) }
    }
}
