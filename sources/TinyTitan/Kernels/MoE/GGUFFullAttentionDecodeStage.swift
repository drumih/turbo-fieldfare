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
    private let qPacked: MTLBuffer
    private let qScratch: MTLBuffer
    private let gateScratch: MTLBuffer
    private let attnOut: MTLBuffer

    package init(
        context: MetalContext, arch: ArchConfig, arena: GGUFResidentArena, maxSeq: Int
    ) throws {
        self.gemv = try BF16GEMV(context: context)
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
        try gemv.encode(
            commandBuffer: commandBuffer, weights: arena.buffer,
            weightsOffset: try offset("attn_q.weight"),
            x: x, xOffset: xOffset, y: qPacked,
            m: UInt32(2 * qDim), n: UInt32(arch.hiddenSize))
        try gemv.encode(
            commandBuffer: commandBuffer, weights: arena.buffer,
            weightsOffset: try offset("attn_k.weight"),
            x: x, xOffset: xOffset, y: kBuffer, yOffset: kvRowOffset,
            m: UInt32(kvDim), n: UInt32(arch.hiddenSize))
        try gemv.encode(
            commandBuffer: commandBuffer, weights: arena.buffer,
            weightsOffset: try offset("attn_v.weight"),
            x: x, xOffset: xOffset, y: vBuffer, yOffset: kvRowOffset,
            m: UInt32(kvDim), n: UInt32(arch.hiddenSize))
        try elementwise.encodeSplitQGate(
            commandBuffer: commandBuffer, packed: qPacked,
            q: qScratch, gate: gateScratch, heads: numHeads, dim: headDim)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer, x: qScratch,
            weight: arena.buffer, weightOffset: try offset("attn_q_norm.weight"),
            out: qScratch, headDim: UInt32(headDim), numHeads: numHeads, eps: 1e-6)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer, x: kBuffer, xOffset: kvRowOffset,
            weight: arena.buffer, weightOffset: try offset("attn_k_norm.weight"),
            out: kBuffer, outOffset: kvRowOffset,
            headDim: UInt32(headDim), numHeads: numKVHeads, eps: 1e-6)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: qScratch,
            position: UInt32(position), headDim: UInt32(headDim),
            numHeads: UInt32(numHeads), rotaryDim: rotaryDim,
            theta: Float(arch.fullRopeTheta))
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: kBuffer, dataOffset: kvRowOffset,
            position: UInt32(position), headDim: UInt32(headDim),
            numHeads: UInt32(numKVHeads), rotaryDim: rotaryDim,
            theta: Float(arch.fullRopeTheta))
        try attention.encodeFull(
            commandBuffer: commandBuffer, q: qScratch,
            k: kBuffer, v: vBuffer, out: attnOut,
            headDim: UInt32(headDim), numQHeads: UInt32(numHeads),
            numKVHeads: UInt32(numKVHeads), seqLen: UInt32(position + 1),
            scale: Float(arch.attentionScale))
        try elementwise.encodeSigmoidGateMul(
            commandBuffer: commandBuffer, out: attnOut, gate: gateScratch, count: qDim)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: arena.buffer,
            weightsOffset: try offset("attn_output.weight"),
            x: attnOut, y: out, yOffset: outOffset,
            m: UInt32(arch.hiddenSize), n: UInt32(qDim))
    }

    package func resetCache() {
        for buffer in kCache.values { memset(buffer.contents(), 0, buffer.length) }
        for buffer in vCache.values { memset(buffer.contents(), 0, buffer.length) }
    }
}
