import Foundation
import Metal

/// Split-out pieces of the gated full-attention decode path: the pre-attention
/// projection/norm/RoPE block and the host-KV gather both branches share.
extension RealForwardRunner {
    /// The windowed indirection for one decode step, or nil when this layer's
    /// rows still live contiguously in VRAM.
    func decodeWindowSelection(
        kv: KVCacheManager,
        layer: Int,
        slot: Int,
        position: Int,
        seqLen: UInt32,
        selection: DecodeKeySelection?
    ) throws -> KVWindowSelection? {
        guard kv.streamsLayer(layer) else { return nil }
        return try kv.prepareDecodeWindow(
            layer: layer, slot: slot, position: position,
            indices: selection?.indices,
            count: selection?.indices == nil
                ? Int(seqLen) : (selection?.count ?? Int(seqLen)))
    }

    /// Where this step's K and V rows are written: straight into the cache
    /// slot, or into the staging rows a quantized cache converts from.
    func gatedFullKVWrites(
        kv: KVCacheManager, layer: Int, position: Int, slot: Int
    ) -> (k: (buffer: MTLBuffer, offset: Int), v: (buffer: MTLBuffer, offset: Int)) {
        guard !kv.precision.isQuantized else {
            return ((kStage, 0), (vStage, 0))
        }
        return (
            kv.kSlot(layer: layer, position: position, slot: slot),
            kv.vSlot(layer: layer, position: position, slot: slot)
        )
    }

    /// One decode step's gated attention after q/k/v are in place: the cache
    /// quantization, attention over `seqLen` keys into `attnOut`, and the
    /// sigmoid output gate. The o-proj that follows is the caller's.
    func encodeGatedFullAttentionAndGate(
        _ cb: inout MTLCommandBuffer,
        kv: KVCacheManager, layer L: Int, position: Int, slot: Int,
        seqLen: UInt32, selection: DecodeKeySelection?
    ) throws {
        guard let elementwise, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail:
                    "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        if kv.precision.isQuantized {
            try encodeQuantizedKV(
                commandBuffer: cb, kv: kv, layer: L,
                position: position, slot: slot, keySource: kStage,
                valueSource: vStage, elementCount: numKV * headDim)
        }
        let keyView = kv.keyView(layer: L, slot: slot, validTokenCount: Int(seqLen))
        let valueView = kv.valueView(layer: L, slot: slot, validTokenCount: Int(seqLen))
        // Host-KV streaming: gather the keys this step will read into the
        // window and let the indirection kernel iterate them. Without it the
        // contiguous kernel keeps reading the same buffer it always has.
        let windowSelection = try decodeWindowSelection(
            kv: kv, layer: L, slot: slot, position: position,
            seqLen: seqLen, selection: selection)
        try attention.encodeFull(
            commandBuffer: cb,
            q: qScratch,
            k: keyView.buffer, kOffset: keyView.offset,
            v: valueView.buffer, vOffset: valueView.offset,
            out: attnOut,
            headDim: UInt32(headDim),
            numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(numKV),
            seqLen: seqLen,
            scale: Float(cfg.attentionScale),
            kvFormat: keyView,
            keepMask: windowSelection == nil ? selection?.mask : nil,
            windowSelection: windowSelection)
        try rotate(&cb, role: "qsa.attention")
        try elementwise.encodeSigmoidGateMul(
            commandBuffer: cb,
            out: attnOut,
            gate: attnGateScratch,
            count: cfg.numHeads * headDim)
        try rotate(&cb, role: "qsa.gatemul")
    }

    /// Packed q_proj split, per-head q/k norms, NeoX sub-dim RoPE.
    func encodeGatedFullPreAttention(
        _ cb: inout MTLCommandBuffer,
        layer L: Int,
        position: Int,
        kWrite: (buffer: MTLBuffer, offset: Int),
        vWrite: (buffer: MTLBuffer, offset: Int),
        qDim: UInt32,
        kvDim: UInt32,
        headDim: Int,
        numKV: Int,
        eps: Float
    ) throws {
        guard let qPackedScratch else {
            throw ModelError.internalInconsistency(
                detail: "attn_output_gate layer \(L) without q scratch")
        }
        try encodeGatedFullQKVProjection(
            cb, layer: L, qOutput: qPackedScratch,
            kOutput: kWrite, vOutput: vWrite,
            qDimension: qDim, kvDimension: kvDim)
        try rotate(&cb, role: "qsa.qkv")
        try encodeGatedFullPostProjection(
            cb, layer: L, position: position,
            qPacked: qPackedScratch, qPackedOffset: 0,
            kWrite: kWrite, vWrite: vWrite,
            headDim: headDim, numKV: numKV, eps: eps)
    }

    /// Split [query ; gate], per-head q/k norms and NeoX sub-dim RoPE after
    /// QKV. The batched path feeds it rows produced by a two-row projection.
    func encodeGatedFullPostProjection(
        _ cb: MTLCommandBuffer,
        layer L: Int,
        position: Int,
        qPacked: MTLBuffer,
        qPackedOffset: Int,
        kWrite: (buffer: MTLBuffer, offset: Int),
        vWrite: (buffer: MTLBuffer, offset: Int),
        headDim: Int,
        numKV: Int,
        eps: Float
    ) throws {
        guard let elementwise, let rope, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail:
                    "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        try elementwise.encodeSplitQGate(
            commandBuffer: cb,
            packed: qPacked, packedOffset: qPackedOffset,
            q: qScratch,
            gate: attnGateScratch,
            heads: cfg.numHeads,
            dim: headDim)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: qScratch,
            weight: qNormW.buffer,
            weightOffset: Int(qNormW.offset),
            out: qScratch,
            headDim: UInt32(headDim),
            numHeads: cfg.numHeads,
            eps: eps)
        try rms.encodeBF16WPerHead(
            commandBuffer: cb,
            x: kWrite.buffer, xOffset: kWrite.offset,
            weight: kNormW.buffer,
            weightOffset: Int(kNormW.offset),
            out: kWrite.buffer, outOffset: kWrite.offset,
            headDim: UInt32(headDim),
            numHeads: numKV,
            eps: eps)
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: qScratch,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(cfg.numHeads),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
        try rope.encodeNeoxSubdim(
            commandBuffer: cb,
            data: kWrite.buffer,
            dataOffset: kWrite.offset,
            position: UInt32(position),
            headDim: UInt32(headDim),
            numHeads: UInt32(numKV),
            rotaryDim: rotaryDim,
            theta: Float(cfg.fullRopeTheta))
    }

}
