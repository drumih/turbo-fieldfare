import Foundation
import Metal
import TinyTitanCore

extension RealForwardRunner {
    /// Two activation rows through one projection into `[row0 m][row1 m]`.
    func encodeRoleGEMVTwoRows(
        commandBuffer cb: MTLCommandBuffer,
        projection p: TensorView,
        weightBits: Int,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        if weightBits == 4 {
            try int4.encodeTwoRows(
                commandBuffer: cb,
                weights: p.buffer, weightsOffset: Int(p.offset),
                scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                biases: p.buffer, biasesOffset: Int(p.biasOffset),
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        guard let dispatcher = affineByWidth[weightBits] else {
            throw ModelError.unsupportedArchitecture(
                detail: "no \(weightBits)-bit two-row GEMV for a \(m)x\(n) projection")
        }
        try dispatcher.encodeTwoRows(
            commandBuffer: cb,
            weights: p.buffer, weightsOffset: Int(p.offset),
            scales: p.buffer, scalesOffset: Int(p.scaleOffset),
            biases: p.buffer, biasesOffset: Int(p.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: m, n: n)
    }

    /// Blit a row between a batch buffer and a staging/cache buffer.
    func copyIntoStaging(
        _ cb: MTLCommandBuffer,
        source: MTLBuffer, sourceOffset: Int,
        destination: MTLBuffer, destinationOffset: Int = 0, bytes: Int
    ) throws {
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: source, sourceOffset: sourceOffset,
            to: destination, destinationOffset: destinationOffset,
            size: bytes)
        blit.endEncoding()
    }

    /// T5 full-attention stage: entry norms and QKV/o_proj are shared across
    /// the batch, while each row keeps its own q/k epilogue, KV slot and
    /// attention. The caller supplies the authoritative hidden rows in
    /// `scratch.hidden`; the result lands in `scratch.o`.
    func encodeBatchGatedAttention(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        commandBuffer cb: inout MTLCommandBuffer
    ) throws {
        precondition(rows.count == scratch.rows)
        let d = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = cfg.numHeads * headDim
        let qRows = cfg.attnOutputGate ? 2 * qDim : qDim
        let kvDim = numKV * headDim
        let eps: Float = 1e-6
        try encodeBatchEntryNorms(
            layer: L, rows: rows, scratch: scratch, commandBuffer: cb, eps: eps)
        try encodeBatchQKV(
            layer: L, rows: rows, scratch: scratch, commandBuffer: cb,
            qRows: qRows, kvDim: kvDim, d: d)
        try encodeBatchRowAttention(
            layer: L, rows: rows, scratch: scratch, commandBuffer: cb,
            headDim: headDim, numKV: numKV, qDim: qDim, qRows: qRows,
            kvDim: kvDim, d: d, half: half, eps: eps)
        try encodeBatchOProj(
            layer: L, rows: rows, scratch: scratch, commandBuffer: cb,
            qDim: qDim, d: d, half: half)
    }

    private func encodeBatchEntryNorms(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        commandBuffer cb: MTLCommandBuffer,
        eps: Float
    ) throws {
        let d = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let inNorm = try model.inputNorm(layer: L)
        for index in rows.indices {
            try encodeResidualEntryDecode(
                commandBuffer: cb,
                hidden: scratch.hidden, hiddenOffset: index * d * half,
                norm: inNorm,
                out: scratch.normed, outOffset: index * d * half,
                sublayer: .attention, layer: L, eps: eps)
        }
    }

    private func encodeBatchQKV(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        commandBuffer cb: MTLCommandBuffer,
        qRows: Int,
        kvDim: Int,
        d: Int
    ) throws {
        let half = MemoryLayout<Float16>.stride
        let qW = try model.qProj(layer: L)
        let kW = try model.kProj(layer: L)
        let vW = cfg.attentionKEqV ? kW : try model.vProj(layer: L)
        var pair = 0
        while pair < rows.count {
            let xOffset = pair * d * half
            let qOffset = pair * qRows * half
            let kvOffset = pair * kvDim * half
            try encodeRoleGEMVTwoRows(
                commandBuffer: cb, projection: qW,
                weightBits: model.qoProjectionWeightBits,
                x: scratch.normed, xOffset: xOffset,
                y: scratch.q, yOffset: qOffset,
                m: UInt32(qRows), n: UInt32(d))
            try encodeRoleGEMVTwoRows(
                commandBuffer: cb, projection: kW,
                weightBits: model.kvProjectionWeightBits,
                x: scratch.normed, xOffset: xOffset,
                y: scratch.k, yOffset: kvOffset,
                m: UInt32(kvDim), n: UInt32(d))
            try encodeRoleGEMVTwoRows(
                commandBuffer: cb, projection: vW,
                weightBits: model.kvProjectionWeightBits,
                x: scratch.normed, xOffset: xOffset,
                y: scratch.v, yOffset: kvOffset,
                m: UInt32(kvDim), n: UInt32(d))
            pair += 2
        }
    }

    private func encodeBatchRowAttention(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        commandBuffer cb: MTLCommandBuffer,
        headDim: Int,
        numKV: Int,
        qDim: Int,
        qRows: Int,
        kvDim: Int,
        d: Int,
        half: Int,
        eps: Float
    ) throws {
        guard let kv, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail: "batch attention requires a KV cache and gate scratch")
        }
        for index in rows.indices {
            let row = rows[index]
            let kSlot = kv.kSlot(layer: L, position: row.position, slot: row.slot)
            let vSlot = kv.vSlot(layer: L, position: row.position, slot: row.slot)
            let quantized = kv.precision.isQuantized
            try copyIntoStaging(
                cb, source: scratch.k, sourceOffset: index * kvDim * half,
                destination: kStage, bytes: kvDim * half)
            try copyIntoStaging(
                cb, source: scratch.v, sourceOffset: index * kvDim * half,
                destination: vStage, bytes: kvDim * half)
            try encodeGatedFullPostProjection(
                cb, layer: L, position: row.position,
                qPacked: scratch.q, qPackedOffset: index * qRows * half,
                kWrite: (buffer: kStage, offset: 0),
                vWrite: (buffer: vStage, offset: 0),
                headDim: headDim, numKV: numKV, eps: eps)
            if quantized {
                try encodeQuantizedKV(
                    commandBuffer: cb, kv: kv, layer: L,
                    position: row.position, slot: row.slot,
                    keySource: kStage, valueSource: vStage,
                    elementCount: kvDim)
            } else {
                try copyIntoStaging(
                    cb, source: kStage, sourceOffset: 0,
                    destination: kSlot.buffer, destinationOffset: kSlot.offset,
                    bytes: kvDim * half)
                try copyIntoStaging(
                    cb, source: vStage, sourceOffset: 0,
                    destination: vSlot.buffer, destinationOffset: vSlot.offset,
                    bytes: kvDim * half)
            }
            let keyView = kv.keyView(
                layer: L, slot: row.slot, validTokenCount: row.position + 1)
            let valueView = kv.valueView(
                layer: L, slot: row.slot, validTokenCount: row.position + 1)
            try attention.encodeFull(
                commandBuffer: cb,
                q: qScratch,
                k: keyView.buffer, kOffset: keyView.offset,
                v: valueView.buffer, vOffset: valueView.offset,
                out: scratch.attn, outOffset: index * qDim * half,
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKV),
                seqLen: UInt32(row.position + 1),
                scale: Float(cfg.attentionScale),
                kvFormat: keyView)
            try requireElementwise().encodeSigmoidGateMul(
                commandBuffer: cb,
                out: scratch.attn, outOffset: index * qDim * half,
                gate: attnGateScratch, count: qDim)
        }
    }

    private func encodeBatchOProj(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        commandBuffer cb: MTLCommandBuffer,
        qDim: Int,
        d: Int,
        half: Int
    ) throws {
        let o = try model.oProj(layer: L)
        var pair = 0
        while pair < rows.count {
            try encodeRoleGEMVTwoRows(
                commandBuffer: cb, projection: o,
                weightBits: model.qoProjectionWeightBits,
                x: scratch.attn, xOffset: pair * qDim * half,
                y: scratch.o, yOffset: pair * d * half,
                m: UInt32(d), n: UInt32(qDim))
            pair += 2
        }
    }

    /// T34 step 1: one full-attention block for the batch.
    ///
    /// `encodeBatchGatedAttention` ends at `scratch.o`. A decode layer then
    /// adds that output into each row's residual and normalizes it for the
    /// FFN, which is the composition the fused loop needs before its per-row
    /// MLP stage. Both halves are row-addressed, so two rows never share the
    /// residual or norm write. Hyper-connection families are refused: their
    /// write gate reads the runner-wide stream buffer and is not part of the
    /// fused eligibility set.
    func encodeBatchAttentionBlock(
        layer L: Int,
        rows: [(position: Int, slot: Int)],
        scratch: BatchedDecodeScratch,
        postAttnNorm: TensorView,
        commandBuffer cb: inout MTLCommandBuffer
    ) throws {
        guard hyperConnection == nil else {
            throw ModelError.unsupportedArchitecture(
                detail: "fused decode does not cover hyper-connection families")
        }
        let d = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let eps: Float = 1e-6
        try encodeBatchGatedAttention(
            layer: L, rows: rows, scratch: scratch, commandBuffer: &cb)
        for index in rows.indices {
            let offset = index * d * half
            try encodeResidualExitDecode(
                commandBuffer: cb,
                hidden: scratch.hidden, hiddenOffset: offset,
                delta: scratch.o, deltaOffset: offset,
                sublayer: .attention, layer: L)
            try encodeResidualEntryDecode(
                commandBuffer: cb,
                hidden: scratch.hidden, hiddenOffset: offset,
                norm: postAttnNorm,
                out: scratch.routedX, outOffset: offset,
                sublayer: .mlp, layer: L, eps: eps)
        }
    }

}
