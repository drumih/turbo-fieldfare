import Foundation
import Metal

/// Decode-order attention for the wide-verify pass (docs/plan-wide-verify.md,
/// task 3), for both layer kinds.
///
/// Gated-DeltaNet layers: the chunked prefill runs batched projections, a
/// multi-row conv and the in-kernel token loop of the delta rule; the identity
/// bisect showed rows after the first drifting at their first linear layer
/// even with per-row projection GEMVs. Here each row runs decode's own conv,
/// q/k norm, delta step and gated norm in turn, which shifts the conv tail and
/// advances the recurrent state exactly as scalar decode.
///
/// Full attention: the chunked prefill attention is a different kernel family
/// from decode's:
/// its own projection GEMM, a fused norm/RoPE epilogue over all rows and a
/// tiled causal attention. Each reorders a reduction, so a verified row drifts
/// from the token decode would have produced at the same position. In
/// decode-order mode each row instead runs decode's own gated pipeline in turn
/// -- split, per-head norms, RoPE, KV write, flash attention over
/// `position + 1` keys, gate -- so row `r` sees exactly the keys and
/// accumulation order scalar decode sees at `startPosition + r`.
///
/// In both, the dense projections around the per-row core (GDN qkv/z/a/b and
/// out_proj; q/k/v and o-proj) run once for all rows through `MultiRowGEMV`
/// (task 7): decode's per-row arithmetic with each weight matrix read once
/// per pass instead of once per row.
extension RealForwardRunner {
    /// Whether layer `L` of this pass takes the decode-order path. It covers the
    /// gated (Qwen) full attention with every key visible and the layer's KV
    /// resident; a QSA selection or a host-streamed layer keeps the prefill
    /// path, which the identity bisect will report as divergent.
    func usesDecodeOrderFullAttention(layer L: Int, keepMask: QSASelection?) -> Bool {
        guard decodeOrderVerify, cfg.attnOutputGate, keepMask == nil,
            cfg.fullAttentionLayerMask[L] == 1, let kv, !kv.streamsLayer(L),
            !splitKernelTiming
        else { return false }
        return true
    }

    /// Decode's gated full-attention step for every row, in row order, on
    /// `cb`, with the dense projections read once for all rows.
    ///
    /// q/k/v for all rows come from one multi-row GEMV (decode's per-row
    /// arithmetic, `MultiRowGEMV`) into the scratch's q and K/V staging rows.
    /// Each row then copies its K/V into decode's write targets and runs
    /// decode's own split/norm/RoPE, KV write, attention over `position + 1`
    /// keys and output gate, so row `r` sees exactly the keys and order scalar
    /// decode sees at `startPosition + r`. The gated outputs are collected and
    /// the o-proj runs once for all rows into `scratch.h1`.
    func encodeFullAttentionDecodeOrderRows(
        cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int,
        startPosition: Int, slot: Int
    ) throws {
        guard let kv else {
            throw ModelError.internalInconsistency(detail: "full attention requires a KV cache")
        }
        precondition(t <= scratch.layout.chunkTokens, "verify rows exceed the chunk scratch")
        let half = MemoryLayout<Float16>.stride
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = cfg.numHeads * headDim
        let kvDim = numKV * headDim
        let packedDim = 2 * qDim
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let formats = gatedQKVFormats(q: q, k: k, v: v)
        try multiRowGEMV.encode(
            commandBuffer: cb,
            segments: [
                .init(
                    view: q, format: formats[0], rows: packedDim, y: scratch.q,
                    yRowStride: packedDim),
                .init(
                    view: k, format: formats[1], rows: kvDim, y: scratch.kStage, yRowStride: kvDim),
                .init(
                    view: v, format: formats[2], rows: kvDim, y: scratch.vStage, yRowStride: kvDim),
            ],
            x: scratch.normed, xRowStride: D, activationRows: t, columns: D)
        for row in 0..<t {
            let position = startPosition + row
            let writes = gatedFullKVWrites(kv: kv, layer: L, position: position, slot: slot)
            try encodeRowCopy(
                cb, from: scratch.kStage, offset: row * kvDim * half,
                to: writes.k.buffer, offset: writes.k.offset, bytes: kvDim * half)
            try encodeRowCopy(
                cb, from: scratch.vStage, offset: row * kvDim * half,
                to: writes.v.buffer, offset: writes.v.offset, bytes: kvDim * half)
            try encodeGatedFullPostProjection(
                cb, layer: L, position: position,
                qPacked: scratch.q, qPackedOffset: row * packedDim * half,
                kWrite: writes.k, vWrite: writes.v,
                headDim: headDim, numKV: numKV, eps: 1e-6)
            var rowCB = cb
            try encodeGatedFullAttentionAndGate(
                &rowCB, kv: kv, layer: L, position: position, slot: slot,
                seqLen: UInt32(position + 1), selection: nil)
            try encodeRowCopy(
                cb, from: attnOut, offset: 0,
                to: scratch.attentionOutput, offset: row * qDim * half, bytes: qDim * half)
        }
        try encodePrimaryGEMVRows(
            commandBuffer: cb, projection: try model.oProj(layer: L),
            x: scratch.attentionOutput, xRowStride: qDim,
            y: scratch.h1, yRowStride: D, m: D, n: qDim, rows: t)
    }

    /// Whether linear layers of this pass take the decode-order path. The
    /// first-token GDN snapshot (MTP pair verify) reads the chunk path's
    /// intermediate buffers, so it keeps that path.
    func usesDecodeOrderLinearAttention(snapshot: Bool) -> Bool {
        decodeOrderVerify && !snapshot && !splitKernelTiming
    }

    /// Decode's linear-attention step for every row, in row order, with the
    /// input and output projections read once for all rows.
    ///
    /// The qkv/z/a/b projections of all rows come from one multi-row GEMV
    /// (decode's per-row arithmetic) into the scratch's rows. Each row then
    /// runs decode's own conv (shifting the tail), q/k norm, delta step
    /// (advancing the recurrent state) and gated norm, exactly as scalar
    /// decode would at that position; the gated rows go through one multi-row
    /// out_proj into `scratch.h1`. With a wide-verify rollback window open,
    /// all rows' raw qkv/a/b projections are retained in one copy.
    func encodeLinearAttentionDecodeOrderRows(
        cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, slot: Int
    ) throws {
        guard let gdn, let gdnState, let gdnConvOut, let gdnY else {
            throw ModelError.internalInconsistency(
                detail:
                    "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        precondition(t <= scratch.layout.chunkTokens, "verify rows exceed the chunk scratch")
        let la = cfg.linearAttention
        let half = MemoryLayout<Float16>.stride
        let inputs = [
            try model.linearInProjQKV(layer: L), try model.linearInProjZ(layer: L),
            try model.linearInProjA(layer: L), try model.linearInProjB(layer: L),
        ]
        let formats = try gdnInputProjectionFormats(inputs)
        let outputs = [
            (scratch.q, la.qkvDim), (scratch.gdnZ, la.valueDim),
            (scratch.gdnA, la.numVHeads), (scratch.gdnB, la.numVHeads),
        ]
        try multiRowGEMV.encode(
            commandBuffer: cb,
            segments: (0..<4).map {
                .init(
                    view: inputs[$0], format: formats[$0], rows: outputs[$0].1,
                    y: outputs[$0].0, yRowStride: outputs[$0].1)
            },
            x: scratch.normed, xRowStride: D, activationRows: t, columns: D)
        try retainDecodeOrderRows(cb, layer: L, scratch: scratch, rows: t)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)
        let tail = gdnState.convTailSlot(layer: L, slot: slot)
        let state = gdnState.stateSlot(layer: L, slot: slot)
        for row in 0..<t {
            try gdn.encodeConvDecode(
                commandBuffer: cb,
                tail: tail.buffer, tailOffset: tail.offset,
                qkv: scratch.q, qkvOffset: row * la.qkvDim * half,
                convWeight: convW.buffer, convWeightOffset: Int(convW.offset),
                out: gdnConvOut)
            try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
            try gdn.encodeDeltaStepDecode(
                commandBuffer: cb,
                convOut: gdnConvOut,
                aProj: scratch.gdnA, aProjOffset: row * la.numVHeads * half,
                bProj: scratch.gdnB, bProjOffset: row * la.numVHeads * half,
                aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                state: state.buffer, stateOffset: state.offset,
                y: gdnY)
            try gdn.encodeGatedNorm(
                commandBuffer: cb,
                y: gdnY,
                z: scratch.gdnZ, zOffset: row * la.valueDim * half,
                weight: gatedNormW.buffer, weightOffset: Int(gatedNormW.offset),
                out: scratch.gdnY, outOffset: row * la.valueDim * half)
        }
        try encodePrimaryGEMVRows(
            commandBuffer: cb, projection: try model.linearOutProj(layer: L),
            x: scratch.gdnY, xRowStride: la.valueDim,
            y: scratch.h1, yRowStride: D, m: D, n: la.valueDim, rows: t)
    }

    /// Hands the pass's GDN projections to an open rollback window: the raw
    /// `in_proj_qkv` rows (the conv's input, read but not written by
    /// `gdn_conv_mix_decode`) and the pre-softplus/pre-sigmoid `a` and `b`
    /// rows (read-only to `gdn_delta_step_decode`). These are exactly the rows
    /// the prefill replay feeds to its conv, tail update and delta kernels.
    /// The scratch holds them row-major at decode's per-row widths (qkvDim,
    /// numVHeads), the layout `encodeRetain` copies, so one copy covers all
    /// rows; nothing later in the layer writes those scratch rows.
    private func retainDecodeOrderRows(
        _ cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers, rows t: Int
    ) throws {
        guard let window = wideVerifyRetain(layer: L) else { return }
        try window.encodeRetain(
            commandBuffer: cb, layer: L,
            qkv: .init(scratch.q), a: .init(scratch.gdnA), b: .init(scratch.gdnB),
            rows: t)
    }
}
