import Foundation
import Metal

/// Decode-order attention for the wide-verify pass (docs/plan-wide-verify.md,
/// task 3), for both layer kinds.
///
/// Gated-DeltaNet layers: the chunked prefill runs batched projections, a
/// multi-row conv and the in-kernel token loop of the delta rule; the identity
/// bisect showed rows after the first drifting at their first linear layer
/// even with per-row projection GEMVs. Here each row runs decode's whole
/// linear-attention step (`encodeLinearAttentionDecode`) in turn, which shifts
/// the conv tail and advances the recurrent state exactly as scalar decode.
///
/// Full attention: the chunked prefill attention is a different kernel family
/// from decode's:
/// its own projection GEMM, a fused norm/RoPE epilogue over all rows and a
/// tiled causal attention. Each reorders a reduction, so a verified row drifts
/// from the token decode would have produced at the same position. In
/// decode-order mode each row instead runs decode's own gated pipeline in turn
/// -- QKV GEMV, split, per-head norms, RoPE, KV write, flash attention over
/// `position + 1` keys, gate, o-proj -- so row `r` sees exactly the keys and
/// accumulation order scalar decode sees at `startPosition + r`.
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

    /// Runs decode's gated full-attention step once per row, in row order, on
    /// `cb`. Row `r`'s normed input is copied into decode's `normed`, and its
    /// o-proj output in `oOut` is copied back into row `r` of `scratch.h1`.
    func encodeFullAttentionDecodeOrderRows(
        cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int,
        startPosition: Int, slot: Int
    ) throws {
        let rowBytes = D * MemoryLayout<Float16>.stride
        for row in 0..<t {
            try encodeRowCopy(
                cb, from: scratch.normed, offset: row * rowBytes,
                to: normed, offset: 0, bytes: rowBytes)
            var rowCB = cb
            let position = startPosition + row
            try encodeGatedFullAttentionDecode(
                &rowCB, layer: L, position: position, slot: slot,
                seqLen: UInt32(position + 1), selection: nil)
            try encodeRowCopy(
                cb, from: oOut, offset: 0,
                to: scratch.h1, offset: row * rowBytes, bytes: rowBytes)
        }
    }

    /// Whether linear layers of this pass take the decode-order path. The
    /// first-token GDN snapshot (MTP pair verify) reads the chunk path's
    /// intermediate buffers, so it keeps that path.
    func usesDecodeOrderLinearAttention(snapshot: Bool) -> Bool {
        decodeOrderVerify && !snapshot && !splitKernelTiming
    }

    /// Runs decode's linear-attention step once per row, in row order, reading
    /// row `r` of `scratch.normed` and writing row `r` of `scratch.h1`.
    func encodeLinearAttentionDecodeOrderRows(
        cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, slot: Int
    ) throws {
        let rowBytes = D * MemoryLayout<Float16>.stride
        for row in 0..<t {
            var rowCB = cb
            try encodeLinearAttentionDecode(
                &rowCB, layer: L, slot: slot,
                input: scratch.normed, inputOffset: row * rowBytes,
                output: scratch.h1, outputOffset: row * rowBytes)
        }
    }

    private func encodeRowCopy(
        _ cb: MTLCommandBuffer,
        from source: MTLBuffer, offset sourceOffset: Int,
        to destination: MTLBuffer, offset destinationOffset: Int,
        bytes: Int
    ) throws {
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: source, sourceOffset: sourceOffset,
            to: destination, destinationOffset: destinationOffset,
            size: bytes)
        blit.endEncoding()
    }
}
