import Foundation
import Metal
import TinyTitanCore

/// The ANE prefill bridge (Track A): stages the normed chunk out, predicts, and rebinds the command buffer so the rest of the layer proceeds exactly as on the GPU path.
///
/// Split from RealForwardRunner.swift in the modularity refactor
/// (docs/modularity-refactor.md) as pure code motion: one concern
/// per file, no signature or behavior changes.
///
/// Coverage note, stated because the unit run reports this file at 0 covered
/// lines and that number has no meaning without its reason: this is the only
/// body in the runtime whose 0% is not a choice. It runs during a real prefill
/// of a full-attention layer the sidecar covers, so reaching it needs a
/// Qwen3.8-Flash-Next install *and* the exported ANE bundle -- and the
/// installed `.ssdai` here carries none (`manifest.json`, `model_weights.bin`,
/// `ngram_table.bin`, `packed_experts/`, `ple_constants.json`, `tokenizer/`,
/// `verified-install.json`). Producing that bundle is `tools/export_ane_prefill.py`,
/// an operator job, so the right answer is not a test that fetches it.
/// `ANEPrefillAttentionTests` covers the bridge it calls; this method's own
/// staging, rebinding and shadow-append is only observable inside a live
/// prefill, and is reported as not checked rather than as covered.
extension RealForwardRunner {
    /// One full-attention layer's prefill attention on the Neural Engine
    /// (Track A). The layer's input norm is already encoded on `cb`; this
    /// stages it out, waits, predicts, and rebinds `cb` so the rest of the
    /// layer (KV quantization, residual, MoE) proceeds exactly as on the GPU
    /// path — `stagingK`/`stagingV` stand in for `kStage`/`vStage` and the
    /// attention output is blitted into `h1` for the generic residual tail.
    func runANEFullAttentionPrefill(
        ane: ANEPrefillAttention,
        cb: inout MTLCommandBuffer,
        layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int,
        kvDim: Int,
        selection: QSASelection?
    ) async throws {
        let halfBytes = MemoryLayout<Float16>.stride
        guard let stage = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        stage.copy(
            from: scratch.normed, sourceOffset: 0,
            to: ane.stagingNormed, destinationOffset: 0,
            size: t * D * halfBytes)
        stage.endEncoding()
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: "prefill_ane_stage", cb)

        try await ane.predict(
            layer: L, history: startPosition,
            tokenCount: t, selection: selection)
        ane.appendShadow(layer: L, startPosition: startPosition, tokenCount: t)
        // Start the next covered layer's model load now: it overlaps the MoE
        // stage the caller is about to encode and run on the GPU, which is
        // roughly an order of magnitude longer than the ~0.5 s load.
        if let next = ((L + 1)..<cfg.numLayers).first(where: {
            ane.coveredLayers.contains($0)
        }) {
            ane.preload(layer: next, history: startPosition)
        }

        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
        if let kv {
            try copyPrefillKVToCache(
                commandBuffer: cb,
                kv: kv,
                layer: L,
                startPosition: startPosition,
                tokenCount: t,
                keySource: ane.stagingK,
                valueSource: ane.stagingV,
                bytesPerToken: kvDim * halfBytes)
        }
        guard let out = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        out.copy(
            from: ane.stagingOut, sourceOffset: 0,
            to: scratch.h1, destinationOffset: 0,
            size: t * D * halfBytes)
        out.endEncoding()
    }
}
