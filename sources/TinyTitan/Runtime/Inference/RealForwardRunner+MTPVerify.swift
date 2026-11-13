import Foundation
import Metal

// The width-2 MTP verify: the pair schedule and the argmax that packages its
// result.
//
// Split out of `RealForwardRunner+MTP.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// `finishVerifyPair` stayed `private`: both callers moved with it.
extension RealForwardRunner {

    /// Verify `[confirmed, draft]` in the existing batched prefill path. The
    /// two target logits and target hidden rows are produced from one 40-layer
    /// backbone traversal, which is where MTP's decode speedup would come from.
    ///
    /// It does not currently come out ahead, and the reason is structural
    /// rather than a tuning problem. On a sparse MoE the cost of a verify pass
    /// tracks the *union* of the experts its rows route to, not the row count:
    /// rows sharing an expert ride along on one weight read (the grouping in
    /// `PrefillMoEGrouping` sorts by expert so this already happens), rows that
    /// do not each pay in full. Measured on Qwen3.6-35B-A3B, 40 layers,
    /// topK=8 of 256:
    ///
    ///     width 1   8.00 experts/layer   cost 1.000x
    ///     width 2  12.68 experts/layer   cost 1.585x   <- verifyGreedyPair
    ///
    /// Against that, acceptance of 57.4% emits 1.574 tokens per pass. Cost
    /// 1.585 versus benefit 1.574: the two cancel, and every other per-pass
    /// overhead turns it into a net loss (~0.85x end to end).
    ///
    /// Widening the block does not rescue it. Benefit is a geometric series
    /// capped at 1/(1-p) = 2.35, while the union keeps growing -- measured
    /// 5.18x at width 13 and 11.25x at width 42. Width 2 is the closest this
    /// model ever gets to break-even, and it still misses.
    ///
    /// So the lever is the verify **path**, not acceptance. **Measured on this
    /// install 2026-09-18** (256 greedy tokens, 86.9% acceptance, 1.869 emitted
    /// per pass, `benchmark/tinytitan_mtp_phases.py`): a pass costs **2.238x** a
    /// scalar token, and the union is not where it goes — `verify_routed_pair`
    /// measured **1.45x** the scalar routed GPU time, at or under the 1.585x
    /// model. The verify backbone alone is **1.946x** a 214.5 ms token, and its
    /// kernels account for ~217 ms of that: the union at the model's price plus
    /// a non-expert prefill path at **1.6-1.7x** where the model assumes 1.0x,
    /// because two rows run through the 32-token prefill kernels rather than the
    /// decode ones. The remaining **~200 ms/pass (0.93x a token)** is host and
    /// commit time — a fresh `MTLArgumentBuffer` and command buffer per tile, the
    /// cache plan, and the sequential fetch awaits nothing overlaps at width 2.
    /// `TINYTITAN_MTP_VERIFY=pair` recovers only 3-5% of that. Acceptance must
    /// still exceed ~0.585 merely to break even; reaching 92.6% did not help,
    /// which is what refutes the older "lever is acceptance" reading.
    /// Parsed once: the schedule cannot change mid-process, and
    /// ProcessInfo.environment is a dictionary copy per call.
    static let mtpVerifyScheduleResult =
        Result { try RuntimeMTPVerifySchedule.environmentValue() }

    func verifyGreedyPair(
        _ tokens: [Int32],
        startPosition: Int
    ) async throws -> TargetPairVerification {
        guard tokens.count == 2 else {
            throw PrefillError.chunkedUnsupported("MTP verification requires exactly two tokens")
        }
        let schedule = try Self.mtpVerifyScheduleResult.get()
        // The pair schedule plans the union of both rows' experts as one
        // cache plan, which needs the slot cache to hold at least 2*topK.
        // Below that (a sub-1 GiB budget) the tile path remains correct.
        let slotCount = model.routedExpertCacheSlotCount() ?? 0
        let pairMoE = schedule == .pair && slotCount >= 2 * cfg.topKExperts
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: config)
        let tBackbone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try await executePrefillChunk(
            tokens: tokens[...],
            startPosition: startPosition,
            outputMode: .logits,
            logits: verificationLogits,
            scratch: scratch,
            config: config,
            writeFinalHead: false,
            snapshotGDNAfterFirstToken: true,
            useTwoRowProjection: true,
            verifyRoutedMoE: pairMoE)
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: scratch.hidden,
            sourceOffset: 0,
            to: verificationHidden,
            destinationOffset: 0,
            size: 2 * residualWidth * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        try encodeVerifyHeadPair(commandBuffer: cb, hidden: scratch.hidden, scratch: scratch)
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: "verify_head", cb)
        return try finishVerifyPair(tBackbone: tBackbone, tHead: tHead)
    }

    /// The verify head for two consecutive residual rows at the start of
    /// `hidden`: final norm and lm_head into `verificationLogits` rows 0 and 1,
    /// with one lm_head read. Used by the pair verify; the W-row verify reads
    /// the head once for all its rows (`encodeVerifyHeadRows`).
    func encodeVerifyHeadPair(
        commandBuffer cb: MTLCommandBuffer,
        hidden: MTLBuffer,
        scratch: PrefillChunkScratchBuffers
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        if let hc = hyperConnection {
            // This stack does not end in an RMSNorm: it ends in the gated
            // mixer that collapses the residual streams. The fused pair head
            // folds a plain RMSNorm into the vocabulary GEMV, which on a wide
            // residual reads stream 0 as if it were the whole hidden state --
            // and since the verify pass produces every emitted token, that is
            // the whole output, not a rounding difference.
            let rowBytes = residualWidth * MemoryLayout<Float16>.stride
            for row in 0..<2 {
                try hc.encodeRead(
                    commandBuffer: cb,
                    streamsBuffer: hidden,
                    streamsOffset: row * rowBytes,
                    hcNorm: finalNorm.buffer,
                    hcNormOffset: Int(finalNorm.offset),
                    down: gateWeightsPublic(try model.hcMixerDown()),
                    up: gateWeightsPublic(try model.hcMixerUp()),
                    blockInput: scratch.normed,
                    blockInputOffset: row * cfg.hiddenSize
                        * MemoryLayout<Float16>.stride,
                    eps: 1e-6)
            }
            for row in 0..<2 {
                try encodeHeadGEMV(
                    commandBuffer: cb,
                    weights: lm.buffer, weightsOffset: Int(lm.offset),
                    scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                    x: scratch.normed,
                    xOffset: row * cfg.hiddenSize * MemoryLayout<Float16>.stride,
                    y: verificationLogits,
                    yOffset: row * cfg.vocabSize * MemoryLayout<Float16>.stride,
                    m: UInt32(cfg.vocabSize), n: UInt32(cfg.hiddenSize))
            }
            return
        }
        // One lm_head weight read for both rows. The former per-row loop
        // read the model's largest tensor twice per verify pass.
        try prefillFinalRowHead.encodeLogitsPair(
            commandBuffer: cb,
            hiddenBlock: hidden,
            rowStrideElements: cfg.hiddenSize,
            normWeight: finalNorm.buffer,
            normWeightOffset: Int(finalNorm.offset),
            weights: lm.buffer,
            weightsOffset: Int(lm.offset),
            scales: lm.buffer,
            scalesOffset: Int(lm.scaleOffset),
            biases: lm.buffer,
            biasesOffset: Int(lm.biasOffset),
            logits: verificationLogits,
            d: UInt32(cfg.hiddenSize),
            vocab: UInt32(cfg.vocabSize),
            rmsEps: 1e-6)
    }

    /// Argmax both verify rows and package the result. Shared by the two
    /// head paths so a family difference in the head cannot become a
    /// difference in what the verify pass reports.
    private func finishVerifyPair(
        tBackbone: UInt64,
        tHead: UInt64
    ) throws -> TargetPairVerification {
        let tArgmax = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let logits = verificationLogits.contents()
            .assumingMemoryBound(to: Float16.self)
        func argmax(row: Int) -> Int32 {
            let base = row * cfg.vocabSize
            var best = 0
            var bestValue = Float(logits[base])
            for index in 1..<cfg.vocabSize {
                let value = Float(logits[base + index])
                if value > bestValue {
                    bestValue = value
                    best = index
                }
            }
            return Int32(best)
        }
        let first = argmax(row: 0)
        let second = argmax(row: 1)
        let tDone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        return TargetPairVerification(
            predictionAfterFirst: first,
            predictionAfterSecond: second,
            hiddenRows: Data(
                bytes: verificationHidden.contents(),
                count: 2 * residualWidth * MemoryLayout<Float16>.stride),
            backboneNanos: tHead &- tBackbone,
            headNanos: tArgmax &- tHead,
            argmaxNanos: tDone &- tArgmax)
    }
}
