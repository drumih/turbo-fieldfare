import Foundation
import Metal

/// What a W-row verify pass produced: each row's greedy prediction and where
/// the time went.
package struct WideVerification: Sendable {
    /// `predictions[i]` is the greedy token after row i's input; empty when
    /// the pass captured the rows' logits instead (`verifyRowLogits`).
    package let predictions: [Int32]
    package let backboneNanos: UInt64
    package let headNanos: UInt64
}

/// The W-row verify pass for the wide-verify probe (docs/plan-wide-verify.md,
/// "dedicated-path probe").
///
/// The MTP pair verify, widened: one backbone traversal over W rows of one
/// sequence, with the routed experts of all W rows planned as one union against
/// the decode slot cache (`encodeRoutedMoEVerifyRows`), so a pass pays for the
/// experts its rows share once and keeps the decode cache pinned. The generic
/// chunked prefill instead unpins that cache and runs the prompt tile scheduler,
/// which is what gate 2 measured at ~1.35 s of fixed cost per pass.
///
/// This is a cost instrument, not yet a decode path: it makes no claim of byte
/// identity with scalar decode (task 3), and it leaves the KV cache and GDN
/// state advanced by W rows -- the caller restores a snapshot.
///
/// Rollback window lifetime (task 9). Every pass opens a wide-verify rollback
/// window before its backbone (`beginWideVerifyRollbackWindow`) and returns
/// with it still open, so a caller can keep only the first k rows. Closing it
/// is the caller's responsibility, by exactly one of
/// `rollbackWideVerify(acceptedRows:tokens:startPosition:)` (a partial accept)
/// or `endWideVerifyRollbackWindow()` (keep all W rows). A caller that never
/// rolls back -- the `verifyprobe` and `passcost` benches -- need not: the
/// window is also closed by `restoreInferenceState`, `prepareForContinuation`,
/// `reset()` and any other `resetTransientState()`, and the next pass reopens
/// it. A window left open after its pass is inert: every linear layer has
/// retained its W rows, so the retention hooks of later passes do nothing.
/// A pass that throws resets the runner, which closes the window.
extension RealForwardRunner {
    /// The widest pass whose expert union is guaranteed to fit the routed slot
    /// cache, or 0 when there is no slot cache to plan against.
    package var wideVerifyMaximumRows: Int {
        guard let slots = model.routedExpertCacheSlotCount(), cfg.topKExperts > 0 else {
            return 0
        }
        return slots / cfg.topKExperts
    }

    package func verifyGreedyRows(
        _ tokens: [Int32],
        startPosition: Int
    ) async throws -> WideVerification {
        let rows = tokens.count
        guard rows >= 2, rows <= wideVerifyMaximumRows else {
            throw PrefillError.chunkedUnsupported(
                "wide verify needs 2...\(wideVerifyMaximumRows) rows, got \(rows)")
        }
        try await forwardStepGate.acquire()
        do {
            let result = try await runVerifyGreedyRows(tokens, startPosition: startPosition)
            await forwardStepGate.release()
            return result
        } catch {
            await forwardStepGate.release()
            throw error
        }
    }

    /// `rowLogits`, when given, receives every row's FP16 logits
    /// (`[rows, vocab]`, see `verifyRowLogits`) and the host argmax is skipped;
    /// without it the pass is the greedy one, unchanged.
    func runVerifyGreedyRows(
        _ tokens: [Int32],
        startPosition: Int,
        rowLogits: MTLBuffer? = nil
    ) async throws -> WideVerification {
        let rows = tokens.count
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)
        let scratch = try ensurePrefillScratch(config: config)
        let tBackbone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // The rollback window's GDN snapshot is queued ahead of the pass and
        // only reads live state; the pass's retention blits only read its
        // projections, so the pass's outputs do not change. The window stays
        // open when this returns (see the lifetime note on `verifyGreedyRows`).
        try beginWideVerifyRollbackWindow(rows: rows, startPosition: startPosition)
        wideVerifyDumpStart = startPosition
        decodeOrderVerify = true
        defer {
            wideVerifyDumpStart = nil
            decodeOrderVerify = false
        }
        do {
            try await executePrefillChunk(
                tokens: tokens[...],
                startPosition: startPosition,
                outputMode: .logits,
                logits: verificationLogits,
                scratch: scratch,
                config: config,
                writeFinalHead: false,
                useTwoRowProjection: rows == 2,
                verifyRoutedMoE: true)
        } catch {
            reset()
            throw error
        }
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        // The head two rows at a time through the pair head: W/2 lm_head reads
        // where a W-row head would need one, so the head time here is an upper
        // bound and is reported separately from the backbone.
        let rowBytes = residualWidth * MemoryLayout<Float16>.stride
        var predictions: [Int32] = []
        predictions.reserveCapacity(rows)
        var row = 0
        while row < rows {
            let pairRows = min(2, rows - row)
            guard let cb = ctx.queue.makeCommandBuffer(),
                let blit = cb.makeBlitCommandEncoder()
            else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(
                from: scratch.hidden, sourceOffset: row * rowBytes,
                to: verificationHidden, destinationOffset: 0,
                size: pairRows * rowBytes)
            blit.endEncoding()
            try encodeVerifyHeadPair(
                commandBuffer: cb, hidden: verificationHidden, scratch: scratch)
            if let rowLogits {
                try encodeRowLogitsCapture(
                    commandBuffer: cb, into: rowLogits, firstRow: row, rows: pairRows)
            }
            cb.commit()
            try waitForCompletion(cb)
            recordKernelGPU(role: "wide_verify_head", cb)
            for offset in 0..<pairRows {
                if rowLogits == nil { predictions.append(verificationArgmax(row: offset)) }
                if activationDumpActive(position: startPosition + row + offset) {
                    dumpActivation(
                        "verify_logits", verificationLogits, count: cfg.vocabSize,
                        position: startPosition + row + offset,
                        offset: offset * cfg.vocabSize * MemoryLayout<Float16>.stride)
                }
            }
            row += pairRows
        }
        return WideVerification(
            predictions: predictions,
            backboneNanos: tHead &- tBackbone,
            headNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tHead)
    }

    private func verificationArgmax(row: Int) -> Int32 {
        let logits = verificationLogits.contents().assumingMemoryBound(to: Float16.self)
        let base = row * cfg.vocabSize
        var best = 0
        var bestValue = Float(logits[base])
        for index in 1..<cfg.vocabSize where Float(logits[base + index]) > bestValue {
            best = index
            bestValue = Float(logits[base + index])
        }
        return Int32(best)
    }
}
