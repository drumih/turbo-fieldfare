import Foundation
import Metal
import TinyTitanCore

// The raw-completion loop's helpers: the MTP-backed streaming pass and the
// single sampling step, both called by `runRawCompletion`.
//
// Split out of `RawCompletion.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. Both widened from
// `private` to internal because their caller stays in RawCompletion.swift.

func runStreamingMTPCompletion(
    decoder: StreamingMTPDecoder,
    tokenizer: GFTokenizer,
    promptIds: [Int32],
    config: GenerationConfig,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig,
    start: RawCompletionStart,
    shouldStop: () -> Bool,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawDecodeResult {
    try config.validate()
    guard config.isPureGreedy else { throw StreamingMTPError.greedyOnly }
    guard case .reset = start else {
        throw GeneratorError.invalidContinuation(
            "MTP continuation snapshots are not yet persisted; start a fresh request")
    }
    guard !promptIds.isEmpty else { throw GeneratorError.emptyPrompt }

    let prefillStart = Date()
    var boundary = try await decoder.prepare(
        promptIds: promptIds,
        config: config,
        prefillConfig: prefillConfig,
        logits: scratch.logits
    ) { done in
        onProgress(.prefill(done: done, total: promptIds.count))
    }
    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    if ProcessInfo.processInfo.environment["TINYTITAN_ANE_MEMORY_TRACE"] == "1" {
        // The other end of the TT-004 question: whatever the ANE's E5RT arena
        // was, is it still resident once decode begins? Compare this line
        // between an ANE-prefilled run and a GPU-prefilled one.
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "[ane-mem] decode-start footprint=%.1f MiB prompt=%d tok\n",
                    ProcessMemory.physFootprintMiB(), promptIds.count
                ).utf8))
    }

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var backedHistory = promptIds
    var uncommitted: [Int32] = []
    var pending: [(token: Int32, backed: Bool)] = [(boundary, false)]
    var toolCallMarkers = ToolCallMarkerCounter()

    decodeLoop: while true {
        while !pending.isEmpty {
            try Task.checkCancellation()
            let item = pending.removeFirst()
            boundary = item.token
            generated += 1
            // Mirrors the scalar loop: `uncommitted` holds the last emitted
            // token iff advance has not yet committed it to the target KV
            // (R10). A token reported backed by the batch is already in the
            // KV, so it never sits uncommitted.
            uncommitted = item.backed ? [] : [item.token]
            toolCallMarkers.observe(
                item.token,
                start: tokenizer.toolCallStartID,
                end: tokenizer.toolCallEndID)

            if tokenizer.stopTokenIDs.contains(item.token)
                || config.extraStopTokens.contains(item.token)
            {
                // Same classification as the scalar loop above.
                if item.token == tokenizer.endOfTurnID {
                    reason = toolCallMarkers.isCompleteToolTurn ? .toolCalls : .endOfTurn
                } else {
                    reason = .eos
                }
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break decodeLoop
            }
            let visible = stopMatcher.push(try detok.push(item.token))
            onProgress(.token(index: generated - 1, id: item.token, delta: visible))
            let hitStop = stopMatcher.isStopped || shouldStop()
            let hitMax = generated >= config.maxNewTokens
            if hitStop || hitMax {
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                if hitStop {
                    reason = stopMatcher.isStopped ? .stopString : .external
                } else {
                    reason = .maxTokens
                }
                break decodeLoop
            }
        }

        // The boundary is reported backed only after the advance that commits
        // it to the target KV succeeds (R10): "reported backed" strictly means
        // "committed by a completed advance", so the final boundary token is
        // always accounted for in `kvBackedTokenIDs`.
        let batch = try await decoder.advance(boundaryToken: boundary)
        backedHistory.append(boundary)
        pending = batch.tokenIDs.enumerated().map { index, token in
            (token, index < batch.backedPrefixCount)
        }
        if batch.backedPrefixCount > 0 {
            backedHistory.append(contentsOf: batch.tokenIDs.prefix(batch.backedPrefixCount))
        }
    }

    return RawDecodeResult(
        prefillTokens: promptIds.count,
        cachedPromptTokens: 0,
        computedPrefillTokens: promptIds.count,
        prefillSeconds: prefillSeconds,
        newTokens: generated,
        decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason,
        kvPosition: decoder.targetPosition,
        kvBackedTokenIDs: backedHistory,
        uncommittedBoundaryTokenIDs: uncommitted)
}

/// Samples one token id.
///
/// `timing` is the runner whose `TINYTITAN_KERNEL_STATS` timeline this command
/// buffer joins, when the producer is one. Without it the sampler's GPU span
/// is invisible to the role summary *and* to the gap accounting, so it lands
/// inside the `head_logits->embed` transition and inflates what reads as idle.
/// That is not hypothetical: it hid a 15.45 ms/token Top-K kernel until the
/// gap was traced by hand.
func sampleOnce(
    scratch: RawCompletionScratch, context: MetalContext,
    history: [Int32], config: GenerationConfig, position: Int,
    timing: RealForwardRunner? = nil
) throws -> Int32 {
    guard let cb = context.queue.makeCommandBuffer() else {
        throw ModelError.residentBufferWrapFailed
    }
    try scratch.sampler.sample(
        commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
        history: history, config: config, position: position,
        outToken: scratch.outToken)
    cb.commit()
    cb.waitUntilCompleted()
    timing?.recordKernelGPU(role: "sample", cb)
    if ProcessInfo.processInfo.environment["TINYTITAN_LOGIT_TRACE"] == "1" {
        // The top-2 of this step's raw head output, for engine-agreement work
        // (TT-002): greedy argmax alone hides *how close* the decision was, and
        // a 4-bit install can lose a near-tie on one engine and not the other.
        // The head writes the logits buffer on any path that is not the fused
        // greedy one, which is what the servers build; a fused-head CLI run
        // would print a stale row, so the callers that need this pass
        // `forceLogitsHead`. Costs one linear scan of the vocabulary.
        let vocab = scratch.sampler.vocab
        let row = scratch.logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        var first = -Float.greatestFiniteMagnitude
        var second = first
        var firstID = 0
        var secondID = 0
        for index in 0..<vocab {
            let value = Float(row[index])
            if value > first {
                second = first
                secondID = firstID
                first = value
                firstID = index
            } else if value > second {
                second = value
                secondID = index
            }
        }
        let chosen = Int(scratch.outToken.contents().load(as: UInt32.self))
        FileHandle.standardError.write(
            Data(
                String(
                    format: "[logit] pos=%d chosen=%d top1=%d:%.4f top2=%d:%.4f margin=%.4f\n",
                    position, chosen, firstID, first, secondID, second, first - second
                ).utf8))
    }
    // Read after completion, while the row max is still the one this dispatch
    // wrote. A row with no finite logit leaves the sampler's in-range fallback
    // in `outToken`; returning it would report a broken model as a valid token,
    // and the generation would then feed that token back and loop on it.
    guard scratch.sampler.lastRowHadFiniteLogit else {
        throw GeneratorError.degenerateLogitsRow
    }
    return try validatedToken(
        scratch.outToken.contents().load(as: UInt32.self),
        vocab: scratch.sampler.vocab)
}

/// Checks a sampled id against the vocabulary before anything uses it.
///
/// The sampler's contract is an in-range id and its tests pin that, but an
/// out-of-range one has been seen intermittently under full-suite GPU load and
/// never identified. A token id indexes the embedding table and extends the KV
/// history, so an unchecked one is silent corruption exactly like the traps this
/// audit converted into reports; this makes it a named error instead, with the
/// id in it, so the next occurrence identifies itself.
func validatedToken(_ raw: UInt32, vocab: Int) throws -> Int32 {
    guard raw < UInt32(vocab) else {
        throw GeneratorError.samplerReturnedOutOfRangeToken(id: raw, vocab: vocab)
    }
    return Int32(bitPattern: raw)
}
