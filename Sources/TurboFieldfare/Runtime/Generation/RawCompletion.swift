import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)

    /// How many of `promptCount` tokens the producer's KV already holds.
    func cachedPromptTokens(promptCount: Int, producer: any LogitProducer) throws -> Int {
        switch self {
        case .reset:
            return 0
        case .resume(let count):
            guard count > 0, count < promptCount else {
                throw GeneratorError.invalidContinuation(
                    "cached prompt token count must be greater than zero and less than the effective prompt")
            }
            guard producer is any ContinuableLogitProducer else {
                throw GeneratorError.invalidContinuation(
                    "producer does not support continuation")
            }
            return count
        }
    }
}

/// What the prefill's seed is for. A decode consumes it, so a producer that
/// returned the wrong kind has to be refused before the loop reads it. A
/// restore has no decode and discards it.
enum RawPrefillSeedUse {
    case decode(isPureGreedy: Bool)
    case discarded
}

struct RawPrefillOutcome {
    let computedPrefillTokens: Int
    let position: Int
    let seed: PrefillSeed?
    let prefillSeconds: Double
    let history: [Int32]
}

/// The prefill half of `runRawCompletion`, on its own so that restoring a
/// stored conversation goes through it too.
///
/// A second implementation would be a second set of chunk boundaries, and chunk
/// boundaries are the one thing a replayed KV has to share with the KV it is
/// reproducing: image spans are indivisible work items, chunks under 32 tokens
/// take a different GEMV than larger ones, and `.prefill` progress is what the
/// app's gauge animates. Restore is a prefill with the decode left off, not a
/// different way to fill a cache.
func runRawPrefill(
    producer: any LogitProducer,
    promptIds: [Int32],
    multimodalInput: MultimodalPrefillInput?,
    prefillConfig requestedPrefillConfig: PrefillRuntimeConfig,
    start: RawCompletionStart,
    outputMode: PrefillOutputMode,
    seedUse: RawPrefillSeedUse,
    historyReserve: Int,
    scratch: RawCompletionScratch,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawPrefillOutcome {
    let cachedPromptTokens = try start.cachedPromptTokens(
        promptCount: promptIds.count, producer: producer)
    let computedPrefillTokens = promptIds.count - cachedPromptTokens
    if let multimodalInput {
        // Under resume the multimodal input is the tail that still has to be
        // prefilled, so it must match the prompt suffix rather than the whole
        // prompt. `prefillMultimodal` already prefills from `startPosition`.
        guard multimodalInput.effectiveTokenIDs
            == Array(promptIds.dropFirst(cachedPromptTokens)) else {
            throw GeneratorError.invalidContinuation(
                "multimodal effective token IDs do not match the prompt")
        }
    }
    // Image spans are served only by the complete chunked prefill path, and a
    // prefill config that cannot serve them is a performance setting, not a
    // decision to drop the images. Coercing here rather than at each call site
    // is what makes it total: the default argument is `.current`, which the
    // environment can turn `.off`, so a caller that passes no config at all
    // reaches this too. The call sites still coerce first, to report it.
    let prefillConfig = multimodalInput == nil
        ? requestedPrefillConfig
        : (requestedPrefillConfig.coercedForImagePrompt() ?? requestedPrefillConfig)

    func checkSeed(_ seed: PrefillSeed, path: String, requiresWrittenLogits: Bool) throws {
        guard case .decode(let isPureGreedy) = seedUse else { return }
        if requiresWrittenLogits, outputMode == .logits, seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill requested logits but producer returned \(seed)")
        }
        if case .greedyToken = seed, !isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion \(path) prefill returned a greedy token for a sampling config")
        }
    }

    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(historyReserve)

    switch start {
    case .reset:
        producer.reset()
    case .resume:
        let continuable = producer as! any ContinuableLogitProducer
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]
    switch (multimodalInput, prefillConfig.mode) {
    case (.some(let input), .chunked) where producer is any MultimodalPrefillRunner:
        let multimodal = producer as! any MultimodalPrefillRunner
        let result = try await multimodal.prefillMultimodal(
            input: input,
            startPosition: position,
            outputMode: outputMode,
            config: prefillConfig,
            into: scratch.logits
        ) { done in
            // The suffix-local count plus what the KV already holds, as the
            // chunked text path reports it. Without the offset a
            // 300-token image turn on a 5,000-token KV showed 1 of 5,300.
            onProgress(.prefill(done: cachedPromptTokens + done,
                                total: promptIds.count))
        }
        try checkSeed(result.seed, path: "multimodal", requiresWrittenLogits: true)
        position = result.newPosition
        prefillSeed = result.seed
        // Only the suffix: under resume the cached prefix is already in history,
        // and appending the whole prompt would duplicate it and desynchronise
        // history from the KV position.
        history.append(contentsOf: prefillTokens)
    case (.some, _):
        // The coercion above leaves an image prompt in chunked mode, so the
        // only way here is a producer that cannot run image spans at all —
        // which no config change can fix.
        throw PrefillError.chunkedUnsupported(
            "multimodal prefill requires a MultimodalPrefillRunner-backed runtime")
    case (.none, .chunked) where producer is any ChunkedPrefillRunner:
        let chunked = producer as! any ChunkedPrefillRunner
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: outputMode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        try checkSeed(result.seed, path: "chunked", requiresWrittenLogits: true)
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case (.none, .chunked):
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case (.none, _):
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    return RawPrefillOutcome(
        computedPrefillTokens: computedPrefillTokens,
        position: position,
        seed: prefillSeed,
        prefillSeconds: Date().timeIntervalSince(prefillStart),
        history: history)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public let withheldTrailingKVTokens: Int

    public init(prefillTokens: Int,
                cachedPromptTokens: Int,
                computedPrefillTokens: Int,
                prefillSeconds: Double,
                newTokens: Int,
                decodeSeconds: Double,
                reason: StopReason,
                kvPosition: Int,
                kvBackedTokenIDs: [Int32],
                uncommittedBoundaryTokenIDs: [Int32],
                withheldTrailingKVTokens: Int = 0) {
        self.prefillTokens = prefillTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.computedPrefillTokens = computedPrefillTokens
        self.prefillSeconds = prefillSeconds
        self.newTokens = newTokens
        self.decodeSeconds = decodeSeconds
        self.reason = reason
        self.kvPosition = kvPosition
        self.kvBackedTokenIDs = kvBackedTokenIDs
        self.uncommittedBoundaryTokenIDs = uncommittedBoundaryTokenIDs
        self.withheldTrailingKVTokens = withheldTrailingKVTokens
    }
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// `@unchecked Sendable`: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             multimodalInput: MultimodalPrefillInput? = nil,
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    // Derived here as well as inside `runRawPrefill` so an invalid resume count
    // is still refused before the context-overflow check, which is the order
    // callers have always seen.
    let cachedPromptTokens = try start.cachedPromptTokens(
        promptCount: promptIds.count, producer: producer)

    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }

    let prefill = try await runRawPrefill(
        producer: producer,
        promptIds: promptIds,
        multimodalInput: multimodalInput,
        prefillConfig: prefillConfig,
        start: start,
        outputMode: fusedGreedy ? .greedyIfAvailable : .logits,
        seedUse: .decode(isPureGreedy: config.isPureGreedy),
        historyReserve: promptIds.count + config.maxNewTokens,
        scratch: scratch,
        onProgress: onProgress)
    let computedPrefillTokens = prefill.computedPrefillTokens
    var history = prefill.history
    var position = prefill.position
    let prefillSeed = prefill.seed

    let decodeStart = Date()
    let prefillSeconds = prefill.prefillSeconds
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var trailingInvisibleTokens = 0

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                         history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        // Cancellation is not a stop-string match: reporting it as one made a
        // user pressing Stop indistinguishable from a configured stop string,
        // and `stopStringFiltered` is computed from `isStopped` rather than the
        // reason, so the two disagreed about the same run.
        let hitStopString = stopMatcher.isStopped
        let cancelled = !hitStopString && shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || cancelled || hitMax {
            if !visible.isEmpty { trailingInvisibleTokens = 0 }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : (cancelled ? .cancelled : .maxTokens)
            break
        }

        history.append(tokenID)
        trailingInvisibleTokens = visible.isEmpty ? trailingInvisibleTokens + 1 : 0
        try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           withheldTrailingKVTokens: reason == .stopString
                               ? trailingInvisibleTokens : 0)
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb)
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
