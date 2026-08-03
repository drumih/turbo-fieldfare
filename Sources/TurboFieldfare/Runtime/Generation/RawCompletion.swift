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
    if let multimodalInput {
        let ranges = multimodalInput.visionTokenRanges
        guard !ranges.isEmpty else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires at least one image embedding overlay")
        }
        var previousEnd = 0
        for (index, overlay) in multimodalInput.embeddingOverlays.enumerated() {
            let range = overlay.tokenRange
            guard range.lowerBound >= previousEnd,
                  range.upperBound <= promptIds.count else {
                throw PrefillError.chunkedUnsupported(
                    "image embedding overlays must be ordered, disjoint, and inside the prompt")
            }
            guard promptIds[range].allSatisfy({ $0 == tokenizer.imageID }) else {
                throw PrefillError.chunkedUnsupported(
                    "image embedding overlay \(index) does not align with image placeholder tokens")
            }
            previousEnd = range.upperBound
        }
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard multimodalInput == nil else {
            throw GeneratorError.invalidContinuation(
                "multimodal prompts cannot resume a text-only KV continuation")
        }
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }
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
    if let multimodalInput {
        guard let multimodal = producer as? any MultimodalChunkedPrefillRunner else {
            throw PrefillError.chunkedUnsupported(
                "image prompts require a multimodal chunked prefill runner")
        }
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        // Correct visual-block attention always uses the chunked path, even if
        // ordinary text prefill was disabled in runtime settings.
        let effectiveConfig = prefillConfig.mode == .chunked
            ? prefillConfig
            : PrefillRuntimeConfig.defaultChunked
        let result = try await multimodal.prefillMultimodal(
            tokens: prefillTokens,
            startPosition: position,
            input: multimodalInput,
            outputMode: mode,
            config: effectiveConfig,
            into: scratch.logits) { done in
                onProgress(.prefill(done: cachedPromptTokens + done,
                                    total: promptIds.count))
            }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion multimodal prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion multimodal prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    } else {
        switch prefillConfig.mode {
        case .chunked where producer is any ChunkedPrefillRunner:
            let chunked = producer as! any ChunkedPrefillRunner
            let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
            let result = try await chunked.prefillChunked(
                tokens: prefillTokens,
                startPosition: position,
                outputMode: mode,
                config: prefillConfig,
                into: scratch.logits) { done in
                    onProgress(.prefill(
                        done: cachedPromptTokens + done,
                        total: promptIds.count))
                }
            if mode == .logits, result.seed != .logitsWritten {
                throw PrefillError.unsupportedPrefillSeed(
                    "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
            }
            if case .greedyToken = result.seed, !config.isPureGreedy {
                throw PrefillError.unsupportedPrefillSeed(
                    "RawCompletion chunked prefill returned a greedy token for a sampling config")
            }
            position = result.newPosition
            prefillSeed = result.seed
            history.append(contentsOf: prefillTokens)
        case .chunked:
            throw PrefillError.chunkedUnsupported(
                PrefillError.chunkedRequiresChunkedRunnerReason)
        case .off:
            for token in prefillTokens {
                try Task.checkCancellation()
                try await producer.produce(
                    token: token,
                    position: position,
                    into: scratch.logits)
                position += 1
                history.append(token)
                onProgress(.prefill(done: position, total: promptIds.count))
            }
        }
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var consecutiveInvisibleTokens = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            tokenID = sampleOnce(scratch: scratch, context: context,
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
        if visible.isEmpty {
            consecutiveInvisibleTokens += 1
        } else {
            consecutiveInvisibleTokens = 0
        }
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        if consecutiveInvisibleTokens >= 64 {
            throw GeneratorError.noVisibleOutput
        }

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : .maxTokens
            break
        }

        history.append(tokenID)
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
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
