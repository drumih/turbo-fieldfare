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
    /// Times the JSON grammar vetoed the GPU-sampled token and the
    /// probability-ordered fallback chose instead. 0 unless `forceJSON`.
    public var grammarVetoes: Int = 0
    /// Times the tool-call grammar vetoed the GPU-sampled token. 0 unless the
    /// tool-call constraint ran.
    public var toolGrammarVetoes: Int = 0
    /// Of those, the ones where the rejected token died while the function name
    /// was still being spelled — the model wanted a tool that was not declared.
    public var toolNameVetoes: Int = 0
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
    /// logits buffer. Anything else — including JSON-constrained decoding,
    /// which must inspect and veto candidates — needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1 && !forceJSON
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
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             toolFilter: ToolCallTokenFilter? = nil,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    // The two grammars speak different languages about the same token stream;
    // there is no meaningful intersection to enforce.
    guard !(config.forceJSON && toolFilter != nil) else {
        throw GeneratorError.invalidGenerationConfig(
            "forceJSON and the tool-call grammar cannot constrain the same generation")
    }
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    // A constrained decode has to see and veto candidates, which the fused
    // head's GPU argmax never surfaces.
    guard !fusedGreedy || (config.isPureGreedy && toolFilter == nil) else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
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
    switch prefillConfig.mode {
    case .chunked where producer is any ChunkedPrefillRunner:
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
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
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
    }

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    let jsonFilter = config.forceJSON
        ? JSONTokenFilter(tokenizer: tokenizer, extraStopIDs: config.extraStopTokens)
        : nil
    let grammar: (any TokenGrammarFilter)? = jsonFilter ?? toolFilter
    var jsonAssembler = UTF8StreamAssembler()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []

    // Under forceJSON the emitted text is the automaton-validated byte stream,
    // NOT detokenizer output: the library decode applies `cleanUp` (` ,`→`,`)
    // and drops undecodable byte-fallback tails, either of which would desync
    // what the grammar guaranteed from what the caller receives.
    func flushTail() -> String {
        if jsonFilter != nil { return jsonAssembler.flush() }
        // Inside an open `<|tool_call>` region the byte-fallback pieces the
        // detokenizer is holding back are raw bytes of an argument — the `é`
        // tail of a path — not assistant prose. The region dies with the
        // generation and is never decoded, so those bytes are not text anyone
        // may show: dropping them is the whole point of holding them back.
        if toolFilter?.isInsideCall == true { return "" }
        return detok.flush()
    }

    while true {
        try Task.checkCancellation()

        let sampled: Int32?
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                sampled = Int32(bitPattern: token)
            case .logitsWritten:
                sampled = try sampleOnce(scratch: scratch, context: context,
                                         history: history, config: config, position: generated,
                                         grammar: grammar)
            }
        } else if fusedGreedy {
            sampled = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else {
            sampled = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated,
                                     grammar: grammar)
        }
        // The grammar admits no token at all. Under the tool-call grammar that
        // is the region byte cap: at `GemmaToolCallParser.maximumBytes` every
        // token overflows it and `<tool_call|>` cannot close an argument object
        // that is still open, so the region can only be abandoned. That is the
        // budget running out inside a call, which the caller already handles —
        // report it as such instead of failing the whole request. Force-json
        // has no bounded region to abandon: there, an empty vocabulary is a
        // real configuration failure.
        guard let tokenID = sampled else {
            guard toolFilter != nil else {
                throw GeneratorError.invalidGenerationConfig(
                    "force-json: no token in the vocabulary can extend the JSON document")
            }
            let tail = stopMatcher.push(flushTail()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = .maxTokens
            break
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if jsonFilter != nil {
                // Grammar-gated stops always mean "the JSON may end here";
                // .toolCalls/.endOfTurn semantics don't apply to forced JSON.
                reason = .eos
            } else if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(flushTail()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta: String
        if let jsonFilter {
            delta = jsonAssembler.push(jsonFilter.lastAcceptedBytes)
        } else {
            delta = detok.push(tokenID)
        }
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        if let jsonFilter, jsonFilter.isComplete {
            let tail = stopMatcher.push(flushTail()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = .eos
            break
        }

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(flushTail()) + stopMatcher.finish()
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
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           grammarVetoes: jsonFilter?.vetoCount ?? 0,
                           toolGrammarVetoes: toolFilter?.vetoCount ?? 0,
                           toolNameVetoes: toolFilter?.nameVetoCount ?? 0)
}

/// nil when the active grammar accepts nothing: the caller decides whether that
/// ends the generation or fails it.
private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int,
                        grammar: (any TokenGrammarFilter)?) throws -> Int32? {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb.error)
    let sampled = Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
    guard let grammar, !grammar.tryAccept(sampled) else { return sampled }
    grammar.noteVeto()
    return grammarFallbackToken(scratch: scratch, filter: grammar)
}

/// The GPU-sampled token broke the active grammar. `scratch.probs` (shared
/// storage, written by the completed softcap+softmax pass) is walked in
/// probability order — first grammar-accepted token wins. A capped candidate
/// set covers the common case; the full vocabulary is scanned only if every
/// high-probability candidate is grammar-invalid. nil when the sweep comes back
/// empty: the grammar has painted itself into a corner.
private func grammarFallbackToken(scratch: RawCompletionScratch,
                                  filter: any TokenGrammarFilter) -> Int32? {
    let vocab = scratch.sampler.vocab
    let probs = scratch.probs.contents().bindMemory(to: Float16.self, capacity: vocab)

    var top: [(id: Int32, p: Float16)] = []
    top.reserveCapacity(257)
    var floor: Float16 = 0
    for id in 0..<vocab {
        let p = probs[id]
        guard p > floor else { continue }
        top.append((Int32(id), p))
        if top.count > 256 {
            let minIndex = top.indices.min { top[$0].p < top[$1].p }!
            top.remove(at: minIndex)
            floor = top.min { $0.p < $1.p }!.p
        }
    }
    top.sort { $0.p > $1.p }
    for candidate in top where filter.tryAccept(candidate.id) {
        return candidate.id
    }

    let remaining = (0..<vocab)
        .map { (id: Int32($0), p: probs[$0]) }
        .sorted { $0.p > $1.p }
    for candidate in remaining where filter.tryAccept(candidate.id) {
        return candidate.id
    }
    return nil
}
