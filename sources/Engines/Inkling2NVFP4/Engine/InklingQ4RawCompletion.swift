import Foundation
import Metal
import TinyTitanCore

/// The Inkling 2 Q4 generation loop behind the shared session (TT-122): the
/// session's `runRawCompletion` stage for this variant.
///
/// The decode is the Inkling route's, moved: greedy without a grammar or a
/// penalty takes the decoder's GPU argmax, anything else reads the decoder's
/// FP32 logits, applies the grammar's mask and draws with `InklingQ4Sampler`.
/// What the session needs on top is the shared contract: resume from a cached
/// prefix, prompt progress, a token stream with its text deltas, stop strings
/// for a raw completion, and the KV-backed ids the prompt cache publishes.
///
/// A token is *committed* once the decoder has been fed it. The token that
/// ended the loop is sampled but never fed, so it is the uncommitted boundary
/// the prompt cache replays (the same accounting as the other engines' loops).
/// lint:allow-long the generation loop: start, prefill, then token-by-token
/// decode with stop handling and progress; the loop body reads and writes the
/// same decode state on every iteration.
package func runInklingQ4Completion(_ request: RawCompletionRequest) async throws
    -> RawDecodeResult
{
    guard let runner = request.producer as? InklingQ4Runner else {
        return try await runInklingQ4BatchedCompletion(request)
    }
    let slot = request.slot
    let config = request.configuration
    try config.validate()
    let prompt = request.promptIDs
    guard !prompt.isEmpty else { throw GeneratorError.emptyPrompt }
    let cached: Int
    switch request.start {
    case .reset:
        cached = 0
        runner.reset(slot: slot)
    case .resume(let count):
        guard count > 0, count < prompt.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt"
            )
        }
        try runner.prepareForContinuation(expectedPosition: count, slot: slot)
        cached = count
    }
    guard prompt.count - cached + config.maxNewTokens <= runner.maxContext - cached else {
        throw GeneratorError.contextOverflow(
            prompt: prompt.count, maxNew: config.maxNewTokens, maxContext: runner.maxContext)
    }
    let tokenizer = request.tokenizer
    let sampler = InklingQ4Sampler(
        temperature: config.temperature, topP: config.topP ?? 1, topK: config.topK ?? 0,
        presencePenalty: config.presencePenalty, repetitionPenalty: config.repetitionPenalty,
        seed: config.seed)
    let generator = sampler.makeGenerator()
    let constraint = config.constraint
    let greedy =
        sampler.isGreedy && sampler.repetitionPenalty == 1 && sampler.presencePenalty == 0
        && constraint == nil
    // A structured answer is one text message: its `<|end_message|>` ends it
    // (the grammar's table admits that token once the document is complete).
    var stops = tokenizer.stopTokenIDs.union(config.extraStopTokens)
    if constraint != nil { stops.formUnion(tokenizer.documentEndTokenIDs) }

    if greedy && slot == 0 && cached == 0 && runner.hasMTP {
        return try await runInklingQ4MTPCompletion(request, runner: runner, stops: stops)
    }

    // Prefill, stopping at each interior checkpoint the session asked for.
    let prefillStart = Date()
    var history = Array(prompt.prefix(cached))
    var segmentStart = cached
    for point in request.checkpointPositions.sorted() where point > cached && point < prompt.count {
        _ = try runner.feed(
            prompt[segmentStart..<point].map(Int.init), at: segmentStart, slot: slot, logits: false)
        history.append(contentsOf: prompt[segmentStart..<point])
        request.onProgress(.prefill(done: point, total: prompt.count))
        segmentStart = point
        request.onPrefillCheckpoint?(point)
    }
    var step = try runner.feed(
        prompt[segmentStart...].map(Int.init), at: segmentStart, slot: slot, logits: !greedy)
    history.append(contentsOf: prompt[segmentStart...])
    request.onProgress(.prefill(done: prompt.count, total: prompt.count))
    let prefillSeconds = Date().timeIntervalSince(prefillStart)

    let decodeStart = Date()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var text = InklingQ4TextStream(tokenizer: tokenizer)
    var generated = 0
    var reason: StopReason = .maxTokens
    var boundary: [Int32] = []
    while true {
        try Task.checkCancellation()
        let token: Int32
        if greedy {
            token = Int32(step.next)
        } else {
            var logits = step.logits ?? []
            if let constraint {
                let mask = constraint.allowedMask()
                guard !mask.isEmpty else { throw GeneratorError.constrainedDecodeStalled }
                mask.apply(toLogits: &logits)
            }
            token = Int32(sampler.pick(logits, history: history, using: generator))
        }
        generated += 1
        boundary = [token]
        if stops.contains(token) {
            reason = token == tokenizer.endOfTurnID || constraint != nil ? .endOfTurn : .eos
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            break
        }
        if let constraint, !constraint.observe(token) {
            throw GeneratorError.constrainedDecodeViolation(id: token)
        }
        let visible = stopMatcher.push(text.push(token))
        request.onProgress(.token(index: generated - 1, id: token, delta: visible))
        if stopMatcher.isStopped || request.shouldStop() || generated >= config.maxNewTokens {
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            if stopMatcher.isStopped {
                reason = .stopString
            } else if generated >= config.maxNewTokens {
                reason = .maxTokens
            } else {
                reason = .external
            }
            break
        }
        history.append(token)
        step = try runner.feed(
            [Int(token)], at: runner.position(forSlot: slot), slot: slot, logits: !greedy)
        boundary = []
    }
    return RawDecodeResult(
        prefillTokens: prompt.count, cachedPromptTokens: cached,
        computedPrefillTokens: prompt.count - cached, prefillSeconds: prefillSeconds,
        newTokens: generated, decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason, kvPosition: runner.position(forSlot: slot), kvBackedTokenIDs: history,
        uncommittedBoundaryTokenIDs: boundary)
}

/// Multi-slot completion loop for `BatchedLogitProducer` (TT-124): drives
/// `resetSequence(slot:)`, `prefillChunked(..., slot:)`, and `produce(..., slot:)`
/// through the slot-aware producer protocol so concurrent requests share one
/// model and interleave safely across slots.
/// Per-slot FP16 logits scratch allocated once per slot at session load for
/// multi-slot batched completions (TT-124).
///
/// unchecked-invariant: owned by one slot in `ServerModelSession.scratches` and
/// only written/read by the single request currently holding that slot.
package final class InklingQ4LogitsScratch: @unchecked Sendable {
    let buffer: MTLBuffer
    let vocabSize: Int

    package init(context: MetalContext, vocabSize: Int) throws {
        let bytes = max(vocabSize, 1) * MemoryLayout<Float16>.size
        guard let buffer = context.device.makeBuffer(length: bytes, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("Inkling2 slot logits scratch")
        }
        self.buffer = buffer
        self.vocabSize = vocabSize
    }
}

private func runInklingQ4BatchedCompletion(_ request: RawCompletionRequest) async throws
    -> RawDecodeResult
{
    let config = request.configuration
    try config.validate()
    let prompt = request.promptIDs
    guard !prompt.isEmpty else { throw GeneratorError.emptyPrompt }
    let slot = request.slot
    let producer = request.producer
    let maxContext = (producer as? any ContextWindowReporting)?.maxContext ?? Int.max
    guard prompt.count + config.maxNewTokens <= maxContext else {
        throw GeneratorError.contextOverflow(
            prompt: prompt.count, maxNew: config.maxNewTokens, maxContext: maxContext)
    }
    await producer.resetSequence(slot: slot)
    let tokenizer = request.tokenizer
    let sampler = InklingQ4Sampler(
        temperature: config.temperature, topP: config.topP ?? 1, topK: config.topK ?? 0,
        presencePenalty: config.presencePenalty, repetitionPenalty: config.repetitionPenalty,
        seed: config.seed)
    let generator = sampler.makeGenerator()
    let constraint = config.constraint
    var stops = tokenizer.stopTokenIDs.union(config.extraStopTokens)
    if constraint != nil { stops.formUnion(tokenizer.documentEndTokenIDs) }

    guard let scratch = request.scratch.payload as? InklingQ4LogitsScratch else {
        throw MetalError.invalidState("Inkling 2 batched completion requires its logits scratch")
    }
    let logitsBuffer = scratch.buffer
    let vocabSize = scratch.vocabSize
    let prefillStart = Date()
    var position = 0
    if let chunked = producer as? any ChunkedPrefillRunner {
        let result = try await chunked.prefillChunked(
            tokens: prompt[...], startPosition: 0, slot: slot, outputMode: .logits,
            config: request.prefillConfiguration, into: logitsBuffer, onProgress: { _ in })
        position = result.newPosition
    } else {
        for (index, token) in prompt.enumerated() {
            try await producer.produce(
                token: token, position: index, slot: slot, into: logitsBuffer)
        }
        position = prompt.count
    }
    var history = prompt
    request.onProgress(.prefill(done: prompt.count, total: prompt.count))
    let prefillSeconds = Date().timeIntervalSince(prefillStart)

    let decodeStart = Date()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var text = InklingQ4TextStream(tokenizer: tokenizer)
    var generated = 0
    var reason: StopReason = .maxTokens
    var boundary: [Int32] = []
    while true {
        try Task.checkCancellation()
        var logits = readFP16Logits(logitsBuffer, count: vocabSize)
        if let constraint {
            let mask = constraint.allowedMask()
            guard !mask.isEmpty else { throw GeneratorError.constrainedDecodeStalled }
            mask.apply(toLogits: &logits)
        }
        let token = Int32(sampler.pick(logits, history: history, using: generator))
        generated += 1
        boundary = [token]
        if stops.contains(token) {
            reason = token == tokenizer.endOfTurnID || constraint != nil ? .endOfTurn : .eos
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            break
        }
        if let constraint, !constraint.observe(token) {
            throw GeneratorError.constrainedDecodeViolation(id: token)
        }
        let visible = stopMatcher.push(text.push(token))
        request.onProgress(.token(index: generated - 1, id: token, delta: visible))
        if stopMatcher.isStopped || request.shouldStop() || generated >= config.maxNewTokens {
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            reason =
                stopMatcher.isStopped
                ? .stopString
                : (generated >= config.maxNewTokens ? .maxTokens : .external)
            break
        }
        history.append(token)
        try await producer.produce(token: token, position: position, slot: slot, into: logitsBuffer)
        position += 1
        boundary = []
    }
    return RawDecodeResult(
        prefillTokens: prompt.count, cachedPromptTokens: 0,
        computedPrefillTokens: prompt.count, prefillSeconds: prefillSeconds,
        newTokens: generated, decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason, kvPosition: position, kvBackedTokenIDs: history,
        uncommittedBoundaryTokenIDs: boundary)
}

private func readFP16Logits(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let n = min(count, buffer.length / MemoryLayout<Float16>.size)
    let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: n)
    return (0..<n).map { Float(ptr[$0]) }
}

/// The visible text of a growing answer as deltas: the decode of every token
/// so far, control tokens skipped, with a trailing U+FFFD held back until the
/// next token completes the character (TT-128's rule).
struct InklingQ4TextStream {
    let tokenizer: GFTokenizer
    private var tokens: [Int32] = []
    private var emitted = ""

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func push(_ token: Int32) -> String {
        tokens.append(token)
        var stable = Substring(tokenizer.decode(tokens, skipSpecialTokens: true))
        while stable.last == "\u{FFFD}" { stable = stable.dropLast() }
        guard stable.hasPrefix(emitted), stable.count > emitted.count else { return "" }
        let delta = String(stable.dropFirst(emitted.count))
        emitted = String(stable)
        return delta
    }

    mutating func flush() -> String {
        let text = tokenizer.decode(tokens, skipSpecialTokens: true)
        guard text.hasPrefix(emitted), text.count > emitted.count else { return "" }
        let delta = String(text.dropFirst(emitted.count))
        emitted = text
        return delta
    }
}
