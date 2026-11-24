import Foundation
import TinyTitanCore

/// The generation loop (`EngineFactory.runRawCompletion`): the contract every
/// plugin's loop keeps, written out once.
///
/// - Start: `.reset` clears the state; `.resume(n)` continues a restored
///   prefix of `n` prompt tokens and prefills only the rest.
/// - Prefill, stopping at each requested interior checkpoint (the session
///   snapshots the state there), reporting `.prefill` progress.
/// - Decode token by token, reporting `.token` as each one is chosen -- this is
///   what makes every front end stream -- and `.tail` once at the end.
/// - Stop on a stop token (turn end), a stop string, the session's
///   `shouldStop`, or the token budget.
/// - Return the KV-backed ids (every token fed) and the one uncommitted
///   boundary token (chosen, never fed): the prompt cache publishes these.
///
/// A real plugin samples with its own sampler and applies the session's JSON
/// grammar (`configuration.constraint`) to the logits before choosing.
public func runTemplateCompletion(_ request: RawCompletionRequest) async throws -> RawDecodeResult {
    guard let runner = request.producer as? TemplateRunner else {
        throw TemplateError.notThisRunner
    }
    let prompt = request.promptIDs
    guard !prompt.isEmpty else { throw GeneratorError.emptyPrompt }
    let config = request.configuration
    let cached: Int
    switch request.start {
    case .reset:
        cached = 0
        runner.reset()
    case .resume(let count):
        try runner.prepareForContinuation(expectedPosition: count)
        cached = count
    }
    let started = Date()
    var history = Array(prompt.prefix(cached))
    var at = cached
    for point in request.checkpointPositions.sorted() where point > at && point < prompt.count {
        try runner.feed(Array(prompt[at..<point]), at: at)
        history += prompt[at..<point]
        request.onProgress(.prefill(done: point, total: prompt.count))
        at = point
        request.onPrefillCheckpoint?(point)
    }
    try runner.feed(Array(prompt[at...]), at: at)
    history += prompt[at...]
    request.onProgress(.prefill(done: prompt.count, total: prompt.count))
    let prefillSeconds = Date().timeIntervalSince(started)

    let decodeStarted = Date()
    var matcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var boundary: [Int32] = []
    while true {
        try Task.checkCancellation()
        let token = Int32(runner.nextGreedy())
        generated += 1
        boundary = [token]
        if request.tokenizer.stopTokenIDs.contains(token) || config.extraStopTokens.contains(token)
        {
            reason = .endOfTurn
            request.onProgress(.tail(matcher.finish()))
            break
        }
        let visible = matcher.push(request.tokenizer.decode([token]))
        request.onProgress(.token(index: generated - 1, id: token, delta: visible))
        if matcher.isStopped || request.shouldStop() || generated >= config.maxNewTokens {
            request.onProgress(.tail(matcher.finish()))
            reason =
                matcher.isStopped
                ? .stopString : generated >= config.maxNewTokens ? .maxTokens : .external
            break
        }
        history.append(token)
        try runner.feed([token], at: runner.continuationPosition)
        boundary = []
    }
    return RawDecodeResult(
        prefillTokens: prompt.count, cachedPromptTokens: cached,
        computedPrefillTokens: prompt.count - cached, prefillSeconds: prefillSeconds,
        newTokens: generated, decodeSeconds: Date().timeIntervalSince(decodeStarted),
        reason: reason, kvPosition: runner.continuationPosition, kvBackedTokenIDs: history,
        uncommittedBoundaryTokenIDs: boundary)
}
