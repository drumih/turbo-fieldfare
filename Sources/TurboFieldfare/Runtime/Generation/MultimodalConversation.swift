import Foundation
import Metal

public enum MultimodalConversationError: Error, CustomStringConvertible {
    case closed
    case busy
    case lineageBroken
    case lineageRecoveryFailed(reason: String)
    case emptyTurn
    case contextExhausted(prompt: Int, maxContext: Int)
    case imageUnavailable(reason: String?)

    public var description: String {
        switch self {
        case .closed: "conversation is closed"
        case .busy: "a turn is already generating on this conversation"
        case .lineageBroken:
            "generation failed partway, so the KV no longer matches this "
                + "conversation; call reset() to start over"
        case .lineageRecoveryFailed(let reason):
            "generation failed partway and the KV could not be restored; "
                + "call reset() to start over. \(reason)"
        case .emptyTurn: "a turn needs text or an image"
        case .contextExhausted(let prompt, let maxContext):
            "conversation needs \(prompt) tokens, beyond the \(maxContext)-token context"
        case .imageUnavailable(let reason):
            reason.map { "image support is unavailable: \($0)" }
                ?? "image support is unavailable: no companion pack is installed"
        }
    }
}

enum MultimodalConversationKVRecovery {
    static func restoreAfterFailure(
        positionBefore: Int,
        generationError: Error,
        rewind: (Int) throws -> Void,
        reset: () -> Void
    ) throws {
        guard positionBefore > 0 else {
            reset()
            return
        }
        do {
            try rewind(positionBefore)
        } catch {
            throw MultimodalConversationError.lineageRecoveryFailed(
                reason: "The turn failed with \(generationError). Rewinding to "
                    + "token \(positionBefore) then failed with \(error).")
        }
    }

    static func trimHiddenStopTokens(
        _ tokenIDs: [Int32],
        withheld: Int,
        rewind: (Int) throws -> Void,
        reset: () -> Void
    ) throws -> [Int32] {
        guard withheld > 0, withheld <= tokenIDs.count else {
            throw MultimodalConversationError.lineageRecoveryFailed(
                reason: "Stop-string cleanup reported \(withheld) hidden tokens "
                    + "for a \(tokenIDs.count)-token KV.")
        }
        let target = tokenIDs.count - withheld
        if target == 0 {
            reset()
        } else {
            do {
                try rewind(target)
            } catch {
                throw MultimodalConversationError.lineageRecoveryFailed(
                    reason: "Removing stop-string tokens required rewinding to "
                        + "token \(target), which failed with \(error).")
            }
        }
        return Array(tokenIDs.prefix(target))
    }
}
public struct MultimodalTurnResult: Sendable {
    public let text: String
    public let promptTokens: Int
    /// Tokens served from the retained KV rather than prefilled again.
    public let cachedTokens: Int
    /// Tokens this turn put through prefill. `RawCompletion` computes it as
    /// `promptIds.count - cachedPromptTokens`, so it is a derivation of the two
    /// figures beside it, not an independent count: asserting that it equals
    /// `promptTokens - cachedTokens` cannot fail and proves nothing. What it is
    /// good for is reporting — the figure a reader wants when asking what a
    /// turn cost. The claim that the KV was reused rests on `cachedTokens`
    /// matching the previous turn's `kvTokens`, and on `prefillSeconds`.
    public let computedPrefillTokens: Int
    /// Tokens the KV holds after this turn. Not `promptTokens +
    /// completionTokens`: a run that stops on max tokens or is cancelled holds
    /// its final token outside the KV, so that sum is one too many.
    public let kvTokens: Int
    public let completionTokens: Int
    /// Why the turn ended; `.cancelled` when `checkCancellation` stopped the
    /// decode mid-turn, in which case `text` holds the partial reply and the
    /// conversation remains resumable.
    public let reason: StopReason
    /// Wall time spent prefilling this turn's own tokens. Reported so a caller
    /// can show what resuming actually saved rather than asserting that it did.
    public let prefillSeconds: Double
    public let decodeSeconds: Double
}

/// A stateful multi-turn conversation that owns its own KV lineage.
///
/// The server has to *match* a stateless request against a cached prefix and
/// fail closed when it cannot. A conversation does not: it appended every token
/// in the KV itself, so it always knows the boundary and always resumes. Each
/// turn prefills only the new tokens, and an image is encoded once, when its
/// turn is appended.
public actor MultimodalConversation {
    private let model: Model
    private let context: MetalContext
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let visionRuntime: VisionRuntime?
    private let visionRuntimeError: Error?
    private let visionResidency: VisionResidencyPolicy
    private let maxContext: Int

    /// Exactly the tokens the KV holds, in order.
    private var kvTokenIDs: [Int32] = []
    private var pending: (parts: [MultimodalContinuationPart], images: [URL])?
    private var closed = false
    /// One generation at a time. `RawCompletionScratch` documents that its
    /// buffers and sampler belong to a single run and that the guard is the
    /// caller's job; the actor alone does not provide it, because `generate`
    /// suspends for the whole decode.
    private var generating = false
    /// Resumed by `finishGeneration()`; see `waitForGeneration()`.
    private var generationWaiters: [CheckedContinuation<Void, Never>] = []
    /// Barrier for `reset()`, set before it awaits the in-flight decode so a
    /// `generate()` racing the reset cannot start and have its KV wiped.
    private var resetting = false
    /// Set when a run throws after prefill. The runner's KV has advanced but
    /// `kvTokenIDs` has not, so the lineage is unusable until reset.
    private var lineageBroken = false
    /// Tokens the model emitted that never entered the KV, which happens when
    /// a run stops on max tokens or is cancelled. The next turn must replay
    /// them, exactly as the server's prefix cache does.
    private var uncommittedBoundary: [Int32] = []
    private var boundaryNeedsReplay = false

    public init(model: Model,
                context: MetalContext,
                tokenizer: GFTokenizer,
                runner: RealForwardRunner,
                scratch: RawCompletionScratch,
                visionRuntime: VisionRuntime? = nil,
                visionRuntimeError: Error? = nil,
                visionResidency: VisionResidencyPolicy = .defaultPolicy,
                maxContext: Int) {
        self.model = model
        self.context = context
        self.tokenizer = tokenizer
        self.runner = runner
        self.scratch = scratch
        self.visionRuntime = visionRuntime
        self.visionRuntimeError = visionRuntimeError
        self.visionResidency = visionResidency
        self.maxContext = maxContext
    }

    public var kvTokenCount: Int { kvTokenIDs.count }
    public var hasStagedTurn: Bool { pending != nil }
    public var isClosed: Bool { closed }
    public var isUsable: Bool { !closed && !lineageBroken }

    /// Marks this conversation unusable without touching the shared runner or
    /// vision runtime. The session calls it when handing out a replacement, so a
    /// stale reference cannot reset or resume onto the live conversation's KV.
    public func invalidate() async {
        closed = true
        pending = nil
        // Wait for any run still decoding: the session resets the shared runner
        // straight after this, and resetting under a live decode either aborts
        // that turn mid-stream or lets two turns drive one runner at once.
        await waitForGeneration()
    }

    /// Suspends until no turn is decoding.
    ///
    /// This was a `while generating { await Task.yield() }` spin, which hot-looped
    /// the cooperative executor for the whole remaining decode — minutes at
    /// single-digit tok/s — and re-entered the actor it was waiting on at every
    /// iteration. Waiters are now resumed by the run itself.
    private func waitForGeneration() async {
        guard generating else { return }
        await withCheckedContinuation { continuation in
            generationWaiters.append(continuation)
        }
    }

    private func finishGeneration() {
        generating = false
        let waiters = generationWaiters
        generationWaiters = []
        for waiter in waiters { waiter.resume() }
    }

    /// Whether a turn is decoding right now.
    public var isGenerating: Bool { generating }

    /// Stages the next user turn without prefilling it.
    public func append(parts: [MultimodalContinuationPart], images: [URL] = []) throws {
        guard !closed else { throw MultimodalConversationError.closed }
        // A turn staged while another is decoding used to be accepted and then
        // dropped: `generate()` clears `pending` when it commits, so the newly
        // staged turn vanished and the next `generate()` reported an empty turn
        // with the user's message gone.
        guard !generating else { throw MultimodalConversationError.busy }
        let imageCount = parts.filter {
            if case .image = $0 { return true } else { return false }
        }.count
        guard !parts.isEmpty, imageCount == images.count else {
            throw MultimodalConversationError.emptyTurn
        }
        pending = (parts, images)
    }

    /// Drops a staged turn that has not been generated yet.
    ///
    /// It cannot undo a generated turn. The KV cache has no rewind — the runner
    /// only validates that a continuation resumes at the current position — so
    /// once a turn is prefilled it is part of the lineage. Use `reset()` to
    /// abandon it, at the cost of re-prefilling the conversation.
    public func clear() {
        pending = nil
    }

    /// Drops the KV and starts over. The model stays loaded.
    ///
    /// Waits for a decode in flight. The actor accepts calls while `generate()`
    /// is suspended mid-run, so resetting here used to wipe the KV under a live
    /// turn: it then attended against an empty cache and either produced garbage
    /// or died on the prefill cursor, leaving `kvTokenIDs` describing a KV that
    /// no longer existed.
    public func reset() async {
        guard !closed else { return }
        // The barrier mirrors `close()` and `invalidate()`, which set theirs
        // before awaiting. Without one, a `generate()` enqueued while this
        // waiter was suspended could win the actor, start decoding, and have
        // its KV wiped underneath it — leaving `kvTokenIDs` restored from the
        // run's own token array while the cursor sat at zero, and
        // `lineageBroken` false, so `isUsable` kept saying yes forever. It
        // also stops back-to-back turns from a higher-priority task starving
        // the reset: the loop alone re-raced each freshly enqueued turn.
        resetting = true
        defer { resetting = false }
        while generating {
            await waitForGeneration()
        }
        guard !closed else { return }
        runner.reset()
        kvTokenIDs.removeAll(keepingCapacity: true)
        pending = nil
        lineageBroken = false
        uncommittedBoundary = []
        boundaryNeedsReplay = false
    }

    /// Ends this conversation and clears the KV it was using. It deliberately
    /// does not touch the session-owned vision runtime: those mappings are
    /// shared, and releasing them here would corrupt another conversation's
    /// state. `TurboFieldfareModelSession.close()` owns that.
    public func close() async {
        guard !closed else { return }
        closed = true
        // Same hazard as `reset()`: the runner is shared and the decode is still
        // using it.
        await waitForGeneration()
        runner.reset()
        kvTokenIDs.removeAll(keepingCapacity: false)
        pending = nil
    }

    /// Appends one user turn and generates the reply, prefilling only the new
    /// tokens. `images` are file URLs in the order they appear among `parts`.
    /// Appends a turn, generates the reply, and commits it.
    public func send(
        parts: [MultimodalContinuationPart],
        images: [URL] = [],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        try append(parts: parts, images: images)
        return try await generate(
            config: config, prefillConfig: prefillConfig,
            checkCancellation: checkCancellation, shouldStop: shouldStop,
            onProgress: onProgress)
    }

    /// Generates a reply to the staged turn. The turn joins the lineage as soon
    /// as it is prefilled, because the KV cannot rewind.
    /// `onProgress` receives the same events `runRawCompletion` reports, so a
    /// caller can stream tokens and prefill progress. Without it a turn was
    /// only observable once it had finished, which is unusable for a UI: the
    /// text was accumulated internally and returned in one piece.
    /// `shouldStop` ends the turn at a token boundary and **keeps** what it
    /// produced: the run returns normally with `.cancelled` as its stop reason,
    /// the partial reply is in the KV, and the next turn resumes on it. That is
    /// a different thing from cancelling the task, which throws out of the
    /// decode loop and makes this rewind the whole turn. A Stop button wants
    /// the first; a teardown wants the second.
    public func generate(
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        guard !closed else { throw MultimodalConversationError.closed }
        guard !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting else { throw MultimodalConversationError.busy }
        guard let staged = pending else {
            throw MultimodalConversationError.emptyTurn
        }
        generating = true
        defer { finishGeneration() }
        let parts = staged.parts
        let images = staged.images
        let imageCount = parts.filter {
            if case .image = $0 { return true } else { return false }
        }.count
        if imageCount > 0, visionRuntime == nil {
            throw MultimodalConversationError.imageUnavailable(
                reason: visionRuntimeError.map(String.init(describing:)))
        }

        let cached = kvTokenIDs.count
        let turn = try await encodeTurn(parts: parts, images: images,
                                        checkCancellation: checkCancellation)
        // A run that stopped on max tokens or was cancelled left its final
        // token outside the KV. Replay it ahead of this turn, exactly as the
        // server's prefix cache does, or the model's context is missing a
        // token it already emitted.
        let boundary = boundaryNeedsReplay ? uncommittedBoundary : []
        let promptIDs = kvTokenIDs + boundary + turn.effectiveTokenIDs
        guard promptIDs.count + 1 <= maxContext else {
            throw MultimodalConversationError.contextExhausted(
                prompt: promptIDs.count, maxContext: maxContext)
        }

        // Same coercion as the other entry points: a turn carrying image spans
        // cannot run under a non-chunked mode, and refusing it after the images
        // are encoded helps nobody.
        var effectivePrefill = prefillConfig
        if turn.prefillInput != nil,
           let coerced = effectivePrefill.coercedForImagePrompt() {
            effectivePrefill = coerced
        }
        var generation = config
        generation.maxNewTokens = min(config.maxNewTokens, maxContext - promptIDs.count)
        var text = ""
        // Marked only once the run has actually written to the KV. Setting it
        // before the attempt condemned the conversation for failures that never
        // touched the cache — a cancellation caught by the first
        // `checkCancellation()`, a rejected prefill config, a resume-position
        // mismatch — where the recorded lineage still matched the KV exactly and
        // the turn could simply have been retried.
        // The runner's own cursor, not a progress callback. Inferring it from
        // the first `.prefill` report missed a chunk that committed and then
        // threw — the KV had moved, nothing said so, and every later turn
        // resumed at a position the runner no longer had, failing forever while
        // `isUsable` still said yes.
        let positionBefore = runner.continuationPosition
        var kvAdvanced = false
        let result: RawDecodeResult
        do {
            result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            multimodalInput: try turn.prefillInput?.prepending(boundary),
            config: generation,
            context: context,
            scratch: scratch,
            prefillConfig: effectivePrefill,
            start: cached == 0 ? .reset : .resume(cachedPromptTokens: cached),
                // Cancellation mid-decode ends the turn at a token boundary
                // and returns the partial result; throwing here instead would
                // condemn the lineage for a KV that is perfectly resumable.
                shouldStop: {
                    if shouldStop?() == true { return true }
                    do { try checkCancellation(); return false } catch { return true }
                }) { progress in
                    onProgress?(progress)
                    switch progress {
                    case .token(_, _, let delta):
                        text += delta
                    case .tail(let tail):
                        // The detokenizer flush at the stop boundary. Dropping
                        // it returned turn text missing its final characters
                        // while the KV kept those very tokens.
                        text += tail
                    case .prefill:
                        // The first progress report is the proof the KV moved.
                        kvAdvanced = true
                    }
                }
        } catch let generationError {
            if runner.continuationPosition != positionBefore { kvAdvanced = true }
            // A failure that moved the KV used to condemn the lineage outright,
            // but the same rewind the stop-string path uses can usually put the
            // cursor back at the turn start: a cancellation caught between
            // prefill chunks or at a token boundary leaves the committed prefix
            // intact under a rewindable tail. The staged turn stays pending, so
            // a successful rewind makes the turn retryable. Only a rewind past
            // the SWA ring slack fails, and only then is the lineage unusable.
            if kvAdvanced {
                do {
                    try MultimodalConversationKVRecovery.restoreAfterFailure(
                        positionBefore: positionBefore,
                        generationError: generationError,
                        rewind: runner.rewind(to:),
                        reset: runner.reset)
                } catch {
                    lineageBroken = true
                    throw error
                }
            }
            throw generationError
        }

        // The KV now holds exactly what the run reported, including the tokens
        // the model generated; keeping the model's own tokens rather than
        // re-tokenising its text is what makes the next turn resumable.
        // The turn is in the KV now, so it is no longer staged. Leaving it
        // pending let a repeated generate() append the same user turn again,
        // behind its own reply, and corrupt the rest of the conversation.
        pending = nil
        kvTokenIDs = result.kvBackedTokenIDs
        // A stop-string match discards the text of the tokens that formed it,
        // but all except the final one were already committed to the KV. Left
        // there, every later turn resumes on a context holding assistant text
        // the caller never saw. Drop them from the record and rewind the
        // runner so KV and transcript agree again; a token that showed a
        // visible prefix before the match began stays, so at most a few
        // characters of one boundary token remain hidden.
        if result.reason == .stopString, result.withheldTrailingKVTokens > 0 {
            do {
                kvTokenIDs = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
                    result.kvBackedTokenIDs,
                    withheld: result.withheldTrailingKVTokens,
                    rewind: runner.rewind(to:),
                    reset: runner.reset)
            } catch {
                lineageBroken = true
                throw error
            }
        }
        uncommittedBoundary = result.uncommittedBoundaryTokenIDs
        boundaryNeedsReplay = result.reason == .maxTokens || result.reason == .cancelled
        return MultimodalTurnResult(
            text: text,
            promptTokens: promptIDs.count,
            cachedTokens: cached,
            // The runner's own count, not a reconstruction from KV lengths.
            // Reconstructing it made the figure depend on the stop reason —
            // end-of-turn, EOS, tool-call and stop-string stops each reported
            // one fewer — so the same run was counted differently here and in
            // `ServerInference`, which has always reported this value.
            computedPrefillTokens: result.computedPrefillTokens,
            kvTokens: kvTokenIDs.count,
            completionTokens: result.newTokens,
            reason: result.reason,
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds)
    }

    private struct EncodedTurn {
        let effectiveTokenIDs: [Int32]
        let prefillInput: MultimodalPrefillInput?
    }

    private func encodeTurn(
        parts: [MultimodalContinuationPart],
        images: [URL],
        checkCancellation: @Sendable () throws -> Void
    ) async throws -> EncodedTurn {
        guard !images.isEmpty, let visionRuntime else {
            var text = ""
            for part in parts { if case .text(let value) = part { text += value } }
            let ids = kvTokenIDs.isEmpty
                ? tokenizer.encode(
                    try tokenizer.applyChatTemplate(
                        [GFTokenizer.Message(role: .user, content: text)]),
                    addBOS: false)
                : tokenizer.encodeTextContinuation(userContent: text)
            return EncodedTurn(effectiveTokenIDs: ids, prefillInput: nil)
        }

        // Token counts come from geometry, so the turn's shape is known before
        // any image is encoded.
        let preprocessor = Gemma4ImagePreprocessor(
            device: context.device, config: visionRuntime.config)
        // Kept, not discarded. Planning for the token count and then letting
        // `encodeImage(at:)` re-plan internally opened, sniffed and parsed every
        // image twice per turn — and a file rewritten between the two opens gave
        // a count that no longer matched the span the tokenizer had laid out,
        // failing the turn with a placeholder mismatch that names nothing the
        // user did.
        let plans = try images.map { try preprocessor.plan(fileURL: $0) }
        let counts = plans.map(\.geometry.softTokenCount)
        let bridge = try tokenizer.encodeMultimodalUserContinuation(
            textAndImages: parts, imageTokenCounts: counts,
            // An empty KV means this is the conversation's first turn, which
            // needs the chat template's opening rather than a continuation's.
            openingConversation: kvTokenIDs.isEmpty)
        var spans: [MultimodalImageSpan] = []
        for (range, plan) in zip(bridge.imageTokenRanges, plans) {
            try checkCancellation()
            let features = try visionRuntime.encodeImage(
                plan: plan, languageModel: model,
                residencyPolicy: visionResidency,
                checkCancellation: checkCancellation)
            guard features.tokenCount == range.count else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            spans.append(MultimodalImageSpan(tokenRange: range, features: features))
        }
        return EncodedTurn(
            effectiveTokenIDs: bridge.effectiveTokenIDs,
            prefillInput: try MultimodalPrefillInput(
                effectiveTokenIDs: bridge.effectiveTokenIDs,
                embeddingTokenIDs: bridge.embeddingTokenIDs,
                imageSpans: spans))
    }
}
