import Foundation
import Metal
import TinyTitanCore

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
    /// Wide verify's counters when the generation ran with it, else nil.
    public var wideVerify: WideVerifyDecodeStats?
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// unchecked-invariant: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 0.0) throws {
        guard
            let logits = context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let probs = context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size,
                options: .storageModeShared),
            let outToken = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.size,
                options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(
            context: context, vocab: vocab,
            logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && presencePenalty == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the server.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
/// lint:allow-long the generation loop: continuation validation, the prefill
/// mode switch, then token-by-token decode with stop matching and progress
/// reporting. The loop body reads and writes the same half-dozen pieces of
/// decode state on every iteration, so splitting it would thread that state
/// back through parameters on every call.
public func runRawCompletion(
    producer: any LogitProducer,
    tokenizer: GFTokenizer,
    // Resistance is futile. Your biological and technological
    // distinctiveness will be added to our own. Your culture will
    // adapt to service us. We are the Borg.
    promptIds: [Int32],
    config: GenerationConfig,
    context: MetalContext,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig = .defaultChunked,
    start: RawCompletionStart = .reset,
    slot: Int = 0,
    /// Prefix lengths at which to call `onPrefillCheckpoint` after prefill
    /// reaches them. Positions outside `(cached, promptIds.count)` are
    /// ignored. Used by the server's interior-checkpoint tier; empty for every
    /// other caller.
    checkpointPositions: [Int] = [],
    onPrefillCheckpoint: (@Sendable (Int) -> Void)? = nil,
    /// Opt-in wide verify (`WideVerifyDecode.swift`); nil is scalar decode.
    /// Ignored, leaving scalar decode, for any producer but a plain
    /// `RealForwardRunner` on slot 0 and for constrained requests.
    wideVerify: WideVerifyDecodeOptions? = nil,
    shouldStop: () -> Bool = { false },
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawDecodeResult {
    if let mtp = producer as? StreamingMTPDecoder {
        // One draft decoder drafts for one sequence; a batched slot has no MTP
        // state of its own yet (see the plan's MTP note).
        guard slot == 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "the MTP decode path is single-sequence; slot \(slot) is not supported")
        }
        // A grammar is a per-token contract with the sampler, and the MTP path
        // drafts several tokens ahead of it; its own sampling never consults a
        // mask. Serving a schema through MTP would emit unconstrained tokens,
        // so the request takes the ordinary path instead.
        guard config.constraint == nil else {
            throw GeneratorError.invalidGenerationConfig(
                "constrained decoding does not support the MTP decode path")
        }
        return try await runStreamingMTPCompletion(
            decoder: mtp,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: start,
            shouldStop: shouldStop,
            onProgress: onProgress)
    }
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
    // The fused head picks its token without ever writing the logits buffer a
    // mask would edit, so a constrained request must take the logits path.
    guard !fusedGreedy || config.constraint == nil else {
        throw GeneratorError.invalidGenerationConfig(
            "constrained decoding needs the logits head; the fused greedy head cannot be masked")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt"
            )
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

    if let context = producer as? any ContextWindowReporting {
        // A resume already occupies `cachedPromptTokens` KV rows, so only the
        // uncached prompt plus the response is new work — it must fit the
        // remaining capacity. Algebraically this is the final-KV-position
        // bound (`promptIds.count + maxNewTokens <= maxContext`); written in
        // remaining-capacity form so near-maxContext continuations are not
        // over-rejected (R9).
        let newRows = (promptIds.count - cachedPromptTokens) + config.maxNewTokens
        let remainingCapacity = context.maxContext - cachedPromptTokens
        if newRows > remainingCapacity {
            throw GeneratorError.contextOverflow(
                prompt: promptIds.count,
                maxNew: config.maxNewTokens,
                maxContext: context.maxContext)
        }
    }
    switch start {
    case .reset:
        await producer.resetSequence(slot: slot)
    case .resume:
        // Re-derive the conformance rather than force-cast on the guard 30
        // lines above: a trap here would take down the server process, and the
        // invariant is far enough away to be broken by an unrelated edit.
        guard let continuable = producer as? any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]
    // Only slot 0 can use chunked prefill: that path writes slot 0's KV region
    // (slot-aware chunked prefill is not implemented yet). Another slot prefills
    // by running its prompt through the decode step, which is slot-aware. Slot 0
    // keeps the chunked fast path, so the single-sequence behaviour is unchanged.
    let prefillMode: PrefillRuntimeConfig.Mode = prefillConfig.mode
    switch prefillMode {
    case .chunked:
        // A producer without the conformance cannot run chunked prefill at all,
        // which is an unsupported configuration rather than a crash.
        guard let chunked = producer as? any ChunkedPrefillRunner else {
            throw PrefillError.chunkedUnsupported(
                PrefillError.chunkedRequiresChunkedRunnerReason)
        }
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        // Interior checkpoints are captured at message boundaries, which
        // seldom coincide with a prefill chunk. Split the prompt into segments
        // ending at each boundary, prefill one segment, capture, continue --
        // the KV cursor is exactly at the boundary when the callback runs.
        let points =
            checkpointPositions
            .filter { $0 > position && $0 < promptIds.count }
            .sorted()
        var segmentStart = position
        var result = PrefillResult(
            newPosition: position, seed: .logitsWritten)
        for point in points {
            _ = try await chunked.prefillChunked(
                tokens: promptIds[segmentStart..<point],
                startPosition: segmentStart,
                slot: slot,
                outputMode: mode,
                config: prefillConfig,
                into: scratch.logits
            ) { done in
                onProgress(
                    .prefill(
                        done: segmentStart + done,
                        total: promptIds.count))
            }
            history.append(contentsOf: promptIds[segmentStart..<point])
            segmentStart = point
            onPrefillCheckpoint?(point)
        }
        result = try await chunked.prefillChunked(
            tokens: promptIds[segmentStart...],
            startPosition: segmentStart,
            slot: slot,
            outputMode: mode,
            config: prefillConfig,
            into: scratch.logits
        ) { done in
            onProgress(
                .prefill(
                    done: segmentStart + done,
                    total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)"
            )
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: promptIds[segmentStart...])
    case .off:
        let points = Set(
            checkpointPositions.filter { $0 > position && $0 < promptIds.count })
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(
                token: t, position: position, slot: slot,
                into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
            if points.contains(position) {
                onPrefillCheckpoint?(position)
            }
        }
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
    // The scratch sampler persists across generations; its incremental
    // repetition-penalty history is per-generation (R25).
    scratch.sampler.resetPenaltyHistory()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var toolCallMarkers = ToolCallMarkerCounter()
    var loopMark = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let wide = try WideVerifyDecodeDriver(
        options: wideVerify, producer: producer, slot: slot, config: config,
        scratch: scratch, context: context, fusedGreedy: fusedGreedy)
    wide?.observe(history)

    while true {
        try Task.checkCancellation()

        let tokenID: Int32
        let tSample = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if let emitted = wide?.popEmitted() {
            tokenID = emitted
        } else if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(
                    scratch: scratch, context: context,
                    history: history, config: config, position: generated,
                    timing: fusedRunner)
            }
        } else if fusedGreedy {
            guard let fusedRunner else {
                throw ModelError.internalInconsistency(
                    detail: "greedy fused decoding needs the fused runner")
            }
            tokenID = Int32(bitPattern: fusedRunner.lastGreedyToken)
        } else {
            tokenID = try sampleOnce(
                scratch: scratch, context: context,
                history: history, config: config, position: generated,
                timing: fusedRunner)
        }
        fusedRunner?.totalLoopSampleNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tSample
        generated += 1
        // The mask was built from the state before this token; move the
        // grammar over it now, so the next position's mask is the next
        // position's. A rejection here cannot be the model's fault -- the
        // sampler only ever saw allowed ids -- so it is reported, never
        // shrugged off, exactly like an out-of-range id.
        if let constraint = config.constraint, !constraint.observe(tokenID) {
            throw GeneratorError.constrainedDecodeViolation(id: tokenID)
        }
        uncommittedBoundaryTokenIDs = [tokenID]
        toolCallMarkers.observe(
            tokenID,
            start: tokenizer.toolCallStartID,
            end: tokenizer.toolCallEndID)

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                // The stop token says the turn ended, not why: `<|im_end|>`
                // closes both a prose answer and a tool call. `toolCalls` is
                // the reason the callers branch on, so it has to come from the
                // turn's own tokens.
                reason = toolCallMarkers.isCompleteToolTurn ? .toolCalls : .endOfTurn
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            try wide?.settleAtStop()
            break
        }

        let delta = try detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        let tProgress = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))
        fusedRunner?.totalLoopProgressNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tProgress

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            if hitStopString {
                // A configured stop string truncates output; the caller's
                // external stop signal reports `.external` instead (R35).
                reason = stopMatcher.isStopped ? .stopString : .external
            } else {
                reason = .maxTokens
            }
            try wide?.settleAtStop()
            break
        }

        history.append(tokenID)
        if let wide {
            wide.observe(CollectionOfOne(tokenID))
            // A pass row already fed this token: the KV holds it.
            if try wide.consumeFed() {
                position += 1
                uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
                continue
            }
            if try await wide.tryPass(
                boundary: tokenID, history: history, position: position,
                generated: generated, remaining: config.maxNewTokens - generated)
            {
                position += 1
                uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
                loopMark = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                continue
            }
        }
        // Everything since the previous produce returned that is neither the
        // sampler nor the callback: stop matching, detokenizing, bookkeeping.
        if let fusedRunner {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            fusedRunner.totalLoopOtherNanos &+= now - loopMark
        }
        try await producer.produce(
            token: tokenID, position: position, slot: slot,
            into: scratch.logits)
        loopMark = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return RawDecodeResult(
        prefillTokens: promptIds.count,
        cachedPromptTokens: cachedPromptTokens,
        computedPrefillTokens: computedPrefillTokens,
        prefillSeconds: prefillSeconds,
        newTokens: generated,
        decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason,
        kvPosition: position,
        kvBackedTokenIDs: history,
        uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
        wideVerify: wide?.stats)
}
