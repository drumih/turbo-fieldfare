import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

/// Serving a small model from the CPU, through the same HTTP surface as the
/// big ones.
///
/// The side-engine began as a way to run a 2B beside a 35B for the memory
/// work, and it stays that. But a model that answers correctly at twenty
/// tokens a second is a model worth serving, and there is no reason a person
/// with a small model and no GPU budget should get a different API, a
/// different tokenizer path or a different set of options.
///
/// So this is a `ServerInferenceBackend` like `ServerModelSession`, and
/// everything above it — the OpenAI and Responses surfaces, the memory
/// subsystem, the watchdogs — works unchanged.
///
/// **What it does not do, deliberately.** No prompt cache: the KV state is
/// rebuilt per request, because a CPU engine's prefill is cheap relative to
/// its decode and the cache's complexity buys little. No expert streaming:
/// these models are dense. No MTP.
public actor CPUModelBackend: ServerInferenceBackend, PromptCacheDescribing {

    private let model: CPUQwen35
    private let tokenizer: GFTokenizer
    /// Where the tokenizer came from, and the reasoning it was rendered at.
    ///
    /// A request that names a different thinking mode or effort -- a
    /// mid-session switch, including turning thinking off -- resolves a
    /// tokenizer for that configuration through the shared
    /// `(folder, thinking, effort)` cache rather than reusing this one.
    private let snapshotDirectory: URL
    /// The folder the tokenizer was loaded from: the model directory for a
    /// snapshot, its `tokenizer/` sidecar for a `.ssdai` install. Kept so a
    /// mid-session re-render reads from the same place the load did.
    private let tokenizerFolder: URL
    private let loadedReasoning: RequestReasoning
    /// The vocabulary-as-bytes table for structured output, built on first use
    /// and shared by every request this backend serves.
    private var jsonTokenTable: JSONTokenTable?
    private let context: Int
    private let defaults: GenerationDefaults.Sampling

    public nonisolated let residentBytes: Int
    /// The thread width in force, so the startup banner can report what the
    /// engine will actually use without importing the kernel.
    public nonisolated let threads: Int
    /// The width the *next* token will really use, which is the live
    /// counterpart to the load-time `threads` above.
    ///
    /// The two can only differ if something armed the engine's contention
    /// signal, and nothing does on this path; `SideEngineModel.threads` is the
    /// same diagnostic on the side engine, which does narrow.
    package var threadWidthInUse: Int { model.threads }
    public nonisolated var maximumContext: Int { context }
    public nonisolated var samplingDefaults: GenerationDefaults.Sampling { defaults }
    /// The CPU engine has no prompt cache, and says so rather than borrowing the
    /// mode the server was asked for. `.off` is what the launch banner reports
    /// for this backend, so the banner and the residency line agree.
    package nonisolated var promptCacheMode: ServerPromptCacheMode { .off }

    /// Loads a snapshot and, unless told otherwise, makes it resident.
    ///
    /// Residency is the point of the CPU path. Left to the page cache a
    /// side model is evicted by whatever else wants the memory, and the next
    /// request pays to fault the whole thing back in — measured on this
    /// project, an unhelpful `madvise` hint alone cost 2.6x throughput. A
    /// model asked for by name should be in memory.
    /// What the CPU path will serve however long the checkpoint says it can.
    ///
    /// Qwen3.5 claims 262,144 positions and the GPU engine honours it. Here
    /// attention is a loop over the cache and the cache is held per token,
    /// so a context that is merely large on a GPU is unusable on four cores:
    /// at 262k the key/value cache alone is over three gigabytes, and every
    /// token would walk all of it. A ceiling that is quietly enforced beats
    /// a promise that is quietly broken.
    public static let contextCeiling = 32_768

    /// Accepts either shape the CPU engine can serve: an affine safetensors
    /// snapshot, which the dense converter writes, or a `.ssdai` install,
    /// which is what every other model in this project is. The two carry the
    /// same quantized tensors -- `tools/ssdai_diff_snapshot.py` checks that
    /// rather than assuming it -- so this is a storage difference, not a
    /// semantic one.
    ///
    /// `thinkingMode` is baked into the tokenizer's generation prompt, as it
    /// is on the GPU path: the template renders the thinking switch, so it is
    /// a load-time setting, not a per-request one. A request can still switch
    /// it; that is `resolvedTokenizer(for:)`.
    public init(
        snapshotDirectory: URL,
        maximumContext: Int = CPUModelBackend.contextCeiling,
        resident: Bool = true,
        thinkingMode: ModelThinkingMode = .off
    ) async throws {
        // A `.ssdai` declares itself with a manifest; a snapshot does not.
        // `manifest.json` is also what the catalog keys on, so the two agree
        // about which shape a directory is.
        let isSSDAI = FileManager.default.fileExists(
            atPath: snapshotDirectory.appendingPathComponent("manifest.json").path)
        let snapshot =
            isSSDAI
            ? try AffineSnapshot(ssdai: snapshotDirectory)
            : try AffineSnapshot(directory: snapshotDirectory)
        guard let family = snapshot.family else {
            throw CPUBackendError.unsupported(
                CPUModelFamily.refusal(modelType: snapshot.modelType))
        }
        _ = family
        residentBytes = resident ? snapshot.makeResident() : 0
        // No contention signal: this engine runs at its full width, and the
        // guard that the served path stays that way is
        // `theServedEngineKeepsTheWidthItsLoadChose`. The coordinator's
        // `generating` is not the signal to reach for here, because it is
        // raised by a CPU client generation too.
        let engine = try CPUQwen35(snapshot: snapshot)
        threads = engine.threads
        model = engine
        // A snapshot keeps its tokenizer at the directory root; a `.ssdai`
        // keeps it under `tokenizer/`. The folder is kept, not just used: a
        // request that switches thinking mode re-renders through this
        // tokenizer, and re-deriving the folder there is how the re-render
        // came to hand `load(from:)` a directory that has no `tokenizer.json`
        // -- which every `.ssdai` install has, so the switch failed for all
        // of them.
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: snapshotDirectory)
        else {
            throw CPUBackendError.unsupported(
                "no tokenizer in \(snapshotDirectory.lastPathComponent)")
        }
        tokenizerFolder = folder
        tokenizer = try await GFTokenizer.load(
            from: folder,
            thinkingMode: thinkingMode)
        self.snapshotDirectory = snapshotDirectory
        self.loadedReasoning = RequestReasoning(
            thinkingMode: thinkingMode,
            effort: nil)
        context = min(
            maximumContext, snapshot.configuration.maxPositions,
            Self.contextCeiling)
        // The family's own, which is what the catalog advertises for it, so
        // a launcher showing the defaults shows what a request will get.
        defaults = family.samplingDefaults
    }

    public enum CPUBackendError: Error, CustomStringConvertible {
        case unsupported(String)
        case promptTooLong(Int, Int)

        public var description: String {
            switch self {
            case .unsupported(let detail): detail
            case .promptTooLong(let count, let limit):
                "prompt is \(count) tokens and the context is \(limit)"
            }
        }
    }

    /// The sampler a request's generation config asks for.
    ///
    /// Its own function because every field on that config has to cross into
    /// the CPU sampler or the two engines answer the same request with two
    /// different distributions -- and `--cpu` is a documented, operator-facing
    /// mode, not a debug switch (AUD-176: the seed and both penalties were read
    /// by the validator, reached this line, and were dropped). The defaults when
    /// a field is absent are the validator's, not this function's invention:
    /// `topP` 1 and `topK` 0 mean "no filter", which is what the GPU path's
    /// kernel does with the same absent values.
    static func cpuSampler(for configuration: GenerationConfig) -> CPUSampler {
        CPUSampler(
            temperature: configuration.temperature,
            topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0,
            presencePenalty: configuration.presencePenalty,
            repetitionPenalty: configuration.repetitionPenalty,
            // The seed the request sent, or nil for the clock -- the same rule
            // the GPU path's `seedFor` applies. Greedy ignores it either way.
            seed: configuration.seed)
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        // A mid-session switch resolves a tokenizer for the requested
        // thinking mode; nil (the common case) reuses the loaded one. The
        // tokenizer carries the think-block and stop token IDs for its own
        // mode, so decode follows the switch as well as the render.
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        let rendered = try renderTokenizer.applyChatTemplate(request.messages)
        let promptIDs = renderTokenizer.encode(rendered, addBOS: false)
        let prompt = promptIDs.map(Int.init)
        guard prompt.count < context else {
            throw CPUBackendError.promptTooLong(prompt.count, context)
        }
        let budget = min(request.maximumCompletionTokens, context - prompt.count)
        var configuration = request.generationConfig
        if let node = request.jsonSchema {
            // One local for the table: the force unwrap here used to be the
            // only thing standing between a nil cache and a crash.
            let table: JSONTokenTable
            if let existing = jsonTokenTable {
                table = existing
            } else {
                let made = JSONTokenTable(tokenizer: tokenizer)
                jsonTokenTable = made
                table = made
            }
            configuration.constraint = JSONConstraint(
                table: table, node: node,
                vocab: model.configuration.vocabulary)
        }
        let sampler = Self.cpuSampler(for: configuration)
        let generator = sampler.makeGenerator()

        model.reset()
        var logits = try model.prefill(tokens: prompt)
        // What the penalties are measured against: the prompt is already in the
        // model's context, so a token the prompt used is a token the history
        // contains. The GPU sampler gets the same thing from its caller.
        var history = promptIDs

        // The same decoder the GPU path runs, so a thought is split the same
        // way on either engine. This path renders no tool template, so the
        // decoder only ever splits thoughts -- including one the model opens
        // itself while the switch is off.
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: renderTokenizer, promptIDs: promptIDs, allowedTools: nil)
        var detokenizer = GFDetokenizer(tokenizer: renderTokenizer)
        var output = AssistantOutput(stops: configuration.stopStrings, onEvent: onEvent)
        var produced = 0
        var reason = "length"
        while produced < budget {
            // Structured output: floor every token the grammar no longer
            // accepts before the sampler sees the row. The same contract as
            // the GPU path's mask, on the same kind of buffer.
            if let constraint = configuration.constraint {
                let mask = constraint.allowedMask()
                guard !mask.isEmpty else { throw GeneratorError.constrainedDecodeStalled }
                mask.apply(toLogits: &logits)
            }
            let next = sampler.pick(logits, history: history, using: generator)
            if let constraint = configuration.constraint, !constraint.observe(Int32(next)) {
                throw GeneratorError.constrainedDecodeViolation(id: Int32(next))
            }
            if next == Int(tokenizer.eosID) {
                reason = "stop"
                break
            }
            produced += 1
            history.append(Int32(next))
            output.publish(
                try events(
                    for: Int32(next), decoder: decoder,
                    detokenizer: &detokenizer))
            if output.isStopped {
                reason = "stop"
                break
            }
            if produced >= budget { break }
            logits = try model.step(token: next)
        }
        output.publish(try decoder.consumeTail(detokenizer.flush()), isToken: false)
        try decoder.finish()
        output.finish()
        return ServerCompletion(
            content: output.content,
            toolCalls: [],
            finishReason: reason,
            usage: OpenAIUsage(
                promptTokens: prompt.count,
                completionTokens: produced,
                totalTokens: prompt.count + produced,
                cachedTokens: 0,
                reasoningTokens: output.reasoningTokens),
            // Named, as the GPU path names it: a Messages client is told
            // which of its stop sequences ended the turn.
            stopSequence: output.matchedStop,
            reasoning: output.reasoning,
            // Same rule as the GPU path: the render decided the switch, so a
            // thought that arrived with thinking off is reported as one.
            unrequestedReasoning: renderTokenizer.thinkingMode.isEnabled
                ? 0 : output.reasoning.count)
    }

    /// One sampled token as decoder events.
    ///
    /// The streaming detokenizer, not a per-token `decode`, and for the reason
    /// the decoder exists: it knows `<think>` and `</think>` by their literal
    /// text on the delta, and a per-token decode that skips special tokens
    /// drops exactly the markers a self-started thought has to be recognized
    /// by. Both engines now detokenize the same way, which is what makes "the
    /// same rule wherever a model runs" true rather than aspirational.
    private func events(
        for token: Int32,
        decoder: StructuredAssistantDecoder,
        detokenizer: inout GFDetokenizer
    ) throws -> [StructuredAssistantEvent] {
        try decoder.consume(tokenID: token, delta: detokenizer.push(token))
    }
}

/// The Messages API's count_tokens, from the same rendering `generate` uses,
/// so a client sizing its context against a CPU model gets the real number.
extension CPUModelBackend: PromptTokenCounting {
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        // Count through the same tokenizer `generate` would use, so a client
        // sizing the context for a mid-session switch gets the real number
        // rather than the loaded mode's.
        try Self.promptTokenCount(
            request, tokenizer: try await resolvedTokenizer(for: request.reasoning))
    }

    /// The tokenizer this request should be rendered with; the loaded one
    /// unless the request named a different thinking mode or effort.
    private func resolvedTokenizer(
        for reasoning: RequestReasoning?
    ) async throws -> GFTokenizer {
        guard let reasoning, !reasoning.matches(loadedReasoning) else {
            return tokenizer
        }
        return try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: reasoning.thinkingMode,
            reasoningEffort: reasoning.effort)
    }

    /// The count from a tokenizer alone, which is how the router answers for
    /// a CPU model that is not the one loaded.
    static func promptTokenCount(
        _ request: ValidatedChatRequest,
        tokenizer: GFTokenizer
    ) throws -> Int {
        let rendered = try tokenizer.applyChatTemplate(request.messages)
        return tokenizer.encode(rendered, addBOS: false).count
    }
}
