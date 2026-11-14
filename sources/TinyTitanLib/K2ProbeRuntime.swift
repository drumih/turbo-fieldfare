import Foundation
import Metal
import TinyTitan
import TinyTitanCore

/// The K2-Horizon probe as a reusable generation engine.
///
/// The graph is the same one `TinyTitanBench k2gpu` drives: dense layers plus
/// the sparse MoVA layers, streamed routed experts and a GPU-resolved value
/// selection. This type owns the per-sequence state that a front end should
/// not have to carry -- the graph, its capacity, and the token stream whose
/// K/V it holds -- so the server backend and the CLI can both use it without
/// duplicating the loop.
///
/// It deliberately knows nothing about HTTP or requests: callers render their
/// own prompt, decide the budget and hand over a `GenerationConfig`.
package actor K2ProbeRuntime {
    private let model: Model
    private let config: ArchConfig
    private let context: MetalContext
    private let contextLimit: Int
    let slotCount: Int
    private let directory: URL

    /// The graph and the tokens whose K/V it currently holds. A request whose
    /// prompt shares a prefix with the last one continues from that prefix
    /// instead of re-forwarding it: the layers read the cache by position, so
    /// identical tokens at identical positions have identical K/V.
    ///
    /// `nonisolated(unsafe)` because the graph is a non-Sendable class the
    /// generation loop awaits; every access is inside this actor's `generate`,
    /// which the actor serializes, so there is no second task to race.
    nonisolated(unsafe) private var graph: K2GraphProbe?
    private var graphCapacity = 0
    private var cachedTokens: [Int32] = []

    /// How many prompt tokens the chunked prefill batches; 0 or 1 is the
    /// token path.
    ///
    /// 16 is the default for 4-bit installs. With the sparse batch fetching the
    /// union of a chunk's selected experts once per layer, a 163-token prompt
    /// measured a median 17.5 s chunked against 23.8 s token-by-token over
    /// eight paired rounds (best 13.4 against 22.5), and the 24-token greedy
    /// stream is byte-identical to the token path. At 8-bit the same prompt is
    /// faster still, but the affine GEMM's reduction order is not the GEMV's
    /// and the greedy output flips three tokens on the golden prompt, which a
    /// speed change may not buy; that install stays on the token path.
    /// `TINYTITAN_K2_CHUNK` overrides either way.
    private var chunkSize: Int {
        let fallback = model.attentionWeightBits == 4 ? 16 : 1
        let requested =
            Int(ProcessInfo.processInfo.environment["TINYTITAN_K2_CHUNK"] ?? "")
            ?? fallback
        return max(1, min(requested, 16))
    }

    /// The graph's initial sequence capacity. K/V costs ~0.2 MB per position
    /// across the 48 layers, so this reserves ~0.4 GB and covers a normal
    /// multi-turn conversation without a rebuild.
    static let minimumGraphCapacity = 2_048
    /// The context this path will serve, however long the checkpoint claims:
    /// 8,192 positions is ~1.6 GB of plain FP16 K/V across 48 layers.
    package static let contextCeiling = 8_192

    /// `sending` because `Model` is a non-Sendable class that the actor owns
    /// outright once constructed; the caller gives up its reference.
    package init(
        model: sending Model, context: MetalContext, config: ArchConfig,
        contextLimit: Int, slotCount: Int, directory: URL
    ) throws {
        precondition(contextLimit > 0 && slotCount > 0)
        self.model = model
        self.config = config
        self.context = context
        self.contextLimit = contextLimit
        self.slotCount = slotCount
        self.directory = directory
    }

    package enum K2RuntimeError: Error, CustomStringConvertible {
        case graphUnavailable
        case promptTooLong(Int, Int)

        package var description: String {
            switch self {
            case .graphUnavailable: "the probe graph is not resident"
            case .promptTooLong(let count, let limit):
                "prompt is \(count) tokens and the context is \(limit)"
            }
        }
    }

    /// Preload the experts a previous run routed to, into the slots they would
    /// have won anyway. The profile is the streamer's LFU seeding; loading the
    /// top few per layer turns the first request after a restart from a cold
    /// miss on every layer into mostly hits.
    ///
    /// Best effort: a missing or unreadable profile, a read failure, or an
    /// install without routed experts just skips warming.
    /// `TINYTITAN_K2_WARM_EXPERTS` sets how many per layer (0 disables;
    /// default 16).
    package func warmExpertCache() async {
        let limit =
            Int(
                ProcessInfo.processInfo.environment["TINYTITAN_K2_WARM_EXPERTS"] ?? "")
            ?? 16
        guard limit > 0, let url = Self.expertProfileURL(for: directory),
            let profile = ExpertUsageProfile.load(
                url: url, layers: config.numLayers,
                expertsPerLayer: config.numExperts),
            !profile.isEmpty
        else { return }
        // The model is a non-Sendable class owned by this actor; the narrow
        // opt-out is for sending it into the fetch's async call, not a claim
        // it is safe to share.
        nonisolated(unsafe) let model = self.model
        let config = self.config
        let perLayer = min(limit, slotCount, config.numExperts)
        let sparseLayers = (0..<config.numLayers).filter {
            !config.denseLayerIndices.contains($0)
        }
        var warmed = 0
        for layer in sparseLayers {
            let ranked = (0..<config.numExperts)
                .sorted {
                    profile.count(layer: layer, expert: $0)
                        > profile.count(layer: layer, expert: $1)
                }
                .prefix(perLayer)
                .filter { profile.count(layer: layer, expert: $0) > 0 }
            guard !ranked.isEmpty,
                let plan = try? model.planRoutedExperts(
                    layer: layer, experts: Array(ranked)),
                (try? await model.fetchRoutedExperts(plan: plan)) != nil
            else { continue }
            warmed += ranked.count
        }
        if warmed > 0 {
            FileHandle.standardError.write(
                Data(
                    ("k2 probe: warmed \(warmed) expert(s) across "
                        + "\(sparseLayers.count) layers from "
                        + "\(url.lastPathComponent)\n").utf8))
        }
    }

    /// The profile beside the install, unless `TINYTITAN_EXPERT_PROFILE`
    /// names another path or turns the mechanism off. Kept in step with
    /// `Model.expertProfileURL`, which is not visible here.
    static func expertProfileURL(for directory: URL) -> URL? {
        guard
            let raw = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_PROFILE"]
        else { return directory.appendingPathComponent("expert-profile.bin") }
        if raw == "0" { return nil }
        let path = (raw as NSString).expandingTildeInPath
        return path.isEmpty ? nil : URL(fileURLWithPath: path)
    }

    package struct K2GenerationResult {
        package var completion: ServerCompletion
        package var stopReason: DecodeStopReason
        package var prefillSeconds: Double
        package var decodeSeconds: Double
    }

    /// Greedy or sampled decode over one prompt. The prompt is validated by
    /// the caller; `budget` is the completion token count it may use.
    /// `tokenizer` is the one the prompt was rendered with: K2's think markers
    /// follow the reasoning effort, so a mid-session switch needs its own
    /// markers for the channel split even though the ids are the same model's.
    package func generate(
        promptIDs: [Int32],
        budget: Int,
        configuration: GenerationConfig,
        allowedTools: Set<String>?,
        tokenizer: GFTokenizer,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> K2GenerationResult {
        guard promptIDs.count < contextLimit else {
            throw K2RuntimeError.promptTooLong(promptIDs.count, contextLimit)
        }
        let boundedBudget = min(budget, contextLimit - promptIDs.count)
        let promptStart = Date()
        try ensureGraph(needed: promptIDs.count + boundedBudget)
        let firstLogits = try await prefill(promptIDs: promptIDs)
        let prefillSeconds = Date().timeIntervalSince(promptStart)

        let sampler = CPUSampler(
            temperature: configuration.temperature,
            topP: configuration.topP ?? 1,
            topK: configuration.topK ?? 0,
            presencePenalty: configuration.presencePenalty,
            repetitionPenalty: configuration.repetitionPenalty,
            seed: configuration.seed)
        let generator = sampler.makeGenerator()
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: tokenizer, promptIDs: promptIDs,
            allowedTools: allowedTools)
        var detokenizer = GFDetokenizer(tokenizer: tokenizer)
        var output = AssistantOutput(
            stops: configuration.stopStrings, onEvent: onEvent)
        let decodeStart = Date()
        let result = try await decode(
            promptIDs: promptIDs, budget: boundedBudget,
            configuration: configuration, sampler: sampler,
            generator: generator, decoder: decoder,
            detokenizer: &detokenizer, output: &output,
            tokenizer: tokenizer, logits: firstLogits)
        output.publish(
            try decoder.consumeTail(detokenizer.flush()), isToken: false)
        try decoder.finish()
        output.finish()
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        // Fold this request's routing into the profile the next start warms
        // from. The write is async and best effort; a read-only install is not
        // an error.
        model.saveExpertUsageProfile()
        return K2GenerationResult(
            completion: completion(
                promptTokens: promptIDs.count, produced: result.produced,
                reason: result.stopReason == .maxTokens ? "length" : "stop",
                output: output),
            stopReason: result.stopReason,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }

    /// Forward the prompt tokens the resident K/V does not already hold.
    /// At least one token is always forwarded, because its logits are what the
    /// sampler reads.
    private func prefill(promptIDs: [Int32]) async throws -> [Float] {
        guard let graph else { throw K2RuntimeError.graphUnavailable }
        let reused = min(
            Self.commonPrefixLength(promptIDs, cachedTokens),
            max(0, promptIDs.count - 1))
        if reused > 0,
            ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1"
        {
            FileHandle.standardError.write(
                Data(
                    ("k2 probe: reused \(reused)/\(promptIDs.count) prompt tokens "
                        + "(capacity \(graphCapacity))\n").utf8))
        }
        // Chunked prefill batches every layer's projections; the routed
        // experts still stream per token, but their command buffers are
        // committed without waiting so the next fetch overlaps the GPU.
        let chunk = chunkSize
        var logits: [Float] = []
        do {
            if chunk > 1 {
                var start = reused
                while start < promptIDs.count {
                    try Task.checkCancellation()
                    let end = min(start + chunk, promptIDs.count)
                    logits = try await graph.forwardChunk(
                        tokenIDs: promptIDs[start..<end].map {
                            UInt32(bitPattern: $0)
                        },
                        basePosition: UInt32(start))
                    start = end
                }
            } else {
                for position in reused..<promptIDs.count {
                    try Task.checkCancellation()
                    logits = try await graph.forward(
                        tokenID: UInt32(bitPattern: promptIDs[position]),
                        position: UInt32(position))
                }
            }
        } catch {
            // The K/V for the positions this request just wrote does not match
            // the recorded token stream, so the next request must not reuse a
            // prefix. Clearing the record is enough: a full prefill overwrites
            // every position it reads.
            cachedTokens = []
            throw error
        }
        cachedTokens = promptIDs
        return logits
    }

    private struct DecodeResult {
        var produced: Int
        var stopReason: DecodeStopReason
    }

    /// The sampling loop. A disconnected client cancels the task; checking
    /// cancellation here stops the GPU work nobody will read. The K/V record
    /// stays valid across a cancellation because it is appended only after a
    /// token's forward completes.
    private func decode(
        promptIDs: [Int32], budget: Int,
        configuration: GenerationConfig, sampler: CPUSampler,
        generator: CPUSampler.Generator, decoder: StructuredAssistantDecoder,
        detokenizer: inout GFDetokenizer, output: inout AssistantOutput,
        tokenizer: GFTokenizer, logits initialLogits: [Float]
    ) async throws -> DecodeResult {
        guard let graph else { throw K2RuntimeError.graphUnavailable }
        var logits = initialLogits
        var history = promptIDs
        var produced = 0
        var position = promptIDs.count
        var stopReason = DecodeStopReason.maxTokens
        let stops: Set<Int> = [
            Int(tokenizer.eosID), Int(tokenizer.endOfTurnID),
        ]
        while produced < budget {
            try Task.checkCancellation()
            if let constraint = configuration.constraint {
                let mask = constraint.allowedMask()
                guard !mask.isEmpty else {
                    throw GeneratorError.constrainedDecodeStalled
                }
                mask.apply(toLogits: &logits)
            }
            let next = sampler.pick(logits, history: history, using: generator)
            if let constraint = configuration.constraint,
                !constraint.observe(Int32(next))
            {
                throw GeneratorError.constrainedDecodeViolation(id: Int32(next))
            }
            if stops.contains(next) {
                stopReason = next == Int(tokenizer.eosID) ? .eos : .endOfTurn
                break
            }
            produced += 1
            history.append(Int32(next))
            output.publish(
                try decoder.consume(
                    tokenID: Int32(next), delta: detokenizer.push(Int32(next))))
            if output.isStopped {
                stopReason = .stopString
                break
            }
            if produced >= budget { break }
            logits = try await graph.forward(
                tokenID: UInt32(bitPattern: Int32(next)),
                position: UInt32(position))
            cachedTokens.append(Int32(next))
            position += 1
        }
        return DecodeResult(produced: produced, stopReason: stopReason)
    }

    /// The resident graph, grown when a request needs more sequence positions.
    /// Growth starts from `minimumGraphCapacity` so a normal conversation does
    /// not rebuild on its second turn; a rebuild drops the K/V cache, which is
    /// why the capacity is not this request's exact size.
    private func ensureGraph(needed: Int) throws {
        let wanted = min(contextLimit, max(needed, Self.minimumGraphCapacity))
        guard graph == nil || needed > graphCapacity else { return }
        let graphStart = Date()
        graph = try K2GraphProbe(
            model: model, context: context, maxSequence: wanted,
            maxChunk: chunkSize)
        graphCapacity = wanted
        cachedTokens = []
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            FileHandle.standardError.write(
                Data(
                    ("k2 probe: graph built in "
                        + String(format: "%.2f", Date().timeIntervalSince(graphStart))
                        + "s (maxSequence \(wanted))\n").utf8))
        }
    }

    /// The completion object, from the accumulated stream.
    private func completion(
        promptTokens: Int, produced: Int, reason: String,
        output: AssistantOutput
    ) -> ServerCompletion {
        ServerCompletion(
            content: output.content,
            toolCalls: output.calls,
            finishReason: reason,
            usage: OpenAIUsage(
                promptTokens: promptTokens,
                completionTokens: produced,
                totalTokens: promptTokens + produced,
                cachedTokens: 0,
                reasoningTokens: output.reasoningTokens),
            stopSequence: output.matchedStop,
            reasoning: output.reasoning,
            // K2 is effort-only: a render always opens a thought block, so any
            // reasoning is requested.
            unrequestedReasoning: 0)
    }

    /// The number of leading token ids the two sequences share.
    private static func commonPrefixLength(_ lhs: [Int32], _ rhs: [Int32]) -> Int {
        var count = 0
        while count < lhs.count, count < rhs.count, lhs[count] == rhs[count] {
            count += 1
        }
        return count
    }
}
