import Foundation
import Metal
import TinyTitan
import TinyTitanLib
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanCPUEngine

/// Serving K2-Horizon from the GPU probe, through the same HTTP surface as the
/// other families.
///
/// The probe (`K2ProbeRuntime`) is the only graph that runs these weights
/// today: `RealForwardRunner` sizes a Qwen-shaped pipeline set and refuses the
/// family. Rather than pretend the runner serves it, this backend is the probe
/// behind the `ServerInferenceBackend` protocol, the way `CPUModelBackend` is
/// the CPU engine. Everything above it — the OpenAI and Responses surfaces,
/// the memory subsystem, the watchdogs — is unchanged.
///
/// **What it does not do, deliberately.** No prompt cache at the HTTP level
/// (the runtime reuses a token-prefix K/V instead), no batching, no ANE
/// prefill, no runner statistics. Those belong to the runner graph, not to
/// this adapter.
///
/// **Memory.** The KV caches are plain FP16 and per layer, about 0.2 MB per
/// position across the 48 layers; the runtime's 8,192-position ceiling is
/// ~1.6 GB on top of the expert cache the model already holds, which is what
/// this 24 GB host can carry. The checkpoint's 524,288-token context is a
/// property of the weights, not something this path can afford to allocate.
public actor K2ProbeBackend: ServerInferenceBackend, PromptCacheDescribing {
    /// The context this path will serve, however long the checkpoint says it
    /// can. See the memory note above.
    public static let contextCeiling = K2ProbeRuntime.contextCeiling

    package static var defaultSamplingProfile: SamplingProfile {
        let d = GenerationDefaults.forFamily(.k2Horizon)
        return SamplingProfile(
            temperature: d.temperature, topK: d.topK, topP: d.topP,
            minP: d.minP, presencePenalty: d.presencePenalty)
    }

    package static func reasoningProfile(
        thinking: ModelThinkingMode, effort: ModelReasoningEffort?
    ) -> ServerReasoningProfile {
        ServerReasoningProfile(
            family: ModelFamily.k2Horizon.rawValue,
            thinkingMode: thinking,
            reasoningEffort: effort,
            reasoningControl: ModelFamily.k2Horizon.engineReasoningControl)
    }

    private let directory: URL
    private let tokenizer: GFTokenizer
    private let tokenizerFolder: URL
    private let loadedReasoning: TinyTitanCore.RequestReasoning
    private let contextLimit: Int
    private let defaults: GenerationDefaults.Sampling
    private let slotCount: Int
    private var jsonTokenTable: JSONTokenTable?
    private let runtime: K2ProbeRuntime

    public nonisolated var maximumContext: Int { contextLimit }
    public nonisolated var samplingDefaults: SamplingProfile {
        SamplingProfile(
            temperature: defaults.temperature, topK: defaults.topK, topP: defaults.topP,
            minP: defaults.minP, presencePenalty: defaults.presencePenalty)
    }
    /// The probe keeps no prompt cache and says so rather than borrowing the
    /// mode the server was asked for.
    package nonisolated var promptCacheMode: ServerPromptCacheMode { .off }
    /// The routed-expert slots per layer in force, for the launch banner.
    package nonisolated var expertCacheSlots: Int { slotCount }

    public enum K2BackendError: Error, CustomStringConvertible {
        case unsupported(String)
        case emptyPrompt

        public var description: String {
            switch self {
            case .unsupported(let detail): detail
            case .emptyPrompt: "the prompt tokenized to nothing"
            }
        }
    }

    /// Loads the install and the tokenizer. The receipt path matches the
    /// bench: a valid receipt avoids re-hashing 15 GB of expert layers on
    /// every start, and an invalid one falls back to the full verification
    /// rather than loading unchecked.
    public init(
        modelDirectory: URL,
        maximumContext: Int,
        thinking: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort?,
        slots: Int? = nil,
        reusingContext: MetalContext? = nil
    ) async throws {
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: modelDirectory)
        let identity = try ManifestReader.peekIdentity(directoryURL: modelDirectory)
        let resolvedSlots = slots ?? (identity.weightBits == 4 ? 64 : 16)
        guard resolvedSlots > 0 else {
            throw K2BackendError.unsupported("expert cache slots must be positive")
        }
        let context = try reusingContext ?? MetalContext.runtimeDefault()
        let loaded: Model
        do {
            loaded = try Model.load(
                directoryURL: modelDirectory, device: context.device,
                expecting: config, streamingMode: .pread(slotCount: resolvedSlots),
                integrityPolicy: .sizeCheckTrustedReceipt)
        } catch let error as ModelError {
            guard case .trustedReceiptInvalid = error else { throw error }
            loaded = try Model.load(
                directoryURL: modelDirectory, device: context.device,
                expecting: config, streamingMode: .pread(slotCount: resolvedSlots))
        }
        guard
            let folder = GFTokenizer.resolvedTokenizerFolder(
                forModelDirectory: modelDirectory)
        else {
            throw K2BackendError.unsupported(
                "no tokenizer in \(modelDirectory.lastPathComponent)")
        }
        self.directory = modelDirectory
        self.slotCount = resolvedSlots
        self.tokenizerFolder = folder
        self.tokenizer = try await GFTokenizer.load(
            from: folder, thinkingMode: thinking, reasoningEffort: reasoningEffort)
        self.loadedReasoning = TinyTitanCore.RequestReasoning(
            thinkingMode: thinking, effort: reasoningEffort)
        self.contextLimit = min(maximumContext, Self.contextCeiling)
        self.defaults = GenerationDefaults.forFamily(.k2Horizon)
        self.runtime = try K2ProbeRuntime(
            model: loaded, context: context, config: config,
            contextLimit: self.contextLimit, slotCount: resolvedSlots,
            directory: modelDirectory)
        await runtime.warmExpertCache()
    }

    package func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        let promptIDs = try encodePrompt(
            renderTokenizer, messages: request.messages, tools: request.tools)
        guard !promptIDs.isEmpty else { throw K2BackendError.emptyPrompt }
        // A prompt past the probe's context is the client's error, not the
        // server's: report it the way the other families do (400,
        // context_length_exceeded) instead of failing the request with a 500.
        guard promptIDs.count < contextLimit else {
            throw ServerRequestError.invalid(
                message:
                    "prompt is \(promptIDs.count) tokens and the model's context is \(contextLimit)",
                param: "messages", code: "context_length_exceeded")
        }
        let budget = min(
            request.maximumCompletionTokens, contextLimit - promptIDs.count)
        let result = try await runtime.generate(
            promptIDs: promptIDs, budget: budget,
            configuration: structuredConfiguration(request),
            allowedTools: request.tools.isEmpty
                ? nil : Set(request.tools.map(\.name)),
            tokenizer: renderTokenizer, onEvent: onEvent)
        return result.completion
    }

    /// The request's generation config with the structured-output constraint,
    /// over a token table built once per backend.
    private func structuredConfiguration(
        _ request: ValidatedChatRequest
    ) -> GenerationConfig {
        var configuration = request.generationConfig
        guard let node = request.jsonSchema else { return configuration }
        let table: JSONTokenTable
        if let existing = jsonTokenTable {
            table = existing
        } else {
            let made = JSONTokenTable(tokenizer: tokenizer)
            jsonTokenTable = made
            table = made
        }
        configuration.constraint = JSONConstraint(
            table: table, node: node, vocab: tokenizer.vocabSize)
        return configuration
    }

    private func resolvedTokenizer(
        for reasoning: TinyTitanCore.RequestReasoning?
    ) async throws -> GFTokenizer {
        guard let reasoning, !reasoning.matches(loadedReasoning) else {
            return tokenizer
        }
        return try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: reasoning.thinkingMode,
            reasoningEffort: reasoning.effort)
    }

    /// The same render the GPU path uses: the tool template when the request
    /// carries tools, the plain chat template otherwise.
    private func encodePrompt(
        _ renderTokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) throws -> [Int32] {
        if tools.isEmpty {
            let rendered = try renderTokenizer.applyChatTemplate(messages)
            return renderTokenizer.encode(rendered, addBOS: false)
        }
        return try renderTokenizer.encodeToolChat(messages: messages, tools: tools)
    }
}

/// The Messages API's count_tokens, from the same rendering `generate` uses.
extension K2ProbeBackend: PromptTokenCounting {
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let renderTokenizer = try await resolvedTokenizer(for: request.reasoning)
        return try encodePrompt(
            renderTokenizer, messages: request.messages, tools: request.tools
        ).count
    }
}

/// 6.x `ModelFamily` bridges for `ModelCatalog.Kind`, kept beside the 6.x
/// probe adapter so `ModelRouter`, `ModelRouterError`, and `main.swift` need
/// not import `TinyTitan` (TT-134).
extension ModelCatalog.Kind {
    package var reasoningProfileTarget: (family: String, control: ReasoningControl) {
        switch self {
        case .gpu(let family): (family.rawValue, family.engineReasoningControl)
        case .cpu: (ModelFamily.qwen36.rawValue, ModelFamily.qwen36.engineReasoningControl)
        case .plugin(let family, let control): (family, control)
        }
    }

    package var isK2Horizon: Bool {
        if case .gpu(.k2Horizon) = self { return true }
        return false
    }

    package static func resolve(directory: URL, cpu: Bool) throws -> Self {
        cpu
            ? .cpu(try ModelCatalog.snapshotFamily(directory))
            : .gpu(try ManifestReader.peekFamily(directoryURL: directory))
    }

    package static func cpuReasoningProfile(
        thinking: ModelThinkingMode
    ) -> ServerReasoningProfile {
        ServerReasoningProfile(
            family: ModelFamily.qwen36.rawValue,
            thinkingMode: thinking,
            reasoningEffort: nil,
            reasoningControl: ModelFamily.qwen36.engineReasoningControl)
    }
}
