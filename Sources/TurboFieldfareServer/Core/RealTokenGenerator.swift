import Foundation
import TurboFieldfare

/// Bridges the Metal runtime to the Metal-free `TokenGenerating` seam.
///
/// One instance per server, owned by the `GenerationSession` actor. Metal
/// state (`runner` / `scratch` / `context`) is touched only from `generate`,
/// which the session's single-in-flight slot serializes. `tokenCount` runs
/// outside the slot so bad requests fail fast with a 400; it only encodes via
/// `GFTokenizer`, whose underlying `Tokenizer` protocol is `Sendable` with
/// immutable post-load state, so concurrent encode/detokenize is safe.
///
/// The runner is built once with `forceLogitsHead: true`: an API server
/// mostly serves sampling configs, and the logits head also serves
/// pure-greedy configs, so one runner covers all requests.
final class RealTokenGenerator: TokenGenerating, @unchecked Sendable {
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let context: MetalContext
    private let scratch: RawCompletionScratch
    private let runtime: RuntimeConfiguration

    init(tokenizer: GFTokenizer,
         runner: RealForwardRunner,
         context: MetalContext,
         scratch: RawCompletionScratch,
         runtime: RuntimeConfiguration) {
        self.tokenizer = tokenizer
        self.runner = runner
        self.context = context
        self.scratch = scratch
        self.runtime = runtime
    }

    /// Load sequence mirrors the CLI (`Sources/TurboFieldfareCLI/Run.swift`).
    static func load(modelPath: String, maxContext: Int) async throws -> RealTokenGenerator {
        let modelURL = URL(fileURLWithPath: modelPath)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let runtime = RuntimeConfiguration(forceLogitsHead: true)
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        return RealTokenGenerator(tokenizer: tokenizer,
                                  runner: runner,
                                  context: context,
                                  scratch: scratch,
                                  runtime: runtime)
    }

    func tokenCount(for input: GenerationInput) throws -> Int {
        try encode(input).count
    }

    func generate(_ input: GenerationInput,
                  sampling: SamplingParams,
                  effectiveMaxTokens: Int,
                  onDelta: (@Sendable (String) -> Void)?) async throws -> GenerationOutcome {
        let promptIds = try encode(input)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxTokens,
            temperature: sampling.temperature,
            topK: 64,
            topP: sampling.topP,
            repetitionPenalty: 1.0,
            seed: sampling.seed,
            stopStrings: sampling.stop,
            extraStopTokens: [])
        var text = ""
        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty {
                        text += delta
                        onDelta?(delta)
                    }
                case .tail(let tail):
                    if !tail.isEmpty {
                        text += tail
                        onDelta?(tail)
                    }
                }
            }
        return GenerationOutcome(text: text,
                                 promptTokens: result.prefillTokens,
                                 completionTokens: result.newTokens,
                                 finishReason: Self.finishReason(for: result.reason))
    }

    private func encode(_ input: GenerationInput) throws -> [Int32] {
        switch input {
        case .chat(let messages):
            let templated = try messages.map { message -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: message.role) else {
                    throw GenerationRequestError.unsupportedRole(message.role)
                }
                return GFTokenizer.Message(role: role, content: message.content)
            }
            let rendered = try tokenizer.applyChatTemplate(templated)
            return tokenizer.encode(rendered, addBOS: false)
        case .raw(let prompt):
            return tokenizer.encode(prompt, addBOS: true)
        }
    }

    static func finishReason(for reason: StopReason) -> FinishReason {
        switch reason {
        case .endOfTurn, .eos, .stopString:
            return .stop
        case .maxTokens:
            return .length
        }
    }
}
