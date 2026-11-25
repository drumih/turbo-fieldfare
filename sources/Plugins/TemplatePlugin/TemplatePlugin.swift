import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

/// The template engine plugin: the golden base every LLM plugin is copied from
/// (TPP/1, `docs/plugin-api-v1.md`).
///
/// It serves a miniature reference model that needs no weights: a counting
/// engine whose next token is the last one plus one (`TemplateRunner`). That
/// keeps every requirement real and testable without a model, so a new plugin
/// starts from code that already satisfies the protocol and replaces one piece
/// at a time:
///
/// 1. `pluginID`, `displayName`, `probe`      -- what install is this plugin's?
/// 2. `parameters(for:)`                       -- which user settings apply?
/// 3. `reasoningControl(for:)`                 -- thinking and effort levels
/// 4. `engineFactory(host:)`                   -- plan, model, runner, loop
///    - `TemplateLoadPlan`                     -- window and memory from the budget
///    - `TemplateModel`   (`EngineModel`)      -- identity and geometry
///    - `TemplateRunner`  (`EngineRunner`)     -- prefill, decode, state, stats
///    - `runTemplateCompletion`                -- the generation loop
///
/// Rules for a plugin (enforced by the `plugin-isolation` gate):
/// - import `TinyTitanPluginAPI`, `TinyTitanCore` and system frameworks only;
/// - keep every model-specific choice -- kernels, tuning, defaults, sampling,
///   chat format -- inside the plugin target;
/// - log through `HostServices.log`, never stdout.
public enum TemplatePlugin: EnginePlugin {
    public static let pluginID = "ai.tinytitan.template"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "TinyTitan template engine"

    /// The manifest family this plugin owns. A real plugin keys on family,
    /// quantization scheme and model id, so two plugins never tie.
    public static let family = "tinytitan-template"

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        install.declaredFamily == family ? .exact : nil
    }

    /// Declare every parameter honestly. The template keeps its state in plain
    /// arrays, so K/V precision and host-KV windows do not apply, and it runs
    /// one sequence; everything not listed is supported.
    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            kvCachePrecision: .only(
                [.fp16], reason: "the template keeps its state unquantized"),
            hostKVWindow: .refused(reason: "the template has no full-attention cache"),
            concurrentSequences: .only([1], reason: "the template runs one sequence"),
            multiTokenPrediction: .refused(reason: "the template has no draft head"),
            expertStreaming: .refused(reason: "the template has no routed experts"))
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        .binaryThinking
    }

    public static func chatDialect(for install: InstallProbe) -> (any ChatDialect)? {
        ChatMLDialect()
    }

    public static func engineFactory(host: HostServices) -> EngineFactory {
        EngineFactory(
            prepareLoad: { options in
                let plan = TemplateLoadPlan.make(
                    requested: options.maxContext, ramBudget: options.expertCacheBudgetBytes)
                host.log(
                    "\(displayName): context \(plan.context) tokens "
                        + "(asked \(options.maxContext)), planned \(plan.plannedBytes) bytes")
                return EngineLoadPlan(
                    architecture: TemplateModel.architecture, expertCacheSlots: 0,
                    tunedExpertCacheBudgetBytes: 0, prefillConfiguration: .defaultChunked,
                    fp16RingEnabled: false, yarnContextTokens: 0, runtimeIdentity: pluginID,
                    engineConfiguration: plan)
            },
            metadata: { directory in
                EngineModelMetadata(
                    modelID: TemplateModel.modelID(directory), family: family, weightBits: 16,
                    sampling: .house, reasoningControl: .binaryThinking)
            },
            resolveArchitecture: { _ in TemplateModel.architecture },
            preview: { directory, chunk in
                EngineSessionPreview(
                    modelID: TemplateModel.modelID(directory), prefillChunkTokens: chunk ?? 64)
            },
            buildModel: { directory, _, _ in TemplateModel(directory: directory) },
            buildRunner: { model, options in
                guard let model = model as? TemplateModel,
                    let plan = options.engineConfiguration as? TemplateLoadPlan
                else { throw TemplateError.wrongModel }
                return TemplateRunner(model: model, maxContext: plan.context)
            },
            runRawCompletion: { request in try await runTemplateCompletion(request) },
            // A real plugin whose install does not keep `tokenizer/` derives the
            // folder here (Inkling 2 Q4 exports it from its GGUF header).
            tokenizerFolder: nil)
    }
}

public enum TemplateError: Error, Equatable, CustomStringConvertible {
    case wrongModel
    case notThisRunner
    case position(expected: Int, got: Int)

    public var description: String {
        switch self {
        case .wrongModel: "the template runner needs the template model and its load plan"
        case .notThisRunner: "the template completion loop needs the template runner"
        case .position(let expected, let got): "position \(got), the state is at \(expected)"
        }
    }
}
