import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanLegacyEngine
import TinyTitanCore

/// What the HTTP layer needs to validate a request for one model before that
/// model is resident: the omitted-sampling defaults, the max_tokens bound and
/// the reasoning profile all belong to the model a request names, not to
/// whichever one happens to be loaded.
public struct ServedModel: Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let maximumContext: Int
    public let sampling: SamplingProfile
    package let reasoningProfile: ServerReasoningProfile

    package init(
        id: String, displayName: String, maximumContext: Int,
        sampling: SamplingProfile,
        reasoningProfile: ServerReasoningProfile
    ) {
        self.id = id
        self.displayName = displayName
        self.maximumContext = maximumContext
        self.sampling = sampling
        self.reasoningProfile = reasoningProfile
    }
}

/// A backend that serves several models by name. Kept apart from
/// `ServerInferenceBackend` for the reason `ResidencyManaging` is: one backend
/// routes, and every other conformer would carry a member that answers "not me".
public protocol ModelRouting: Sendable {
    /// Every model a request may name, in listing order.
    var servedModels: [ServedModel] { get }
    /// Includes addressable engine aliases that are deliberately not listed.
    func servedModel(named name: String) -> ServedModel?
}

extension ModelRouting {
    /// The model a request's `model` field names, or nil for an unknown name.
    /// An exact id wins over the "<model>-fast" alias, so a catalog id that
    /// itself ends in "-fast" stays reachable.
    public func servedModel(named name: String) -> ServedModel? {
        if let exact = servedModels.first(where: { $0.id == name }) { return exact }
        guard name.hasSuffix("-fast") else { return nil }
        let base = String(name.dropLast("-fast".count))
        return servedModels.first { $0.id == base }
    }
}

/// The reasoning one model runs under for the server-wide level.
public struct ReasoningChoice: Sendable, Equatable {
    public let requested: TinyTitanCore.ReasoningLevel
    public let effective: TinyTitanCore.ReasoningLevel
    public let thinking: ModelThinkingMode
    public let effort: ModelReasoningEffort?
}

/// Fits one server-wide reasoning level to models that expose different ones.
///
/// `effectiveLevel` now lives in `TinyTitanLib` (2026-10-02, phase A1 of
/// `docs/plan-embedded-library.md`), because the request validator there
/// applies the same mapping; this extension keeps the catalog-aware half,
/// which reads the server's `ModelCatalog.Kind`.
extension ReasoningFallback {
    package static func choice(
        for kind: ModelCatalog.Kind,
        requested: TinyTitanCore.ReasoningLevel
    ) throws -> ReasoningChoice {
        let effective = effectiveLevel(
            requested, supported: kind.supportedReasoningLevels,
            whenOn: kind.levelWhenOn)
        let runtime = try kind.runtimeReasoning(for: effective)
        return ReasoningChoice(
            requested: requested, effective: effective,
            thinking: runtime.thinking, effort: runtime.effort)
    }
}

extension ModelRouter {
    /// Loads entries the way the single-model paths do: a GPU install through
    /// `ModelSessionPlan` with every server flag, a CPU snapshot through
    /// `CPUModelBackend`. Reasoning comes from the router, fitted per model.
    package static func standardLoader(arguments: ServerArguments) -> Loader {
        let metal = SharedMetalContext()
        return { entry, reasoning in
            switch entry.kind {
            case .gpu(.k2Horizon):
                // K2 has a graph, but not the runner's: the probe is the
                // engine, so the catalog path takes the same backend the
                // single-model path does. No prompt cache, per-request graph.
                let slots = Int(
                    ProcessInfo.processInfo.environment["TINYTITAN_K2_SLOTS"] ?? ""
                )
                .flatMap { $0 > 0 ? $0 : nil }
                return try await K2ProbeBackend(
                    modelDirectory: entry.path,
                    maximumContext: arguments.maxContext,
                    thinking: reasoning.thinking,
                    reasoningEffort: reasoning.effort,
                    slots: slots)
            case .gpu:
                // One plan factory for both front ends: a loader-built plan
                // that drops a field is how `slots` silently stayed at one on
                // the catalog path while the coordinator admitted four.
                let plan = ModelSessionPlan.from(
                    arguments: arguments,
                    modelDirectory: entry.path,
                    thinking: reasoning.thinking,
                    reasoningEffort: reasoning.effort,
                    mtpModelDirectory: nil)
                // TT-097 Task 11: the shipped variant installs route to their
                // dedicated engines; the identity read is the same bounded
                // manifest read the plan's reasoning profile performs, and
                // every identity without a shipped variant stays on the 6.x
                // engine (TT-087).
                return try await plan.makeSession(
                    reusingContext: try await metal.context(),
                    factory: EngineFactories.gpuFactory(directoryURL: entry.path))
            case .cpu:
                // `entry.quant` is the routed-expert width the catalog read
                // from the install — the fact the variant CPU selection needs
                // beside the hidden size the backend reads from the snapshot.
                return try await CPUModelBackend(
                    snapshotDirectory: entry.path,
                    maximumContext: arguments.maxContext,
                    resident: arguments.cpuResident,
                    thinkingMode: reasoning.thinking,
                    routedExpertWeightBits: entry.quant)
            }
        }
    }
}

extension ModelRouter {
    /// Counts with the named model's tokenizer, rendered as its engine
    /// renders a prompt, without loading any weights.
    package static let standardCounter: Counter = { entry, choice, request in
        _ = try GFTokenizer.requireModelDirectory(entry.path)
        switch entry.kind {
        case .gpu:
            guard let folder = GFTokenizer.tokenizerFolder(forModelDirectory: entry.path) else {
                throw GFTokenizerError.missingToolTemplate
            }
            let tokenizer = try await GFTokenizer.load(
                from: folder,
                thinkingMode: choice.thinking,
                reasoningEffort: choice.effort)
            return try ServerModelSession.promptTokenCount(request, tokenizer: tokenizer)
        case .cpu:
            // The folder, not the model directory: a `.ssdai` install keeps
            // `tokenizer.json` in a `tokenizer/` sidecar, so handing
            // `load(from:)` the model directory fails for every installed CPU
            // model (the GPU branch above resolves the same way).
            guard
                let folder = GFTokenizer.resolvedTokenizerFolder(
                    forModelDirectory: entry.path)
            else {
                throw GFTokenizerError.missingToolTemplate
            }
            let tokenizer = try await GFTokenizer.load(
                from: folder,
                thinkingMode: choice.thinking)
            return try CPUModelBackend.promptTokenCount(request, tokenizer: tokenizer)
        }
    }
}

/// One Metal context for the process, built on the first GPU load. A command
/// queue has no deinit-safe teardown (see `MetalContext.deinit`) and the
/// shader library is costly to compile, so every switch reuses it.
private actor SharedMetalContext {
    private var built: MetalContext?

    func context() throws -> MetalContext {
        if let built { return built }
        let context = try MetalContext.runtimeDefault()
        built = context
        return context
    }
}

extension ServerArguments {
    /// The thinking settings a single-model server loads with. Without
    /// `--reasoning` these are the `--thinking` / `--reasoning-effort` values
    /// verbatim, as before; with it, the level is fitted to the model's family
    /// exactly as the router fits it.
    public func singleModelReasoning(
        directory: URL
    ) throws -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
        guard let reasoningLevel else { return (thinkingMode, reasoningEffort) }
        let kind: ModelCatalog.Kind =
            cpu
            ? .cpu(try ModelCatalog.snapshotFamily(directory))
            : .gpu(try ManifestReader.peekFamily(directoryURL: directory))
        let choice = try ReasoningFallback.choice(for: kind, requested: reasoningLevel)
        ServerLog.residency(
            "reasoning=\(choice.effective.rawValue)"
                + (choice.effective == reasoningLevel
                    ? "" : " (server level \(reasoningLevel.rawValue))"))
        return (choice.thinking, choice.effort)
    }
}
