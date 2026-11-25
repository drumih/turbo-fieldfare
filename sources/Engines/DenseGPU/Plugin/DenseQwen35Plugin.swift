import Foundation
import Metal
import TinyTitanCore
import TinyTitanPluginAPI

/// The dense Qwen 3.5 family (2B, 4B, 9B at 4 and 8 bits) on the dense GPU
/// engine, as a TPP/1 engine plugin (TT-132).
///
/// Standalone: probe, parameters, load plan, identity, model, runner and
/// completion loop are this target's own. The load plan is the 6.x planner's
/// arithmetic on the target's copies, so the session receives the plan it did
/// before the plugin, and the goldens stay identical.
public enum DenseQwen35Plugin: EnginePlugin {
    public static let pluginID = "ai.tinytitan.qwen3.5-dense"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Qwen 3.5 dense"

    /// A dense install without an explicit window gets 1M with YaRN; the
    /// thresholded kernel keeps every position below the native 262,144 on the
    /// native path.
    static let defaultContext = 1_048_576

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        guard install.manifest != nil,
            let family = try? ManifestReader.peekFamily(directoryURL: install.directory),
            family == .qwen35Dense
        else { return nil }
        return .exact
    }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            hostKVWindow: .refused(
                reason: "the dense engine keeps its K/V in Metal; it has no host-KV arena"),
            multiTokenPrediction: .refused(reason: "the dense engine has no MTP draft path"),
            expertStreaming: .refused(reason: "a dense model has no routed experts"))
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        coreControl(ModelFamily.qwen35Dense.reasoningControl)
    }

    public static func chatDialect(for install: InstallProbe) -> (any ChatDialect)? {
        ChatMLDialect()
    }

    static func coreControl(_ control: ModelReasoningControl) -> ReasoningControl {
        switch control {
        case .binaryThinking:
            .binaryThinking
        case .thinkingWithEffortLevels(let defaultEffort, let supported):
            .thinkingWithEffortLevels(defaultEffort: defaultEffort, supported: supported)
        case .effortOnly(let defaultEffort, let supported):
            .effortOnly(defaultEffort: defaultEffort, supported: supported)
        }
    }

    /// lint:allow-long one factory literal: every stage is a short closure over
    /// this target's own types; splitting them would only move the list.
    public static func engineFactory(host: HostServices) -> EngineFactory {
        EngineFactory(
            makeContext: { device in try MetalContext.runtimeDefault(device: device) },
            prepareLoad: { options in try plan(options, host: host) },
            metadata: { directory in try metadata(directory) },
            validateReasoning: { metadata, thinking, effort in
                let family = ModelFamily(rawValue: metadata.family) ?? .qwen35Dense
                try family.validateReasoning(thinkingMode: thinking, effort: effort)
            },
            resolveArchitecture: { directory in
                try ArchConfig.resolved(
                    forFamily: try ManifestReader.peekFamily(directoryURL: directory),
                    directoryURL: directory)
            },
            preview: { directory, chunk in try preview(directory, requestedChunk: chunk) },
            readManifest: { directory, arch in
                try ManifestReader.load(directoryURL: directory, expecting: arch)
            },
            defaultMaxContextTokens: defaultContext,
            buildModel: { directory, _, context in
                guard let context else {
                    throw MetalError.invalidState("the dense Qwen 3.5 engine requires Metal")
                }
                // Geometry comes from the install manifest (the dense-only load
                // path resolves it); a dense install has no routed experts.
                return try Model.load(directoryURL: directory, device: context.device)
            },
            buildRunner: { model, options in
                guard let model = model as? Model, let context = options.context else {
                    throw MetalError.invalidState(
                        "the dense Qwen 3.5 runner requires its model and a Metal context")
                }
                // The extension profile is fixed at 1M independent of the cap,
                // so a larger --max-context changes capacity, not the rope
                // function; with YaRN off the profile is unused.
                let runtime = try RuntimeConfiguration(
                    forceLogitsHead: true, kvCachePrecision: options.kvCachePrecision,
                    ropeScalingMode: options.ropeScalingMode,
                    yarnContextTokens: RuntimeConfiguration.defaultYaRNContextTokens)
                return try RealForwardRunner(
                    model: model, context: context, maxContext: options.maxContext,
                    slots: options.slots, runtimeConfiguration: runtime)
            },
            makeScratch: { context, vocabulary, softcap in
                TinyTitanCore.RawCompletionScratch(
                    payload: try RawCompletionScratch(
                        context: context, vocab: vocabulary, logitSoftcap: softcap))
            },
            runRawCompletion: { request in try await runCompletion(request) })
    }

    static func metadata(_ directory: URL) throws -> EngineModelMetadata {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let sampling = ModelProfile.resolve(identity: identity).sampling
        return EngineModelMetadata(
            modelID: identity.modelID, family: identity.family.rawValue,
            weightBits: identity.weightBits,
            sampling: SamplingProfile(
                temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP,
                minP: sampling.minP, presencePenalty: sampling.presencePenalty, thinking: nil),
            reasoningControl: coreControl(identity.family.reasoningControl))
    }

    /// The 6.x preview: the dense family's default chunk is the production one
    /// here (the load plan's is the long-prompt chunk), as before the plugin.
    static func preview(_ directory: URL, requestedChunk: Int?) throws -> EngineSessionPreview {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let chunk =
            requestedChunk
            ?? ModelProfile.resolve(identity: identity).prefillChunkTokens
            ?? RuntimeConfiguration.production.prefillChunkTokens
        var id = identity.modelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        if id == "unknown/snapshot" { id = "qwen3.5-dense" }
        return EngineSessionPreview(
            modelID: id + "_\(identity.weightBits)-Bit", prefillChunkTokens: chunk)
    }
}
