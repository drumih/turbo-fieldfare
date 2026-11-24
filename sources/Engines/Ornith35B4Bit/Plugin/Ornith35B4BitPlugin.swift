import Foundation
import Metal
import TinyTitanCore
import TinyTitanPluginAPI

/// Ornith 1.5 35B-A3B at 4 bits as a TPP/1 engine plugin (TT-132).
///
/// Standalone: the probe, the parameter table, the load plan (expert slots from
/// `--ram-budget` or the model profile's tuned budget, the prefill chunk, the
/// runtime identity), the identity and sampling, the model, runner and
/// completion loop are all this target's own copies. The load plan is the 6.x
/// planner's arithmetic on those copies, so the session sees the same plan and
/// the golden baseline stays identical.
public enum Ornith35B4BitPlugin: EnginePlugin {
    public static let pluginID = "ai.tinytitan.ornith-1.5-35b-a3b-4bit"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Ornith 1.5 35B-A3B 4-bit"

    /// The variant's parity was measured at 32 slots; the load plan's slot
    /// count replaces it whenever the session resolves one.
    static let defaultStreamingSlots = 32

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        guard install.manifest != nil,
            let identity = try? ManifestReader.peekIdentity(directoryURL: install.directory),
            identity.family == .qwen36, identity.modelID.hasPrefix("ornith-1.5-35b"),
            identity.weightBits == 4
        else { return nil }
        return .exact
    }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            multiTokenPrediction: .refused(
                reason: "the Ornith 4-bit engine has no MTP draft path; the 6.x engine's "
                    + "was not carried into the variant"))
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        coreControl(ModelFamily.qwen36.reasoningControl)
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
                let family = ModelFamily(rawValue: metadata.family) ?? .qwen36
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
            buildModel: { directory, arch, context in
                try loadModel(directory, arch, context, slots: nil)
            },
            // The session's resolved slot count (profile default, --ram-budget or
            // --expert-cache-slots) reaches the model here; the environment
            // override stays the operator's last word.
            buildModelWithSlots: { directory, arch, context, slots in
                try loadModel(directory, arch, context, slots: slots)
            },
            buildRunner: { model, options in
                guard let model = model as? Model, let context = options.context else {
                    throw MetalError.invalidState(
                        "the Ornith runner requires its model and a Metal context")
                }
                let runtime = try RuntimeConfiguration(
                    forceLogitsHead: true, kvCachePrecision: options.kvCachePrecision)
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

    static func loadModel(
        _ directory: URL, _ arch: ArchConfig, _ context: MetalContext?, slots: Int?
    ) throws -> any EngineModel {
        guard let context else {
            throw MetalError.invalidState("the Ornith engine requires Metal")
        }
        let requested =
            ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init) ?? slots ?? defaultStreamingSlots
        return try Model.load(
            directoryURL: directory, device: context.device, expecting: arch,
            streamingMode: .pread(slotCount: requested))
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

    static func preview(_ directory: URL, requestedChunk: Int?) throws -> EngineSessionPreview {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let chunk =
            requestedChunk
            ?? ModelProfile.resolve(identity: identity).prefillChunkTokens
            ?? RuntimeConfiguration.qwenLongPrefillChunkTokens
        var id = identity.modelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        return EngineSessionPreview(
            modelID: id + "_\(identity.weightBits)-Bit", prefillChunkTokens: chunk)
    }
}
