import Foundation
import Metal
import TinyTitanCore
import TinyTitanPluginAPI

/// Inkling 2 Mini from its original NVFP4/MXFP8 checkpoint (`.ssdai`), as a
/// TPP/1 engine plugin (TT-125).
///
/// It runs the same decode graph as the Q4 plugin (a copy, per the 7.0
/// duplication rule) over the checkpoint install. `InklingQ4Install` maps the
/// checkpoint's tensors onto the graph's roles with the transforms measured in
/// `docs/plan-inkling2-7x-engine.md` (TT-125), and the expert cache repacks each
/// expert as it is read: NVFP4 into the GGUF blocks the kernels read (byte-exact
/// with the Q4 install), MXFP8 into 34-byte blocks the MXFP8 kernels read.
public enum Inkling2NVFP4Plugin: EnginePlugin {
    public static let pluginID = "ai.tinytitan.inkling2-nvfp4"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Inkling 2 Mini"

    /// Thinking off, or one of the template's effort levels (`high`, 0.9, is
    /// its own default).
    public static let reasoning = ReasoningControl.thinkingWithEffortLevels(
        defaultEffort: .high, supported: [.low, .medium, .high, .xhigh])

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        InklingQ4Install.recognizes(install.directory) ? .exact : nil
    }

    /// K/V stays 16-bit: 8- and 4-bit K/V were built and measured on the
    /// llama.cpp reference cases (96 tokens) at 95/96 and 91/96 top-1 and
    /// rejected under the quality rule (TT-123).
    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport(
            kvCachePrecision: .only(
                [.fp16],
                reason: "its K/V cache stays 16-bit; 8-bit measured 95/96 and 4-bit 91/96 "
                    + "top-1 agreement with llama.cpp (TT-123)"),
            ropeScaling: .only(
                [.none], reason: "Inkling 2 has a 1M native window and no YaRN factor"),
            hostKVWindow: .refused(reason: "it needs a QSA model"),
            concurrentSequences: .supported,
            multiTokenPrediction: .refused(
                reason: "bundled DSpark MTP head is opt-in via TINYTITAN_INKLING_MTP=on; off by "
                    + "default because 8-token verify measured 3.404 vs 8.047 tok/s (0.423x) "
                    + "under SSD expert streaming (TT-129)"),
            expertStreaming: .supported)
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
        reasoning
    }

    /// lint:allow-long one factory literal: every stage is a short closure over
    /// the variant, and splitting them would only move the list elsewhere.
    public static func engineFactory(host: HostServices) -> EngineFactory {
        @Sendable func metadata(_ directory: URL) throws -> EngineModelMetadata {
            let manifest = try SSDAIManifestReader.read(directoryURL: directory)
            return EngineModelMetadata(
                modelID: manifest.modelID, family: "inkling2", weightBits: 4,
                sampling: .house, reasoningControl: reasoning)
        }
        return EngineFactory(
            prepareLoad: { options in
                let manifest = try SSDAIManifestReader.read(directoryURL: options.modelDirectory)
                let workingSet = Int(
                    options.context?.device.recommendedMaxWorkingSetSize
                        ?? MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
                let plan = try InklingQ4LoadPlan.make(
                    geometry: try InklingQ4LoadPlan.Geometry(directory: options.modelDirectory),
                    requested: options.maxContext, ramBudget: options.expertCacheBudgetBytes,
                    explicitSlots: options.expertCacheSlots, workingSet: workingSet,
                    maxContextExplicit: options.maxContextExplicit)
                host.log(
                    "Inkling 2 NVFP4: 16-bit K/V, \(plan.slots) expert slots per layer, context "
                        + "\(plan.context) tokens (asked \(options.maxContext)), planned "
                        + "\(plan.plannedBytes >> 20) MiB")
                return EngineLoadPlan(
                    architecture: InklingQ4EngineModel.architecture(manifest.arch),
                    expertCacheSlots: plan.slots,
                    tunedExpertCacheBudgetBytes: InklingQ4LoadPlan.defaultExpertBytes,
                    prefillConfiguration: .defaultChunked, fp16RingEnabled: false,
                    yarnContextTokens: 0, runtimeIdentity: "inkling2-nvfp4",
                    engineConfiguration: plan)
            },
            metadata: metadata,
            validateReasoning: { _, thinking, effort in
                guard let effort else { return }
                guard thinking.isEnabled else {
                    throw EngineFactoryError.unsupported(
                        "reasoning effort \(effort.rawValue) needs thinking on for Inkling 2")
                }
                guard case .thinkingWithEffortLevels(_, let supported) = reasoning,
                    supported.contains(effort)
                else {
                    throw EngineFactoryError.unsupported(
                        "Inkling 2 has no reasoning effort \(effort.rawValue)")
                }
            },
            resolveArchitecture: { directory in
                InklingQ4EngineModel.architecture(
                    try SSDAIManifestReader.read(directoryURL: directory).arch)
            },
            preview: { directory, requestedChunk in
                EngineSessionPreview(
                    modelID: try metadata(directory).modelID,
                    prefillChunkTokens: requestedChunk ?? 256)
            },
            buildModel: { directory, _, _ in
                try InklingQ4EngineModel(directory: directory, slots: 0)
            },
            buildModelWithSlots: { directory, _, _, slots in
                try InklingQ4EngineModel(directory: directory, slots: slots)
            },
            buildRunner: { model, options in
                guard let model = model as? InklingQ4EngineModel, let context = options.context,
                    let plan = options.engineConfiguration as? InklingQ4LoadPlan
                else {
                    throw MetalError.invalidState(
                        "the Inkling 2 Q4 runner needs its model, plan and Metal context")
                }
                let decoder = try model.makeDecoder(
                    context: context, maxContext: plan.context,
                    maxChunk: options.prefillChunkTokens ?? 256,
                    sequenceSlots: options.slots)
                return InklingQ4Runner(model: model, decoder: decoder)
            },
            makeScratch: { context, vocabSize, _ in
                RawCompletionScratch(
                    payload: try InklingQ4LogitsScratch(context: context, vocabSize: vocabSize))
            },
            runRawCompletion: { request in try await runInklingQ4Completion(request) },
            perSlotBytes: { _, _, loadPlan in
                (loadPlan.engineConfiguration as? InklingQ4LoadPlan)?.perSlotBytes ?? 0
            },
            tokenizerFolder: nil)
    }
}
