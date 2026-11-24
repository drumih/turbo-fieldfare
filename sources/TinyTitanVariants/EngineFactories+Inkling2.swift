import Foundation
import Inkling2GGUFQ4
import Metal
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib

extension EngineFactories {
    /// Inkling 2's reasoning control: thinking off, or one of its template's
    /// effort levels (`InklingChatTemplate.effortValue`), `high` (0.9) being
    /// the template's own default.
    public static let inkling2ReasoningControl = ReasoningControl.thinkingWithEffortLevels(
        defaultEffort: .high, supported: [.low, .medium, .high, .xhigh])

    /// Why Inkling 2 keeps its K/V cache at 16 bits (TT-123). Quantized K/V
    /// was built and measured on the llama.cpp reference cases (96 tokens,
    /// teacher-forced): 8 bits agreed on 95 and 4 bits on 91 with one scale per
    /// 32 channels, 94 and 90 with one per head. A memory saving that changes a
    /// token is rejected, so 8 and 4 bits are not offered: an explicit request is
    /// refused (`inkling2RefusesKVBits`) and a default one is served at 16 bits
    /// with this notice.
    public static func inkling2KVNotice(_ requested: KVCachePrecision) -> String {
        "Inkling 2 Q4: --kv-bits \(requested.rawValue) does not apply; K/V stays 16-bit "
            + "(8-bit measured 95/96 and 4-bit 91/96 top-1 against llama.cpp, TT-123)"
    }

    /// The refusal for an explicit `--kv-bits 8` or `4` on an Inkling 2 install.
    public static func inkling2RefusesKVBits(_ requested: KVCachePrecision) -> String? {
        requested == .fp16
            ? nil
            : "--kv-bits \(requested.rawValue) is not supported for Inkling 2: its K/V cache "
                + "stays 16-bit because 8-bit measured 95/96 and 4-bit 91/96 top-1 agreement "
                + "with llama.cpp (TT-123); drop the flag or pass --kv-bits 16"
    }

    /// Whether `directoryURL` is an Inkling 2 Q4 `.ssdai` install.
    public static func isInkling2Q4(directoryURL: URL) -> Bool {
        InklingQ4Install.recognizes(directoryURL)
    }

    /// The Inkling 2 Q4 install as a shared-session engine (TT-122): the
    /// variant's decoder behind `ServerModelSession`, so the prompt cache, the
    /// library and CLI paths and every API surface apply to it as they do to
    /// the other models.
    ///
    /// The load plan is the variant's own (`InklingQ4LoadPlan`): `--ram-budget`
    /// is the whole-process target (`expertCacheBudgetBytes`), sized into a
    /// window and expert slots; `--expert-cache-slots` wins over the budget's
    /// slot count. The tokenizer is derived from the install's verified GGUF
    /// header into the user's cache directory, as the Inkling route did.
    /// lint:allow-long one factory literal: every stage is a short closure over
    /// the variant, and splitting them would only move the list elsewhere.
    public static func variantGPUInkling2Q4() -> EngineFactory {
        @Sendable func metadata(_ directory: URL) throws -> EngineModelMetadata {
            let manifest = try SSDAIManifestReader.read(directoryURL: directory)
            return EngineModelMetadata(
                modelID: manifest.modelID, family: "inkling2", weightBits: 4,
                sampling: .house, reasoningControl: inkling2ReasoningControl)
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
                    explicitSlots: options.expertCacheSlots, workingSet: workingSet)
                if options.kvCachePrecision != .fp16 {
                    ServerLog.diagnostic(inkling2KVNotice(options.kvCachePrecision))
                }
                ServerLog.diagnostic(
                    "Inkling 2 Q4: 16-bit K/V, \(plan.slots) expert slots per layer, context "
                        + "\(plan.context) tokens (asked \(options.maxContext)), planned "
                        + "\(plan.plannedBytes >> 20) MiB")
                return EngineLoadPlan(
                    architecture: InklingQ4EngineModel.architecture(manifest.arch),
                    expertCacheSlots: plan.slots,
                    tunedExpertCacheBudgetBytes: InklingQ4LoadPlan.defaultExpertBytes,
                    prefillConfiguration: .defaultChunked, fp16RingEnabled: false,
                    yarnContextTokens: 0, runtimeIdentity: "inkling2-gguf-q4",
                    engineConfiguration: plan)
            },
            metadata: metadata,
            validateReasoning: { _, thinking, effort in
                guard let effort else { return }
                guard thinking.isEnabled else {
                    throw EngineFactoryError.unsupported(
                        "reasoning effort \(effort.rawValue) needs thinking on for Inkling 2")
                }
                guard case .thinkingWithEffortLevels(_, let supported) = inkling2ReasoningControl,
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
                    throw ServerInferenceError.unsupportedModel(
                        "the Inkling 2 Q4 runner needs its model, plan and Metal context")
                }
                let decoder = try model.makeDecoder(
                    context: context, maxContext: plan.context,
                    maxChunk: options.prefillChunkTokens ?? 256)
                return InklingQ4Runner(model: model, decoder: decoder)
            },
            runRawCompletion: { request in try await runInklingQ4Completion(request) },
            // One sequence until per-slot state exists (TT-124): a cost no
            // budget can meet keeps the session's width at 1.
            perSlotBytes: { _, options, _ in options.slots > 1 ? Int.max / 2 : 0 },
            tokenizerFolder: { directory in try inkling2Q4TokenizerFolder(directory) })
    }

    /// The install's tokenizer as a folder the session can load: exported once
    /// from the verified GGUF header (`tokenizer.json`, its config and the chat
    /// template) into the user's cache directory, keyed by the install's
    /// identity and the header's size so a different install never reuses it.
    static func inkling2Q4TokenizerFolder(_ directory: URL) throws -> URL {
        let header = directory.appendingPathComponent(InklingQ4Install.ggufHeaderFile)
        let manifest = try SSDAIManifestReader.read(directoryURL: directory)
        let size =
            (try? FileManager.default.attributesOfItem(atPath: header.path)[.size] as? Int) ?? 0
        let snapshot = (manifest.sourceSnapshotHash ?? "unpinned")
            .replacingOccurrences(of: "sha256:", with: "").prefix(16)
        let caches =
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let folder =
            caches
            .appendingPathComponent("TinyTitan/gguf-tokenizers", isDirectory: true)
            .appendingPathComponent("\(manifest.modelID)-\(snapshot)-\(size)", isDirectory: true)
        let files = [
            GGUFTokenizerSidecar.tokenizerFileName, GGUFTokenizerSidecar.templateFileName,
        ]
        if !files.allSatisfy({
            FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
        }) {
            try GGUFTokenizerSidecar.write(
                from: try GGUFModelReader.readHeader(at: header), to: folder)
        }
        return folder
    }
}
