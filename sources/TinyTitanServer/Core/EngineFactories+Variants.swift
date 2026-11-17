import Foundation
import Ornith35B4Bit
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib

extension EngineFactories {

    // MARK: - TT-097 Phase 3 Task 10: the shipped variants as server engines

    /// Which GPU engine a catalog identity selects. The table is explicit and
    /// small: a shipped variant joins it when it qualifies (registry
    /// `shipped: true`, `catalog --check` green), and every identity that
    /// misses falls back to the 6.x engine — which stays the serving engine
    /// for every model without a shipped variant (TT-087). `Equatable` so a
    /// test can pin the selection without running a model.
    public enum GPUVariantSelection: Equatable {
        case legacy
        case ornith35B4Bit
    }

    /// The selection for a GPU catalog identity, decided from the manifest
    /// identity the entry's directory carries: the shipped Ornith 1.5 35B-A3B
    /// install at 4 bits is the one variant the registry ships today.
    public static func gpuSelection(
        family: TinyTitan.ModelFamily,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) -> GPUVariantSelection {
        if family == TinyTitan.ModelFamily.qwen36,
            manifestModelID.hasPrefix("ornith-1.5-35b"),
            routedExpertWeightBits == 4
        {
            return .ornith35B4Bit
        }
        return .legacy
    }

    /// The GPU factory a catalog identity selects. The router reads the
    /// install manifest identity (the same bounded read its plan performs)
    /// and dispatches through the selection table: the shipped variant
    /// engines for their installs, the 6.x engine for everything else —
    /// which keeps serving every model without a shipped variant (TT-087).
    public static func gpuFactory(
        family: TinyTitan.ModelFamily,
        manifestModelID: String,
        routedExpertWeightBits: Int
    ) -> EngineFactory {
        switch gpuSelection(
            family: family,
            manifestModelID: manifestModelID,
            routedExpertWeightBits: routedExpertWeightBits)
        {
        case .ornith35B4Bit: return variantGPUOrnith()
        case .legacy: return sixXGPU()
        }
    }

    /// The Ornith 1.5 35B-A3B 4-bit variant as the server's GPU engine. The
    /// planner, metadata, preview and architecture stages are the
    /// family-generic production ones shared with the 6.x factory; the model
    /// and runner stages close over the variant target's own copies, which
    /// conform to the same Core contracts (`EngineModel`/`EngineRunner`) the
    /// shared session consumes. The variant's runtime configuration is built
    /// here — engine-specific configuration stays inside the factory
    /// (`SessionBuildOptions` deliberately does not expose a runtime module).
    public static func variantGPUOrnith() -> EngineFactory {
        let base = sixXGPU()
        return EngineFactory(
            prepare: base.prepare,
            makeContext: base.makeContext,
            prepareLoad: base.prepareLoad,
            metadata: base.metadata,
            validateReasoning: base.validateReasoning,
            resolveArchitecture: base.resolveArchitecture,
            preview: base.preview,
            buildModel: { directory, arch, context in
                guard let context else {
                    throw ServerInferenceError.unsupportedModel(
                        "the ornith variant engine requires Metal")
                }
                // The variant's own streaming default; the env override the
                // 6.x planner honours is read here too so an operator tuning
                // residency gets the same knob.
                let slots =
                    ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
                    .flatMap(Int.init) ?? Ornith35B4BitModelDefaults.streamingSlots
                return try Ornith35B4Bit.Model.load(
                    directoryURL: directory,
                    device: context.device,
                    expecting: arch,
                    streamingMode: .pread(slotCount: slots))
            },
            buildRunner: { model, options in
                guard let model = model as? Ornith35B4Bit.Model,
                    let context = options.context
                else {
                    throw ServerInferenceError.unsupportedModel(
                        "the ornith variant runner requires its model and Metal context")
                }
                let runtime = try Ornith35B4Bit.RuntimeConfiguration(
                    forceLogitsHead: true,
                    kvCachePrecision: options.kvCachePrecision)
                return try Ornith35B4Bit.RealForwardRunner(
                    model: model,
                    context: context,
                    maxContext: options.maxContext,
                    slots: options.slots,
                    runtimeConfiguration: runtime)
            },
            // The variant's own scratch buffers and generation loop: the Core
            // contract erases them, and the loop itself is engine-generic
            // through the session's `any LogitProducer` — the variant
            // difference is only which concrete scratch and loop the closures
            // name.
            makeScratch: { context, vocabulary, softcap in
                RawCompletionScratch(
                    payload: try Ornith35B4Bit.RawCompletionScratch(
                        context: context, vocab: vocabulary, logitSoftcap: softcap))
            },
            runRawCompletion: { request in
                guard
                    let scratch = request.scratch.payload
                        as? Ornith35B4Bit.RawCompletionScratch
                else {
                    throw ServerInferenceError.unsupportedModel(
                        "the ornith variant completion requires its scratch buffers")
                }
                let result = try await Ornith35B4Bit.runRawCompletion(
                    producer: request.producer,
                    tokenizer: request.tokenizer,
                    promptIds: request.promptIDs,
                    config: request.configuration,
                    context: request.context,
                    scratch: scratch,
                    prefillConfig: request.prefillConfiguration,
                    // The variant's start enum is its own copy of the promoted
                    // shape; map the two cases across the eraser.
                    start: {
                        switch request.start {
                        case .reset: return .reset
                        case .resume(let cached): return .resume(cachedPromptTokens: cached)
                        }
                    }(),
                    slot: request.slot,
                    checkpointPositions: request.checkpointPositions,
                    onPrefillCheckpoint: request.onPrefillCheckpoint,
                    shouldStop: request.shouldStop,
                    onProgress: { progress in
                        switch progress {
                        case .prefill(let done, let total):
                            request.onProgress(.prefill(done: done, total: total))
                        case .token(let index, let id, let delta):
                            request.onProgress(.token(index: index, id: id, delta: delta))
                        case .tail(let text):
                            request.onProgress(.tail(text))
                        }
                    })
                return RawDecodeResult(
                    prefillTokens: result.prefillTokens,
                    cachedPromptTokens: result.cachedPromptTokens,
                    computedPrefillTokens: result.computedPrefillTokens,
                    prefillSeconds: result.prefillSeconds,
                    newTokens: result.newTokens,
                    decodeSeconds: result.decodeSeconds,
                    reason: result.reason,
                    kvPosition: result.kvPosition,
                    kvBackedTokenIDs: result.kvBackedTokenIDs,
                    uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs)
            })
    }

    // MARK: - the dense CPU variants

    /// Which dense variant target a CPU install's width selects. Each dense
    /// model owns its dedicated code path (the 7.0 directive), so the width —
    /// not a shared engine — decides the target. 6 bits is the withdrawn
    /// legacy format and is refused by name, like the registry does.
    public enum DenseCPUVariantSelection: Equatable {
        case qwen35Dense4Bit
        case qwen35Dense8Bit
    }

    public static func variantCPUEngineSelection(weightBits: Int) throws
        -> DenseCPUVariantSelection
    {
        switch weightBits {
        case 4: return .qwen35Dense4Bit
        case 8: return .qwen35Dense8Bit
        default:
            throw ServerInferenceError.unsupportedModel(
                "no dense CPU variant ships at \(weightBits) bits; supported: 4, 8")
        }
    }
}

/// The variant engine's own defaults, kept beside the factory that uses them.
/// `Ornith35B4Bit.ExpertStreamingMode`'s 32-slot default is the value the
/// variant's parity was measured at.
private enum Ornith35B4BitModelDefaults {
    static let streamingSlots = 32
}
