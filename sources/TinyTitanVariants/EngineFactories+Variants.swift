import Foundation
import DenseGPU
import Ornith35B4Bit
import Qwen35Dense2B4Bit
import Qwen35Dense2B8Bit
import Qwen35Dense4B4Bit
import Qwen35Dense4B8Bit
import Qwen35Dense9B4Bit
import Qwen35Dense9B8Bit
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib
import TinyTitanPluginAPI

/// The engine a caller asks for explicitly.
public enum EngineChoice: String, Sendable, CaseIterable, Equatable {
    case cpu
    case gpu
}

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
        case qwen35DenseGPU
    }

    /// The selection for a GPU catalog identity, decided from the manifest
    /// identity the entry's directory carries: the shipped Ornith 1.5 35B-A3B
    /// install at 4 bits routes to its variant, every dense Qwen 3.5 install
    /// routes to the shared dense-GPU engine, and everything else stays on
    /// the 6.x engine.
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
        if family == TinyTitan.ModelFamily.qwen35Dense {
            return .qwen35DenseGPU
        }
        return .legacy
    }

    /// The GPU factory a catalog identity selects. The router reads the
    /// install manifest identity (the same bounded read its plan performs)
    /// and dispatches through the selection table: the shipped variant
    /// engines for their installs, the 6.x engine for everything else —
    /// which keeps serving every model without a shipped variant (TT-087).
    ///
    /// Resolved through the plugin registry (TPP/1, TT-130): the strongest
    /// plugin probe wins. An install no plugin claims specifically and that has
    /// no readable manifest is refused here, as before the registry.
    public static func gpuFactory(directoryURL: URL) throws -> EngineFactory {
        let resolved = try pluginRegistry.resolve(directoryURL)
        if resolved.strength == .fallback {
            _ = try TinyTitan.ManifestReader.peekIdentity(directoryURL: directoryURL)
        }
        return resolved.engineFactory()
    }

    /// The CPU factory for a dense Qwen 3.5 install: the per-width variant
    /// engine the server serves, built from the install's own hidden size and
    /// quant width. A family the CPU engine does not implement is refused by
    /// name, before the model loads.
    public static func routingCPUFactory(directoryURL: URL) throws -> CPUEngineFactory {
        guard
            let identity = try? TinyTitan.ManifestReader.peekIdentity(
                directoryURL: directoryURL)
        else {
            throw EngineFactoryError.unsupported(
                "--engine cpu needs an .ssdai install with a readable manifest; "
                    + "\(directoryURL.path) has none")
        }
        guard identity.family == .qwen35Dense else {
            throw EngineFactoryError.unsupported(
                "\(identity.modelID) declares family \(identity.family.rawValue); "
                    + "the CPU engine implements the dense Qwen 3.5 family only, so "
                    + "this install is GPU-only")
        }
        let arch = try sixXGPU().resolveArchitecture(directoryURL)
        return try variantCPUEngine(
            hiddenSize: arch.hiddenSize,
            weightBits: identity.weightBits,
            snapshotDirectory: directoryURL,
            isSSDAI: true)
    }

    /// The GPU factory a single-model front end (the CLI) and an embedder
    /// use: the same selection table the server's router uses, keyed on the
    /// install's manifest identity. A directory without a readable manifest
    /// (an affine snapshot, a GGUF directory) falls back to the 6.x factory,
    /// which is what those paths used before this surface existed.
    public static func routingFactory(directoryURL: URL) -> EngineFactory {
        (try? pluginRegistry.resolve(directoryURL))?.engineFactory() ?? sixXGPU()
    }

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
        case .ornith35B4Bit: return Ornith35B4BitPlugin.engineFactory(host: .standard)
        case .qwen35DenseGPU: return variantDenseGPU()
        case .legacy: return sixXGPU()
        }
    }

    /// The shared dense-GPU engine as the server's engine: one factory for
    /// every dense Qwen 3.5 install, geometry taken from each install's
    /// manifest by the engine's own dense-only load path (`Model.load`
    /// refuses non-dense families and resolves the width's architecture from
    /// its manifest). The planner, metadata and preview stages are the
    /// family-generic production ones shared with the 6.x factory; the model
    /// and runner stages close over the DenseGPU target's own copies, and the
    /// scratch/completion pair names the dense engine's own generation loop —
    /// the same engine-generic-through-the-producer shape the ornith factory
    /// uses.
    public static func variantDenseGPU() -> EngineFactory {
        let base = sixXGPU()
        return EngineFactory(
            prepare: base.prepare,
            makeContext: base.makeContext,
            prepareLoad: base.prepareLoad,
            metadata: base.metadata,
            validateReasoning: base.validateReasoning,
            resolveArchitecture: base.resolveArchitecture,
            preview: base.preview,
            defaultMaxContextTokens: 1_048_576,
            buildModel: { directory, _, context in
                guard let context else {
                    throw ServerInferenceError.unsupportedModel(
                        "the dense-GPU engine requires Metal")
                }
                // Geometry comes from the install manifest (the engine's
                // dense-only load path resolves it); no expert streaming is
                // configured because a dense install has no routed experts.
                return try DenseGPU.Model.load(
                    directoryURL: directory,
                    device: context.device)
            },
            buildRunner: { model, options in
                guard let model = model as? DenseGPU.Model,
                    let context = options.context
                else {
                    throw ServerInferenceError.unsupportedModel(
                        "the dense-GPU runner requires its model and Metal context")
                }
                // The dense GPU engine owns the thresholded YaRN kernels. The
                // extension profile is fixed at 1M independent of the cap, so
                // --max-context 4M/8M/16M only changes capacity, not the rope
                // function. With YaRN off the profile is unused.
                let runtime = try DenseGPU.RuntimeConfiguration(
                    forceLogitsHead: true,
                    kvCachePrecision: options.kvCachePrecision,
                    ropeScalingMode: options.ropeScalingMode,
                    yarnContextTokens: DenseGPU.RuntimeConfiguration
                        .defaultYaRNContextTokens)
                return try DenseGPU.RealForwardRunner(
                    model: model,
                    context: context,
                    maxContext: options.maxContext,
                    slots: options.slots,
                    runtimeConfiguration: runtime)
            },
            makeScratch: { context, vocabulary, softcap in
                RawCompletionScratch(
                    payload: try DenseGPU.RawCompletionScratch(
                        context: context, vocab: vocabulary, logitSoftcap: softcap))
            },
            runRawCompletion: { request in
                guard
                    let scratch = request.scratch.payload
                        as? DenseGPU.RawCompletionScratch
                else {
                    throw ServerInferenceError.unsupportedModel(
                        "the dense-GPU completion requires its scratch buffers")
                }
                let result = try await DenseGPU.runRawCompletion(
                    producer: request.producer,
                    tokenizer: request.tokenizer,
                    promptIds: request.promptIDs,
                    config: request.configuration,
                    context: request.context,
                    scratch: scratch,
                    prefillConfig: request.prefillConfiguration,
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
        case qwen35Dense2B4Bit
        case qwen35Dense2B8Bit
        case qwen35Dense4B4Bit
        case qwen35Dense4B8Bit
        case qwen35Dense9B4Bit
        case qwen35Dense9B8Bit
    }

    /// The dense CPU target a served snapshot selects: hidden size names the
    /// model, the width its quant twin. An unknown combination is refused by
    /// name rather than run through a sibling's statics.
    public static func variantCPUEngineSelection(
        hiddenSize: Int,
        weightBits: Int
    ) throws -> DenseCPUVariantSelection {
        switch (hiddenSize, weightBits) {
        case (2_048, 4): return .qwen35Dense2B4Bit
        case (2_048, 8): return .qwen35Dense2B8Bit
        case (2_560, 4): return .qwen35Dense4B4Bit
        case (2_560, 8): return .qwen35Dense4B8Bit
        case (4_096, 4): return .qwen35Dense9B4Bit
        case (4_096, 8): return .qwen35Dense9B8Bit
        default:
            throw ServerInferenceError.unsupportedModel(
                "no dense CPU variant ships for hiddenSize \(hiddenSize) at "
                    + "\(weightBits) bits; supported sizes: 2048, 2560, 4096 at 4 or 8 bits")
        }
    }

    /// The dense CPU variant for a snapshot's size and width. Each dense
    /// model owns its dedicated code path (the 7.0 directive): the hidden
    /// size decides the model (2B 2048 / 4B 2560 / 9B 4096) and the width the
    /// quant twin. The variant's engine consumes its own module's snapshot
    /// type, so the closure builds it from the install directory — the 6.x
    /// snapshot the backend holds is a different module's class and cannot
    /// cross. An unknown size/width combination is refused by name.
    public static func variantCPUEngine(
        hiddenSize: Int,
        weightBits: Int,
        snapshotDirectory: URL,
        isSSDAI: Bool
    ) throws -> CPUEngineFactory {
        func engine(_ build: @Sendable @escaping () throws -> any CPUEngineRunner)
            -> CPUEngineFactory
        {
            CPUEngineFactory(build: build)
        }
        switch try variantCPUEngineSelection(hiddenSize: hiddenSize, weightBits: weightBits) {
        case .qwen35Dense2B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense2B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense2B4Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense2B4Bit.CPUQwen35(snapshot: snapshot)
            }
        case .qwen35Dense2B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense2B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense2B8Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense2B8Bit.CPUQwen35(snapshot: snapshot)
            }
        case .qwen35Dense4B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense4B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense4B4Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense4B4Bit.CPUQwen35(snapshot: snapshot)
            }
        case .qwen35Dense4B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense4B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense4B8Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense4B8Bit.CPUQwen35(snapshot: snapshot)
            }
        case .qwen35Dense9B4Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense9B4Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense9B4Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense9B4Bit.CPUQwen35(snapshot: snapshot)
            }
        case .qwen35Dense9B8Bit:
            return engine {
                let snapshot =
                    isSSDAI
                    ? try Qwen35Dense9B8Bit.AffineSnapshot(ssdai: snapshotDirectory)
                    : try Qwen35Dense9B8Bit.AffineSnapshot(directory: snapshotDirectory)
                return try Qwen35Dense9B8Bit.CPUQwen35(snapshot: snapshot)
            }
        }
    }
}

/// Load the Ornith variant model with the resolved slot count.
///
/// A nil `slots` is the legacy `buildModel` path: the env override, then the
/// variant's own 32-slot default. The session's resolved count (profile,
/// `--ram-budget`, `--expert-cache-slots`) arrives through
/// `buildModelWithSlots`. The env stays the operator's last word.
