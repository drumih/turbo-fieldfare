import Foundation
import TinyTitan
import TinyTitanCore
import TinyTitanLib

/// unchecked-invariant: the lock protects configuration and the MTP decoder
/// while the sequential session loader moves through factory stages.
public final class EngineFactories: @unchecked Sendable {
    private let lock = NSLock()
    private var configuration: DefaultEngineConfiguration?
    private var decoder: StreamingMTPDecoder?

    public init() {}

    public static func sixXGPU() -> EngineFactory {
        let factories = EngineFactories()
        return EngineFactory(
            prepare: { options in factories.prepare(options) },
            makeContext: { device in try MetalContext.runtimeDefault(device: device) },
            prepareLoad: { options in try factories.prepareLoad(options) },
            metadata: { directory in try factories.metadata(directory) },
            validateReasoning: { metadata, thinking, effort in
                let family = ModelFamily(rawValue: metadata.family) ?? .qwen36
                try family.validateReasoning(thinkingMode: thinking, effort: effort)
            },
            resolveArchitecture: { directory in
                let family = try ManifestReader.peekFamily(directoryURL: directory)
                return try ArchConfig.resolved(forFamily: family, directoryURL: directory)
            },
            preview: { directory, requestedChunk in
                try factories.preview(directory, requestedChunk: requestedChunk)
            },
            readManifest: { directory, arch in
                try ManifestReader.load(directoryURL: directory, expecting: arch)
            },
            buildModel: { directory, arch, context in
                try factories.buildModel(
                    directory: directory, arch: arch, context: context)
            },
            buildRunner: { model, options in
                try factories.buildRunner(model: model, options: options)
            },
            mtpDecoder: { factories.mtpDecoder },
            makeScratch: { context, vocabulary, softcap in
                RawCompletionScratch(
                    payload: try TinyTitan.RawCompletionScratch(
                        context: context, vocab: vocabulary, logitSoftcap: softcap))
            },
            runRawCompletion: { request in try await factories.runRawCompletion(request) },
            makeWarmup: { model, requested in factories.makeWarmup(model, requested: requested) },
            perSlotBytes: { model, options, plan in
                LegacyBatchedMemoryBudget.perSlotBytes(
                    config: model.config, maxContext: options.maxContext,
                    precision: options.kvCachePrecision,
                    fp16RingEnabled: plan.fp16RingEnabled,
                    slidingWindow: model.config.slidingWindow,
                    maxPrefillChunkTokens: plan.prefillConfiguration.chunkTokens,
                    vocab: model.config.vocabSize,
                    hostKVWindowTokens: options.hostKVWindowTokens)
            })
    }

    private var mtpDecoder: StreamingMTPDecoder? {
        lock.withLock { decoder }
    }

    private func prepare(_ options: SessionBuildOptions) {
        guard let configuration = options.engineConfiguration as? DefaultEngineConfiguration
        else { return }
        lock.withLock { self.configuration = configuration }
    }

    private func buildModel(
        directory: URL, arch: ArchConfig, context: MetalContext?
    ) throws -> any EngineModel {
        let configuration = try configured()
        guard let context else {
            throw ServerInferenceError.unsupportedModel("6.x model load requires Metal")
        }
        return try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: arch,
            streamingMode: .pread(slotCount: configuration.loadSlots),
            expertCachePolicy: configuration.loadRuntime.modelExpertCachePolicy,
            expertIOTimeline: configuration.loadRuntime.expertIOTimeline,
            integrityPolicy: configuration.integrityPolicy)
    }

    private func buildRunner(
        model: any EngineModel, options: SessionBuildOptions
    ) throws -> any EngineRunner {
        let configuration = try configured()
        guard let model = model as? Model,
            let runtime = configuration.runtime,
            let context = options.context
        else {
            throw ServerInferenceError.unsupportedModel(
                "6.x runner requires its model, runtime, and Metal context")
        }
        if let mtpModelDirectory = options.mtpModelDirectory {
            let sidecarFamily = try ManifestReader.peekFamily(directoryURL: mtpModelDirectory)
            let sidecarArch: ArchConfig
            do {
                sidecarArch = try ArchConfig.resolved(
                    forFamily: sidecarFamily, directoryURL: mtpModelDirectory)
            } catch {
                throw ServerInferenceError.unsupportedModel(
                    "MTP sidecar declares family \(sidecarFamily.rawValue), "
                        + "which this runtime does not implement")
            }
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: sidecarArch,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                expertIOTimeline: runtime.expertIOTimeline,
                integrityPolicy: configuration.requestedIntegrityPolicy
                    ?? .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: options.maxContext,
                memoryBudgetMiB: options.mtpMemoryMiB,
                runtimeConfiguration: runtime)
            lock.withLock { self.decoder = decoder }
            return decoder.target
        }
        lock.withLock { decoder = nil }
        return try RealForwardRunner(
            model: model,
            context: context,
            maxContext: options.maxContext,
            slots: options.slots,
            runtimeConfiguration: runtime)
    }

    private func prepareLoad(_ options: SessionBuildOptions) throws -> EngineLoadPlan {
        let directory = options.modelDirectory
        let loadRuntime = try RuntimeConfiguration(
            forceLogitsHead: true,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue(),
            expertIOTimeline: try RuntimeExpertIOTimeline.environmentValue())
        let slotOverride = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init)
        let family = try ManifestReader.peekFamily(directoryURL: directory)
        let arch: ArchConfig
        do {
            arch = try ArchConfig.resolved(forFamily: family, directoryURL: directory)
        } catch {
            throw ServerInferenceError.unsupportedModel("\(error)")
        }
        let tunedBudget: Int
        if let identity = try? ManifestReader.peekIdentity(directoryURL: directory) {
            tunedBudget = RuntimeConfiguration.affordableExpertCacheBudget(
                ModelProfile.resolve(identity: identity).expertCacheBudgetBytes)
        } else {
            tunedBudget = RuntimeConfiguration.defaultExpertCacheBudgetBytes
        }
        let manifest = try? ManifestReader.load(directoryURL: directory, expecting: arch)
        let cachePerSlot =
            manifest.map {
                Double($0.expertStride) * Double($0.arch.numLayers)
            } ?? 0
        let residentFloor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: manifest?.files["model_weights.bin"].map { Int($0.size) } ?? 0)
        let appliedSlots = { options.expertCacheSlots ?? slotOverride ?? $0 }
        let derivedSlots: Int
        if let target = options.expertCacheBudgetBytes {
            derivedSlots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: manifest?.expertStride ?? 0,
                layers: manifest?.arch.numLayers ?? 0, cacheBytes: target - residentFloor)
            for line in ExpertCacheDiagnostics.targetLines(
                targetBytes: target, appliedSlots: appliedSlots(derivedSlots),
                derivedSlots: derivedSlots, cacheBytesPerSlot: cachePerSlot,
                residentFloorBytes: residentFloor)
            {
                ServerLog.diagnostic(line)
            }
        } else if let manifest {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride, layers: manifest.arch.numLayers,
                budgetBytes: tunedBudget)
            ServerLog.diagnostic(
                ExpertCacheDiagnostics.profileLine(
                    appliedSlots: appliedSlots(derivedSlots), derivedSlots: derivedSlots,
                    cacheBytesPerSlot: cachePerSlot, residentFloorBytes: residentFloor))
        } else {
            derivedSlots = RuntimeConfiguration.minimumExpertCacheSlots
        }
        let loadSlots = appliedSlots(derivedSlots)
        let integrity = options.integrityPolicy ?? .resolved(directoryURL: directory)
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let profile = ModelProfile.resolve(identity: identity)
        let fallback = defaultPrefillChunkTokens(
            family: family, fallback: loadRuntime.prefillChunkTokens)
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: options.rdAdvicePolicy
                ?? ProcessInfo.processInfo.environment["TINYTITAN_RDADVISE_POLICY"]
                .map(RDAdvicePolicyMode.parse)
                ?? loadRuntime.rdadvisePolicy,
            prefillChunkTokens: options.prefillChunkTokens ?? profile.prefillChunkTokens
                ?? fallback,
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: options.forceLogitsHead,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            expertIOTimeline: loadRuntime.expertIOTimeline,
            kvCachePrecision: options.kvCachePrecision,
            hostKVWindowTokens: options.hostKVWindowTokens,
            ropeScalingMode: options.ropeScalingMode,
            yarnContextTokens: options.ropeScalingMode == .yarn
                ? options.maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        let runtimeIdentity = [
            String(runtime.expertCacheSlots), runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue, runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens), runtime.headPath.rawValue,
            String(runtime.kvCachePrecision.rawValue), runtime.ropeScalingMode.rawValue,
            String(runtime.yarnContextTokens),
        ].joined(separator: ":")
        let configuration = DefaultEngineConfiguration(
            loadRuntime: loadRuntime, integrityPolicy: integrity,
            requestedIntegrityPolicy: options.integrityPolicy, loadSlots: loadSlots,
            runtime: runtime)
        prepare(
            SessionBuildOptions(
                modelDirectory: directory, context: options.context,
                maxContext: options.maxContext, slots: options.slots,
                promptCacheMaximumEntries: options.promptCacheMaximumEntries,
                promptCacheMemoryLimitBytes: options.promptCacheMemoryLimitBytes,
                promptCacheDiskDirectory: options.promptCacheDiskDirectory,
                promptCacheDiskLimitBytes: options.promptCacheDiskLimitBytes,
                prefillChunkTokens: options.prefillChunkTokens,
                kvCachePrecision: options.kvCachePrecision,
                hostKVWindowTokens: options.hostKVWindowTokens,
                ropeScalingMode: options.ropeScalingMode, thinkingMode: options.thinkingMode,
                reasoningEffort: options.reasoningEffort,
                expertCacheSlots: options.expertCacheSlots,
                expertCacheBudgetBytes: options.expertCacheBudgetBytes,
                mtpModelDirectory: options.mtpModelDirectory, mtpMemoryMiB: options.mtpMemoryMiB,
                integrityPolicy: options.integrityPolicy, rdAdvicePolicy: options.rdAdvicePolicy,
                forceLogitsHead: options.forceLogitsHead, engineConfiguration: configuration))
        return EngineLoadPlan(
            architecture: arch, expertCacheSlots: loadSlots,
            tunedExpertCacheBudgetBytes: tunedBudget,
            prefillConfiguration: runtime.prefillConfig, fp16RingEnabled: runtime.fp16RingEnabled,
            yarnContextTokens: runtime.yarnContextTokens, runtimeIdentity: runtimeIdentity,
            engineConfiguration: configuration)
    }

    private func metadata(_ directory: URL) throws -> EngineModelMetadata {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let family = identity.family
        return EngineModelMetadata(
            modelID: identity.modelID, family: family.rawValue,
            weightBits: identity.weightBits,
            sampling: family.engineSamplingProfile(identity: identity),
            reasoningControl: family.engineReasoningControl)
    }

    private func preview(_ directory: URL, requestedChunk: Int?) throws -> EngineSessionPreview {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let family = identity.family
        let chunk =
            requestedChunk
            ?? ModelProfile.resolve(identity: identity).prefillChunkTokens
            ?? (family == .qwen36
                ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                : RuntimeConfiguration.production.prefillChunkTokens)
        var id = identity.modelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        if id == "unknown/snapshot" {
            switch family {
            case .qwen36: id = "qwen3.6-35b-a3b"
            case .qwen36MTP: id = "qwen3.6-35b-a3b-mtp"
            case .qwen38flash: id = "qwen3.8-flash-next"
            case .qwen38flashMTP: id = "qwen3.8-flash-next-mtp"
            case .qwen4flash: id = "qwen4-omni-120b-think"
            case .qwen4flashMTP: id = "qwen4-omni-120b-think-mtp"
            case .qwen35Dense: id = "qwen3.5-dense"
            case .k2Horizon: id = "k2-horizon"
            case .laguna: id = "laguna-s-2-1"
            }
        }
        return EngineSessionPreview(
            modelID: id + "_\(identity.weightBits)-Bit", prefillChunkTokens: chunk)
    }

    private func runRawCompletion(
        _ request: RawCompletionRequest
    ) async throws -> TinyTitanCore.RawDecodeResult {
        guard let scratch = request.scratch.payload as? TinyTitan.RawCompletionScratch else {
            throw ServerInferenceError.unsupportedModel(
                "6.x completion requires its scratch buffers")
        }
        let options = request.wideVerify as? TinyTitanCore.WideVerifyDecodeOptions
        let result = try await TinyTitan.runRawCompletion(
            producer: request.producer,
            tokenizer: request.tokenizer,
            promptIds: request.promptIDs,
            config: request.configuration,
            context: request.context,
            scratch: scratch,
            prefillConfig: request.prefillConfiguration,
            start: request.start,
            slot: request.slot,
            checkpointPositions: request.checkpointPositions,
            onPrefillCheckpoint: request.onPrefillCheckpoint,
            wideVerify: options,
            shouldStop: request.shouldStop,
            onProgress: { progress in
                switch progress {
                case .prefill(let done, let total):
                    request.onProgress(.prefill(done: done, total: total))
                case .token(let index, let id, let delta):
                    request.onProgress(.token(index: index, id: id, delta: delta))
                case .tail(let text): request.onProgress(.tail(text))
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
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            wideVerify: result.wideVerify.map {
                var stats = WideVerifyDecodeStats()
                stats.maxRows = $0.maxRows
                stats.passes = $0.passes
                stats.scalarSteps = $0.scalarSteps
                stats.draftedTokens = $0.draftedTokens
                stats.acceptedDrafts = $0.acceptedDrafts
                stats.rejectingPasses = $0.rejectingPasses
                stats.emittedTokens = $0.emittedTokens
                stats.passesCutAtStop = $0.passesCutAtStop
                stats.passNanos = $0.passNanos
                return stats
            })
    }

    private func configured() throws -> DefaultEngineConfiguration {
        try lock.withLock {
            guard let configuration else {
                throw ServerInferenceError.unsupportedModel("6.x factory is not configured")
            }
            return configuration
        }
    }
}
