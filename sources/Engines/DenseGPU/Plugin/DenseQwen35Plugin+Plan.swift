import Foundation
import TinyTitanCore
import TinyTitanPluginAPI

extension DenseQwen35Plugin {
    /// The plugin's own load plan: the 6.x planner's arithmetic on this
    /// target's copies of the runtime configuration, manifest reader and model
    /// profile, so the plan (slots, tuned budget, prefill chunk, runtime
    /// identity) is the one the session received before the plugin existed.
    /// lint:allow-long the planner is one derivation: architecture, slots from
    /// the budget or the profile, then the runtime it implies.
    static func plan(_ options: SessionBuildOptions, host: HostServices) throws -> EngineLoadPlan {
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
        let arch = try ArchConfig.resolved(forFamily: family, directoryURL: directory)
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let profile = ModelProfile.resolve(identity: identity)
        let tunedBudget = RuntimeConfiguration.affordableExpertCacheBudget(
            profile.expertCacheBudgetBytes)
        let manifest = try? ManifestReader.load(directoryURL: directory, expecting: arch)
        let cachePerSlot =
            manifest.map { Double($0.expertStride) * Double($0.arch.numLayers) } ?? 0
        let residentFloor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: manifest?.files["model_weights.bin"].map { Int($0.size) } ?? 0)
        let appliedSlots = { options.expertCacheSlots ?? slotOverride ?? $0 }
        let derivedSlots: Int
        if let target = options.expertCacheBudgetBytes {
            derivedSlots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: manifest?.expertStride ?? 0,
                layers: manifest?.arch.numLayers ?? 0, cacheBytes: target - residentFloor)
            for line in DenseExpertCacheDiagnostics.targetLines(
                targetBytes: target, appliedSlots: appliedSlots(derivedSlots),
                derivedSlots: derivedSlots, cacheBytesPerSlot: cachePerSlot,
                residentFloorBytes: residentFloor)
            {
                host.log(line)
            }
        } else if let manifest {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride, layers: manifest.arch.numLayers,
                budgetBytes: tunedBudget)
            host.log(
                DenseExpertCacheDiagnostics.profileLine(
                    appliedSlots: appliedSlots(derivedSlots), derivedSlots: derivedSlots,
                    cacheBytesPerSlot: cachePerSlot, residentFloorBytes: residentFloor))
        } else {
            derivedSlots = RuntimeConfiguration.minimumExpertCacheSlots
        }
        let loadSlots = appliedSlots(derivedSlots)
        // This target's own copy of the policy enum; the session's maps across
        // by its raw value.
        let requestedAdvice = options.rdAdvicePolicy.flatMap {
            RDAdvicePolicyMode(rawValue: $0.rawValue)
        }
        let environmentAdvice = ProcessInfo.processInfo.environment["TINYTITAN_RDADVISE_POLICY"]
            .map(RDAdvicePolicyMode.parse)
        let rdAdvice = requestedAdvice ?? environmentAdvice ?? loadRuntime.rdadvisePolicy
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: rdAdvice,
            prefillChunkTokens: options.prefillChunkTokens ?? profile.prefillChunkTokens
                ?? RuntimeConfiguration.qwenLongPrefillChunkTokens,
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
        var identityParts: [String] = []
        identityParts.append(String(runtime.expertCacheSlots))
        identityParts.append(runtime.expertCachePolicy.rawValue)
        identityParts.append(runtime.rdadvisePolicy.rawValue)
        identityParts.append(runtime.prefillPolicy.rawValue)
        identityParts.append(String(runtime.prefillChunkTokens))
        identityParts.append(runtime.headPath.rawValue)
        identityParts.append(String(runtime.kvCachePrecision.rawValue))
        identityParts.append(runtime.ropeScalingMode.rawValue)
        identityParts.append(String(runtime.yarnContextTokens))
        let runtimeIdentity = identityParts.joined(separator: ":")
        return EngineLoadPlan(
            architecture: arch, expertCacheSlots: loadSlots,
            tunedExpertCacheBudgetBytes: tunedBudget,
            prefillConfiguration: runtime.prefillConfig, fp16RingEnabled: runtime.fp16RingEnabled,
            yarnContextTokens: runtime.yarnContextTokens, runtimeIdentity: runtimeIdentity,
            engineConfiguration: runtime)
    }

    /// The variant's generation loop behind the session's completion stage.
    static func runCompletion(_ request: TinyTitanCore.RawCompletionRequest) async throws
        -> TinyTitanCore.RawDecodeResult
    {
        guard let scratch = request.scratch.payload as? RawCompletionScratch else {
            throw MetalError.invalidState("the dense completion requires its scratch buffers")
        }
        let result = try await runRawCompletion(
            producer: request.producer, tokenizer: request.tokenizer,
            promptIds: request.promptIDs, config: request.configuration,
            context: request.context, scratch: scratch,
            prefillConfig: request.prefillConfiguration,
            // The variant's start enum is its own copy of the promoted shape.
            start: {
                switch request.start {
                case .reset: return .reset
                case .resume(let cached): return .resume(cachedPromptTokens: cached)
                }
            }(),
            slot: request.slot, checkpointPositions: request.checkpointPositions,
            onPrefillCheckpoint: request.onPrefillCheckpoint, shouldStop: request.shouldStop,
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
        return TinyTitanCore.RawDecodeResult(
            prefillTokens: result.prefillTokens, cachedPromptTokens: result.cachedPromptTokens,
            computedPrefillTokens: result.computedPrefillTokens,
            prefillSeconds: result.prefillSeconds, newTokens: result.newTokens,
            decodeSeconds: result.decodeSeconds, reason: result.reason,
            kvPosition: result.kvPosition, kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs)
    }
}
