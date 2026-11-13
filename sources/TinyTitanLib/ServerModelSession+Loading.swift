// The `load` factory: verification, tokenizer, scratch and cache setup.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import Metal
import TinyTitan

extension ServerModelSession {
    /// Interior checkpoints are on by default; the A/B benchmark can turn
    /// them off without a second binary. They are only useful when a later
    /// request edits the last turn, and their capture cost is real, so the
    /// switch is part of the measured surface.
    package static func checkpointsEnabled(environment: [String: String]) -> Bool {
        environment["TINYTITAN_PROMPT_CHECKPOINTS"] != "0"
    }

    /// Both stores live under one disk root with independent eviction, so
    /// when both exist they split the configured budget between them:
    /// together they stay inside the operator's limit rather than doubling
    /// it. A lone store keeps the whole budget.
    package static func promptCachePerStoreDiskLimit(
        total: Int, checkpointsEnabled: Bool
    ) -> Int {
        checkpointsEnabled ? total / 2 : total
    }

    /// A previous run with checkpoints enabled leaves their records in the
    /// directory the turn store also indexes. With the feature off nothing
    /// would charge or evict them, so they would sit beside a full-budget turn
    /// store; delete what can be deleted and return the bytes that refused,
    /// for the caller to keep inside the shared limit.
    package static func reclaimStaleCheckpoints(
        directory: URL?, totalDiskLimitBytes: Int
    ) async throws -> Int {
        guard let directory else { return 0 }
        let stale = try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 0, diskDirectory: directory,
                diskLimitBytes: 0))
        await stale.removeAll()
        return min(await stale.diskBytes, totalDiskLimitBytes)
    }

    /// lint:allow-long a sequential construction pipeline: tokenizer, Metal
    /// context, runtime config, model, optional MTP sidecar, runner, scratch.
    /// Each step consumes the last, so extracting any of them would return a
    /// tuple straight back into the next -- the same shape as Model.load.
    package static func load(
        modelDirectory: URL,
        maxContext: Int,
        slots: Int = 1,
        promptCacheMode: ServerPromptCacheMode = .multiPrefix,
        promptCacheMaximumEntries: Int = 4,
        promptCacheMemoryLimitBytes: Int = 256 * 1_048_576,
        promptCacheDiskDirectory: URL? = nil,
        promptCacheDiskLimitBytes: Int = 8_192 * 1_048_576,
        prefillChunkTokens requestedPrefillChunkTokens: Int? = nil,
        kvCachePrecision: KVCachePrecision = .int8,
        hostKVWindowTokens: Int? = nil,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        integrityPolicy requestedIntegrityPolicy: ModelIntegrityPolicy? = nil,
        expertCacheSlots requestedExpertCacheSlots: Int? = nil,
        expertCacheBudgetBytes: Int? = nil,
        rdadvisePolicy requestedRDAdvisePolicy: RDAdvicePolicyMode? = nil,
        forceLogitsHead: Bool = true,
        mtpModelDirectory: URL? = nil,
        mtpMemoryMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
        device: MTLDevice? = nil,
        reusingContext: MetalContext? = nil,
        warmExperts: Bool = true
    ) async throws -> ServerModelSession {
        _ = try GFTokenizer.requireModelDirectory(modelDirectory)
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        // Reasoning effort is defined per family; reject it before the
        // tokenizer bakes an unsupported control into its rendering. An
        // unreadable manifest is left for Model.load, which reports it better.
        if reasoningEffort != nil,
            let family = try? ManifestReader.peekFamily(directoryURL: modelDirectory)
        {
            try family.validateReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort)
        }
        let tokenizer = try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
        // A caller managing model residency supplies its own context so one
        // MTLCommandQueue and one compiled shader library survive across
        // unload/reload cycles (MetalContext.deinit documents that queue
        // teardown is not deinit-safe). Nil for every ordinary caller.
        //
        // A caller that named a device gets a context on that device; with
        // neither, the system default, which is what the server wants.
        let context: MetalContext
        if let reusingContext {
            context = reusingContext
        } else if let device {
            context = try MetalContext(device: device)
        } else {
            context = try MetalContext()
        }
        let loadRuntime = try RuntimeConfiguration(
            forceLogitsHead: true,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue(),
            expertIOTimeline: try RuntimeExpertIOTimeline.environmentValue())
        let slotOverride = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init)
        // Precedence: --expert-cache-slots flag, then the env override, then a
        // count derived from the model's own expert stride against a 1 GiB budget.
        //
        // Derived rather than fixed because the right count depends on the
        // quantisation: 1 GiB is 16 slots at 4-bit and 8 at 8-bit, which are the
        // measured optima for each. The previous fixed default of 64 was slower
        // *and* larger than either -- benchmarked at the shipped 262144 context,
        // 4-bit managed 9.85 tok/s at 64 slots against 13.61 at 16.
        // The architecture comes from the manifest rather than being assumed,
        // exactly as the CLI resolves it: a payload of any other family should
        // load, not fail on a dimension mismatch.
        let modelFamily = try ManifestReader.peekFamily(directoryURL: modelDirectory)
        let expectedArch: ArchConfig
        do {
            // The family's preset, or -- for a family with more than one
            // geometry, like the dense Qwen 3.5 models -- the manifest's own
            // declaration. `ServerInferenceError` keeps the shape callers
            // expect; the reason travels in the message.
            expectedArch = try ArchConfig.resolved(
                forFamily: modelFamily,
                directoryURL: modelDirectory)
        } catch {
            throw ServerInferenceError.unsupportedModel("\(error)")
        }
        let derivedSlots: Int
        // An explicit --ram-budget names what the whole server may hold; this
        // only supplies the cache default: clamped so a family tuned on a
        // 24 GiB machine cannot hand a smaller one a budget it has no room
        // for, and grown on a larger one, which has working set the tune left
        // unused.
        let tunedBudget: Int
        if let identity = try? ManifestReader.peekIdentity(directoryURL: modelDirectory) {
            tunedBudget = RuntimeConfiguration.affordableExpertCacheBudget(
                ModelProfile.resolve(identity: identity).expertCacheBudgetBytes)
        } else {
            tunedBudget = RuntimeConfiguration.defaultExpertCacheBudgetBytes
        }
        let loadedManifest = try? ManifestReader.load(
            directoryURL: modelDirectory,
            expecting: expectedArch)
        // Bytes one expert occupies in the cache, across every layer.
        let cachePerSlotBytes =
            loadedManifest.map {
                Double($0.expertStride) * Double($0.arch.numLayers)
            } ?? 0
        let residentFloor = RuntimeConfiguration.residentFloorBytes(
            residentWeightBytes: loadedManifest?.files["model_weights.bin"]
                .map { Int($0.size) } ?? 0)
        // One definition of the precedence, shared by the lines below and by the
        // load. The diagnostics have to print the count that is actually
        // allocated: before this, an operator's `--expert-cache-slots` changed
        // the rung and the log still named the derived one.
        let appliedSlots = { (derived: Int) in
            requestedExpertCacheSlots ?? slotOverride ?? derived
        }
        if let explicitTarget = expertCacheBudgetBytes {
            // The flag is a target for the whole process, not just the cache:
            // the weights and the runtime are resident either way, so the cache
            // is what is left, and the slot count steps down to stay inside the
            // number the user named.
            derivedSlots = RuntimeConfiguration.expertCacheSlotsFitting(
                expertStrideBytes: loadedManifest?.expertStride ?? 0,
                layers: loadedManifest?.arch.numLayers ?? 0,
                cacheBytes: explicitTarget - residentFloor)
            for line in ExpertCacheDiagnostics.targetLines(
                targetBytes: explicitTarget,
                appliedSlots: appliedSlots(derivedSlots),
                derivedSlots: derivedSlots,
                cacheBytesPerSlot: cachePerSlotBytes,
                residentFloorBytes: residentFloor
            ) {
                ServerLog.diagnostic(line)
            }
        } else if let manifest = loadedManifest {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: tunedBudget)
            ServerLog.diagnostic(
                ExpertCacheDiagnostics.profileLine(
                    appliedSlots: appliedSlots(derivedSlots),
                    derivedSlots: derivedSlots,
                    cacheBytesPerSlot: cachePerSlotBytes,
                    residentFloorBytes: residentFloor))
        } else {
            // Unreadable manifest means the load below will fail with a better
            // message than anything this could throw, so pick the safe small end.
            derivedSlots = RuntimeConfiguration.minimumExpertCacheSlots
        }
        let loadSlots = appliedSlots(derivedSlots)
        // `nil` keeps the long-standing rule: an installer receipt means the
        // payload's digests were pinned at install time, so the check is against
        // the manifest and the path; without one, hash the payload. An explicit
        // policy is the caller's and applies to the draft head too — it is a
        // file the engine computes on, not a document.
        let integrityPolicy =
            requestedIntegrityPolicy ?? .resolved(directoryURL: modelDirectory)
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: expectedArch,
            streamingMode: .pread(slotCount: loadSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            expertIOTimeline: loadRuntime.expertIOTimeline,
            integrityPolicy: integrityPolicy)
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: requestedRDAdvisePolicy
                ?? ProcessInfo.processInfo.environment["TINYTITAN_RDADVISE_POLICY"]
                .map(RDAdvicePolicyMode.parse)
                ?? loadRuntime.rdadvisePolicy,
            prefillChunkTokens: requestedPrefillChunkTokens
                ?? ModelProfile.resolve(
                    modelID: model.modelID, family: model.config.family,
                    weightBits: model.routedExpertWeightBits
                ).prefillChunkTokens
                ?? defaultPrefillChunkTokens(
                    family: model.config.family,
                    fallback: loadRuntime.prefillChunkTokens),
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: forceLogitsHead,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            expertIOTimeline: loadRuntime.expertIOTimeline,
            kvCachePrecision: kvCachePrecision,
            hostKVWindowTokens: hostKVWindowTokens,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        // MTP owns the target's runner and is single-sequence, so it cannot
        // batch. Otherwise the requested width is capped by what the worst-case
        // per-slot stores can hold beside the wired expert cache: the whole
        // point of the cap is that the cache cannot be paged out to rescue an
        // over-commit. The clamp is per load, so a catalog switch re-evaluates
        // it against the model actually being loaded.
        let hasPerSequenceState =
            model.config.sparseIndexer.enabled || model.config.ple.enabled
        let requestedSlots = BatchedMemoryBudget.effectiveRequestedSlots(
            requested: mtpModelDirectory == nil ? slots : 1,
            hasPerSequenceState: hasPerSequenceState)
        if slots > 1, hasPerSequenceState {
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan batch width \(slots) is not supported for a "
                        + "QSA/PLE model (per-sequence indexer/ngram state is "
                        + "not slot-keyed); serving 1 at once\n").utf8))
        }
        let perSlotBytes = BatchedMemoryBudget.perSlotBytes(
            config: model.config,
            maxContext: maxContext,
            precision: runtime.kvCachePrecision,
            fp16RingEnabled: runtime.fp16RingEnabled,
            slidingWindow: model.config.slidingWindow,
            maxPrefillChunkTokens: runtime.prefillChunkTokens,
            vocab: model.config.vocabSize,
            hostKVWindowTokens: runtime.hostKVWindowTokens)
        let slotBudget = BatchedMemoryBudget.slotBudgetBytes(
            physicalMemory: ProcessInfo.processInfo.physicalMemory,
            // A dense family has no routed experts, so it has no expert cache to
            // hold back; see `expertCacheHeldBack`.
            expertCacheBudgetBytes: BatchedMemoryBudget.expertCacheHeldBack(
                numExperts: model.config.numExperts,
                configured: expertCacheBudgetBytes ?? tunedBudget))
        let effectiveSlots = BatchedMemoryBudget.effectiveSlots(
            requested: requestedSlots,
            perSlotBytes: perSlotBytes,
            budgetBytes: slotBudget)
        if effectiveSlots < requestedSlots {
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan batch width \(requestedSlots) exceeds the memory budget "
                        + "(per-slot \(perSlotBytes / 1_048_576) MiB, budget "
                        + "\(slotBudget / 1_048_576) MiB); serving \(effectiveSlots) at once\n")
                        .utf8))
        }
        let mtpDecoder: StreamingMTPDecoder?
        let runner: RealForwardRunner
        if let mtpModelDirectory {
            let sidecarFamily = try ManifestReader.peekFamily(
                directoryURL: mtpModelDirectory)
            // Resolved rather than looked up: a 4.0 draft derived from a
            // differently-shaped target must come from its own manifest, the
            // same way the target did.
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
                integrityPolicy: requestedIntegrityPolicy
                    ?? .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: maxContext,
                memoryBudgetMiB: mtpMemoryMiB,
                runtimeConfiguration: runtime)
            mtpDecoder = decoder
            runner = decoder.target
        } else {
            mtpDecoder = nil
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                slots: effectiveSlots,
                runtimeConfiguration: runtime)
        }
        // K0: one line per model load naming the prefill tensor-ops path and
        // why it was chosen. Written before the first request so the reason is
        // in the startup log even if no generation runs.
        ServerLog.diagnostic(runner.prefillTensorOpsLogLine)
        let scratches = try (0..<effectiveSlots).map { _ in
            try RawCompletionScratch(
                context: context, vocab: model.config.vocabSize,
                logitSoftcap: Float(model.config.finalLogitSoftcap))
        }
        // Bounded like every other document copied into a model directory: the
        // installed template measures 8,952 bytes, and the digest below is taken
        // over whatever bytes come back, so the read is capped before it allocates.
        let templateDigest = SHA256.hash(
            data: try BoundedMetadataRead.read(
                fileAt: templateURL,
                maxBytes: ManifestReader.defaultMaxBytes)
        )
        .map { String(format: "%02x", $0) }
        .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
            String(runtime.kvCachePrecision.rawValue),
            runtime.ropeScalingMode.rawValue,
            String(runtime.yarnContextTokens),
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: runtime.kvCachePrecision.label,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let effectivePromptCacheMode = Self.effectivePromptCacheMode(
            requested: promptCacheMode,
            mtpEnabled: mtpDecoder != nil,
            slots: effectiveSlots)
        let promptStateStore: ServerPromptStateStore?
        let promptCheckpointStore: PromptCheckpointStore?
        let promptCache: ServerPromptCache
        if effectivePromptCacheMode == .multiPrefix {
            // Interior checkpoints are on by default; the A/B benchmark can
            // turn them off without a second binary. They are only useful
            // when a later request edits the last turn, and their capture
            // cost is real, so the switch is part of the measured surface.
            let checkpointsEnabled = Self.checkpointsEnabled(
                environment: ProcessInfo.processInfo.environment)
            // Both stores index the same directory and evict only their own
            // records, so their limits add: the pair may use the whole budget,
            // never twice it. Checkpoints that could not be deleted stay
            // charged against the turn store's share rather than vanishing
            // from the accounting.
            let checkpointSurvivorBytes =
                checkpointsEnabled
                ? 0
                : try await Self.reclaimStaleCheckpoints(
                    directory: promptCacheDiskDirectory,
                    totalDiskLimitBytes: promptCacheDiskLimitBytes)
            let storage = ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: promptCacheMemoryLimitBytes,
                diskDirectory: promptCacheDiskDirectory,
                diskLimitBytes: Self.promptCachePerStoreDiskLimit(
                    total: promptCacheDiskLimitBytes - checkpointSurvivorBytes,
                    checkpointsEnabled: checkpointsEnabled))
            let store = try ServerPromptStateStore(configuration: storage)
            let persisted = store.loadEntries(domain: promptCacheDomain)
            if persisted.count > promptCacheMaximumEntries {
                store.remove(
                    entryIDs:
                        persisted
                        .dropLast(promptCacheMaximumEntries)
                        .map(\.id))
            }
            promptStateStore = store
            if checkpointsEnabled {
                // Process-wide per storage configuration: a model reload in
                // the same server must see the system prefix the previous
                // session captured, even with the disk tier off (K1).
                let checkpoints = try PromptCheckpointStore.sharedInstance(
                    configuration: storage)
                // Index now rather than on first capture: a restart can find
                // more checkpoint records than their half allows, and leaving
                // them until then lets both stores exceed the shared budget
                // in the meantime.
                await checkpoints.indexDiskIfNeeded()
                promptCheckpointStore = checkpoints
            } else {
                promptCheckpointStore = nil
            }
            promptCache = ServerPromptCache(
                maximumEntries: promptCacheMaximumEntries,
                entries: persisted)
        } else {
            promptStateStore = nil
            promptCheckpointStore = nil
            promptCache = ServerPromptCache(maximumEntries: 1)
        }
        let session = ServerModelSession(
            context: context,
            model: model,
            tokenizer: tokenizer,
            tokenizerFolder: tokenizerFolder,
            loadedReasoning: RequestReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort),
            runner: runner,
            mtpDecoder: mtpDecoder,
            scratches: scratches,
            prefillConfig: runtime.prefillConfig,
            expertCacheSlots: loadSlots,
            slots: effectiveSlots,
            maxContext: maxContext,
            promptCacheMode: effectivePromptCacheMode,
            promptCacheDomain: promptCacheDomain,
            promptCache: promptCache,
            promptStateStore: promptStateStore,
            promptCheckpointStore: promptCheckpointStore,
            concisePrompt: conciseModeEnabled()
                ? ConcisePrompt.standard : nil)
        // In the background: the session is ready now, and the first request
        // stops the warm-up rather than waiting behind it. A bench that drives
        // the runner directly passes `warmExperts: false`, because nothing
        // would keep it out of the planner.
        if warmExperts { await session.startExpertWarmup() }
        return session
    }
}
