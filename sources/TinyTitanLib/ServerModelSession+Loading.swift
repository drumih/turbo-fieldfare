// The `load` factory: verification, tokenizer, scratch and cache setup.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import Metal
import TinyTitanCore

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
        factory: EngineFactory,
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
        mtpMemoryMiB: Int = 384,
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
        if reasoningEffort != nil {
            let metadata = try factory.metadata(modelDirectory)
            try factory.validateReasoning(metadata, thinkingMode, reasoningEffort)
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
        } else {
            context = try factory.makeContext(device)
        }
        let baseOptions = SessionBuildOptions(
            modelDirectory: modelDirectory,
            context: context,
            maxContext: maxContext,
            slots: slots,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryLimitBytes: promptCacheMemoryLimitBytes,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskLimitBytes: promptCacheDiskLimitBytes,
            prefillChunkTokens: requestedPrefillChunkTokens,
            kvCachePrecision: kvCachePrecision,
            hostKVWindowTokens: hostKVWindowTokens,
            ropeScalingMode: ropeScalingMode,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            expertCacheSlots: requestedExpertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            mtpModelDirectory: mtpModelDirectory,
            mtpMemoryMiB: mtpMemoryMiB,
            integrityPolicy: requestedIntegrityPolicy,
            rdAdvicePolicy: requestedRDAdvisePolicy,
            forceLogitsHead: forceLogitsHead)
        let loadPlan = try factory.prepareLoad(baseOptions)
        let expectedArch = loadPlan.architecture
        let loadSlots = loadPlan.expertCacheSlots
        let tunedBudget = loadPlan.tunedExpertCacheBudgetBytes
        let model: any EngineModel
        if let buildWithSlots = factory.buildModelWithSlots {
            model = try buildWithSlots(modelDirectory, expectedArch, context, loadSlots)
        } else {
            model = try factory.buildModel(modelDirectory, expectedArch, context)
        }
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
        let perSlotBytes = factory.perSlotBytes(model, baseOptions, loadPlan)
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
        let options = SessionBuildOptions(
            modelDirectory: modelDirectory,
            context: context,
            maxContext: maxContext,
            slots: effectiveSlots,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryLimitBytes: promptCacheMemoryLimitBytes,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskLimitBytes: promptCacheDiskLimitBytes,
            prefillChunkTokens: requestedPrefillChunkTokens,
            kvCachePrecision: kvCachePrecision,
            hostKVWindowTokens: hostKVWindowTokens,
            ropeScalingMode: ropeScalingMode,
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort,
            expertCacheSlots: requestedExpertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            mtpModelDirectory: mtpModelDirectory,
            mtpMemoryMiB: mtpMemoryMiB,
            integrityPolicy: requestedIntegrityPolicy,
            rdAdvicePolicy: requestedRDAdvisePolicy,
            forceLogitsHead: forceLogitsHead,
            engineConfiguration: loadPlan.engineConfiguration)
        factory.prepare(options)
        let runner = try factory.buildRunner(model, options)
        let mtpDecoder = factory.mtpDecoder()
        // K0: one line per model load naming the prefill tensor-ops path and
        // why it was chosen. Written before the first request so the reason is
        // in the startup log even if no generation runs.
        ServerLog.diagnostic(runner.prefillTensorOpsLogLine)
        let scratches = try (0..<effectiveSlots).map { _ in
            try factory.makeScratch(
                context, model.config.vocabSize, Float(model.config.finalLogitSoftcap))
        }
        // Bounded like every other document copied into a model directory: the
        // installed template measures 8,952 bytes, and the digest below is taken
        // over whatever bytes come back, so the read is capped before it allocates.
        let templateDigest = SHA256.hash(
            data: try BoundedMetadataRead.read(
                fileAt: templateURL,
                maxBytes: 64 * 1024 * 1024)
        )
        .map { String(format: "%02x", $0) }
        .joined()
        let runtimeDigest = SHA256.hash(data: Data(loadPlan.runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: kvCachePrecision.label,
            fp16RingEnabled: loadPlan.fp16RingEnabled,
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
            metadata: try factory.metadata(modelDirectory),
            defaultModelID: try factory.preview(modelDirectory, requestedPrefillChunkTokens)
                .modelID,
            tokenizer: tokenizer,
            tokenizerFolder: tokenizerFolder,
            loadedReasoning: RequestReasoning(
                thinkingMode: thinkingMode,
                effort: reasoningEffort),
            runner: runner,
            mtpDecoder: mtpDecoder,
            runCompletion: factory.runRawCompletion,
            makeWarmup: factory.makeWarmup,
            scratches: scratches,
            prefillConfig: loadPlan.prefillConfiguration,
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
