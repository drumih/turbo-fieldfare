// Cache publication, token counting, prompt encoding and diagnostics.
//
// Split out of ServerInference.swift (2026-09-27) under the 500-line-per-file
// rule; declarations and their doc comments are unchanged, except that
// `private` became internal where a declaration now lives beside an extension
// in another file.
import CryptoKit
import Foundation
import TinyTitan
import TinyTitanCore

extension ServerModelSession {
    /// Publish this turn's KV range to the prompt cache, and persist a snapshot
    /// so a later request can resume from it without re-prefilling.
    ///
    /// Every failure path here degrades to "no cache entry" rather than to a
    /// broken one: an entry whose snapshot cannot be captured or verified is
    /// removed again, so the next hit re-prefills instead of attempting a
    /// doomed restore.
    func publishCacheEntry(
        cacheRequest: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool
    ) {
        if mtpDecoder != nil {
            // Native MTP keeps a second KV stream. Until both states are
            // persisted atomically, do not publish target-only cache entries.
            promptCache.invalidate()
            activePromptCacheEntryID = nil
        } else if promptCacheMode == .singlePrefix {
            let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopStringFiltered)
            if publication == nil { promptCache.invalidate() }
        } else if promptCacheMode == .multiPrefix {
            let previousActive = activePromptCacheEntryID
            if let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopStringFiltered)
            {
                promptStateStore?.remove(entryIDs: publication.evictedEntryIDs)
                do {
                    guard let promptStateStore else {
                        throw ServerPromptStateStoreError.missing(
                            publication.entry.id)
                    }
                    // S2: capture is bounded by the store's hard snapshot cap;
                    // the payload is a plain Data copy, so the disk write can
                    // proceed off the actor (dedicated store disk queue) while
                    // the next request starts. Concurrent saves serialize on
                    // the queue, so a later generation's snapshot can never
                    // clobber an in-flight write. The entry is already in the
                    // in-memory cache; a request that races the write simply
                    // misses and re-prefills (restore failure self-heals).
                    let snapshot = try runner.captureInferenceState(
                        maximumBytes: promptStateStore.maximumSnapshotBytes)
                    guard snapshot.descriptor.position == publication.entry.kvPosition else {
                        throw InferenceStateSnapshotError.invalidPosition(
                            snapshot.descriptor.position)
                    }
                    let auxiliary = try runner.captureAuxiliaryState()
                    if let auxiliary,
                        auxiliary.position != publication.entry.kvPosition
                    {
                        throw InferenceAuxiliaryStateError.invalidPosition(
                            expected: publication.entry.kvPosition,
                            actual: auxiliary.position)
                    }
                    let entry = publication.entry
                    Task.detached(priority: .utility) { [promptStateStore] in
                        let saved = await promptStateStore.save(
                            entry: entry,
                            snapshot: snapshot,
                            auxiliary: auxiliary)
                        if let diskError = saved.diskError {
                            FileHandle.standardError.write(
                                Data(
                                    ("TinyTitan prompt_cache disk_write_failed error=\(diskError)\n")
                                        .utf8))
                        }
                        ServerLog.diagnostic(
                            "TinyTitan prompt_cache stored "
                                + "tokens=\(entry.kvPosition) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "aux_bytes=\(auxiliary?.payload.count ?? 0) "
                                + "ram_bytes=\(saved.memoryBytes) "
                                + "disk_bytes=\(saved.diskBytes) "
                                + "entry=\(entry.id.uuidString.lowercased())")
                    }
                } catch {
                    // S24: a snapshot that cannot be captured or verified is
                    // never left published without backing; drop the entry so
                    // the next hit re-prefills instead of a doomed restore.
                    FileHandle.standardError.write(
                        Data(
                            ("TinyTitan prompt_cache snapshot_failed error=\(error)\n").utf8))
                    promptCache.remove(entryIDs: [publication.entry.id])
                    activePromptCacheEntryID = nil
                }
                if let previousActive,
                    previousActive != publication.entry.id,
                    promptStateStore?.contains(previousActive) != true
                {
                    promptCache.remove(entryIDs: [previousActive])
                }
                activePromptCacheEntryID = publication.entry.id
            } else {
                activePromptCacheEntryID = nil
            }
        }
    }

    func usesToolTemplate(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> Bool {
        Self.usesToolTemplate(messages: messages, tools: tools)
    }

    static func usesToolTemplate(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> Bool {
        !tools.isEmpty
            || messages.contains {
                $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
            }
    }

    /// Prompt tokens of a request as generation would render it — the same
    /// encoding, minus the generation.
    ///
    /// The tokenizer is resolved per request for the same reason `generate`
    /// resolves one: a request that names a different thinking mode or effort is
    /// rendered at that level, and an effort sentence is tens of tokens on a model
    /// that has levels. Counting with the session's tokenizer reported the loaded
    /// level's number for a request that would not be rendered at it.
    package func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        try Self.promptTokenCount(
            request,
            tokenizer: try await resolvedTokenizer(for: request.reasoning),
            concisePrompt: concisePrompt)
    }

    /// The count from a tokenizer alone, which is how the router answers for a
    /// GPU model that is not the one loaded.
    ///
    /// `concisePrompt` is passed in rather than read because it is a property of a
    /// *loaded session*; the router's path has no session, so it counts without
    /// one and is the one case that can differ from what a concise-mode server
    /// would spend.
    package static func promptTokenCount(
        _ request: ValidatedChatRequest,
        tokenizer: GFTokenizer,
        concisePrompt: String? = nil
    ) throws -> Int {
        // The same two transformations `preparePrompt` applies, in the same order.
        // This used to encode the request verbatim, so the count was inflated for
        // the `<model>-fast` alias and whenever `TINYTITAN_STRIP_CLI_PROMPT` is set,
        // and under-reported in concise mode — the opposite direction in each case,
        // which is why neither showed up as a single discrepancy.
        let filteredMessages: [GFTokenizer.Message]
        let filteredTools: [GFTokenizer.FunctionDefinition]
        if request.stripCLIPrompt || CLIStrip.isEnabled() {
            let filtered = CLIStrip.filter(
                messages: request.messages,
                tools: request.tools)
            filteredMessages = filtered.messages
            filteredTools = filtered.tools
        } else {
            filteredMessages = request.messages
            filteredTools = request.tools
        }
        let messages =
            concisePrompt.map {
                ConcisePrompt.appendingSystemPrompt($0, to: filteredMessages)
            } ?? filteredMessages
        return try encodePrompt(
            tokenizer: tokenizer, messages: messages, tools: filteredTools,
            usesToolTemplate: usesToolTemplate(
                messages: filteredMessages,
                tools: filteredTools)
        ).count
    }

    /// The vocabulary-as-bytes table, built on first use.
    func structuredOutputTable() -> JSONTokenTable {
        if let jsonTokenTable { return jsonTokenTable }
        let table = JSONTokenTable(tokenizer: tokenizer)
        jsonTokenTable = table
        return table
    }

    /// The tokenizer this request should be rendered with.
    ///
    /// Almost every request takes the session's own, which keeps the render,
    /// the special tokens and the prompt cache exactly as they were. A request
    /// that named a different thinking mode or effort -- a mid-session switch,
    /// including turning thinking off -- gets a tokenizer for that
    /// configuration. The load coordinator caches by
    /// `(folder, thinking, effort)`, so this is a dictionary lookup once a
    /// level has been used, and a real load the first time.
    ///
    /// The returned tokenizer carries the think-block and stop token IDs for
    /// its own mode, so decode and the assistant decoder follow the switch
    /// rather than only the prompt text.
    func resolvedTokenizer(
        for reasoning: RequestReasoning?
    ) async throws -> GFTokenizer {
        guard let reasoning, !reasoning.matches(loadedReasoning) else {
            return tokenizer
        }
        return try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: reasoning.thinkingMode,
            reasoningEffort: reasoning.effort)
    }

    func encodePrompt(
        with renderTokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool
    ) throws -> [Int32] {
        try Self.encodePrompt(
            tokenizer: renderTokenizer, messages: messages, tools: tools,
            usesToolTemplate: usesToolTemplate)
    }

    static func encodePrompt(
        tokenizer: GFTokenizer,
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(messages: messages, tools: tools)
        }
        let rendered = try tokenizer.applyChatTemplate(messages)
        return tokenizer.encode(rendered, addBOS: false)
    }

    /// Optional per-request diagnostics: MTP acceptance, the TINYTITAN_RUNNER_STATS
    /// stage split, and the TINYTITAN_KERNEL_STATS GPU breakdown. All three are
    /// env-gated and read-only, so they stay out of the generation path proper.
    func emitGenerationDiagnostics(
        activeProducer: any LogitProducer,
        result: RawDecodeResult,
        snapshot runnerSnapshot: RunnerCounterSnapshot
    ) {
        if let activeMTP = activeProducer as? StreamingMTPDecoder {
            let stats = activeMTP.statistics
            let decodeRate =
                result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            ServerLog.diagnostic(
                String(
                    format:
                        "TinyTitan mtp drafted=%d accepted=%d acceptance=%.1f%% "
                        + "target_passes=%d emitted_per_pass=%.3f "
                        + "prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f "
                        + "memory_required_mib=%.1f memory_budget_mib=%.1f",
                    stats.draftedTokens,
                    stats.acceptedTokens,
                    stats.acceptanceRate * 100,
                    stats.targetBackbonePasses,
                    stats.emittedTokensPerTargetPass,
                    result.prefillSeconds,
                    result.decodeSeconds,
                    decodeRate,
                    Double(activeMTP.memoryPlan.requiredBytes) / 1_048_576,
                    Double(activeMTP.memoryPlan.budgetBytes) / 1_048_576))
            if ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil,
                stats.targetBackbonePasses > 0
            {
                // Per-pass phase attribution for the Track B1 investigation:
                // where a verify pass's wall time actually goes. Milliseconds
                // averaged over the request's target passes.
                let passes = Double(stats.targetBackbonePasses)
                let ms: (UInt64) -> Double = { Double($0) / passes / 1_000_000 }
                ServerLog.diagnostic(
                    String(
                        format:
                            "TinyTitan mtp-phases per_pass_ms proposal=%.3f checkpoint=%.3f "
                            + "verify=%.3f verify_backbone=%.3f verify_head=%.3f "
                            + "verify_argmax=%.3f commit=%.3f rollback=%.3f passes=%d",
                        ms(stats.proposalNanos),
                        ms(stats.checkpointNanos),
                        ms(stats.verifyNanos),
                        ms(stats.verifyBackboneNanos),
                        ms(stats.verifyHeadNanos),
                        ms(stats.verifyArgmaxNanos),
                        ms(stats.commitNanos),
                        ms(stats.rollbackNanos),
                        stats.targetBackbonePasses))
            }
        } else {
            let decodeRate =
                result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            ServerLog.diagnostic(
                String(
                    format:
                        "TinyTitan generation prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f",
                    result.prefillSeconds,
                    result.decodeSeconds,
                    decodeRate))
        }
        if let wide = result.wideVerify {
            ServerLog.diagnostic(wide.summary)
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_RUNNER_STATS"] != nil {
            emitRunnerDiagnostics(result: result, snapshot: runnerSnapshot)
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_STATS"] != nil {
            emitKernelDiagnostics(result: result)
        }
        // Decode-only expert I/O. Gated by the runner's own baseline rather than
        // by a flag here: `decodeExpertIO()` is non-nil only when
        // TINYTITAN_DECODE_IO_TRACE took the handover snapshot. This line used
        // to be written by the CLI (removed in 0cdba76 when generation moved
        // onto the library), leaving the flag, the baseline and the accessor
        // with no reporter while benchmark/expert_cache_slots.py still parsed it.
        if let decode = runner.decodeExpertIO() {
            ServerLog.diagnostic(
                ExpertCacheDiagnostics.decodeIOLine(
                    hits: decode.hits, misses: decode.misses, bytes: decode.bytes,
                    tokens: result.newTokens))
        }
    }

    func emitRunnerDiagnostics(
        result: RawDecodeResult,
        snapshot: RunnerCounterSnapshot
    ) {
        let tokens = max(1, result.newTokens)
        let ms: (UInt64, UInt64) -> Double = { delta, base in
            Double(delta > base ? delta - base : 0) / Double(tokens) / 1_000_000
        }
        let missIoNanos = runner.totalMissIoNanos - snapshot.missIo
        let exposedIoNanos = runner.totalExposedIoNanos - snapshot.exposedIo
        let hiddenPercent =
            missIoNanos == 0
            ? 100.0
            : 100 * (1 - Double(exposedIoNanos) / Double(missIoNanos))
        let expert = runner.expertStreamingStatistics()
            .subtracting(snapshot.expertStreaming)
        let gpuHits = runner.totalGPUClassifiedHits - snapshot.gpuClassifiedHits
        let gpuMisses = runner.totalGPUClassifiedMisses - snapshot.gpuClassifiedMisses
        let gpuAllHit = runner.totalGPUResidencyAllHitLayers - snapshot.gpuAllHitLayers
        ServerLog.diagnostic(
            String(
                format: "TinyTitan runner cb1_ms=%.3f io_ms=%.3f cb2_ms=%.3f "
                    + "head_ms=%.3f head_fused_ms=%.3f rdadvise_ms=%.3f "
                    + "wait_ms=%.3f body_ms=%.3f rdadvise_calls=%llu rdadvise_mib=%.1f "
                    + "expert_hit_rate=%.4f expert_hits=%llu expert_misses=%llu "
                    + "expert_evictions=%llu expert_reloads=%llu expert_read_mib=%.1f "
                    + "expert_load_p50_ms=%.3f expert_load_p95_ms=%.3f "
                    + "expert_load_p99_ms=%.3f io_hidden_pct=%.2f hit_fixup_layers=%llu "
                    + "router_readback_ms=%.4f cache_plan_ms=%.4f io_queue_ms=%.4f "
                    + "io_completion_to_fixup_ms=%.4f io_host_waits=%llu "
                    + "io_host_waits_avoided=%llu gpu_classified_hits=%llu "
                    + "gpu_classified_misses=%llu gpu_all_hit_layers=%llu "
                    + "prefetch_issued_per_token=%.2f prefetch_adopted_per_token=%.2f "
                    + "pre_ms=%.3f pre_release_ms=%.3f pre_pin_ms=%.3f pre_reserve_ms=%.3f "
                    + "embed_ms=%.3f gather_ms=%.3f loop_sample_ms=%.3f "
                    + "loop_progress_ms=%.3f loop_other_ms=%.3f",
                ms(runner.totalCb1Nanos, snapshot.cb1),
                ms(runner.totalIoNanos, snapshot.io),
                ms(runner.totalCb2Nanos, snapshot.cb2),
                ms(runner.totalHeadNanos, snapshot.head),
                ms(runner.totalHeadFusedNanos, snapshot.headFused),
                ms(runner.totalRDAdviseNanos, snapshot.rdadvise),
                ms(runner.totalWaitNanos, snapshot.wait),
                ms(runner.totalBodyNanos, snapshot.body),
                runner.totalRDAdviseCalls - snapshot.rdadviseCalls,
                Double(runner.totalRDAdviseBytes - snapshot.rdadviseBytes) / 1_048_576,
                expert.hitRate, expert.hits, expert.misses, expert.evictions,
                expert.reloads, Double(expert.bytesRead) / 1_048_576,
                Double(expert.loadLatencyPercentile(0.50)) / 1_000_000,
                Double(expert.loadLatencyPercentile(0.95)) / 1_000_000,
                Double(expert.loadLatencyPercentile(0.99)) / 1_000_000,
                hiddenPercent, runner.totalHitFixupLayers - snapshot.hitFixupLayers,
                ms(runner.totalRouterReadbackNanos, snapshot.routerReadback),
                ms(runner.totalCachePlanNanos, snapshot.cachePlan),
                ms(runner.totalIOQueueNanos, snapshot.ioQueue),
                ms(runner.totalIOCompletionToFixupSubmitNanos, snapshot.ioCompletionToFixup),
                runner.totalExpertIOHostWaits - snapshot.ioHostWaits,
                runner.totalExpertIOHostWaitsAvoided - snapshot.ioHostWaitsAvoided,
                gpuHits, gpuMisses, gpuAllHit,
                Double(runner.totalPrefetchIssued &- snapshot.prefetchIssued) / Double(tokens),
                Double(runner.totalPrefetchAdopted &- snapshot.prefetchAdopted) / Double(tokens),
                ms(runner.totalPreambleNanos, snapshot.preamble),
                ms(runner.totalPreambleReleaseNanos, snapshot.preambleRelease),
                ms(runner.totalPreamblePinNanos, snapshot.preamblePin),
                ms(runner.totalPreambleReserveNanos, snapshot.preambleReserve),
                ms(runner.totalEmbedNanos, snapshot.embed),
                ms(runner.totalGatherNanos, snapshot.gather),
                ms(runner.totalLoopSampleNanos, snapshot.loopSample),
                ms(runner.totalLoopProgressNanos, snapshot.loopProgress),
                ms(runner.totalLoopOtherNanos, snapshot.loopOther)))
        if let ring = runner.prefetchRingSummary { ServerLog.diagnostic("TinyTitan \(ring)") }
    }

    func emitKernelDiagnostics(result: RawDecodeResult) {
        let tokens = max(1, result.newTokens)
        let summary = runner.kernelGPUTimingSummary()
        let totalGPU = summary.reduce(0) { $0 + $1.millis }
        for entry in summary {
            ServerLog.diagnostic(
                String(
                    format: "TinyTitan kernel role=%@ gpu_ms=%.3f per_token_ms=%.3f count=%d",
                    entry.role, entry.millis, entry.millis / Double(tokens), entry.count))
        }
        // Role sums overlap by design. Merged busy/span is the actual queue
        // occupancy and distinguishes useful concurrency from idle gaps.
        let occupancy = runner.kernelGPUOccupancy()
        ServerLog.diagnostic(
            String(
                format: "TinyTitan kernel total_gpu_ms=%.3f gpu_share_of_decode=%.1f%%",
                totalGPU,
                result.decodeSeconds > 0
                    ? totalGPU / (result.decodeSeconds * 1000) * 100 : 0))
        for gap in runner.kernelGPUGaps().prefix(8) {
            ServerLog.diagnostic(
                String(
                    format: "TinyTitan gap %@ total_ms=%.1f per_token_ms=%.3f count=%d",
                    gap.transition, gap.millis, gap.millis / Double(tokens), gap.count))
        }
        ServerLog.diagnostic(
            String(
                format: "TinyTitan kernel busy_ms=%.3f span_ms=%.3f "
                    + "occupancy=%.1f%% busy_share_of_decode=%.1f%% busy_per_token_ms=%.3f",
                occupancy.busyMillis, occupancy.spanMillis,
                occupancy.spanMillis > 0
                    ? occupancy.busyMillis / occupancy.spanMillis * 100 : 0,
                result.decodeSeconds > 0
                    ? occupancy.busyMillis / (result.decodeSeconds * 1000) * 100 : 0,
                occupancy.busyMillis / Double(tokens)))
    }
}
