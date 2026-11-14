// Slot admission and prompt-cache preparation and resolution.
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
    /// Take a slot for one generation. Actor-isolated, so the free list and the
    /// waiter queue never race; the coordinator's width normally keeps a slot
    /// free, and the wait exists only so a wider coordinator degrades to
    /// queueing instead of failing.
    func acquireSlot() async throws -> Int {
        while true {
            switch claimSlotOrQueue() {
            case .taken(let slot): return slot
            case .queued(let slot):
                await withTaskCancellationHandler {
                    await withCheckedContinuation { slot.handOver($0) }
                } onCancel: {
                    Task { await self.cancelSlot(slot) }
                }
                if slot.received == .cancel || Task.isCancelled { throw CancellationError() }
            }
        }
    }

    /// Pop a free slot or register the caller on a fresh one, in one step with no
    /// suspension: `releaseSlot` cannot hand a slot over between the check and the
    /// registration. See `SuspensionSlot` for why the continuation itself is not
    /// what gets registered.
    private func claimSlotOrQueue() -> SlotGate {
        if let slot = freeSlots.popLast() { return .taken(slot) }
        let waiter = SuspensionSlot()
        slotWaiters.append(waiter)
        return .queued(waiter)
    }

    private enum SlotGate {
        case taken(Int)
        case queued(SuspensionSlot)
    }

    func cancelSlot(_ slot: SuspensionSlot) {
        if let index = slotWaiters.firstIndex(where: { $0 === slot }) {
            slotWaiters.remove(at: index)
        }
        slot.signal(.cancel)
    }

    func releaseSlot(_ slot: Int) {
        freeSlots.append(slot)
        if !slotWaiters.isEmpty {
            slotWaiters.removeFirst().signal(.wake)
        }
    }

    /// TINYTITAN_CONCISE_MODE=1 (or "on") enables concise mode; the per-quant
    /// system prompt is then injected into every completion.
    static func conciseModeEnabled() -> Bool {
        switch ProcessInfo.processInfo.environment["TINYTITAN_CONCISE_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return true
        default: return false
        }
    }

    /// Render a validated request into prompt tokens.
    ///
    /// TINYTITAN_STRIP_CLI_PROMPT: drop the coding-CLI's system/developer guidance,
    /// tool definitions, tool-call history, and in-message <system-reminder>
    /// scaffolding, keeping only the real user/assistant conversation (see
    /// CLIStrip). Guards ensure the real prompt can never be stripped into an
    /// empty turn or an empty request. Runs when the request names the
    /// "<model>-fast" alias or TINYTITAN_STRIP_CLI_PROMPT is set.
    ///
    /// Returns the encoded prompt alongside the `cacheRequest` — the post-strip
    /// view the prompt cache must key on. Cache entries describe a KV range
    /// prefilled from the filtered messages, and the cache's text-continuation
    /// path re-renders the tail with the same template, so matching or
    /// publishing against the raw request would splice an unstripped tail onto
    /// a stripped prefix, silently losing the "-fast" alias's strip on every
    /// cached continuation turn.
    ///
    /// `renderTokenizer` is the one this request's reasoning resolves to, so a
    /// mid-session switch renders through the right template instead of the
    /// one the model happened to load with.
    func preparePrompt(
        _ request: ValidatedChatRequest,
        renderTokenizer: GFTokenizer
    ) throws -> (
        promptIDs: [Int32],
        cacheRequest: ValidatedChatRequest,
        needsToolTemplate: Bool
    ) {
        let filteredMessages: [GFTokenizer.Message]
        let filteredTools: [GFTokenizer.FunctionDefinition]
        var stripStats: CLIStrip.Stats?
        if request.stripCLIPrompt || CLIStrip.isEnabled() {
            let filtered = CLIStrip.filter(
                messages: request.messages,
                tools: request.tools)
            filteredMessages = filtered.messages
            filteredTools = filtered.tools
            stripStats = filtered.stats
        } else {
            filteredMessages = request.messages
            filteredTools = request.tools
        }
        let cacheRequest = request.replacingMessages(
            filteredMessages,
            tools: filteredTools)
        let needsToolTemplate = usesToolTemplate(
            messages: filteredMessages,
            tools: filteredTools)
        let effectiveMessages =
            concisePrompt.map {
                ConcisePrompt.appendingSystemPrompt($0, to: filteredMessages)
            } ?? filteredMessages
        let promptIDs = try encodePrompt(
            with: renderTokenizer,
            messages: effectiveMessages,
            tools: filteredTools,
            usesToolTemplate: needsToolTemplate)
        if let stats = stripStats {
            ServerLog.strip(
                stats: stats,
                promptTokens: promptIDs.count)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return (promptIDs, cacheRequest, needsToolTemplate)
    }

    /// Decide where this request's prefill starts: from scratch, or resumed on
    /// a cache entry whose KV is live or restorable.
    ///
    /// Mutates the cache and `activePromptCacheEntryID`, so it must run on the
    /// actor and before any generation begins.
    /// Whether a reasoning-level change forbids reusing any cached prefix.
    ///
    /// A request rendered at a different level than the session loaded at must
    /// not splice onto a cached KV range, and this is what makes the claim on
    /// `ValidatedChatRequest.reasoning` true.
    ///
    /// Comparing rendered token IDs is enough for the direct-prefix path, which
    /// is what that claim was written against: a level change renders different
    /// IDs, so it misses. It is *not* enough for a text continuation, which
    /// re-renders only the tail -- `matchTextContinuation` calls
    /// `applyChatTemplate` again -- and that used to happen with the session's
    /// tokenizer. The model then saw a generation prompt built for the loaded
    /// level while the decoder was built for the requested one, so either
    /// chain-of-thought leaked into `content` or the whole answer was reported
    /// as reasoning with `content` empty.
    ///
    /// Re-prefilling is the correct cost of a switch: the cached KV belongs to a
    /// different render, and the honest outcome is a miss.
    func reasoningForbidsCacheReuse(_ requested: RequestReasoning?) -> Bool {
        guard let requested else { return false }
        return !requested.matches(loadedReasoning)
    }

    func resolveCacheStart(
        cacheRequest: ValidatedChatRequest,
        promptIDs: [Int32],
        requestedReasoning: RequestReasoning?
    ) async throws -> (effectivePromptIDs: [Int32], start: RawCompletionStart) {
        if reasoningForbidsCacheReuse(requestedReasoning) {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            return (promptIDs, .reset)
        }
        let effectivePromptIDs: [Int32]
        var completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: cacheRequest,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer)
            {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(_, let effective, let cached):
                if runner.continuationPosition != cached {
                    // S15: the live KV no longer sits at the cached entry's
                    // position; re-prefill instead of resuming from a stale
                    // or mismatched in-memory state.
                    promptCache.invalidate()
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                } else {
                    effectivePromptIDs = effective
                    completionStart = .resume(cachedPromptTokens: cached)
                }
            }
        } else if promptCacheMode == .multiPrefix {
            let resolved = await resolveMultiPrefixCacheStart(
                cacheRequest: cacheRequest,
                promptIDs: promptIDs)
            effectivePromptIDs = resolved.effectivePromptIDs
            completionStart = resolved.start
        } else {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        // S12: an identical-prompt replay whose render equals the entry's
        // KV-backed prefix has nothing to prefill (cached == prompt count).
        // The continuation API requires cached < prompt count (it must
        // prefill at least one token), so resume as a full prefill; the
        // entry stays active for later extending requests.
        if case .resume(let cached) = completionStart,
            cached >= effectivePromptIDs.count
        {
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return (effectivePromptIDs, completionStart)
    }

    /// The multi-prefix branch of `resolveCacheStart`: a live/turn-end hit, an
    /// interior checkpoint, or a full prefill. Split out so the orchestrator
    /// stays inside the function-length gate; the branches are unchanged.
    private func resolveMultiPrefixCacheStart(
        cacheRequest: ValidatedChatRequest,
        promptIDs: [Int32]
    ) async -> (
        effectivePromptIDs: [Int32], start: RawCompletionStart
    ) {
        switch promptCache.match(
            domain: promptCacheDomain,
            request: cacheRequest,
            renderedPromptIDs: promptIDs,
            tokenizer: tokenizer)
        {
        case .miss:
            activePromptCacheEntryID = nil
            // The turn-end cache only describes a completed assistant turn.
            // An edited last turn has no such entry, but the conversation
            // prefix before it may still be on disk as an interior checkpoint.
            if let restored = await restoreInteriorCheckpoint(
                promptIDs: promptIDs,
                messages: cacheRequest.messages)
            {
                return (promptIDs, .resume(cachedPromptTokens: restored))
            }
            // No same-conversation checkpoint, but the system-end prefix may
            // have been captured by an *earlier* conversation. This is the
            // warm-subagent path: restore the shared system prefix and prefill
            // only the user turn behind it.
            if let shared = await restoreSharedSystemPrefix(
                promptIDs: promptIDs,
                messages: cacheRequest.messages)
            {
                return (promptIDs, .resume(cachedPromptTokens: shared))
            }
            return (promptIDs, .reset)
        case .hit(let entryID, let effective, let cached):
            if entryID == activePromptCacheEntryID,
                runner.continuationPosition == cached
            {
                // S15: tier=live is only trusted when the in-memory KV still
                // matches the entry (same entry id and the KV cursor sits
                // exactly at the request's expected position). Anything else
                // falls through to a snapshot restore or a full prefill
                // instead of resuming from a stale or mismatched KV.
                ServerLog.diagnostic(
                    "TinyTitan prompt_cache hit tier=live "
                        + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
            } else {
                do {
                    guard let promptStateStore else {
                        throw ServerPromptStateStoreError.missing(entryID)
                    }
                    let tier = try await promptStateStore.restore(
                        entryID: entryID,
                        into: runner)
                    ServerLog.diagnostic(
                        "TinyTitan prompt_cache hit tier=\(tier) "
                            + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())"
                    )
                } catch {
                    // Drop the stale entry and prefill from scratch rather
                    // than trust it.
                    FileHandle.standardError.write(
                        Data(
                            ("TinyTitan prompt_cache restore_failed "
                                + "entry=\(entryID.uuidString.lowercased()) error=\(error)\n")
                                .utf8))
                    promptStateStore?.remove(entryIDs: [entryID])
                    promptCache.remove(entryIDs: [entryID])
                    activePromptCacheEntryID = nil
                    return (promptIDs, .reset)
                }
            }
            activePromptCacheEntryID = entryID
            return (effective, .resume(cachedPromptTokens: cached))
        }
    }

    /// Restore the longest interior checkpoint this prompt can use.
    ///
    /// Candidates are tried longest first, so an edited last turn reuses the
    /// whole conversation prefix and never regenerates a prior turn. A
    /// content-addressed key means a one-token difference in the prefix is a
    /// miss, not a wrong restore; a corrupt or stale record is deleted and the
    /// next candidate (or full prefill) is used.
    func restoreInteriorCheckpoint(
        promptIDs: [Int32],
        messages: [GFTokenizer.Message]
    ) async -> Int? {
        guard promptCacheMode == .multiPrefix,
            let store = promptCheckpointStore,
            runner.supportsInferenceStateRestore
        else { return nil }
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: promptIDs,
            messages: messages,
            imStartID: tokenizer.imStartID,
            gridTokens: Self.checkpointGridTokens,
            includeMessageBoundaries: !Self.checkpointGridOnly
        ).sorted { $0.position > $1.position }
        // Interior checkpoints describe one conversation. Two conversations
        // can render a content-identical prefix (the system prompt is the
        // common case), but only the system-end boundary is shareable across
        // them; anything longer must belong to this conversation. The shared
        // path below handles that one boundary.
        let conversation = PromptCheckpointPlanner.conversationKey(
            domain: promptCacheDomain, messages: messages)
        for boundary in boundaries {
            let prefix = Array(promptIDs.prefix(boundary.position))
            let key = PromptCheckpointKey.make(
                domain: promptCacheDomain, prefixTokenIDs: prefix)
            do {
                // Read the small metadata before the payload: a record this
                // conversation cannot use is skipped without promoting it
                // into memory, so the shared path's tier still names disk.
                if let metadata = await store.metadataIfPresent(
                    key, domain: promptCacheDomain),
                    metadata.conversationKey != conversation
                {
                    continue
                }
                let loaded = try await store.load(
                    key, domain: promptCacheDomain)
                guard loaded.metadata.position == boundary.position,
                    loaded.metadata.prefixTokenIDs == prefix
                else {
                    throw PromptCheckpointStoreError.invalidMetadata(
                        "checkpoint position or prefix does not match the request")
                }
                guard loaded.metadata.conversationKey == conversation else {
                    // A valid record for another conversation, not a corrupt
                    // one: leave it for the shared system path (or for that
                    // conversation) instead of deleting it.
                    continue
                }
                try runner.restoreInferenceState(
                    loaded.snapshot, auxiliary: loaded.auxiliary)
                ServerLog.diagnostic(
                    "TinyTitan prompt_checkpoint hit kind=\(boundary.kind.rawValue) "
                        + "cached_tokens=\(boundary.position) "
                        + "state_bytes=\(loaded.snapshot.payload.count)")
                return boundary.position
            } catch PromptCheckpointStoreError.missing(_) {
                continue
            } catch {
                // A record that cannot be trusted is removed, and the next
                // candidate is tried. Falling back to a full prefill is the
                // only safe outcome; a partially restored runner was reset by
                // `restoreInferenceState`'s own failure path.
                await store.remove(key)
                ServerLog.diagnostic(
                    "TinyTitan prompt_checkpoint restore_failed "
                        + "tokens=\(boundary.position) error=\(error)")
            }
        }
        return nil
    }
}
