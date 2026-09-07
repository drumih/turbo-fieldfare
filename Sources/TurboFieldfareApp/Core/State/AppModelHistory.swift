import Foundation
import TurboFieldfare

/// What the stored copies of one turn's pictures came to.
///
/// The count of failures is the part that matters: the turn's recorded IDs
/// still hold each picture's placeholder run, so a picture with no stored copy
/// means a later replay would hand the model a span of image tokens with
/// nothing behind them, and the context would silently differ from the one the
/// answer came out of. Two `try?`s used to drop exactly that on the floor.
struct TurnImageWriteOutcome: Sendable {
    var records: [ConversationImageRecord] = []
    var failed = 0
    /// Why the first of them failed, in words that name no file: the count of
    /// lost pictures is a diagnostic, their names are the user's.
    var reason: String?
}

/// The shape of an image-write failure, without the path or the display name
/// the error itself carries.
private func imageWriteReason(_ error: Error) -> String {
    switch error {
    case ConversationImageWriterError.noMetalDevice:
        return "no Metal device is available to prepare them"
    case ConversationImageWriterError.modelInputUnavailable:
        return "preprocessing produced no model-input copy"
    case ConversationImageWriterError.encodeFailed:
        return "the copy could not be written into the conversation's folder"
    default:
        return "\(type(of: error))"
    }
}

/// The app's side of persisted conversations: writing the open chat as it is
/// used, listing what is on disk, and replaying one back into the KV.
///
/// The rule the whole file obeys is the one `AppConversation` already states —
/// the transcript equals the model's context. A turn is written when it is in
/// the KV and not before, so a stored conversation is always a sequence the
/// model actually saw, and replaying it cannot invent a turn.
extension AppModel {
    // MARK: - Store lifecycle

    func replaceConversationBinding(for modelDirectory: URL, reason: String) {
        guard settingsPersistenceEnabled else { return }
        let modelDirectory = modelDirectory.standardizedFileURL
        conversationBindingGeneration &+= 1

        let identity: ConversationIdentity?
        if AppModelInstallationProbe.status(at: modelDirectory) == .complete {
            do {
                identity = try conversationIdentityProvider(modelDirectory)
            } catch {
                identity = nil
                historyDiagnostic = "the installed model identity could not be read after \(reason): \(error)"
            }
        } else {
            identity = nil
            historyDiagnostic = nil
        }

        pendingTurnImageWrite?.cancel()
        pendingTurnImageWrite = nil
        persistenceTail?.cancel()
        persistenceTail = nil
        quarantinedPersistenceLineages.removeAll()
        storedConversationID = nil
        serviceEpoch = nil
        pendingServiceRecoveryConversationID = nil
        machine = ConversationScreenMachine()
        history.replaceEntries([])
        history.selection = pendingRestoredConversationID
        history.setUnreadableCount(0)
        history.setDiscardedLegacyTrashCount(0)
        history.setReadOnlyStore(false)
        conversationBinding = ConversationStoreBinding(
            modelDirectory: modelDirectory,
            identity: identity,
            generation: conversationBindingGeneration,
            storeProvider: conversationStoreProvider)
        activateConversationStore()
    }

    private func isCurrentConversationBinding(
        generation: UInt64, store: ConversationStore
    ) -> Bool {
        guard let binding = conversationBinding else { return false }
        return binding.generation == generation && binding.store === store
    }

    func activateConversationStore() {
        guard let binding = conversationBinding else { return }
        let store = binding.store
        let generation = binding.generation
        Task { [weak self] in
            do {
                try await store.activate()
                let discarded = await store.discardedLegacyTrashCount
                if discarded > 0,
                   self?.isCurrentConversationBinding(
                    generation: generation, store: store) == true {
                    self?.history.setDiscardedLegacyTrashCount(discarded)
                    self?.recordHistoryDiagnostic(
                        "removed \(discarded) conversation(s) an older build had "
                            + "left in .trash")
                }
            } catch {
                // A store that cannot be locked for an unknown reason stays
                // read-only rather than being written to anyway.
                if self?.isCurrentConversationBinding(
                    generation: generation, store: store) == true {
                    self?.recordHistoryDiagnostic("\(error)")
                }
            }
            await self?.refreshHistory(store: store, generation: generation)
        }
    }

    /// Re-reads the list. Cheap on purpose: it touches only `conversation.json`
    /// per directory, never a transcript.
    func refreshHistory() async {
        guard let binding = conversationBinding else { return }
        await refreshHistory(store: binding.store, generation: binding.generation)
    }

    private func refreshHistory(store: ConversationStore, generation: UInt64) async {
        guard isCurrentConversationBinding(generation: generation, store: store) else {
            return
        }
        let isReadOnly = await store.isReadOnly
        guard isCurrentConversationBinding(generation: generation, store: store) else {
            return
        }
        history.setReadOnlyStore(isReadOnly)
        let entries: [ConversationMeta]
        do {
            entries = try await store.list()
        } catch {
            guard isCurrentConversationBinding(generation: generation, store: store) else {
                return
            }
            // The last good list stays on screen. Emptied instead, a store that
            // could not be enumerated once — a permission change, a volume that
            // went away, a root that is now a file — looked exactly like a user
            // with no chats at all, and the window then offered New Chat over
            // conversations that were still there.
            recordHistoryDiagnostic(
                "the list of conversations could not be read, so the sidebar is "
                    + "showing the last one that could: \(error)")
            history.setUnreadableCount(max(history.unreadableCount, 1))
            return
        }
        guard isCurrentConversationBinding(generation: generation, store: store) else {
            return
        }
        let skipped = await store.skippedByLastList
        let reasons = await store.skipReasons
        guard isCurrentConversationBinding(generation: generation, store: store) else {
            return
        }
        history.setUnreadableCount(skipped.count)
        if !skipped.isEmpty {
            // Names and reasons, never content: a skipped conversation's
            // transcript is exactly what could not be read anyway.
            recordHistoryDiagnostic(
                "\(skipped.count) conversation(s) could not be read: "
                    + skipped.map {
                        "\($0.uuidString) (\(reasons[$0] ?? "no reason recorded"))"
                    }.joined(separator: ", "))
        }
        history.replaceEntries(entries)
        if let saved = pendingRestoredConversationID {
            pendingRestoredConversationID = nil
            if history.entry(saved) != nil {
                openConversation(id: saved)
            } else {
                history.selection = nil
                persistHistorySelection()
            }
        }
    }

    /// Says so when a conversation's `conversation.json` had to be rebuilt.
    ///
    /// The conversation survives it — the records are the conversation — but a
    /// cache that could not be read is a real fault of the store, and healing
    /// one silently leaves nobody able to say it ever happened.
    func recordMetaRebuild(_ opened: ConversationOpenResult) {
        guard let reason = opened.metaRebuildReason else { return }
        recordHistoryDiagnostic(
            "conversation \(opened.meta.id.uuidString) was rebuilt from its "
                + "records: \(reason)")
    }

    func recordHistoryDiagnostic(_ message: String) {
        // Names and counts only. A diagnostic that carried transcript text
        // would put the user's conversation into whatever collects it.
        historyDiagnostic = message
    }

    // MARK: - Writing the open chat

    /// The directory the open chat is written to, created on first use.
    ///
    /// Lazy so a New Chat the user never sends leaves nothing on disk: a
    /// sidebar full of empty rows is worse than no sidebar.
    func ensureStoredConversation(firstMessage: String) async -> UUID? {
        guard let binding = conversationBinding,
              let identity = binding.identity else {
            return nil
        }
        let store = binding.store
        guard !quarantinedPersistenceLineages.contains(where: {
            $0.epoch == conversation.epoch
                && $0.bindingGeneration == binding.generation
        }) else { return nil }
        // Read-only first, before the existing id: a store without the writer
        // lock must not be written to at all, and returning the id of a chat
        // another window owns had this one write image files into that
        // window's directory before the append itself was refused.
        guard await !store.isReadOnly else { return nil }
        if let existing = storedConversationID { return existing }
        // Storage may recover after an unsaved turn, but a new file cannot
        // stand in for the complete prefix still held by the live KV.
        guard conversation.committedTurns == 0 else {
            let reason = "Earlier exchanges in this chat were not saved. Start a new chat to resume saving."
            recordHistoryDiagnostic(reason)
            error = .conversationPersistenceFailed(reason)
            return nil
        }
        do {
            let meta = try await store.create(
                title: ConversationTitle.fromFirstMessage(firstMessage),
                identity: identity,
                session: ConversationSessionSettings(
                    contextTokens: maxContextTokens,
                    expertCacheSlots: runtimeOptions.expertCacheSlots,
                    visionResidencyPolicy: runtimeOptions.visionResidencyPolicy.rawValue),
                sampling: currentSampling())
            storedConversationID = meta.id
            history.selection = meta.id
            persistHistorySelection()
            await refreshHistory()
            return meta.id
        } catch {
            recordHistoryDiagnostic("\(error)")
            self.error = .conversationPersistenceFailed("\(error)")
            return nil
        }
    }

    func currentSampling() -> ConversationSampling {
        ConversationSampling(
            temperature: temperature, topKEnabled: topKEnabled, topK: topK,
            topPEnabled: topPEnabled, topP: topP,
            maxNewTokens: maxNewTokensOverride ?? effectiveMaxContextTokens)
    }

    /// Starts writing the stored copies of a turn's images while the model is
    /// generating.
    ///
    /// Off the main actor and overlapped with the reply on purpose: preparing a
    /// model-input copy is a full decode and resize, seconds per image on the
    /// 8 GB host, and the user is already waiting for the answer. A turn the
    /// runtime then rewinds leaves its files behind, which is what the store's
    /// orphan sweep is for.
    func beginStoringTurnImages(_ attachments: [StagedImage], in id: UUID) {
        pendingTurnImageWrite?.cancel()
        guard !attachments.isEmpty, let store = conversationStore else {
            pendingTurnImageWrite = nil
            return
        }
        // The writer reads links of its own. The turn's retained links belong
        // to the transcript, and New Chat or a replay of another chat releases
        // them the moment the reply lands — while the writer, seconds into a
        // decode and resize, still had the thumbnail to read from the same
        // path. That failed the write, which marked the turn's record as
        // missing a picture and refused the conversation every replay after.
        var links: [StagedImage] = []
        for attachment in attachments {
            do {
                links.append(try attachmentStore.retain(attachment))
            } catch {
                let failed = attachments.count
                let reason = imageWriteReason(error)
                for link in links { attachmentStore.remove(link) }
                pendingTurnImageWrite = Task {
                    TurnImageWriteOutcome(failed: failed, reason: reason)
                }
                return
            }
        }
        let taken = links
        let releasing = attachmentStore
        pendingTurnImageWrite = Task.detached(priority: .utility) {
            let directory = await store.imagesURL(for: id)
            return Self.writeStoredImages(taken, into: directory, releasing: releasing)
        }
    }

    /// Writes the stored copies of a turn's pictures from the writer's own
    /// links, and releases those links whatever came of it.
    nonisolated static func writeStoredImages(
        _ links: [StagedImage], into directory: URL,
        releasing attachmentStore: AppImageAttachmentStore
    ) -> TurnImageWriteOutcome {
        let outcome = writeStoredImages(links, into: directory)
        for link in links { attachmentStore.remove(link) }
        return outcome
    }

    /// Writes the stored copies of a turn's pictures, one outcome for all.
    nonisolated static func writeStoredImages(
        _ links: [StagedImage], into directory: URL
    ) -> TurnImageWriteOutcome {
        let writer: ConversationImageWriter
        do {
            writer = try ConversationImageWriter()
        } catch {
            // Every picture of the turn, not none of them. Discarded here,
            // each one vanished from the record while its placeholder
            // tokens stayed in the turn's IDs, and a later replay prefilled
            // that span with no pixels behind it.
            return TurnImageWriteOutcome(
                failed: links.count, reason: imageWriteReason(error))
        }
        var outcome = TurnImageWriteOutcome()
        for link in links {
            do {
                outcome.records.append(
                    try writer.write(attachment: link, into: directory))
            } catch {
                outcome.failed += 1
                if outcome.reason == nil {
                    outcome.reason = imageWriteReason(error)
                }
            }
        }
        return outcome
    }

    /// Waits for the turn writes already queued to reach disk.
    ///
    /// Quit and delete both need it: the last exchange's write starts when
    /// the reply lands and, with pictures, runs for seconds after the send
    /// pipeline has returned. Quitting under it lost the exchange; deleting
    /// under it left a headerless transcript behind.
    public func awaitPendingPersistence() async {
        await persistenceTail?.value
    }

    /// Whether the pending write finished before the termination deadline.
    public func awaitPendingPersistence(timeout: Duration) async -> Bool {
        guard !Task.isCancelled else { return false }
        guard let pending = persistenceTail else { return true }
        // A task group waits for every child even after cancelAll. Observing
        // an independent Task.value cannot be cancelled, so race notifications
        // instead and leave the write free to finish during termination.
        let (events, continuation) = AsyncStream<Bool>.makeStream(
            bufferingPolicy: .bufferingOldest(1))
        let completion = Task {
            await pending.value
            continuation.yield(true)
            continuation.finish()
        }
        let deadline = Task {
            do { try await Task.sleep(for: timeout) }
            // Sleep throws only on cancellation, including the cleanup below
            // when persistence wins the race.
            catch { return }
            continuation.yield(false)
            continuation.finish()
        }
        defer {
            completion.cancel()
            deadline.cancel()
            continuation.finish()
        }
        var iterator = events.makeAsyncIterator()
        return await iterator.next() ?? false
    }

    /// Writes a completed turn: the user's half, then the model's, then the
    /// small file the sidebar reads.
    ///
    /// Called only from the terminal path that commits the turn to
    /// `AppConversation`, so what lands on disk is exactly what the KV holds. A
    /// turn that threw is written nowhere, because the runtime rewound it.
    func enqueueCompletedTurnPersistence(userText: String,
                                         assistantText: String,
                                         diagnostics: AppDiagnostics?) {
        guard let binding = conversationBinding,
              let id = storedConversationID else { return }
        let operation = CompletedTurnPersistence(
            key: PersistenceLineageKey(
                conversationID: id,
                epoch: conversation.epoch,
                bindingGeneration: binding.generation),
            store: binding.store,
            userText: userText,
            assistantText: assistantText,
            diagnostics: diagnostics,
            sampling: currentSampling(),
            boundary: ConversationBoundary(
                tokens: conversation.boundaryTokenIDs,
                needsReplay: conversation.boundaryNeedsReplay),
            imageWrite: pendingTurnImageWrite)
        pendingTurnImageWrite = nil
        let prior = persistenceTail
        persistenceTail = Task { [weak self] in
            await withTaskCancellationHandler {
                await prior?.value
            } onCancel: {
                // The tail is the one binding replacement can reach. Carry
                // cancellation backward through the chain so an older write
                // cannot land in the model root that was just left behind.
                prior?.cancel()
            }
            guard !Task.isCancelled else {
                await self?.discardImages(for: operation)
                return
            }
            await self?.persistCompletedTurn(operation)
        }
    }

    private func persistCompletedTurn(_ operation: CompletedTurnPersistence) async {
        let store = operation.store
        let id = operation.key.conversationID
        guard isCurrentConversationBinding(
            generation: operation.key.bindingGeneration, store: store),
              !quarantinedPersistenceLineages.contains(operation.key) else {
            await discardImages(for: operation)
            return
        }
        let written = await operation.imageWrite?.value
        guard !Task.isCancelled,
              isCurrentConversationBinding(
                generation: operation.key.bindingGeneration, store: store),
              !quarantinedPersistenceLineages.contains(operation.key) else {
            await discardImages(for: operation)
            return
        }
        let images = written?.records ?? []
        let lostImages = written?.failed ?? 0
        let now = Date()
        do {
            // Both halves in one call: the meta is projected from the records
            // once, at the end, so the sidebar never sees a conversation with a
            // question and no answer in it.
            let meta = try await store.append([
                .turn(ConversationTurnRecord(
                    role: .user, at: now, text: operation.userText, images: images,
                    tokens: operation.diagnostics?.promptTokenIDs,
                    promptTokens: operation.diagnostics?.promptTokenCount,
                    cachedTokens: operation.diagnostics?.cachedPromptTokens,
                    sampling: operation.sampling,
                    // Recorded on the turn that cites them, so the projection
                    // can refuse the whole conversation a replay and the row
                    // can say why instead of failing on its first turn back.
                    imageWriteFailed: lostImages > 0 ? true : nil)),
                .turn(ConversationTurnRecord(
                    role: .assistant, at: now, text: operation.assistantText,
                    tokens: operation.diagnostics?.generatedTokenIDs,
                    generatedTokens: operation.diagnostics?.generatedTokens,
                    stopReason: operation.diagnostics.map {
                        String(describing: $0.stopReason)
                    },
                    boundary: operation.boundary)),
            ], to: id)
            // The runtime's count and the record's are two answers to one
            // question, and they used to be reconciled by preferring the
            // runtime's and writing it into the meta by hand — which is how a
            // row came to read 1,065 tokens over 22 tokens of record. The
            // record wins now, because it is what a replay actually puts back;
            // a disagreement is a fault worth seeing rather than papering over.
            if let reported = operation.diagnostics?.conversationTokens,
               reported != meta.kvTokens {
                recordHistoryDiagnostic(
                    "the runtime reports \(reported) token(s) in this "
                        + "conversation and its records hold "
                        + "\(meta.kvTokens.map(String.init) ?? "no countable"): "
                        + "the record decides")
            }
        } catch {
            quarantinedPersistenceLineages.insert(operation.key)
            if isCurrentConversationBinding(
                generation: operation.key.bindingGeneration, store: store) {
                recordHistoryDiagnostic("\(error)")
                if storedConversationID == id,
                   conversation.epoch == operation.key.epoch {
                    self.error = .conversationPersistenceFailed("\(error)")
                    storedConversationID = nil
                    history.selection = nil
                    persistHistorySelection()
                }
            }
        }
        // Last, so it is the diagnostic on screen: a conversation that can no
        // longer be reopened is worth more than a count that disagreed.
        if lostImages > 0 {
            recordHistoryDiagnostic(
                "\(lostImages) image(s) in this turn have no stored copy, so this "
                    + "conversation cannot be put back into the model's context: "
                    + (written?.reason ?? "no reason recorded"))
        }
        await refreshHistory(store: store, generation: operation.key.bindingGeneration)
    }

    private func discardImages(for operation: CompletedTurnPersistence) async {
        operation.imageWrite?.cancel()
        _ = await operation.imageWrite?.value
        do {
            _ = try await operation.store.sweepOrphanImages(
                in: operation.key.conversationID)
        } catch {
            if isCurrentConversationBinding(
                generation: operation.key.bindingGeneration,
                store: operation.store) {
                let original = historyDiagnostic ?? "the lineage is quarantined"
                recordHistoryDiagnostic(
                    original + "; its orphan images could not be collected: \(error)")
            }
        }
    }

    // MARK: - Reopening

    /// The state a stored row is in right now, against the context in force.
    public func continuability(of meta: ConversationMeta) -> ConversationContinuability {
        guard let identity = conversationIdentity else {
            return .cannotReplay(reason: .differentModel)
        }
        return ConversationContinuability.evaluate(
            meta: meta, currentContext: maxContextTokens, identity: identity)
    }

    /// Opens a stored conversation and shows it.
    ///
    /// Reading a conversation and being able to continue it are two different
    /// costs, and this only pays the first. The transcript is a file, so it
    /// appears at once; the model's context is a full prefill of every token
    /// the conversation holds — tens of seconds at 8K — and nothing needs it
    /// until the next message. So the replay is deferred to the first send,
    /// where the user is already waiting for the model and the cost is part of
    /// a turn rather than the price of a click.
    ///
    /// A conversation that could not continue anyway is never replayed at all;
    /// the row says why instead.
    public func openConversation(id: UUID) {
        guard !isTurnInFlight, conversationStore != nil else { return }
        guard let meta = history.entry(id) else { return }
        if id == storedConversationID,
           serviceEpoch == conversation.epoch,
           continuability(of: meta) == .continuable {
            applySampling(meta.sampling)
        }
        // Reading copies only. The live conversation is left exactly as it is —
        // it is what the KV holds, and browsing must not drop it — so what is on
        // screen becomes the stored copy of the row that was clicked, and
        // `isShowingStoredCopy` keeps the live turn from being drawn under it.
        perform(machine.apply(.rowClicked(
            id: id,
            heldID: storedConversationID,
            kvMatchesHeld: serviceEpoch == conversation.epoch,
            state: continuability(of: meta),
            renderID: UUID())))
    }

    /// Reads a stored conversation off disk and puts it on screen.
    ///
    /// One loader for both the chat that can be continued and the one that
    /// cannot: they used to be two, which is how a failure reported in one of
    /// them and not the other left a row selected over the ordinary empty state
    /// — the thing an unwritten chat looks like.
    private func loadDocument(id: UUID, renderID: UUID) {
        guard let binding = conversationBinding else { return }
        let store = binding.store
        let generation = binding.generation
        let samplingAtStart = currentSampling()
        // The previous read's failure is not this row's. Cleared here rather
        // than as its own effect: a load is the only thing that can replace it.
        error = nil
        Task { [weak self] in
            guard let self else { return }
            let opened: ConversationOpenResult
            do {
                opened = try await store.open(id: id)
            } catch {
                guard self.isCurrentConversationBinding(
                    generation: generation, store: store) else { return }
                guard screen.conversationID == id,
                      screen.documentReadID == renderID else { return }
                recordHistoryDiagnostic("\(error)")
                perform(machine.apply(.documentFailed(
                    id: id, renderID: renderID, .conversationUnreadable("\(error)"))))
                return
            }
            guard self.isCurrentConversationBinding(
                generation: generation, store: store) else { return }
            let directory = await store.directoryURL(for: id)
            guard isCurrentConversationBinding(generation: generation, store: store),
                  screen.conversationID == id,
                  screen.documentReadID == renderID else { return }
            recordMetaRebuild(opened)
            history.replaceEntryAndResort(opened.meta)
            let effects = machine.apply(.documentLoaded(
                id: id, renderID: renderID,
                document: ConversationDocument.load(opened, directory: directory),
                state: continuability(of: opened.meta)))
            // Opening restores defaults, but a later edit belongs to the user.
            perform(effects.filter { effect in
                if case .applySampling = effect {
                    return currentSampling() == samplingAtStart
                }
                return true
            })
        }
    }

    /// Reads a stored conversation off disk and puts its tokens back into the
    /// KV, ahead of the turn that asked for it. Reached from a send, never from
    /// a click.
    ///
    /// Returns what happened rather than applying it: the send pipeline is
    /// holding the message this replay is for, and it is the only thing
    /// entitled to decide what happens to it next. Applying the outcome here
    /// and letting the effect hand the message onward is what made the send
    /// path call itself.
    func replayOutcome(id: UUID) async -> ConversationScreenMachine.Event {
        guard let binding = conversationBinding,
              let lifecycle = client as? AppModelLifecycleClient else {
            // Refused rather than returned as a success: the send is waiting on
            // this, and handing it back as if the conversation were in the KV
            // put the send straight back into the same branch, forever.
            let refusal = AppInferenceError.conversationRestoreFailed(
                "this build cannot restore a conversation into the model's context")
            return .replayRefused(id: id, error: refusal)
        }
        let store = binding.store
        let generation = binding.generation
        let document: ConversationDocument
        do {
            let opened = try await store.open(id: id)
            guard isCurrentConversationBinding(
                generation: generation, store: store) else {
                return .replayNotStarted(id: id, error: .conversationRestoreFailed(
                    "the model location changed while the conversation was being read"))
            }
            recordMetaRebuild(opened)
            // The row's answer, taken again from the record just read. The
            // click took it from the cached meta, which a crash between a
            // turn's append and its cache write leaves a turn behind; a send
            // before the click's own read lands would otherwise ask the
            // service to drop the held chat for a restore the context then
            // refuses.
            let fresh = continuability(of: opened.meta)
            switch fresh {
            case .continuable:
                break
            case .needsContext(let required):
                let reserve: Int = ConversationGenerationReserve.tokens
                let prompt: Int = required - reserve
                let refusal = AppInferenceError.conversationContextExhausted(
                    prompt: prompt, reserve: reserve, maxContext: maxContextTokens)
                return .replayRefused(id: id, error: refusal, state: fresh)
            case .cannotReplay(let reason):
                let refusal = AppInferenceError.conversationRestoreFailed(
                    "the stored record can no longer be replayed: \(reason)")
                return .replayRefused(id: id, error: refusal, state: fresh)
            }
            document = ConversationDocument.load(
                opened, directory: await store.directoryURL(for: id))
            if document.turns.contains(where: { !$0.images.isEmpty }) {
                recheckVisionPackAtCurrentLocation()
                guard isVisionPackInstalled else {
                    let directory = URL(
                        fileURLWithPath: modelPathText, isDirectory: true)
                    let location: String
                    do {
                        location = try VisionPackLocation.companionURL(
                            forTextModel: directory).path
                    } catch {
                        location = "an unresolved companion path (\(error))"
                    }
                    // Not `.replayFailed`: the service has not been asked for
                    // anything, so the held conversation is still in the KV
                    // and must not be ended for a restore that never ran.
                    return .replayNotStarted(id: id, error: .conversationRestoreFailed(
                        "image support is unavailable at \(location)"))
                }
            }
        } catch {
            // The transcript could not be read at all, which is a different
            // failure from one that reads and cannot be replayed — and a
            // permanent property of what is on disk either way.
            let refusal: AppInferenceError = (error as? AppInferenceError)
                ?? AppInferenceError.conversationRestoreFailed("\(error)")
            return .replayRefused(id: id, error: refusal)
        }
        let lineage: AppConversationLineage
        switch document.lineage {
        case .success(let value):
            lineage = value
        case .failure(let recordError):
            // The record itself cannot be replayed — a turn with no tokens, an
            // image whose placeholders are not where it says. That is a
            // permanent property of what is on disk, so the row says so.
            let refusal = AppInferenceError.conversationRestoreFailed(
                recordError.description)
            return .replayRefused(id: id, error: refusal)
        }
        do {
            let epoch = UUID()
            let kvTokens = try await lifecycle.restoreConversation(
                lineage, epoch: epoch, options: runtimeOptions,
                maxContextTokens: maxContextTokens
            ) { [weak self] done, total in
                Task { @MainActor in self?.applyRestoreProgress(done: done, total: total) }
            }
            // The KV is only given up here, which is why the machine emits the
            // release with this event and not at the click. Released before the
            // replay instead, a restore that then failed left the held
            // conversation on screen pointing at pictures that had already been
            // deleted off disk.
            return .replaySucceeded(
                id: id, document: document, epoch: epoch, kvTokens: kvTokens)
        } catch let restoreError {
            // The record was fine; putting it back failed. That can be the
            // service refusing a digest, the service being gone, a load in
            // flight — none of them a property of the conversation, and all of
            // them fixable. Marking the row as having no replay record said the
            // opposite, and pointed at "re-read into a new chat" when what was
            // needed was the service back.
            let appError = (restoreError as? AppInferenceError)
                ?? .conversationRestoreFailed("\(restoreError)")
            if case .connectionLost = appError {
                // The transport has already dropped the connection, so this is
                // the same loss a generation reports, and it has to leave the
                // same state: a model that needs reloading. Folded into an
                // ordinary failed replay, the window kept advertising a ready
                // model with no Retry Load, and every later send failed on a
                // connection that was already gone.
                recordLostServiceConnection(appError)
                // So Retry Load keeps this row on screen for the send that
                // follows, rather than archiving it as a released KV would.
                pendingServiceRecoveryConversationID = id
            }
            return .replayFailed(id: id, error: appError)
        }
    }

    private func adoptRestoredConversation(document: ConversationDocument,
                                           epoch: UUID,
                                           kvTokens: Int) {
        // No image release here. This is only ever reached from a deferred
        // replay, and by then `openConversation` has already released whatever
        // the window was showing — so the only images left are the ones the
        // turn about to be sent is holding. Releasing those deleted the staged
        // files out from under the send: the turn came back "could not retain
        // <name>: errno 2" and was refused.
        conversation.adoptRestored(
            epoch: epoch,
            turns: document.turns,
            kvTokens: kvTokens,
            boundaryTokenIDs: document.meta.boundary.tokens,
            boundaryNeedsReplay: document.meta.boundary.needsReplay)
        storedConversationID = document.id
        serviceEpoch = epoch
        quarantinedPersistenceLineages.removeAll()
        // The live fields are left exactly as they are. They were seeded here
        // with the restored conversation's newest pair back when a reopen
        // replayed on the click and the window then sat at rest showing it.
        // A replay now only ever runs inside a send, so what they hold is the
        // message that started it — overwriting that dropped the user's prompt
        // and deleted the images it was carrying. The newest stored pair is
        // drawn from the history instead: `transcriptHistory` only holds a pair
        // back while no turn is in flight, and by the time this returns one is
        // about to be.
        diagnostics = nil
        error = nil
        livePrefillDone = 0
        livePrefillTotal = 0
        // Back out of the prefill phase the click put the window into, or the
        // gauge keeps claiming a replay that has already finished.
        phase = .idle
    }

    /// The staged copies belonging to the conversation the KV is about to stop
    /// holding.
    ///
    /// The only place the window's own pictures are released. Ten call sites
    /// used to do this, each with its own idea of when it applied, and one of
    /// them deleting a stored conversation's files is the defect the two image
    /// types now make impossible — a `StoredImage` cannot reach `remove` at all.
    ///
    /// `includingLiveTurn` is false after a replay: by then the live fields
    /// hold the message that started it, and the send is about to hard-link
    /// exactly those files. New Chat takes them too, because nothing is left
    /// drawing any of it.
    private func releaseImagesOfTheConversationBeingLeft(includingLiveTurn: Bool) {
        for pair in conversation.outOfContextPairs {
            for image in pair.user.images {
                if let staged = image.staged { attachmentStore.remove(staged) }
            }
        }
        for turn in conversation.turns {
            for image in turn.images {
                if let staged = image.staged { attachmentStore.remove(staged) }
            }
        }
        guard includingLiveTurn else { return }
        for image in outputImageAttachments {
            if let staged = image.staged { attachmentStore.remove(staged) }
        }
        outputImageAttachments = []
    }

    private func applySampling(_ sampling: ConversationSampling) {
        temperature = sampling.temperature
        topKEnabled = sampling.topKEnabled
        topK = sampling.topK
        topPEnabled = sampling.topPEnabled
        topP = sampling.topP
        // The public app budgets responses automatically. A recorded request
        // limit must not become a fixed cap when a saved chat is opened.
    }

    private func applyRestoreProgress(done: Int, total: Int) {
        phase = .prefill
        livePrefillDone = done
        livePrefillTotal = total
    }

    // MARK: - Performing what a transition asks for

    /// The one place a screen transition turns into work.
    ///
    /// Ten `attachmentStore.remove` sites, two loaders, two hand-backs and four
    /// assignments to `history.selection` used to do this, each with its own
    /// idea of when it applied. A transition that forgets one of them is now a
    /// missing effect in a table test rather than a defect nobody can see.
    func perform(_ effects: [ConversationScreenMachine.Effect]) {
        for effect in effects {
            switch effect {
            case .select(let id):
                history.selection = id
                persistHistorySelection()
            case .dropOutOfContextTurns:
                conversation.dropOutOfContextPairs()
            case .loadDocument(let id, let renderID):
                loadDocument(id: id, renderID: renderID)
            case .applySampling(let sampling):
                // A send may be queued before its replay screen is installed.
                // Its settings cannot be replaced by the click's late read.
                if !isTurnInFlight { applySampling(sampling) }
            case .reportError(let failure):
                error = failure
            case .replay, .startTurn:
                // Read by the send pipeline, which is the thing holding the
                // message: `.replay` is a stage it awaits and `.startTurn` is
                // its permission to go on. Performing either here is what made
                // the send path call itself.
                continue
            case .releaseImagesOfHeldConversation(let includingLiveTurn):
                releaseImagesOfTheConversationBeingLeft(
                    includingLiveTurn: includingLiveTurn)
            case .endFailedReplayLineage:
                endLineageAfterFailedReplay()
            case .adoptRestored(let document, let epoch, let kvTokens):
                adoptRestoredConversation(
                    document: document, epoch: epoch, kvTokens: kvTokens)
            case .restoreComposer(let pending, let failure):
                // Withdrawn from the transcript as well: what is drawn there is
                // this very message under a prefill placeholder, with nothing
                // underneath it, because the replay it was waiting for never
                // landed.
                restoreComposer(pending, error: failure,
                                withdrawingFromTranscript: true)
            }
        }
    }

    private func persistHistorySelection() {
        persistSettings()
    }

    /// The held chat's turns move out of context and keep their pictures:
    /// they are still drawn there, and New Chat releases them along with the
    /// rest. Releasing them here first carried pairs whose thumbnails pointed
    /// at files that no longer existed.
    private func endLineageAfterFailedReplay() {
        let carried = conversation.outOfContextPairs + conversation.completedPairs
        conversation.startNew(carryingOutOfContext: carried)
        storedConversationID = nil
        serviceEpoch = nil
    }

    func prepareServiceRecoveryForReplay(includingLiveTurn: Bool = false) {
        releaseImagesOfTheConversationBeingLeft(includingLiveTurn: includingLiveTurn)
        conversation.startNew(carryingOutOfContext: [])
        serviceEpoch = nil
        outputPromptText = ""
        outputText = ""
        outputImageAttachments = []
        generationTranscriptMailbox?.reset()
    }

    // MARK: - Rename, delete, undo

    public func renameConversation(id: UUID, to title: String) {
        guard canMutateConversation(id: id), let store = conversationStore else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        Task { [weak self] in
            do { try await store.rename(id: id, to: trimmed) }
            catch { self?.recordHistoryDiagnostic("\(error)") }
            await self?.refreshHistory()
        }
    }

    /// Collects the image files of a turn the runtime rewound.
    ///
    /// A turn's images are written while the reply is still generating, so a
    /// turn that never reaches the KV leaves them behind — recorded nowhere,
    /// named by digest, and invisible to every other pass. `sweepOrphanImages`
    /// existed for exactly this and had no caller, so those bytes stayed for the
    /// life of the conversation.
    ///
    /// Only after a rewind, never after a completed turn: the sweep parses the
    /// whole transcript, and doing that on every turn would make each one cost
    /// more as the conversation grows.
    func sweepImagesOfRewoundTurn() async {
        guard let store = conversationStore, let id = storedConversationID else {
            return
        }
        do {
            let removed = try await store.sweepOrphanImages(in: id)
            if removed > 0 {
                recordHistoryDiagnostic(
                    "collected \(removed) image file(s) from a turn that was "
                        + "rewound")
            }
        } catch {
            recordHistoryDiagnostic("\(error)")
        }
    }

    /// Ends the image write a rewound turn started.
    ///
    /// Only `persistCompletedTurn` used to clear `pendingTurnImageWrite`, which
    /// a cancelled or failed turn never reaches — so the app read as
    /// permanently mid-write to anything that asked whether a turn was
    /// finished, and the orphan sweep that follows a rewind raced the very
    /// write it exists to collect after. Awaited rather than only cancelled:
    /// a cancelled `Task` is not a finished one, and the sweep deletes every
    /// file no record names.
    func discardPendingTurnImages() async {
        guard let write = pendingTurnImageWrite else { return }
        pendingTurnImageWrite = nil
        write.cancel()
        _ = await write.value
    }

    /// The folder a stored conversation lives in, for opening in Finder.
    ///
    /// The folder rather than a file inside it: what is worth looking at is the
    /// transcript next to its images and its meta, and a directory shows all
    /// three. Nil is a real answer here — a row can outlive its directory if
    /// something outside the app moved it — so it is recorded rather than
    /// returned silently.
    public func conversationFolderURL(for id: UUID) async -> URL? {
        guard let store = conversationStore else {
            recordHistoryDiagnostic("no conversation store is open")
            return nil
        }
        let url = await store.directoryURL(for: id)
        guard FileManager.default.fileExists(atPath: url.path) else {
            recordHistoryDiagnostic("conversation \(id) has no folder on disk")
            return nil
        }
        return url
    }

    /// Deletes a conversation and everything in it, permanently.
    ///
    /// There is no undo, because there is no copy: the alternative was a
    /// `.trash` directory that kept every deleted transcript and image for the
    /// life of the store with nothing to collect them. The row's menu asks
    /// first — an alert for something irreversible is the case the guidelines
    /// make an exception for.
    public func deleteConversation(id: UUID) {
        guard canMutateConversation(id: id), let binding = conversationBinding else { return }
        let store = binding.store
        let generation = binding.generation
        conversationDeletionTask = Task { [weak self] in
            guard let self else { return }
            defer { conversationDeletionTask = nil }
            // The last turn's write may still be running: its image copies
            // are made after the reply lands, and `isTurnInFlight` is already
            // false by then. Deleting under it had the writer recreate the
            // directory and the append leave a headerless transcript behind,
            // which the list then counted as unreadable forever.
            await awaitPendingPersistence()
            guard isCurrentConversationBinding(generation: generation, store: store) else {
                return
            }
            do {
                try await store.delete(id: id)
                guard isCurrentConversationBinding(generation: generation, store: store) else {
                    return
                }
                if storedConversationID == id {
                    // The open chat was the one deleted, so its lineage ends
                    // here. Keeping the KV and letting the next turn open a
                    // fresh file broke the rule the whole design rests on —
                    // the transcript equals the model's context. The new file
                    // held one exchange while its meta described the whole
                    // conversation behind it: a row reading "1,065 tokens" over
                    // 22 tokens of record, which the sidebar believes, the
                    // context gauge believes, and continuability decides
                    // against. Deleting a conversation ends it.
                    if let viewed = screen.conversationID, viewed != id {
                        // Only the deleted lineage ends. The reader and its
                        // composer belong to a different, still-durable chat.
                        prepareServiceRecoveryForReplay(includingLiveTurn: true)
                        storedConversationID = nil
                        pendingServiceRecoveryConversationID = nil
                        quarantinedPersistenceLineages.removeAll()
                        diagnostics = nil
                    } else {
                        newChat()
                    }
                }
                // If the chat being read is the one deleted, what comes back on
                // screen is the chat the KV is still holding, and that is the
                // row the sidebar highlights: nil left the window showing a
                // conversation whose row was not selected, which is the one
                // state every other path here is careful not to produce. Never
                // the row just deleted, which `newChat` above has usually
                // already cleared.
                let remainingHeldID = storedConversationID == id
                    ? nil : storedConversationID
                let deletedViewedRow = screen.conversationID == id
                perform(machine.apply(.deleted(id: id, heldID: remainingHeldID)))
                // The machine's deleted transition returns to a live held row.
                // After an unload there is still a held durable row but no KV;
                // return to its stored copy instead so the next send replays it.
                if deletedViewedRow, let remainingHeldID,
                   serviceEpoch != conversation.epoch {
                    openConversation(id: remainingHeldID)
                }
            } catch {
                guard isCurrentConversationBinding(generation: generation, store: store) else {
                    return
                }
                recordHistoryDiagnostic("\(error)")
            }
            await refreshHistory(store: store, generation: generation)
        }
    }

    public func canMutateConversation(id: UUID) -> Bool {
        conversationStore != nil && history.entry(id) != nil
            && !history.isReadOnlyStore && !isTurnInFlight
            && conversationDeletionTask == nil
    }

    /// Puts a conversation the app cannot replay into the composer as one
    /// message, to be read fresh.
    ///
    /// A different lineage, and labelled as one everywhere it appears: turn one
    /// renders through the chat template, which strips historical thought spans
    /// and trims content, so the model sees a summary of the conversation and
    /// not the conversation. Presenting it as "Continue" is the one thing this
    /// action must never do.
    /// Whether there is anything to re-read.
    ///
    /// The prompt is built from the turns on screen, so a conversation that
    /// could not be read at all has nothing to build from. Offered anyway, the
    /// button started a new chat with an empty composer, which looks like the
    /// action failing silently.
    public var canRereadIntoNewChat: Bool {
        guard case .reading(_, let document, _, _) = screen else { return false }
        return canStartNewChat && !(document?.pairs.isEmpty ?? true)
    }

    public func rereadIntoNewChat() {
        guard canRereadIntoNewChat,
              let document = screen.document else { return }
        let text = Self.rereadPrompt(from: document.pairs)
        newChat()
        promptText = text
    }

    static func rereadPrompt(
        from pairs: [(user: AppChatTurn, assistant: AppChatTurn)]
    ) -> String {
        pairs.map { pair in
            "You: \(pair.user.text)\n\nAssistant: \(pair.assistant.text)"
        }.joined(separator: "\n\n")
    }
}
