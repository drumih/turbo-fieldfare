import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareDecodeProtocol

@testable import TurboFieldfareAppCore

final class FakeInferenceClient: AppModelLifecycleClient, Sendable {
    private struct State: Sendable {
        var loadedKey: AppLoadedRuntimeKey?
        var conversationEpochs: [UUID] = []
        var resetFailure: AppInferenceError?
        var stopRequested = false
        var restoredLineages: [AppConversationLineage] = []
        var restoreFailure: AppInferenceError?
        var generationFailure: AppInferenceError?
        /// What the KV would hold, so a stored conversation records a count.
        var conversationTokens = 0
        /// The service's own admission rule, running here rather than being
        /// restated. The app's agreement with it is the one thing no app-side
        /// test could observe while this double ignored the ticket entirely:
        /// every turn was admitted, so a stale epoch or a skipped position read
        /// exactly like a correct one.
        var gate = DecodeConversationGate()
        /// Every turn the gate refused, in order. A correct app produces none.
        var gateRejections: [DecodeConversationGate.Rejection] = []
        /// How many placeholder tokens one attached image contributes to a
        /// turn's recorded IDs. Zero until a caller measures a real
        /// preprocessed image, because replay locates an image by finding its
        /// placeholder run inside the turn that cites it.
        var imageSoftTokens = 0
    }

    enum CancelSemantics: Sendable { case hardAbort, cooperativeStop }

    private final class StateBox: Sendable {
        let value = Mutex(State())
    }

    private let state = StateBox()
    private let generationTasks = GenerationTaskRegistry()
    private let eventDelay: Duration
    private let cancelSemantics: CancelSemantics

    init(eventDelay: Duration = .milliseconds(80),
         cancelSemantics: CancelSemantics = .hardAbort) {
        self.eventDelay = eventDelay
        self.cancelSemantics = cancelSemantics
    }

    func failNextReset(with error: AppInferenceError) {
        state.value.withLock { $0.resetFailure = error }
    }

    func ensureLoaded(modelDirectory: URL,
                      maxContextTokens: Int,
                      options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        let start = Date()
        onState(.loading(.validatingDirectory))
        try await Task.sleep(for: eventDelay)
        onState(.loading(.preparingRunner))
        try await Task.sleep(for: eventDelay)
        try Task.checkCancellation()
        state.value.withLock {
            $0.loadedKey = AppLoadedRuntimeKey(modelDirectory: modelDirectory,
                                               maxContextTokens: maxContextTokens,
                                               options: options,
                                               forceLogitsHead: forceLogitsHead)
            // A load builds a new runner and an empty KV, so whatever epoch was
            // open names tokens that no longer exist. The service ends the
            // lineage here for exactly that reason, and an app that resumed
            // onto the rebuilt cache would have every turn refused from then on.
            $0.gate.endLineage()
            $0.conversationTokens = 0
        }
        onState(.ready(
            modelDirectory: modelDirectory.standardizedFileURL,
            loadSeconds: Date().timeIntervalSince(start)))
    }

    func unload() async {
        cancel()
        state.value.withLock {
            $0.loadedKey = nil
            // The runner and its KV are gone, so the epoch names tokens that no
            // longer exist. The service ends the lineage here for that reason.
            $0.gate.endLineage()
        }
    }

    /// What the gate holds right now, so a test can assert the app and the
    /// service agree about which conversation is open and how far into it.
    var gateOpenEpoch: UUID? { state.value.withLock { $0.gate.openEpoch } }
    var gateCommittedTurns: Int { state.value.withLock { $0.gate.committedTurns } }
    var gateRejections: [DecodeConversationGate.Rejection] {
        state.value.withLock { $0.gateRejections }
    }

    /// Whether a generation is still running. The app clears `isRunning` on the
    /// terminal event, which the stream yields before it finishes, so the two
    /// are not the same question.
    var isGenerating: Bool { generationTasks.hasActiveGeneration }

    /// Moves the loaded session's context without ending the lineage.
    ///
    /// The window requires a reload after a context change and the walk draws
    /// that reload as its own operation, so this is only the double keeping up
    /// with a setting the app has already applied. `ensureLoaded` ends the
    /// lineage the way a real load does, and calling it here would take the KV
    /// away from a conversation the window is still holding.
    func setLoadedContext(_ tokens: Int) {
        state.value.withLock { $0.loadedKey?.maxContextTokens = tokens }
    }

    /// Makes an attached image cost `tokens` placeholder IDs in the turn that
    /// carries it, the way a rendered prompt does.
    ///
    /// Replay finds an image by locating its placeholder run inside the turn
    /// that cites it, so a stored conversation with pictures cannot be replayed
    /// at all unless the recorded IDs actually contain those runs. The count
    /// has to come from a real preprocessed image, because it is what the
    /// stored record claims.
    func setImageSoftTokens(_ tokens: Int) {
        state.value.withLock { $0.imageSoftTokens = tokens }
    }

    var conversationEpochs: [UUID] {
        state.value.withLock { $0.conversationEpochs }
    }

    func resetConversation(epoch: UUID) async throws {
        if let failure = state.value.withLock({ value -> AppInferenceError? in
            defer { value.resetFailure = nil }
            return value.resetFailure
        }) {
            throw failure
        }
        state.value.withLock {
            $0.conversationEpochs.append(epoch)
            $0.conversationTokens = 0
            $0.gate.reset(to: epoch)
        }
    }

    /// Every lineage the app asked to replay, so a test can assert it sent the
    /// stored IDs rather than a re-render of the transcript.
    var restoredLineages: [AppConversationLineage] {
        state.value.withLock { $0.restoredLineages }
    }

    /// Makes the next `restoreConversation` fail, as a digest mismatch or a
    /// dead connection does.
    func failNextRestore(with error: AppInferenceError) {
        state.value.withLock { $0.restoreFailure = error }
    }

    func failNextGeneration(with error: AppInferenceError) {
        state.value.withLock { $0.generationFailure = error }
    }

    func restoreConversation(
        _ lineage: AppConversationLineage,
        epoch: UUID,
        options: AppRuntimeOptions,
        maxContextTokens: Int,
        onPrefillProgress: @escaping @Sendable (Int, Int) -> Void
    ) async throws -> Int {
        if let failure = state.value.withLock({ value -> AppInferenceError? in
            defer { value.restoreFailure = nil }
            guard let failure = value.restoreFailure else { return nil }
            // Restore replaces the old KV before later work can fail. Model
            // that destructive ordering so app tests cannot accidentally keep
            // taking the former held-row fast path after a failed replay.
            value.gate.endLineage()
            value.conversationTokens = 0
            return failure
        }) {
            throw failure
        }
        // A real replay is a full prefill, so the double takes visible time
        // too: without it no test can observe the window during which a reopen
        // is in flight.
        try? await Task.sleep(for: eventDelay)
        onPrefillProgress(lineage.tokenIDs.count, lineage.tokenIDs.count)
        state.value.withLock {
            $0.restoredLineages.append(lineage)
            $0.conversationEpochs.append(epoch)
            $0.conversationTokens = lineage.tokenIDs.count
            // Opened at the turn count the transcript carries, not at zero: a
            // reopened conversation's next turn is numbered against what is
            // already in the KV.
            $0.gate.restore(to: epoch, committedTurns: lineage.committedTurns)
        }
        return lineage.tokenIDs.count
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            do {
                try request.validate(requireModelDirectory: false)
                let expected = AppLoadedRuntimeKey(
                    modelDirectory: request.modelDirectory,
                    maxContextTokens: request.maxContextTokens,
                    options: request.runtimeOptions,
                    forceLogitsHead: !request.isPureGreedy)
                guard let loaded = state.value.withLock({ $0.loadedKey }) else {
                    throw AppInferenceError.modelNotLoaded
                }
                guard loaded == expected else { throw AppInferenceError.reloadRequired }
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }

            let admission: DecodeConversationGate.Admission
            switch state.value.withLock({ $0.gate.admit(Self.ticket(for: request)) }) {
            case .success(let value):
                admission = value
            case .failure(let rejection):
                state.value.withLock { $0.gateRejections.append(rejection) }
                // The turn was composed against a conversation the KV is no
                // longer holding, so retrying it would fail identically. That
                // is what `conversationLineageLost` means to the app, and what
                // `DecodeServiceInferenceClient` turns a `.lineageLost` event
                // into.
                let error = AppInferenceError.conversationLineageLost(
                    String(describing: rejection))
                continuation.yield(.failed(error, partial: nil))
                continuation.finish(throwing: error)
                return
            }

            let id = UUID()
            guard generationTasks.reserve(id) else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                await streamResponse(request: request,
                                     continuation: continuation,
                                     generationID: id,
                                     admission: admission)
            }
            generationTasks.attach(task, to: id)

            continuation.onTermination = { [generationTasks] _ in
                generationTasks.take(id)?.cancel()
            }
        }
    }

    func cancel() {
        switch cancelSemantics {
        case .hardAbort:
            generationTasks.takeCurrent()?.cancel()
        case .cooperativeStop:
            state.value.withLock { $0.stopRequested = true }
        }
    }

    /// The wire request the gate actually inspects, built from the app's own
    /// ticket. Only the three fields the decision reads are carried across.
    private static func ticket(for request: AppGenerationRequest) -> DecodeGenerationRequest {
        DecodeGenerationRequest(
            prompt: request.prompt,
            maxNewTokens: request.maxNewTokens,
            maxContextTokens: request.maxContextTokens,
            temperature: request.temperature,
            conversationEpoch: request.conversationEpoch,
            turnIndex: request.turnIndex)
    }

    private func streamResponse(
        request: AppGenerationRequest,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        generationID: UUID,
        admission: DecodeConversationGate.Admission
    ) async {
        let start = Date()
        var firstTokenDate: Date?
        var prefillEnd = start
        var generated = 0
        var stoppedEarly = false
        state.value.withLock { $0.stopRequested = false }

        do {
            if let failure = state.value.withLock({ value -> AppInferenceError? in
                defer { value.generationFailure = nil }
                guard let failure = value.generationFailure else { return nil }
                value.loadedKey = nil
                value.gate.endLineage()
                value.conversationTokens = 0
                return failure
            }) {
                continuation.yield(.failed(failure, partial: nil))
                continuation.finish(throwing: failure)
                return
            }
            for step in 1...3 {
                try await Task.sleep(for: eventDelay)
                try Task.checkCancellation()
                continuation.yield(.prefillProgress(done: step, total: 3))
            }
            prefillEnd = Date()

            let response = "Simulated response. Your prompt was: \(request.prompt)"
            let words = response.split(whereSeparator: \.isWhitespace)
            let pieces = words.enumerated().map { index, word in
                index == 0 ? String(word) : " " + word
            }.prefix(request.maxNewTokens)

            for (index, piece) in pieces.enumerated() {
                try await Task.sleep(for: eventDelay)
                try Task.checkCancellation()
                if state.value.withLock({ $0.stopRequested }) {
                    stoppedEarly = true
                    break
                }
                if firstTokenDate == nil { firstTokenDate = Date() }
                generated += 1
                continuation.yield(.token(AppTokenEvent(
                    index: index,
                    textDelta: piece,
                    elapsedDecodeSeconds: Date().timeIntervalSince(prefillEnd))))
            }

            let end = Date()
            let decodeSeconds = max(end.timeIntervalSince(prefillEnd), 0)
            let softTokens = state.value.withLock { $0.imageSoftTokens }
            // Stand-in IDs, distinct per half so a test can tell which record a
            // stored turn came from. The real client reports what the KV took;
            // what matters here is that the app carries them to disk unchanged.
            //
            // Each image contributes a marker, its placeholder run and a second
            // marker, in that order, because that is the shape replay searches
            // for: one run per image, separated by something that is not a
            // placeholder. Emitting the runs back to back would merge them into
            // a single longer run and no image would be locatable.
            var promptTokenIDs: [Int32] = []
            for _ in request.imageAttachments {
                promptTokenIDs.append(1_500)
                promptTokenIDs += Array(
                    repeating: MultimodalPromptRenderer.imageTokenID, count: softTokens)
                promptTokenIDs.append(1_501)
            }
            let textTokens = max(
                1, request.prompt.split(whereSeparator: \.isWhitespace).count)
            promptTokenIDs += (0..<textTokens).map { Int32(1_000 + $0) }
            // The count and the IDs are one fact. A conversation whose meta
            // claims more tokens than its records hold is exactly the row the
            // history tests exist to catch, so the double must not invent one.
            let promptTokens = promptTokenIDs.count
            let generatedTokenIDs = (0..<generated).map { Int32(2_000 + $0) }
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: generated,
                stopReason: stoppedEarly ? .cancelled
                    : (generated >= request.maxNewTokens ? .maxTokens : .eos),
                promptTokenCount: promptTokens,
                conversationTokens: state.value.withLock {
                    $0.conversationTokens += promptTokens + generated
                    return $0.conversationTokens
                },
                promptTokenIDs: promptTokenIDs,
                generatedTokenIDs: generatedTokenIDs,
                prefillSeconds: prefillEnd.timeIntervalSince(start),
                timeToFirstTokenSeconds: firstTokenDate.map { $0.timeIntervalSince(prefillEnd) },
                decodeSeconds: decodeSeconds,
                tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            // After the turn ran, not at admission: a turn that failed before
            // touching the KV must not advance the position the next one has to
            // match.
            state.value.withLock { $0.gate.commit(admission) }
            continuation.finish()
        } catch is CancellationError {
            let end = Date()
            let decodeSeconds = max(end.timeIntervalSince(prefillEnd), 0)
            continuation.yield(.cancelled(AppDiagnostics(
                generatedTokens: generated,
                stopReason: .cancelled,
                promptTokenCount: max(1, request.prompt.split(whereSeparator: \.isWhitespace).count),
                prefillSeconds: prefillEnd.timeIntervalSince(start),
                timeToFirstTokenSeconds: firstTokenDate.map { $0.timeIntervalSince(prefillEnd) },
                decodeSeconds: decodeSeconds,
                tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch {
            let appError = error as? AppInferenceError ?? .unknown("\(error)")
            continuation.yield(.failed(appError, partial: nil))
            continuation.finish(throwing: appError)
        }

        generationTasks.clear(generationID)
    }
}
