import Darwin
import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    private struct Connection {
        var input: FileHandle?
        var responses: DecodeServiceResponseRouter?
        var loadedDirectory: URL?
        var launchLabel: String?
        var socketPath: String?
    }

    private let connection = Mutex(Connection())
    private let serviceURL: URL
    private let inferenceMemory = Mutex<UInt64?>(nil)
    private let inferenceTowerMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public var currentInferenceTowerBytes: UInt64? {
        inferenceTowerMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil) {
        self.serviceURL = serviceURL ?? Self.defaultServiceURL()
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
    }

    init(testInput: FileHandle, responseOutput: FileHandle) {
        self.serviceURL = Self.defaultServiceURL()
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        let responses = makeRouter(output: responseOutput)
        connection.withLock {
            $0.input = testInput
            $0.responses = responses
        }
    }

    var connectionIsInstalled: Bool {
        connection.withLock { $0.input != nil && $0.responses != nil }
    }

    var installedRouter: DecodeServiceResponseRouter? {
        connection.withLock { $0.responses }
    }

    private func invalidateConnection(
        expecting router: DecodeServiceResponseRouter? = nil
    ) {
        let dead = connection.withLock { state -> Connection? in
            if let router, state.responses !== router { return nil }
            defer { state = Connection() }
            return state
        }
        guard let dead else { return }
        if let input = dead.input {
            try? input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.shutdown))
            try? input.close()
        }
        dead.responses?.closeStream()
        if let label = dead.launchLabel { Self.removeLaunchJob(label: label) }
        if let socketPath = dead.socketPath { unlink(socketPath) }
        inferenceMemory.withLock { $0 = nil }
        inferenceTowerMemory.withLock { $0 = nil }
    }

    private func write(_ command: DecodeServiceCommand,
                       to input: FileHandle,
                       expecting responses: DecodeServiceResponseRouter) throws {
        do {
            try input.write(contentsOf: DecodeFrameCodec.encode(command))
        } catch {
            invalidateConnection(expecting: responses)
            throw AppInferenceError.unknown(
                "could not reach the decode service: \(error)")
        }
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        let handles = try await Task.detached(priority: .userInitiated) { [self] in
            try ensureProcess()
        }.value
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead)
        try write(.load(request), to: handles.input, expecting: handles.responses)
        let event = try await handles.responses.next(matching: request.requestID)
        switch event.kind {
        case .ready:
            break
        case .failed:
            throw AppInferenceError.modelLoadFailed(
                event.error ?? "decode service load failed")
        default:
            throw AppInferenceError.modelLoadFailed(
                "decode service returned \(event.kind.rawValue) for a load request")
        }
        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
        inferenceTowerMemory.withLock { $0 = event.visionTowerMappedBytes }
        connection.withLock { $0.loadedDirectory = modelDirectory.standardizedFileURL }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    public func unload() async {
        guard let handles = currentHandles() else { return }
        let requestID = UUID()
        try? write(.unload(requestID), to: handles.input, expecting: handles.responses)
        guard let event = try? await handles.responses.next(matching: requestID),
              event.kind == .unloaded else { return }
        connection.withLock { $0.loadedDirectory = nil }
        inferenceMemory.withLock { $0 = nil }
        inferenceTowerMemory.withLock { $0 = nil }
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [self] in
                do {
                    try request.validate()
                    guard let handles = currentHandles() else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    let generationID = UUID()
                    generationTranscriptMailbox.reset()
                    let command = DecodeGenerationRequest(
                        prompt: request.prompt,
                        imageAttachments: request.imageAttachments.map {
                            DecodeImageAttachment(
                                id: $0.id,
                                path: $0.fileURL.path,
                                displayName: $0.displayName,
                                encodedBytes: $0.encodedBytes,
                                sha256: $0.sha256)
                        },
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID,
                        conversationEpoch: request.conversationEpoch,
                        turnIndex: request.turnIndex)
                    try write(.generate(command), to: handles.input,
                              expecting: handles.responses)

                    var expectedSequence: UInt64 = 1
                    var lastMetricYield = Date.distantPast
                    var hasYieldedVisibleText = false
                    while true {
                        let event = try await handles.responses.next(matching: generationID)
                        // Only when the event carries a figure: an event
                        // without one says nothing about memory, and clearing
                        // the last reading made the display flicker to empty.
                        if let bytes = event.currentMemoryBytes {
                            inferenceMemory.withLock { $0 = bytes }
                        }
                        if let tower = event.visionTowerMappedBytes {
                            inferenceTowerMemory.withLock { $0 = tower }
                        }
                        guard event.generationID == generationID else { continue }

                        if event.kind == .memory {
                            // The stored reading is not observed by the UI, so
                            // yield an event that causes it to redraw.
                            continuation.yield(.memorySample)
                            continue
                        }
                        if event.kind == .prefill || event.kind == .snapshot {
                            guard event.sequence == expectedSequence else {
                                throw AppInferenceError.unknown(
                                    "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
                            }
                            expectedSequence &+= 1
                        }
                        if event.kind == .prefill,
                           let done = event.prefillDone,
                           let total = event.prefillTotal {
                            continuation.yield(.prefillProgress(done: done, total: total))
                            continue
                        }
                        if event.kind == .snapshot {
                            generationTranscriptMailbox.append(event.textDelta)
                            let now = Date()
                            let beginsVisibleText = !hasYieldedVisibleText
                                && event.textDelta.contains { !$0.isWhitespace }
                            if beginsVisibleText
                                || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                                lastMetricYield = now
                                hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                                continuation.yield(.token(AppTokenEvent(
                                    index: max(0, event.tokenCount - 1),
                                    textDelta: beginsVisibleText ? event.textDelta : "",
                                    elapsedDecodeSeconds: event.decodeSeconds)))
                            }
                            continue
                        }

                        let diagnostics = Self.diagnostics(
                            event, options: request.runtimeOptions)
                        switch event.kind {
                        case .finished:
                            continuation.yield(.finished(diagnostics))
                            continuation.finish()
                        case .cancelled:
                            continuation.yield(.cancelled(diagnostics))
                            continuation.finish()
                        case .failed:
                            let error = AppInferenceError.unknown(
                                event.error ?? "decode service failed")
                            continuation.yield(.failed(error, partial: diagnostics))
                            continuation.finish(throwing: error)
                        case .lineageLost:
                            // Carried across as its own case, not flattened into
                            // `.unknown`: the app has to clear the conversation
                            // rather than offer a retry that would fail the same
                            // way every time.
                            let error = AppInferenceError.conversationLineageLost(
                                event.error ?? "the conversation's KV no longer matches it")
                            continuation.yield(.failed(error, partial: diagnostics))
                            continuation.finish(throwing: error)
                        default:
                            continue
                        }
                        return
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] _ in
                task.cancel()
                self?.cancel()
            }
        }
    }

    /// Opens `epoch` on the service and waits for it to say so.
    ///
    /// Not fire-and-forget: the app numbers the next turn from zero the moment
    /// this returns, and a turn numbered against a reset the service never
    /// applied is refused by its gate — which is the safe outcome, but the user
    /// sees a rejected message instead of a new chat.
    public func resetConversation(epoch: UUID) async throws {
        guard let handles = currentHandles() else {
            // Not "nothing is loaded" — `currentHandles` is nil when the
            // *connection* is gone. Returning normally let the caller record an
            // epoch the service had never heard of, and because it then matched
            // the app's own, the reset was never retried: every later turn was
            // refused and no recovery path could fire.
            throw AppInferenceError.unknown(
                "the decode service connection is gone; the new chat was not opened")
        }
        let requestID = UUID()
        try write(.resetConversation(
            DecodeResetConversationRequest(epoch: epoch, requestID: requestID)),
            to: handles.input, expecting: handles.responses)
        let event = try await handles.responses.next(matching: requestID)
        guard event.kind == .conversationReset else {
            throw AppInferenceError.unknown(
                event.error ?? "decode service refused to start a new conversation")
        }
    }

    public func cancel() {
        guard let handles = currentHandles() else { return }
        try? write(.cancel, to: handles.input, expecting: handles.responses)
    }

    deinit {
        let state = connection.withLock { value -> Connection in
            defer { value = Connection() }
            return value
        }
        if let input = state.input {
            try? input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.shutdown))
            try? input.close()
        }
        state.responses?.closeStream()
        if let label = state.launchLabel { Self.removeLaunchJob(label: label) }
        if let socketPath = state.socketPath { unlink(socketPath) }
    }

    private func ensureProcess() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        if let handles = currentHandles() { return handles }
        return try launchIndependentService()
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let identifier = "\(getuid()).\(getpid()).\(UUID().uuidString.lowercased())"
        let label = "com.turbofieldfare.decode.\(identifier)"
        let socketPath = "/private/tmp/turbofieldfare-decode-\(identifier).sock"
        let propertyListURL = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("\(label).plist")
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                serviceURL.path,
                "--socket", socketPath,
                "--launch-label", label,
            ],
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Interactive",
        ]
        let propertyListData = try PropertyListSerialization.data(
            fromPropertyList: propertyList, format: .xml, options: 0)
        try propertyListData.write(to: propertyListURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: propertyListURL) }

        let launcher = Process()
        let errors = Pipe()
        launcher.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        launcher.arguments = [
            "bootstrap", "gui/\(getuid())", propertyListURL.path,
        ]
        launcher.standardOutput = FileHandle.nullDevice
        launcher.standardError = errors
        try launcher.run()
        launcher.waitUntilExit()
        guard launcher.terminationStatus == 0 else {
            let data = try? errors.fileHandleForReading.readToEnd()
            let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                ?? "launchd could not start the decode service"
            throw AppInferenceError.modelLoadFailed(message)
        }

        // Demand the job without restarting a RunAtLoad winner; the socket is
        // authoritative. `bootstrap` alone does not always leave the job
        // running, and the failure then looked like a socket timeout with no
        // indication that launchd was the cause.
        var kickstartError: String?
        do {
            let starter = Process()
            let starterErrors = Pipe()
            starter.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            starter.arguments = Self.kickstartArguments(uid: getuid(), label: label)
            starter.standardOutput = FileHandle.nullDevice
            starter.standardError = starterErrors
            try starter.run()
            starter.waitUntilExit()
            if starter.terminationStatus != 0 {
                let data = try? starterErrors.fileHandleForReading.readToEnd()
                let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                kickstartError = detail.flatMap { $0.isEmpty ? nil : $0 }
                    ?? "exit status \(starter.terminationStatus)"
            }
        } catch {
            kickstartError = String(describing: error)
        }

        var lastError: Error?
        for _ in 0..<200 {
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                let responses = makeRouter(output: handles.output)
                connection.withLock {
                    $0.input = handles.input
                    $0.responses = responses
                    $0.launchLabel = label
                    $0.socketPath = socketPath
                }
                return (handles.input, responses)
            } catch {
                lastError = error
                usleep(10_000)
            }
        }
        Self.removeLaunchJob(label: label)
        throw AppInferenceError.modelLoadFailed(
            Self.socketFailureMessage(
                socketError: lastError.map(String.init(describing:)),
                kickstartError: kickstartError))
    }

    /// `kickstart` demands the job; it deliberately omits `-k`, which would
    /// restart a job that a RunAtLoad bootstrap already won. The socket is what
    /// decides readiness, so restarting a healthy service would only lose it.
    static func kickstartArguments(uid: uid_t, label: String) -> [String] {
        ["kickstart", "gui/\(uid)/\(label)"]
    }

    /// The socket timeout stays the primary diagnostic, because that is what the
    /// caller actually observed; a launchctl failure is appended when there was
    /// one, since it usually explains the timeout.
    static func socketFailureMessage(socketError: String?, kickstartError: String?)
        -> String {
        let kickstartDetail = kickstartError.map {
            "; launchctl kickstart failed: \($0)"
        } ?? ""
        return "decode service socket did not become ready: \(socketError ?? "unknown error")\(kickstartDetail)"
    }

    private func currentHandles()
        -> (input: FileHandle, responses: DecodeServiceResponseRouter)? {
        let handles = connection.withLock { state
            -> (input: FileHandle, responses: DecodeServiceResponseRouter)? in
            guard let input = state.input, let responses = state.responses else {
                return nil
            }
            return (input, responses)
        }
        guard let handles else { return nil }
        if handles.responses.isTerminated {
            invalidateConnection(expecting: handles.responses)
            return nil
        }
        return handles
    }

    private func makeRouter(output: FileHandle) -> DecodeServiceResponseRouter {
        DecodeServiceResponseRouter(output: output) { [weak self] router, _ in
            self?.invalidateConnection(expecting: router)
        }
    }

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled ? .cancelled
                : (event.kind == .failed || event.kind == .lineageLost) ? .failed
                : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            cachedPromptTokens: event.cachedPromptTokens,
            computedPrefillTokens: event.computedPrefillTokens,
            conversationTokens: event.conversationTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            visionTowerMappedBytes: event.visionTowerMappedBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics))
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }

    private static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue,
            visionResidencyPolicy: options.visionResidencyPolicy.rawValue)
    }

    private static func removeLaunchJob(label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }

    private static func defaultServiceURL() -> URL {
        return Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("TurboFieldfareDecodeService")
    }
}
