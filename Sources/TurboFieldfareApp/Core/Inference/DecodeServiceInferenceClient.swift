import Darwin
import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    private struct Connection {
        var input: FileHandle?
        var output: FileHandle?
        var loadedDirectory: URL?
        var launchLabel: String?
        var socketPath: String?
    }

    private let connection = Mutex(Connection())
    private let serviceURL: URL
    private let inferenceMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil) {
        self.serviceURL = serviceURL ?? Self.defaultServiceURL()
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
        try handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.load(request)))
        let event = try await readEvent(from: handles.output)
        guard event.generationID == request.requestID, event.kind == .ready else {
            throw AppInferenceError.modelLoadFailed(
                event.error ?? "decode service load failed")
        }
        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
        connection.withLock { $0.loadedDirectory = modelDirectory.standardizedFileURL }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    public func unload() async {
        guard let handles = currentHandles() else { return }
        let requestID = UUID()
        try? handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.unload(requestID)))
        _ = try? await readEvent(from: handles.output)
        connection.withLock { $0.loadedDirectory = nil }
        inferenceMemory.withLock { $0 = nil }
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
                        prompt: request.prompt, maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID)
                    try handles.input.write(contentsOf: DecodeFrameCodec.encode(
                        DecodeServiceCommand.generate(command)))

                    var expectedSequence: UInt64 = 1
                    var lastMetricYield = Date.distantPast
                    while true {
                        let event = try DecodeFrameCodec.read(
                            DecodeServiceEvent.self, from: handles.output)
                        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
                        guard event.generationID == generationID else { continue }

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
                            if now.timeIntervalSince(lastMetricYield) >= 0.5 {
                                lastMetricYield = now
                                continuation.yield(.token(AppTokenEvent(
                                    index: max(0, event.tokenCount - 1),
                                    textDelta: "",
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

    public func cancel() {
        guard let input = currentHandles()?.input else { return }
        try? input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.cancel))
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
        if let label = state.launchLabel { Self.removeLaunchJob(label: label) }
        if let socketPath = state.socketPath { unlink(socketPath) }
    }

    private func ensureProcess() throws -> (input: FileHandle, output: FileHandle) {
        if let handles = currentHandles() { return handles }
        return try launchIndependentService()
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, output: FileHandle) {
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

        var lastError: Error?
        for _ in 0..<200 {
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                connection.withLock {
                    $0.input = handles.input
                    $0.output = handles.output
                    $0.launchLabel = label
                    $0.socketPath = socketPath
                }
                return handles
            } catch {
                lastError = error
                usleep(10_000)
            }
        }
        Self.removeLaunchJob(label: label)
        throw AppInferenceError.modelLoadFailed(
            "decode service socket did not become ready: \(lastError.map(String.init(describing:)) ?? "unknown error")")
    }

    private func currentHandles() -> (input: FileHandle, output: FileHandle)? {
        connection.withLock { state in
            guard let input = state.input, let output = state.output else {
                return nil
            }
            return (input, output)
        }
    }

    private func readEvent(from output: FileHandle) async throws
        -> DecodeServiceEvent {
        try await Task.detached(priority: .userInitiated) {
            try DecodeFrameCodec.read(DecodeServiceEvent.self, from: output)
        }.value
    }

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled
                ? .cancelled
                : event.kind == .failed ? .failed : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
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
            turboQuantKVEnabled: options.turboQuantKVEnabled,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue)
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
