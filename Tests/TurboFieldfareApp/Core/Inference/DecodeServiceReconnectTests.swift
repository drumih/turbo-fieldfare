import Darwin
import Foundation
import Synchronization
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

/// A stand-in decode service: it speaks the wire protocol over a real socket
/// and can be killed, which is the whole point. A real service would need a
/// model, and the defect under test is in the transport, not the model.
private final class FakeDecodeService: @unchecked Sendable {
    let socketPath: String
    private let loads = Mutex(0)
    private let cancels = Mutex(0)
    private let generations = Mutex(0)
    private let shutdowns = Mutex(0)
    /// Makes the load slow enough to be cancelled mid-flight, the way a real
    /// multi-minute model load is.
    let loadDelay: Duration
    private var thread: Thread?
    private let stopped = Mutex(false)
    private let handles = Mutex<(input: FileHandle, output: FileHandle)?>(nil)

    init(loadDelay: Duration = .zero) {
        self.socketPath = "/private/tmp/turbofieldfare-fake-decode-\(UUID().uuidString).sock"
        self.loadDelay = loadDelay
    }

    var loadCount: Int { loads.withLock { $0 } }
    var cancelCount: Int { cancels.withLock { $0 } }
    var generationCount: Int { generations.withLock { $0 } }
    var shutdownCount: Int { shutdowns.withLock { $0 } }

    func start() {
        let thread = Thread { [self] in serve() }
        thread.name = "FakeDecodeService"
        thread.start()
        self.thread = thread
        // The file appears at bind(), before listen(), so waiting for it is not
        // waiting for readiness. The client under test retries its connect, so
        // this only has to give the listener a moment to exist at all.
        let deadline = ContinuousClock.now + .seconds(5)
        while !FileManager.default.fileExists(atPath: socketPath),
              ContinuousClock.now < deadline {
            usleep(5_000)
        }
    }

    /// Kills the service the way a jetsam or a crash would: the socket simply
    /// stops, mid-session, with no protocol-level goodbye.
    func kill() {
        stopped.withLock { $0 = true }
        let current = handles.withLock { value -> (FileHandle, FileHandle)? in
            defer { value = nil }
            return value.map { ($0.input, $0.output) }
        }
        try? current?.0.close()
        try? current?.1.close()
        unlink(socketPath)
    }

    private func serve() {
        guard let accepted = try? DecodeUnixSocket.listenAndAccept(path: socketPath)
        else { return }
        handles.withLock { $0 = accepted }
        while !(stopped.withLock { $0 }) {
            guard let command = try? DecodeFrameCodec.read(
                DecodeServiceCommand.self, from: accepted.input) else { return }
            switch command {
            case .load(let request):
                loads.withLock { $0 += 1 }
                if loadDelay != .zero {
                    let seconds = Double(loadDelay.components.seconds)
                        + Double(loadDelay.components.attoseconds) / 1e18
                    usleep(useconds_t(seconds * 1_000_000))
                }
                try? accepted.output.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceEvent(kind: .ready, generationID: request.requestID)))
            case .resetConversation(let request):
                try? accepted.output.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceEvent(kind: .conversationReset,
                                       generationID: request.requestID,
                                       conversationTokenCount: 0,
                                       conversationEpoch: request.epoch)))
            case .restoreConversation(let request):
                try? accepted.output.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceEvent(kind: .conversationRestored,
                                       generationID: request.requestID,
                                       conversationTokenCount: request.tokenIDs.count,
                                       conversationEpoch: request.epoch)))
            case .generate(let request):
                generations.withLock { $0 += 1 }
                try? accepted.output.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceEvent(kind: .finished,
                                       generationID: request.generationID,
                                       textDelta: "answer",
                                       tokenCount: 1,
                                       stopReason: "eos")))
            case .unload(let id):
                try? accepted.output.write(contentsOf: DecodeFrameCodec.encode(
                    DecodeServiceEvent(kind: .unloaded, generationID: id)))
            case .cancel:
                cancels.withLock { $0 += 1 }
            case .shutdown:
                shutdowns.withLock { $0 += 1 }
                return
            }
        }
    }
}

/// The app talks to the decode service over a socket, so when that service dies
/// the client used to latch its handles and its router's terminal error and
/// fail identically on every later request — Ready in the UI, nothing but
/// errors underneath, recoverable only by quitting the app.
@Suite(.serialized) struct DecodeServiceReconnectTests {
    private func withSocketEnvironment<T>(
        _ path: String, _ body: () async throws -> T
    ) async rethrows -> T {
        setenv("TURBO_FIELDFARE_DECODE_SOCKET_PATH", path, 1)
        defer { unsetenv("TURBO_FIELDFARE_DECODE_SOCKET_PATH") }
        return try await body()
    }

    @Test func terminationDisconnectsButDoesNotShutdownAnExternalService() async throws {
        let options = AppRuntimeOptions()
        let model = FileManager.default.temporaryDirectory
            .appendingPathComponent("fake-model-\(UUID())")
        let service = FakeDecodeService()
        service.start()
        defer { service.kill() }

        try await withSocketEnvironment(service.socketPath) {
            let first = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })
            try await first.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: options, forceLogitsHead: false, onState: { _ in })
            first.shutdownForTermination()
            #expect(service.shutdownCount == 0)
        }
        #expect(service.loadCount == 1)
        #expect(service.shutdownCount == 0)
        #expect(FileManager.default.fileExists(atPath: service.socketPath),
                "disconnecting unlinked an externally owned service socket")
    }

    @Test func aDeadServiceIsReplacedInsteadOfFailingForever() async throws {
        let options = AppRuntimeOptions()
        let model = FileManager.default.temporaryDirectory
            .appendingPathComponent("fake-model-\(UUID().uuidString)")

        let first = FakeDecodeService()
        first.start()
        defer { first.kill() }
        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })

        try await withSocketEnvironment(first.socketPath) {
            try await client.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: options, forceLogitsHead: false, onState: { _ in })
        }
        #expect(first.loadCount == 1)

        // The service dies mid-session.
        first.kill()

        // A second service is standing by on a new socket, exactly as launchd
        // would provide after the app relaunches it.
        let second = FakeDecodeService()
        second.start()
        defer { second.kill() }

        try await withSocketEnvironment(second.socketPath) {
            // The first attempt may still fail — the client can be holding a
            // handle to the dead socket — but it must recover by itself, not
            // stay broken until the app is quit.
            var lastError: Error?
            for _ in 0..<3 {
                do {
                    try await client.ensureLoaded(
                        modelDirectory: model, maxContextTokens: 512,
                        options: options, forceLogitsHead: false, onState: { _ in })
                    lastError = nil
                    break
                } catch {
                    lastError = error
                }
            }
            let detail = String(describing: lastError)
            #expect(lastError == nil,
                    "the client never rebuilt its connection: \(detail)")
        }
        #expect(second.loadCount >= 1, "the replacement service was never reached")
    }

    /// Cancelling a load has to reach the service. It runs commands serially,
    /// so a load nobody wants still blocks the unload the UI is waiting on —
    /// which is why cancelling used to freeze every model and companion action
    /// for the rest of the load. The connection must survive it: throwing the
    /// whole service away for a cancel would cost a relaunch and a reload.
    @Test func cancellingALoadReachesTheServiceAndKeepsTheConnection() async throws {
        let options = AppRuntimeOptions()
        let model = FileManager.default.temporaryDirectory
            .appendingPathComponent("fake-model-\(UUID().uuidString)")
        // Long enough that a loaded machine cannot finish it before the
        // cancel: the test waits for the service to report the load, so the
        // only thing this delay buys is the window to cancel in.
        let service = FakeDecodeService(loadDelay: .seconds(10))
        service.start()
        defer { service.kill() }
        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })

        try await withSocketEnvironment(service.socketPath) {
            let loading = Task {
                try await client.ensureLoaded(
                    modelDirectory: model, maxContextTokens: 512,
                    options: options, forceLogitsHead: false, onState: { _ in })
            }
            // Cancel once the service has certainly received the load, rather
            // than after a fixed sleep that a busy machine can overrun.
            let handedOver = ContinuousClock.now + .seconds(20)
            while service.loadCount == 0, ContinuousClock.now < handedOver {
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(service.loadCount == 1)
            loading.cancel()
            let started = ContinuousClock.now
            await #expect(throws: CancellationError.self) { try await loading.value }
            #expect(ContinuousClock.now - started < .seconds(60),
                    "the cancel never reached the transport wait")

            let deadline = ContinuousClock.now + .seconds(30)
            while service.cancelCount == 0, ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(20))
            }
            #expect(service.cancelCount >= 1,
                    "the service was never told to abandon the load")

            // Same connection, still usable.
            try await client.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: options, forceLogitsHead: false, onState: { _ in })
        }
        #expect(service.loadCount == 2,
                "the client rebuilt the connection instead of reusing it")
    }

    /// A service that accepts the connection and then says nothing must not
    /// hang the caller: the wait is bounded, and the connection is dropped so
    /// the next request starts clean.
    /// A consumer that stops listening is not a broken transport.
    ///
    /// Dropping a generation stream cancels the wait underneath it, and treating
    /// that as a dead connection would shut the service down, bootout its launch
    /// job and unlink its socket — so abandoning one answer would cost a
    /// relaunch and a full model reload before the next one.
    @Test func abandoningAGenerationKeepsTheService() async throws {
        let options = AppRuntimeOptions()
        // A real directory: the client validates the request before it writes
        // anything, so a placeholder path would fail the stream on the spot and
        // never reach the transport this covers.
        let model = FileManager.default.temporaryDirectory
            .appendingPathComponent("fake-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: model) }
        let service = FakeDecodeService()
        service.start()
        defer { service.kill() }
        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })

        try await withSocketEnvironment(service.socketPath) {
            try await client.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: options, forceLogitsHead: false, onState: { _ in })
            #expect(service.loadCount == 1)

            // The service never answers a generate, so the stream is still
            // waiting when its consumer walks away.
            let reading = Task {
                for try await _ in client.generate(AppGenerationRequest(
                    modelDirectory: model, prompt: "hello",
                    maxNewTokens: 8, maxContextTokens: 512,
                    runtimeOptions: options)) {}
            }
            try await Task.sleep(for: .milliseconds(300))
            reading.cancel()
            _ = try? await reading.value
            try await Task.sleep(for: .milliseconds(300))
            #expect(service.cancelCount >= 1,
                    "the abandoned stream never reached the service, so this proves nothing")

            // Same socket, same process: the next load must reach it.
            try await client.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: options, forceLogitsHead: false, onState: { _ in })
        }
        #expect(service.loadCount == 2,
                "the service was torn down by a caller that stopped listening")
    }

    @Test func aMuteServiceFailsTheRequestAndDropsTheConnection() async throws {
        let options = AppRuntimeOptions()
        let model = FileManager.default.temporaryDirectory
            .appendingPathComponent("fake-model-\(UUID().uuidString)")
        let path = "/private/tmp/turbofieldfare-mute-\(UUID().uuidString).sock"
        // Accepts, then never answers.
        let listener = Thread {
            _ = try? DecodeUnixSocket.listenAndAccept(path: path)
            while !Thread.current.isCancelled { usleep(50_000) }
        }
        listener.start()
        defer { listener.cancel(); unlink(path) }
        let deadline = ContinuousClock.now + .seconds(5)
        while !FileManager.default.fileExists(atPath: path),
              ContinuousClock.now < deadline {
            usleep(5_000)
        }

        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })
        await withSocketEnvironment(path) {
            let waiting = Task {
                try await client.ensureLoaded(
                    modelDirectory: model, maxContextTokens: 512,
                    options: options, forceLogitsHead: false, onState: { _ in })
            }
            try? await Task.sleep(for: .milliseconds(200))
            waiting.cancel()
            let started = ContinuousClock.now
            await #expect(throws: (any Error).self) { try await waiting.value }
            #expect(ContinuousClock.now - started < .seconds(10),
                    "cancelling a load never reached the transport wait")
        }
    }

    /// Prefill is outside the load key, so changing it must not turn a
    /// completed answer into a failure or require another model load.
    @Test func flippingPrefillAfterLoadDoesNotFailTheFinishedRun() async throws {
        let loaded = AppRuntimeOptions(prefillEnabled: true)
        // Generation validates its request, unlike a load, so this one needs a
        // directory that actually looks like a model.
        let model = try makeCompleteModelInstall("decode-service-prefill-toggle")
        defer { try? FileManager.default.removeItem(at: model) }

        let service = FakeDecodeService()
        service.start()
        defer { service.kill() }
        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })

        try await withSocketEnvironment(service.socketPath) {
            try await client.ensureLoaded(
                modelDirectory: model, maxContextTokens: 512,
                options: loaded, forceLogitsHead: false, onState: { _ in })

            // The Inspector's Prefill toggle, flipped after the load.
            var request = AppGenerationRequest(
                modelDirectory: model, prompt: "hello", maxNewTokens: 1,
                maxContextTokens: 512,
                runtimeOptions: AppRuntimeOptions(prefillEnabled: false))
            request.imageAttachments = []
            #expect(request.runtimeOptions.prefillEnabled != loaded.prefillEnabled)

            var events: [AppInferenceEvent] = []
            for try await event in client.generate(request) { events.append(event) }

            #expect(service.generationCount == 1)
            let finished = events.contains {
                if case .finished = $0 { return true }
                return false
            }
            #expect(finished, "a completed run was reported as a failure")
        }
    }
}

/// `currentHandles()` is nil when the connection is gone, not when the model is
/// unloaded. Returning normally let the caller record an epoch the service had
/// never heard of, and because it then matched the app's own, the reset was
/// never retried: every later turn was refused with no way back.
@Suite struct DecodeServiceResetFailureTests {
    @Test func aresetWithNoConnectionFailsRatherThanReportingSuccess() async throws {
        let client = DecodeServiceInferenceClient(testConnection: {
            let path = try #require(ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DECODE_SOCKET_PATH"])
            return try DecodeUnixSocket.connect(path: path)
        })
        await #expect(throws: AppInferenceError.self) {
            try await client.resetConversation(epoch: UUID())
        }
    }
}
