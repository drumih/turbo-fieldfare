import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

/// Prefill is the long phase for a large prompt or an image, and it is exactly
/// when someone watches the memory figure. It was sampled only when a token
/// arrived, so it stayed blank for the entire wait.
private final class PrefillOnlyClient: AppInferenceClient, AppInferenceMemoryReporting,
    @unchecked Sendable {
    private let lock = NSLock()
    private var memory: UInt64?

    var currentInferenceMemoryBytes: UInt64? {
        lock.lock(); defer { lock.unlock() }
        return memory
    }

    var currentInferenceResidentBytes: UInt64? {
        lock.lock(); defer { lock.unlock() }
        return memory.map { $0 + 1_144_373_248 }
    }

    /// The tower the process is holding, which is the only figure that moves
    /// between the two residency policies.
    var currentInferenceTowerBytes: UInt64? {
        lock.lock(); defer { lock.unlock() }
        return tower
    }

    private var tower: UInt64?

    func reportTower(_ bytes: UInt64?) {
        lock.lock(); defer { lock.unlock() }
        tower = bytes
    }

    /// Stands in for the decode service reporting memory alongside progress.
    func reportMemory(_ bytes: UInt64) {
        lock.lock(); defer { lock.unlock() }
        memory = bytes
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream(AppInferenceEvent.self) { continuation in
            Task { [self] in
                for step in 1...3 {
                    reportMemory(UInt64(step) * 1_000_000_000)
                    continuation.yield(.prefillProgress(done: step, total: 3))
                    try? await Task.sleep(for: .milliseconds(10))
                }
                continuation.yield(.finished(AppDiagnostics(
                    generatedTokens: 0, stopReason: .eos,
                    timeToFirstTokenSeconds: nil,
                    decodeSeconds: 0, tokensPerSecond: 0,
                    peakMemoryBytes: currentInferenceMemoryBytes,
                    runtimeOptions: request.runtimeOptions)))
                continuation.finish()
            }
        }
    }

    func cancel() {}
}

@Suite struct AppLiveMemoryDuringPrefillTests {
    @MainActor
    @Test func memoryIsReportedBeforeTheFirstToken() async throws {
        let client = PrefillOnlyClient()
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.promptText = "describe this"
        model.send()

        let deadline = Date().addingTimeInterval(60)
        while model.liveMemoryBytes == nil,
              model.isTurnInFlight || model.phase == .prefill,
              Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.liveMemoryBytes != nil,
                "memory stayed blank through prefill")
        #expect(model.liveMemoryBytes ?? 0 > 0)
    }

    /// What the HUD reads is what matters. It used to consult the client
    /// directly, which is invisible to observation, so the figure only
    /// refreshed when a generated token happened to redraw the view — never
    /// during prefill.
    @MainActor
    @Test func thedisplayedFigureMovesDuringPrefill() async throws {
        let client = PrefillOnlyClient()
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)

        // Driven event by event rather than by polling a live run. Sampling a
        // background run at intervals is a race the test loses under load: the
        // three prefill events can all land between two samples, and the test
        // then reports the bug it is meant to catch. The property is that each
        // prefill event moves what the HUD reads, which is exactly this.
        var seen: [UInt64] = []
        for step in 1...3 {
            client.reportMemory(UInt64(step) * 1_000_000_000)
            model.apply(.prefillProgress(done: step, total: 3))
            seen.append(try #require(model.currentProcessMemoryBytes))
        }

        #expect(seen == [1_000_000_000, 2_000_000_000, 3_000_000_000],
                "the displayed memory did not follow prefill: \(seen)")
        #expect(model.liveTokenCount == 0,
                "this run must not reach decode, or it proves nothing")
    }

    /// The same for a service that has nothing else to say. Image encoding
    /// produces no progress and no tokens for seconds at a time, and the figure
    /// used to sit at its pre-run value for the whole of it.
    @MainActor
    @Test func thedisplayedFigureMovesWithNoProgressAtAll() async throws {
        let client = PrefillOnlyClient()
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)

        // A first reading through the ordinary path, so the model holds a figure
        // of its own. Without it the display falls back to asking the client
        // directly and the test passes whatever the model does — which is how
        // the first version of this test managed to prove nothing.
        client.reportMemory(1_500_000_000)
        model.apply(.prefillProgress(done: 1, total: 2))
        #expect(model.liveMemoryBytes == 1_500_000_000)

        // Now the silent window: the service reports more memory and sends
        // nothing else, exactly as it does while encoding an image.
        client.reportMemory(2_100_000_000)
        model.apply(.memorySample)
        #expect(model.liveMemoryBytes == 2_100_000_000,
                "the held figure went stale through a silent phase")
        #expect(model.currentProcessMemoryBytes == 2_100_000_000,
                "a memory-only reading did not reach the display")
        #expect(model.livePrefillTotal == 2,
                "a memory reading moved the prefill counters")

        // And the display reads the tracked property, not the client. A figure
        // the client knows but the model has not observed cannot redraw the
        // view, so showing it would put a number on screen that only changes
        // when something else happens to trigger a redraw.
        client.reportMemory(9_000_000_000)
        #expect(model.currentProcessMemoryBytes == 2_100_000_000,
                "the display reached past the observed value into the client")
    }

    /// All three figures move together, and each answers a different question:
    /// what the process is charged, what is actually in RAM, and what the image
    /// tower is holding. Showing only the first is what made a 26B model read
    /// 2.26 GB and look impossible.
    @MainActor
    @Test func allThreeMemoryFiguresFollowARun() async throws {
        let client = PrefillOnlyClient()
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)

        client.reportMemory(2_260_000_000)
        client.reportTower(1_144_373_248)
        model.apply(.memorySample)

        #expect(model.currentProcessMemoryBytes == 2_260_000_000)
        #expect(model.currentProcessResidentBytes == 2_260_000_000 + 1_144_373_248,
                "the in-RAM figure did not include the mapped pages")
        #expect(model.visionTowerMappedBytes == 1_144_373_248,
                "the tower figure did not reach the model during the run")

        // Load On Demand releases after each image, and that has to show.
        client.reportTower(0)
        model.apply(.memorySample)
        #expect(model.visionTowerMappedBytes == 0,
                "a released tower still reported as held")
    }

    /// An event that carries no memory reading says nothing about memory, and
    /// must not blank out the last one.
    @Test func anEventWithoutAReadingKeepsTheLastOne() async throws {
        let pipe = Pipe()
        let id = UUID()
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .prefill, generationID: id,
                               prefillDone: 1, prefillTotal: 2,
                               currentMemoryBytes: 4_000_000_000)))
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(kind: .prefill, generationID: id,
                               prefillDone: 2, prefillTotal: 2)))
        try pipe.fileHandleForWriting.close()

        // The decoded events are what the client applies; assert the rule the
        // client follows rather than reaching through a live connection.
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        var reported: UInt64?
        for _ in 0..<2 {
            let event = try await router.next(matching: id, timeout: .seconds(5))
            if let bytes = event.currentMemoryBytes { reported = bytes }
        }
        #expect(reported == 4_000_000_000,
                "a reading-free event erased the last known figure")
        try? pipe.fileHandleForReading.close()
    }
}

/// With several images attached, the prompt and its thumbnails are tall enough
/// to push the answer out of view, so the transcript has to put the newest turn
/// on screen when a run starts rather than only when the reader is already at
/// the bottom.
@Suite struct AppRunIdentityTests {
    @MainActor
    @Test func eachRunGetsANewIdentity() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.maxNewTokensOverride = 1

        let before = model.runIdentity
        model.promptText = "first"
        model.send()
        await SendWaiting.generationStarts(model)
        let afterFirst = model.runIdentity
        #expect(afterFirst != before, "starting a run did not change its identity")

        await SendWaiting.turnEnds(model)
        model.promptText = "second"
        model.send()
        await SendWaiting.generationStarts(model)
        #expect(model.runIdentity != afterFirst,
                "a second run reused the first run's identity")
    }

    /// A refused run must not claim a new turn: nothing was added to scroll to.
    @MainActor
    @Test func arefusedRunKeepsTheIdentity() throws {
        let model = AppModel(client: MockInferenceClient())
        model.loadState = .notLoaded
        model.promptText = "go"
        let before = model.runIdentity
        model.send()
        #expect(model.runIdentity == before)
    }
}

/// Loading takes minutes and holds gigabytes, so it stays opt-in — but when it
/// is opted into, launching must actually start it.
@Suite struct AppLaunchLoadTests {
    @MainActor
    @Test func launchDoesNotLoadByDefault() throws {
        let directory = try makeCompleteModelInstall("launch-default")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        #expect(!model.loadModelOnLaunch)

        model.loadModelAtLaunchIfEnabled()
        #expect(model.loadState == .notLoaded,
                "launch started a load nobody asked for")
    }

    @MainActor
    @Test func launchLoadsWhenTheSettingIsOn() async throws {
        let directory = try makeCompleteModelInstall("launch-on")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        model.setLoadModelOnLaunch(true)
        #expect(model.loadModelOnLaunch)

        model.loadModelAtLaunchIfEnabled()
        let deadline = Date().addingTimeInterval(60)
        while !model.loadState.isReady, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.loadState.isReady, "the launch load never ran")
    }

    /// A model that is not installed must not be loaded at launch, and the
    /// attempt must not leave an error on screen at startup.
    @MainActor
    @Test func launchLeavesAnUninstalledModelAlone() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gturbo")
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        model.setLoadModelOnLaunch(true)
        model.loadModelAtLaunchIfEnabled()
        #expect(model.loadState == .notLoaded)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func thesettingSurvivesRelaunch() throws {
        let directory = try makeCompleteModelInstall("launch-persist")
        defer { try? FileManager.default.removeItem(at: directory) }
        var settings = MacAppSettings()
        settings.loadModelOnLaunch = true
        let encoded = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: encoded)
        #expect(decoded.loadModelOnLaunch)

        // Settings written before this existed simply have it off.
        var object = try #require(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "loadModelOnLaunch")
        let older = try JSONSerialization.data(withJSONObject: object)
        #expect(!(try JSONDecoder().decode(MacAppSettings.self, from: older))
            .loadModelOnLaunch)
    }
}

/// Changing a runtime setting and pressing Generate without reloading was
/// refused outright: the request carried the new options while the session was
/// loaded with the old ones, and the service compares them. The settings still
/// apply on reload — they just no longer break the run in the meantime.
@Suite struct AppPendingSettingsRunTests {
    @MainActor
    private func readyModel() throws -> (AppModel, URL) {
        let directory = try makeCompleteModelInstall("pending-settings")
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        return (model, directory)
    }

    @MainActor
    @Test func arunUsesTheLoadedSessionsOptionsNotThePendingOnes() async throws {
        let (model, directory) = try readyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        model.runtimeOptions.visionResidencyPolicy = .onDemand
        model.setMaxContextTokens(AppContextLengthOption.fourK.tokens)
        model.loadModel()
        let deadline = Date().addingTimeInterval(60)
        while !model.loadState.isReady, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        try #require(model.loadState.isReady)

        // The user changes settings and presses Generate without reloading.
        model.setMaxContextTokens(AppContextLengthOption.sixteenK.tokens)
        model.promptText = "go"
        #expect(model.hasStaleLoadedRuntime, "the reload prompt should be showing")

        let request = try model.makeRequest()
        #expect(request.maxContextTokens == AppContextLengthOption.fourK.tokens,
                "the run carried a context the loaded session does not have")
    }

    /// With nothing loaded there is no session to match, so the request simply
    /// carries the current settings.
    @MainActor
    @Test func withoutALoadedSessionTheCurrentSettingsAreUsed() throws {
        let (model, directory) = try readyModel()
        defer { try? FileManager.default.removeItem(at: directory) }
        model.setMaxContextTokens(AppContextLengthOption.sixteenK.tokens)
        model.promptText = "go"
        let request = try model.makeRequest()
        #expect(request.maxContextTokens == AppContextLengthOption.sixteenK.tokens)
    }
}

/// When inference runs in another process, its memory is the only memory worth
/// showing. The UI's own footprint must never appear in a row labelled as the
/// model's.
private final class SilentReporterClient: AppInferenceClient,
    AppInferenceMemoryReporting, @unchecked Sendable {
    var currentInferenceMemoryBytes: UInt64? { nil }
    var currentInferenceResidentBytes: UInt64? { nil }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream(AppInferenceEvent.self) { $0.finish() }
    }

    func cancel() {}
}

@Suite struct AppInferenceOnlyMemoryTests {
    @MainActor
    @Test func anUnreportingServiceShowsNothingRatherThanTheAppsOwnMemory() throws {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: SilentReporterClient())
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)

        #expect(model.currentProcessMemoryBytes == nil,
                "the app's own footprint was shown as the model's")
        #expect(model.currentProcessResidentBytes == nil)
    }

    /// With in-process inference the app *is* the inference process, so its own
    /// figure is the right one.
    @MainActor
    @Test func inProcessInferenceReportsThisProcess() throws {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(client: MockInferenceClient())
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)

        #expect((model.currentProcessMemoryBytes ?? 0) > 0)
        #expect((model.currentProcessResidentBytes ?? 0) > 0)
    }
}
