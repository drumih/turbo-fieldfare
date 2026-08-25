import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppModelServerControlTests {
    @MainActor
    @Test func startServerDisabledWithoutInstalledModel() {
        // The directory must be absent *at construction*, not patched in
        // afterward: `installationStatus` is computed once in `init` from
        // whatever directory is passed there, so on a machine with a real
        // model at `AppModelLocation.defaultURL()` (the default this
        // initializer would otherwise fall back to), reassigning
        // `modelPathText` post-init leaves `isModelInstalled` stuck `true`.
        let missingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gturbo")
        let model = AppModel(modelDirectory: missingDirectory,
                             client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())

        #expect(!model.canStartServer)
        model.startServer()
        #expect(model.serverState == .stopped)
    }

    @MainActor
    @Test func startServerSendsArgumentsBuiltFromCurrentSettings() throws {
        let directory = try makeCompleteModelInstall("server-control-arguments")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)
        model.maxContextTokens = 32_768
        model.runtimeOptions.expertCacheSlots = 24
        model.setServerPort(9090)
        model.setServerQueueLimit(8)

        model.startServer()

        #expect(model.serverState == .starting)
        let expected = AppServerArguments.build(
            modelPath: model.modelPathText,
            maxContextTokens: 32_768,
            runtimeOptions: model.runtimeOptions,
            visionPackPath: nil,
            port: 9090,
            queueLimit: 8)
        #expect(controller.startCalls == [expected])
    }

    @MainActor
    @Test func startServerIncludesVisionPackPathWhenInstalled() throws {
        let directory = try makeVisionReadyModelInstall("server-control-vision")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)
        #expect(model.isVisionPackInstalled)

        model.startServer()

        let companion = try VisionPackLocation.companionURL(forTextModel: directory)
        #expect(controller.startCalls.last?.suffix(2) == ["--vision-pack", companion.path])
    }

    @MainActor
    @Test func applyServerStateIgnoresAStaleGeneration() throws {
        let directory = try makeCompleteModelInstall("server-control-stale")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())

        model.startServer()
        #expect(model.serverState == .starting)
        model.applyServerState(.running(port: 1234), generation: 0)

        #expect(model.serverState == .starting,
                "a callback from a previous start overwrote the current one")
    }

    @MainActor
    @Test func runningStateEnablesStopAndDisablesStart() async throws {
        let directory = try makeCompleteModelInstall("server-control-running")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        #expect(model.canStopServer)
        #expect(!model.canStartServer)
        #expect(!model.canEditServerSettings)
    }

    @MainActor
    @Test func stopServerSendsStopAndReturnsToStoppedOnGracefulExit() async throws {
        let directory = try makeCompleteModelInstall("server-control-stop")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        model.stopServer()
        #expect(model.serverState == .stopping)
        #expect(controller.stopCallCount == 1)

        controller.emit(.stopped)
        try await waitUntil(deadline: 5) { model.serverState == .stopped }
        #expect(model.canStartServer)
        #expect(model.canEditServerSettings)
    }

    @MainActor
    @Test func aCrashWhileRunningIsReportedAsFailedAndCanBeRestarted() async throws {
        let directory = try makeCompleteModelInstall("server-control-crash")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        controller.emit(.failed(message: "server exited unexpectedly with status 134"))
        try await waitUntil(deadline: 5) {
            model.serverState == .failed(message: "server exited unexpectedly with status 134")
        }
        #expect(model.canStartServer)
    }

    @MainActor
    @Test func stopServerForTerminationStopsARunningServer() async throws {
        let directory = try makeCompleteModelInstall("server-control-termination")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        model.stopServerForTermination()

        #expect(controller.stopCallCount == 1)
    }

    @MainActor
    @Test func stopServerForTerminationDoesNothingWhenAlreadyStopped() {
        let missingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gturbo")
        let controller = MockServerController()
        let model = AppModel(modelDirectory: missingDirectory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.stopServerForTermination()

        #expect(controller.stopCallCount == 0)
    }

    @MainActor
    @Test func isServerStoppingDuringStartOnlyTrueWhenStopWasRequestedWhileStarting() async throws {
        let directory = try makeCompleteModelInstall("server-control-stopping-during-start")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        model.stopServer()

        #expect(model.isServerStoppingDuringStart)

        controller.emit(.stopped)
        try await waitUntil(deadline: 5) { model.serverState == .stopped }

        // A normal stop from `.running` must not carry the flag.
        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }
        model.stopServer()

        #expect(!model.isServerStoppingDuringStart)
    }

    @MainActor
    @Test func aLateRunningCallbackAfterAFailureIsIgnored() async throws {
        let directory = try makeCompleteModelInstall("server-control-late-running")
        defer { try? FileManager.default.removeItem(at: directory) }
        let controller = MockServerController()
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient(),
                             serverController: controller)

        model.startServer()
        controller.emit(.running(port: 8080))
        try await waitUntil(deadline: 5) { model.serverState == .running(port: 8080) }

        controller.emit(.failed(message: "crashed"))
        try await waitUntil(deadline: 5) {
            model.serverState == .failed(message: "crashed")
        }

        // A late `.running` racing in after the terminal `.failed` (the two
        // are reported through independent, unsynchronized callback paths in
        // `ProcessServerController`) must not resurrect the dead process.
        controller.emit(.running(port: 9999))
        try await Task.sleep(nanoseconds: 50_000_000)
        #expect(model.serverState == .failed(message: "crashed"))
    }

    @MainActor
    @Test func setServerPortRejectsOutOfRangeValues() {
        let missingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gturbo")
        let model = AppModel(modelDirectory: missingDirectory,
                             client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())
        let defaultPort = model.serverPort

        model.setServerPort(0)
        #expect(model.serverPort == defaultPort)

        model.setServerPort(70_000)
        #expect(model.serverPort == defaultPort)
    }

    @MainActor
    @Test func setServerQueueLimitRejectsOutOfRangeValues() {
        let missingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-\(UUID().uuidString).gturbo")
        let model = AppModel(modelDirectory: missingDirectory,
                             client: MockLifecycleInferenceClient(),
                             serverController: MockServerController())
        let defaultQueueLimit = model.serverQueueLimit

        model.setServerQueueLimit(0)
        #expect(model.serverQueueLimit == defaultQueueLimit)
    }

    @MainActor
    private func waitUntil(deadline seconds: TimeInterval,
                           _ condition: @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition(), Date() < deadline {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        if !condition() {
            Issue.record("timed out waiting for condition")
        }
    }
}
