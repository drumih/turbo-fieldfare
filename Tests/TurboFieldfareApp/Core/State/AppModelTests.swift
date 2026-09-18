import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppInferenceErrorDiagnosticTests {
    /// What crosses a process boundary is the cause, not the sentence. The
    /// decode service forwarded `"\(error)"` — which is the sentence — and the
    /// client wrapped it again, so a failed replay read "This conversation
    /// could not be reopened: This conversation could not be reopened: image
    /// support is unavailable: …". Found running case E46 with the vision pack
    /// moved aside.
    @Test func adiagnosticMessageCarriesTheCauseWithoutTheSentence() {
        let error = AppInferenceError.conversationRestoreFailed(
            "image support is unavailable")
        #expect(error.userMessage
            == "This conversation could not be reopened: image support is unavailable")
        #expect(error.diagnosticMessage == "image support is unavailable")
        // Re-wrapping the diagnostic says it once; re-wrapping the sentence
        // said it twice.
        let rewrapped = AppInferenceError.conversationRestoreFailed(
            error.diagnosticMessage)
        #expect(rewrapped.userMessage == error.userMessage)
    }

    /// A case with no message of its own falls back to its own wording rather
    /// than to an empty string.
    @Test func acaseWithoutAMessageStillReportsSomething() {
        #expect(!AppInferenceError.modelNotLoaded.diagnosticMessage.isEmpty)
        #expect(AppInferenceError.modelNotLoaded.diagnosticMessage
            == AppInferenceError.modelNotLoaded.userMessage)
    }
}

@Suite struct AppModelTests {

    @MainActor @Test func cacheSelectionClampsContextAndPersistsBothSettings() throws {
        let root = try makeSettingsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("model.gturbo")
        let model = AppModel(modelDirectory: directory, client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true, hostMemoryBytes: 8 << 30)
        model.setMaxContextTokens(131_072)
        model.setExpertCacheSlots(24)
        #expect(model.maxContextTokens == 65_536)
        #expect(model.contextClampNotice?.contains("131,072") == true)
        #expect(!model.contextOptions.contains(.oneTwentyEightK))
        let saved = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
        #expect(saved.contextTokens == 65_536)
        #expect(saved.expertCacheSlots == 24)
        model.setExpertCacheSlots(16)
        #expect(model.contextOptions.contains(.oneTwentyEightK))
        #expect(model.maxContextTokens == 65_536)
    }

    @MainActor @Test func storedCacheGrowthClampsOnLaunchAndModelChange() throws {
        let root = try makeSettingsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo")
        let second = root.appendingPathComponent("second/model.gturbo")
        for directory in [first, second] {
            try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 131_072,
                                                            expertCacheSlots: 24),
                                             forModelDirectory: directory)
        }
        let model = AppModel(modelDirectory: first, client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true, hostMemoryBytes: 8 << 30)
        #expect(model.maxContextTokens == 65_536)
        #expect(model.contextClampNotice != nil)
        model.setModelURL(second)
        #expect(model.maxContextTokens == 65_536)
        #expect(model.contextClampNotice != nil)
    }

    @MainActor @Test func loadRefusesCacheGrowthThatBypassedThePicker() throws {
        let directory = try makeCompleteModelInstall("cache-context-refusal")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        let model = AppModel(modelDirectory: directory, client: client, hostMemoryBytes: 8 << 30)
        model.setMaxContextTokens(131_072)
        model.runtimeOptions.expertCacheSlots = 32
        model.loadModel()
        guard case .failed(let error) = model.loadState else {
            Issue.record("unbacked cache/context combination reached loading")
            return
        }
        #expect("\(error)".contains("16 GB"))
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 8_192,
                "the reply limit follows the default context, 8K since 2026-08-17")
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
        #expect(request.runtimeOptions.expertCacheSlots == 16)
        #expect(request.runtimeOptions.expertCachePolicy == .lfu)
        #expect(request.runtimeOptions.rdadvisePolicy == .off)
        #expect(request.runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func runDisabledWhenPromptEmpty() {
        let model = AppModel()
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "   "
        #expect(!model.canRun)
    }

    @MainActor
    @Test func runDisabledUntilModelReady() {
        let model = AppModel()
        model.promptText = "go"
        #expect(!model.canRun)
    }

    @MainActor
    @Test func disablingTopKNeutralizesBothTruncationControls() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.topKEnabled = false
        model.topPEnabled = true

        let request = try model.makeRequest()
        #expect(request.topK == nil)
        #expect(request.topP == nil)
    }

    @MainActor
    @Test func prefillToggleSurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        model.runtimeOptions.prefillEnabled = false
        #expect(try !model.makeRequest().runtimeOptions.prefillEnabled)

        model.runtimeOptions.prefillEnabled = true
        #expect(try model.makeRequest().runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func adaptiveRDAdvicePolicySurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.runtimeOptions.rdadvisePolicy = .adaptive

        let request = try model.makeRequest()
        #expect(request.runtimeOptions.rdadvisePolicy == .adaptive)
    }

    @MainActor
    @Test func loadAffectingRuntimeChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func contextChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        // 8K is the default now, so changing to it changes nothing.
        model.setMaxContextTokens(AppContextLengthOption.sixteenK.tokens)
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.setMaxContextTokens(AppContextLengthOption.sixtyFourK.tokens)

        #expect(try model.makeRequest().maxNewTokens == AppContextLengthOption.sixtyFourK.tokens)
    }

    @MainActor
    @Test func requestTimePrefillChangeDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.runtimeOptions.prefillEnabled = false

        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func newlineShortcutDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setNewlineShortcut(.shiftReturn)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func promptExamplesPreferenceDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setShowPromptExamples(false)

        #expect(!model.showPromptExamples)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func promptExamplesOnlyShowBeforeTheFirstTurnWithAnEmptyComposer() async {
        let model = readyModel(client: MockInferenceClient(response: "answer"))

        #expect(model.shouldShowPromptExamples)
        model.promptText = "draft"
        #expect(!model.shouldShowPromptExamples)

        model.promptText = ""
        model.setShowPromptExamples(false)
        #expect(!model.shouldShowPromptExamples)

        model.setShowPromptExamples(true)
        model.promptText = "first turn"
        model.send()
        await waitForIdle(model)

        #expect(model.promptText.isEmpty)
        #expect(model.hasOutputTranscript)
        #expect(!model.shouldShowPromptExamples,
                "an empty composer must not cover an existing conversation with examples")

        model.newChat()
        #expect(model.shouldShowPromptExamples)
    }

    @MainActor
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.send()

        await SendWaiting.turnEnds(model)

        #expect(!model.isRunning)
        #expect(model.outputText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.send()
        // The composer clears on the click; the transcript picks the message up
        // one stage later, when the request has been built and its images
        // retained.
        #expect(model.promptText.isEmpty)
        await SendWaiting.generationStarts(model)

        #expect(model.outputPromptText == "original prompt")
        #expect(model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\noriginal prompt")

        model.promptText = "edited prompt"
        await waitForIdle(model)

        #expect(model.outputPromptText == "original prompt")
        #expect(model.outputResponsePlainText == "answer")
        #expect(model.outputConversationPlainText
            == "You:\noriginal prompt\n\nAnswer:\nanswer")
        #expect(!model.outputConversationPlainText.contains("edited prompt"))
    }

    @MainActor
    @Test func clearAfterSendingPreservesTranscriptAndNextDraft() async throws {
        let client = MockInferenceClient(
            response: "answer",
            tokenDelayNanos: 20_000_000)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1

        model.send()
        await SendWaiting.generationStarts(model)

        #expect(model.isRunning)
        #expect(model.outputPromptText == "original prompt")
        #expect(model.promptText.isEmpty)

        model.promptText = "next draft"
        await waitForIdle(model)

        #expect(model.promptText == "next draft")
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\noriginal prompt\n\nAnswer:\n"))
    }

    @MainActor
    @Test func failedValidationDoesNotClearPrompt() async {
        let model = readyModel(client: MockInferenceClient(response: "answer"))
        model.promptText = "keep invalid prompt"
        model.maxNewTokensOverride = 0

        model.send()
        // The composer is handed over before the request is built, so the
        // message comes back a hop later rather than never having left.
        await SendWaiting.turnEnds(model)

        #expect(!model.isRunning)
        #expect(model.promptText == "keep invalid prompt")
        #expect(model.outputPromptText.isEmpty)
        #expect(model.error != nil)
    }

    @MainActor
    @Test func staleReadySessionDisablesGenerationUntilReload() throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("stale-runtime")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "go"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.canRun)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
        #expect(!model.canRun)
        #expect(model.canReloadModel)
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func cancelAfterPartialOutputCanBeCleared() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "stop after token"
        model.maxNewTokensOverride = 10
        model.send()

        for _ in 0..<200 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.liveTokenCount > 0)
        model.cancel()
        #expect(model.isCancellationPending)
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(!model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\nstop after token\n\nAnswer:\n"))

        model.newChat()
        #expect(!model.hasOutputTranscript)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func cancelDuringPrefillKeepsPromptSnapshotUntilClear() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 20
        let model = readyModel(client: client)
        model.promptText = "prefill prompt"
        model.send()

        for _ in 0..<200 where model.livePrefillDone == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.outputPromptText == "prefill prompt")
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.outputPromptText == "prefill prompt")
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\nprefill prompt")
        #expect(model.hasOutputTranscript)

        model.newChat()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.send()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        let model = AppModel(client: MockInferenceClient(),
                             installer: MockModelInstallerClient())
        let testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("app-model-path-\(UUID().uuidString)", isDirectory: true)
        let oldURL = testDirectory.appendingPathComponent("old.gturbo")
        let newURL = testDirectory.appendingPathComponent("new.gturbo")
        model.modelPathText = oldURL.path
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)
        model.diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 1,
            tokensPerSecond: 1,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        model.error = .unknown("old error")

        model.setModelURL(newURL)

        #expect(model.modelPathText == newURL.standardizedFileURL.path)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
        #expect(model.presentation.label == "Model required")
        #expect(!model.canRun)
    }

    /// A settings file can carry a context this Mac cannot back — written on a
    /// larger machine, or on a build whose admission rule was looser. Opening
    /// the app on it would put every load into `.failed` with no way back
    /// except a menu the user has to guess at, so it is clamped and said.
    @MainActor
    @Test func unavailableStoredContextIsClampedWithNotice() throws {
        let root = try makeSettingsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 262_144),
                                         forModelDirectory: modelDirectory)

        let model = AppModel(modelDirectory: modelDirectory,
                             client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true,
                             hostMemoryBytes: 8_589_934_592)

        #expect(model.maxContextTokens == 131_072)
        let notice = try #require(model.contextClampNotice,
                                  "the context was changed without saying so")
        #expect(notice.contains("262,144"))
        #expect(notice.contains("16 GB"))
        #expect(notice.contains("128K"))

        model.setMaxContextTokens(model.maxContextTokens)
        #expect(model.contextClampNotice == notice)

        // Context edits must go through the history-aware setter after the merge;
        // direct assignment clears the notice but loses persistence and redraw.
        model.setMaxContextTokens(65_536)
        #expect(model.contextClampNotice == nil)
        #expect(MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory).contextTokens == 65_536)
    }

    /// The same clamp on the settings-apply path, which `init` does not cover:
    /// switching model directories adopts that directory's settings, and one
    /// of them can carry a context this Mac cannot back. The notice also has
    /// to survive being set — assigning the context clears it, so an
    /// assignment made after the notice would wipe it.
    @MainActor
    @Test func achangedModelDirectoryClampsItsStoredContextToo() throws {
        let root = try makeSettingsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo", isDirectory: true)
        let second = root.appendingPathComponent("second/model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 8_192),
                                         forModelDirectory: first)
        try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 262_144),
                                         forModelDirectory: second)
        let model = AppModel(modelDirectory: first,
                             client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true,
                             hostMemoryBytes: 8_589_934_592)
        try #require(model.contextClampNotice == nil)

        model.setModelURL(second)

        #expect(model.maxContextTokens == 131_072)
        #expect(model.contextClampNotice?.contains("262,144") == true,
                "the clamp on the settings-apply path said nothing")
    }

    @MainActor
    @Test func availableStoredContextIsKept() throws {
        let root = try makeSettingsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(MacAppSettings(contextTokens: 262_144),
                                         forModelDirectory: modelDirectory)

        let model = AppModel(modelDirectory: modelDirectory,
                             client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true,
                             hostMemoryBytes: 25_769_803_776)

        #expect(model.maxContextTokens == 262_144)
        #expect(model.contextClampNotice == nil)
    }

    /// The refusal has to happen before the load state moves, or the app sends
    /// the decode service a context that gets the service killed part-way
    /// through — which reaches the user as a lost connection, not as a reason.
    @MainActor
    @Test func loadRefusesUnavailableContextBeforeLoading() async throws {
        let directory = try makeCompleteModelInstall("context-refusal")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        let model = AppModel(modelDirectory: directory,
                             client: client,
                             hostMemoryBytes: 8_589_934_592)
        model.setMaxContextTokens(262_144)
        try #require(model.canLoadModel)

        model.loadModel()

        guard case .failed(let error) = model.loadState else {
            Issue.record("an unbacked context was allowed to start loading")
            return
        }
        #expect("\(error)".contains("262,144"))
        #expect(client.ensureLoadedCallCount() == 0,
                "the loader was reached with a context this host cannot back")
        #expect(!model.loadState.isLoading)
    }

    @Test func contextOptionsNoteNamesHiddenSizes() throws {
        let note = try #require(
            AppModel.contextOptionsNote(hostMemoryBytes: 8_589_934_592))
        #expect(note.contains("262,144"))
        #expect(note.contains("16 GB"))
        #expect(note.contains("8 GB"))
        // 128K is offered on this host, so it is not something to explain away.
        #expect(!note.contains("131,072"))
    }

    @Test func contextOptionsNoteIsAbsentWhenEverySizeIsOffered() {
        #expect(AppModel.contextOptionsNote(hostMemoryBytes: 25_769_803_776) == nil)
    }

    private func makeSettingsRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelContext-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: true)
        return root
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        await SendWaiting.turnEnds(model)
    }
}
