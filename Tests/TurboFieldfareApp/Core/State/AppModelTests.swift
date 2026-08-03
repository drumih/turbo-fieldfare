import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelTests {
    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 4_096)
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
        model.maxContextTokens = AppContextLengthOption.eightK.tokens
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens

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
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.run()

        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(!model.isRunning)
        #expect(model.outputText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
        #expect(model.promptText.isEmpty)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.run()

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
    @Test func completedTurnIsIncludedInTheNextGenerationRequest() async throws {
        let client = MockInferenceClient(response: "first answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 8
        model.promptText = "first question"
        model.run()
        await waitForIdle(model)

        model.promptText = "follow-up question"
        model.run()
        await waitForIdle(model)

        let requests = client.requests()
        #expect(requests.count == 2)
        #expect(requests[1].messages == [
            AppChatMessage(role: .user, content: "first question"),
            AppChatMessage(
                role: .assistant,
                content: "first answer Prompt received: first question"),
            AppChatMessage(role: .user, content: "follow-up question"),
        ])
        #expect(model.outputConversationPlainText.contains("first question"))
        #expect(model.outputConversationPlainText.contains("follow-up question"))
    }

    @MainActor
    @Test func newChatKeepsThePreviousConversationAvailable() async throws {
        let model = readyModel(client: MockInferenceClient(response: "first answer", tokenDelayNanos: 1))
        model.maxNewTokensOverride = 8
        model.promptText = "First question"
        model.run()
        await waitForIdle(model)

        model.promptText = "Second question"
        model.run()
        await waitForIdle(model)

        let firstChatID = model.selectedChatID
        #expect(model.chats.count == 1)
        #expect(model.chats[0].title == "First question")

        model.newChat()

        #expect(model.chats.count == 2)
        #expect(model.selectedChatID != firstChatID)
        let newChatID = model.selectedChatID
        #expect(model.chats.first?.id == newChatID)
        #expect(model.conversation.isEmpty)
        #expect(!model.hasOutputTranscript)

        model.selectChat(firstChatID)

        #expect(model.selectedChatID == firstChatID)
        #expect(model.chats.first?.id == newChatID)
        #expect(model.conversation == [
            AppChatMessage(role: .user, content: "First question"),
            AppChatMessage(
                role: .assistant,
                content: "first answer Prompt received: First question"),
            AppChatMessage(role: .user, content: "Second question"),
            AppChatMessage(
                role: .assistant,
                content: "first answer Prompt received: Second question"),
        ])
    }

    @MainActor
    @Test func repeatedNewChatReusesTheSingleEmptyChat() {
        let model = AppModel()
        let initialID = model.selectedChatID

        model.promptText = "unsent draft"
        model.newChat()
        model.newChat()

        #expect(model.chats.count == 1)
        #expect(model.selectedChatID == initialID)
        #expect(model.promptText.isEmpty)
        #expect(model.conversation.isEmpty)
        #expect(model.outputText.isEmpty)
    }

    @MainActor
    @Test func savedConversationRestoresAcrossModelInstances() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelChatPersistence-\(UUID().uuidString)",
                                    isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel(
            modelDirectory: directory,
            client: MockInferenceClient(response: "answer", tokenDelayNanos: 1),
            settingsPersistenceEnabled: true)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 8
        model.promptText = "Remember this"
        model.run()
        await waitForIdle(model)
        model.promptText = "Follow up"
        model.run()
        await waitForIdle(model)

        let restored = AppModel(
            modelDirectory: directory,
            client: MockInferenceClient(),
            settingsPersistenceEnabled: true)

        #expect(restored.conversation.count == 4)
        #expect(restored.conversation[0] ==
            AppChatMessage(role: .user, content: "Remember this"))
        #expect(restored.conversation[2] ==
            AppChatMessage(role: .user, content: "Follow up"))
    }

    @MainActor
    @Test func systemInstructionsArePrependedWithoutAppearingInTheTranscript() throws {
        let model = readyModel(client: MockInferenceClient())
        model.systemPromptText = "Reply with exactly one sentence."
        model.promptText = "What is the capital of France?"

        let request = try model.makeRequest()
        #expect(request.messages == [
            AppChatMessage(role: .system, content: "Reply with exactly one sentence."),
            AppChatMessage(role: .user, content: "What is the capital of France?"),
        ])
        #expect(!model.outputConversationPlainText.contains("exactly one sentence"))
    }

    @MainActor
    @Test func regenerateRemovesThePreviousAssistantTurnBeforeRetrying() async throws {
        let client = MockInferenceClient(response: "retry answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 8
        model.promptText = "retry this"
        model.run()
        await waitForIdle(model)

        #expect(model.canRegenerate)
        model.regenerateLastResponse()
        await waitForIdle(model)

        let requests = client.requests()
        #expect(requests.count == 2)
        #expect(requests[1].messages == [
            AppChatMessage(role: .user, content: "retry this"),
        ])
    }

    @MainActor
    @Test func latestPromptEditReplacesOnlyTheNewestTurnAndPreservesContext() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 8

        model.promptText = "first question"
        model.run()
        await waitForIdle(model)
        let firstAnswer = model.conversation[1]

        model.promptText = "second question"
        model.run()
        await waitForIdle(model)

        #expect(model.canEditLastPrompt)
        #expect(model.submitEditedLastPrompt("edited second question"))
        #expect(!model.canEditLastPrompt)
        await waitForIdle(model)

        let requests = client.requests()
        #expect(requests.count == 3)
        #expect(requests[2].messages == [
            AppChatMessage(role: .user, content: "first question"),
            firstAnswer,
            AppChatMessage(role: .user, content: "edited second question"),
        ])
        #expect(model.conversation.count == 4)
        #expect(model.conversation[0] ==
            AppChatMessage(role: .user, content: "first question"))
        #expect(model.conversation[1] == firstAnswer)
        #expect(model.conversation[2] ==
            AppChatMessage(role: .user, content: "edited second question"))
        #expect(model.conversation[3].role == .assistant)
        #expect(model.conversation[3].content.contains("edited second question"))
        #expect(!model.conversation.contains(where: { $0.content == "second question" }))
    }

    @MainActor
    @Test func latestPromptEditRequiresACompletedAnswerAndNonemptyReplacement() async throws {
        let client = MockInferenceClient(
            response: "answer",
            tokenDelayNanos: 20_000_000)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 4

        #expect(!model.canEditLastPrompt)
        #expect(!model.submitEditedLastPrompt("not sent"))

        model.promptText = "original question"
        model.run()
        #expect(!model.canEditLastPrompt)
        await waitForIdle(model)

        let completedConversation = model.conversation
        #expect(model.canEditLastPrompt)
        #expect(model.canSubmitEditedLastPrompt)
        #expect(!model.submitEditedLastPrompt("   \n"))
        #expect(model.conversation == completedConversation)

        model.loadState = .notLoaded
        #expect(model.canEditLastPrompt)
        #expect(!model.canSubmitEditedLastPrompt)
        #expect(!model.submitEditedLastPrompt("edited while unloaded"))
        #expect(model.conversation == completedConversation)
        #expect(client.requests().count == 1)
    }

    @MainActor
    @Test func failedLatestPromptEditRestoresTheOriginalPromptAndAnswer() async throws {
        let client = MockInferenceClient(response: "original answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 8
        model.promptText = "original question"
        model.run()
        await waitForIdle(model)
        let originalConversation = model.conversation

        client.failureMessage = "edited answer failed"
        #expect(model.submitEditedLastPrompt("edited question"))
        await waitForIdle(model)

        #expect(model.error?.userMessage == "edited answer failed")
        #expect(model.conversation == originalConversation)
        #expect(model.outputPromptText == "original question")
        #expect(model.outputText == originalConversation.last?.content)
    }

    @MainActor
    @Test func successfulLatestPromptEditPersistsAcrossRestoration() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "AppModelEditedChat-\(UUID().uuidString)",
                isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 8
        model.promptText = "original question"
        model.run()
        await waitForIdle(model)

        #expect(model.submitEditedLastPrompt("edited question"))
        await waitForIdle(model)

        let restored = AppModel(
            modelDirectory: directory,
            client: MockInferenceClient(),
            settingsPersistenceEnabled: true)
        #expect(restored.conversation.count == 2)
        #expect(restored.conversation[0] ==
            AppChatMessage(role: .user, content: "edited question"))
        #expect(restored.conversation[1].role == .assistant)
        #expect(restored.conversation[1].content.contains("edited question"))
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
        model.run()

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

        model.clearOutput()
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
        model.run()

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

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func immediateStopFinalizesTheSubmittedTurn() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 8
        let model = readyModel(client: client)
        model.promptText = "stop immediately"

        model.run()
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        #expect(model.conversation == [
            AppChatMessage(role: .user, content: "stop immediately"),
        ])
    }

    @MainActor
    @Test func cancelledTurnPersistsAcrossRestoration() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelCancelledChat-\(UUID().uuidString)",
                                    isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let client = MockInferenceClient(
            response: "one two three four five",
            tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.promptText = "keep my stopped answer"
        model.run()

        for _ in 0..<200 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        model.cancel()
        await waitForIdle(model)

        let restored = AppModel(
            modelDirectory: directory,
            client: MockInferenceClient(),
            settingsPersistenceEnabled: true)
        #expect(restored.conversation.first ==
            AppChatMessage(role: .user, content: "keep my stopped answer"))
        #expect(restored.conversation.last?.role == .assistant)
        #expect(!(restored.conversation.last?.content ?? "").isEmpty)
    }

    @MainActor
    @Test func failedRegenerationRestoresThePreviousAnswer() async throws {
        let client = MockInferenceClient(response: "original answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.maxNewTokensOverride = 8
        model.promptText = "regenerate safely"
        model.run()
        await waitForIdle(model)
        let originalConversation = model.conversation

        client.failureMessage = "replacement failed"
        model.regenerateLastResponse()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "replacement failed")
        #expect(model.conversation == originalConversation)
        #expect(model.outputText == originalConversation.last?.content)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.run()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func imageAttachmentPersistsAcrossSubmitChatSwitchAndLatestEdit() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "AppModelImageAttachment-\(UUID().uuidString)",
            isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("question.png")
        try Self.onePixelPNG.write(to: source)

        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = AppModel(
            modelDirectory: directory,
            client: client,
            settingsPersistenceEnabled: true)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.maxNewTokensOverride = 8

        #expect(await model.attachImage(at: source))
        let image = try #require(model.pendingImage)
        let managedURL = try #require(model.imageURL(for: image))
        model.promptText = "What is shown?"
        model.run()

        #expect(model.pendingImage == nil)
        #expect(model.outputMessages.last?.images == [image])
        await waitForIdle(model)
        #expect(model.conversation.first?.images == [image])
        #expect(FileManager.default.fileExists(atPath: managedURL.path))

        let imageChatID = model.selectedChatID
        model.newChat()
        model.selectChat(imageChatID)
        #expect(model.conversation.first?.images == [image])

        #expect(model.submitEditedLastPrompt("Describe the image."))
        await waitForIdle(model)
        #expect(client.requests().last?.messages.last?.images == [image])
        #expect(model.conversation.first?.content == "Describe the image.")
        #expect(model.conversation.first?.images == [image])

        let restored = AppModel(
            modelDirectory: directory,
            client: MockInferenceClient(),
            settingsPersistenceEnabled: true)
        #expect(restored.conversation.first?.images == [image])
        #expect(restored.imageURL(for: image) == managedURL)
    }

    @MainActor
    @Test func removingPendingImageDeletesItsManagedCopy() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "AppModelPendingImage-\(UUID().uuidString)",
            isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("question.png")
        try Self.onePixelPNG.write(to: source)
        let model = AppModel(modelDirectory: directory)

        #expect(await model.attachImage(at: source))
        let image = try #require(model.pendingImage)
        let managedURL = try #require(model.imageURL(for: image))
        #expect(FileManager.default.fileExists(atPath: managedURL.path))

        model.removePendingImage()

        #expect(model.pendingImage == nil)
        #expect(!FileManager.default.fileExists(atPath: managedURL.path))
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        let model = AppModel(client: MockInferenceClient())
        let oldURL = FileManager.default.temporaryDirectory.appendingPathComponent("old.gturbo")
        let newURL = FileManager.default.temporaryDirectory.appendingPathComponent("new.gturbo")
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

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private static let onePixelPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")!
}
