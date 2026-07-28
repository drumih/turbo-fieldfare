import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelServiceReportingTests {
    @MainActor
    @Test func serviceMemoryAndCanonicalTranscriptOverrideUIProcessState() {
        let client = ReportingInferenceClient(memoryBytes: 2_100_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        client.generationTranscriptMailbox.append("lossless output")

        #expect(model.currentProcessMemoryBytes == 2_100_000_000)
        #expect(model.outputResponsePlainText == "lossless output")
        #expect(model.outputConversationPlainText == "Answer:\nlossless output")

        model.clearOutput()
        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
    }

    @MainActor
    @Test func startingAnotherRunClearsPreviousServiceTranscriptSynchronously() {
        let client = ReportingInferenceClient(memoryBytes: 2_100_000_000)
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        client.generationTranscriptMailbox.append("previous completion")
        model.promptText = "new prompt"

        model.run()

        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputPromptText == "new prompt")
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\nnew prompt")
    }

    @MainActor
    @Test func switchingCompletedChatsUsesPersistedAnswerAfterMailboxReset() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "AppModelMailboxChatSwitchTests-\(UUID().uuidString)",
                isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "model.gturbo",
            isDirectory: true)
        let first = AppChat(
            title: "First",
            messages: [
                AppChatMessage(role: .user, content: "First question"),
                AppChatMessage(role: .assistant, content: "First answer"),
            ])
        let second = AppChat(
            title: "Second",
            messages: [
                AppChatMessage(role: .user, content: "Second question"),
                AppChatMessage(role: .assistant, content: "Second answer"),
            ])
        try AppChatFileStore.save(
            AppChatArchive(
                selectedChatID: first.id,
                chats: [first, second]),
            forModelDirectory: modelDirectory)
        let client = ReportingInferenceClient(memoryBytes: 0)
        let model = AppModel(
            modelDirectory: modelDirectory,
            client: client,
            settingsPersistenceEnabled: true)

        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputText == "First answer")
        #expect(model.outputResponsePlainText == "First answer")

        client.generationTranscriptMailbox.append("stale mailbox")
        model.selectChat(id: second.id)

        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(model.outputText == "Second answer")
        #expect(model.outputResponsePlainText == "Second answer")
        #expect(model.outputConversationPlainText.contains("Second answer"))
        #expect(!model.outputConversationPlainText.contains("First answer"))
        model.flushChatPersistence()
    }
}

private final class ReportingInferenceClient: AppInferenceClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    let currentInferenceMemoryBytes: UInt64?
    let generationTranscriptMailbox = GenerationTranscriptMailbox()

    init(memoryBytes: UInt64) {
        currentInferenceMemoryBytes = memoryBytes
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}
