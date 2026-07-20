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
