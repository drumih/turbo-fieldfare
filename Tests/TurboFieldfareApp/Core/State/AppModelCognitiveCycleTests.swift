import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct AppModelCognitiveCycleTests {
    private func makeModel(client: AppInferenceClient) -> AppModel {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("cognitive-\(UUID().uuidString).json")
        let model = AppModel(client: client,
                             conversationStore: ConversationStore(storageURL: url))
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        return model
    }

    private func waitUntilIdle(_ model: AppModel, timeoutMilliseconds: Int = 15_000) async {
        var waited = 0
        while model.runState == .running && waited < timeoutMilliseconds {
            try? await Task.sleep(for: .milliseconds(20))
            waited += 20
        }
    }

    @Test
    func cognitiveCycleRunsFourPasses() async throws {
        let mock = MockInferenceClient(response: "pass output",
                                       tokenDelayNanos: 1_000_000)
        let model = makeModel(client: mock)
        model.promptText = "Explain Metal"
        model.cognitiveModeEnabled = true

        model.run()
        await waitUntilIdle(model)

        #expect(model.runState == .idle)
        #expect(model.error == nil)
        #expect(model.outputText.contains("— Plan —"))
        #expect(model.outputText.contains("— Draft —"))
        #expect(model.outputText.contains("— Critique —"))
        #expect(model.outputText.contains("— Final answer —"))
        // Four passes, each echoing the pass prompt back through the mock.
        #expect(model.outputText.contains("Plan your answer"))
    }

    @Test
    func cognitiveCycleStoresFinalRevisionInHistory() async throws {
        let mock = MockInferenceClient(response: "revision text",
                                       tokenDelayNanos: 1_000_000)
        let model = makeModel(client: mock)
        model.promptText = "Explain Metal"
        model.cognitiveModeEnabled = true

        model.run()
        await waitUntilIdle(model)

        #expect(model.conversationStore.conversations.count == 1)
        let saved = model.conversationStore.conversations[0]
        // History stores the final revision, not the whole cycle transcript.
        #expect(!saved.response.contains("— Plan —"))
        #expect(saved.response.contains("revision text"))
    }

    @Test
    func standardGenerationRunsWhenCognitiveDisabled() async throws {
        let mock = MockInferenceClient(response: "plain answer",
                                       tokenDelayNanos: 1_000_000)
        let model = makeModel(client: mock)
        model.promptText = "Explain Metal"

        model.run()
        await waitUntilIdle(model)

        #expect(model.error == nil)
        #expect(!model.outputText.contains("— Plan —"))
        #expect(model.outputText.contains("plain answer"))
    }

    @Test
    func cognitiveCycleStopsOnCancel() async throws {
        // Long response so cancellation lands mid-pass.
        let longResponse = String(repeating: "word ", count: 300)
        let mock = MockInferenceClient(response: longResponse,
                                       tokenDelayNanos: 2_000_000)
        let model = makeModel(client: mock)
        model.promptText = "Explain Metal"
        model.cognitiveModeEnabled = true

        model.run()
        try? await Task.sleep(for: .milliseconds(80))
        model.cancel()
        await waitUntilIdle(model)

        #expect(model.runState == .idle)
        #expect(model.error == .cancelled)
        #expect(!model.outputText.contains("— Final answer —"))
    }
}
