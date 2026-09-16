import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppTextSizeTests {
    @Test func menuHasExactlyTheFiveAgreedPresets() {
        #expect(AppTextSize.allCases.map(\.rawValue) == [100, 125, 150, 175, 200])
        #expect(AppTextSize.allCases.map(\.label) == ["100%", "125%", "150%", "175%", "200%"])
    }

    @Test(arguments: AppTextSize.allCases)
    func presetsRoundTrip(_ size: AppTextSize) throws {
        let settings = MacAppSettings(textSize: size)
        #expect(try JSONDecoder().decode(MacAppSettings.self,
                                        from: JSONEncoder().encode(settings)) == settings)
        #expect(size.scale == Double(size.rawValue) / 100)
    }

    // Throwing this appearance field into the store's recovery path used to
    // delete the settings file, resetting unrelated context/sampling choices.
    @Test(arguments: ["110", "999", "null", "\"private arbitrary string\"", "{}", "1.5"])
    func malformedAppearanceKeepsOtherSettingsAndFile(_ value: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("model.gturbo")
        let settings = MacAppSettings(contextTokens: 4096, temperature: 0.7,
                                      showPromptExamples: false, textSize: .largest)
        try MacAppSettingsFileStore.save(settings, forModelDirectory: directory)
        let file = MacAppSettingsFileStore.fileURL(forModelDirectory: directory)
        let encoded = try #require(String(data: JSONEncoder().encode(settings), encoding: .utf8))
        let broken = Data(encoded.replacingOccurrences(of: "\"textSize\":200",
                                                       with: "\"textSize\":\(value)").utf8)
        #expect(broken != Data(encoded.utf8))
        try broken.write(to: file)
        let read = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
        #expect(read.textSize == .standard)
        #expect(read.contextTokens == 4096)
        #expect(read.temperature == 0.7)
        #expect(!read.showPromptExamples)
        #expect(try Data(contentsOf: file) == broken)
    }

    @Test func absentFieldIsBackwardCompatible() throws {
        let original = MacAppSettings(contextTokens: 4096)
        var object = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(original)) as? [String: Any])
        object.removeValue(forKey: "textSize")
        let decoded = try JSONDecoder().decode(MacAppSettings.self,
            from: JSONSerialization.data(withJSONObject: object))
        #expect(decoded == original)
        #expect(decoded.version == MacAppSettings.currentVersion)
    }

    @MainActor @Test func selectionPersistsBeforeSendAndFollowsModelLocation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo")
        let second = root.appendingPathComponent("second/model.gturbo")
        try MacAppSettingsFileStore.save(.init(textSize: .large), forModelDirectory: second)
        let model = AppModel(modelDirectory: first, client: MockLifecycleInferenceClient(),
                             settingsPersistenceEnabled: true)
        model.setTextSize(.largest)
        model.setTextSize(.largest)
        #expect(MacAppSettingsFileStore.loadOrCreate(forModelDirectory: first).textSize == .largest)
        let reopened = AppModel(modelDirectory: first, client: MockLifecycleInferenceClient(),
                                settingsPersistenceEnabled: true)
        #expect(reopened.textSize == .largest)
        model.setModelURL(second)
        #expect(model.textSize == .large)
        model.setModelURL(first)
        #expect(model.textSize == .largest)
    }

    @MainActor @Test func appearanceDoesNotInvalidateLoadedSessionOrDraft() throws {
        let directory = try makeCompleteModelInstall("text-size")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        let model = AppModel(modelDirectory: directory, client: client)
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        model.promptText = "Preserve my draft"
        let key = model.loadedRuntimeKey
        let epoch = model.displayedTranscriptID
        let conversationEpoch = model.conversation.epoch
        for size in AppTextSize.allCases {
            model.setTextSize(size)
            #expect(model.loadedRuntimeKey == key)
            #expect(model.displayedTranscriptID == epoch)
            #expect(model.conversation.epoch == conversationEpoch)
            #expect(model.promptText == "Preserve my draft")
            #expect(model.canRun)
            #expect(!model.hasStaleLoadedRuntime)
            #expect(client.ensureLoadedCallCount() == 0)
        }
    }

    @MainActor @Test func sizeCanChangeDuringMockStreamingWithoutCancelling() async {
        let client = MockInferenceClient(response: "one two three four", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = AppModel(client: client)
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.promptText = "go"
        model.send()
        await SendWaiting.generationStarts(model)
        model.setTextSize(.largest)
        #expect(model.textSize == .largest)
        #expect(model.isRunning)
        await SendWaiting.turnEnds(model)
        #expect(model.outputText.contains("four"))
        #expect(model.error == nil)
    }
}
