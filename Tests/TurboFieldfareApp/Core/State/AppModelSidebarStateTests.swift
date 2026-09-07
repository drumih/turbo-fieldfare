import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The button, the View menu item and Cmd+Ctrl+S are one action against one
/// persisted setting. Three places reading three different states is how a
/// toggle ends up saying "Hide" while the sidebar is already hidden.
@Suite struct AppModelSidebarStateTests {
    @MainActor
    @Test func terminationPersistsUnsavedSamplingChanges() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("quit-settings-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        // Slider bindings update the model without invoking a persistence action.
        model.temperature = 0.35
        model.topK = 17
        model.topP = 0.8
        model.shutdownForTermination()
        let reopened = AppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        #expect(reopened.temperature == 0.35)
        #expect(reopened.topK == 17)
        #expect(reopened.topP == 0.8)
    }

    @MainActor
    @Test func theToggleFlipsAndPersists() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sidebar-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel(modelDirectory: directory,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        #expect(model.isSidebarVisible, "the list is shown by default")

        model.toggleSidebar()
        #expect(!model.isSidebarVisible)

        // A new model over the same directory has to come back hidden: the
        // window remembering its shape is the whole reason this is persisted.
        let reopened = AppModel(modelDirectory: directory,
                                settingsPersistenceEnabled: true)
        #expect(!reopened.isSidebarVisible)

        model.setSidebarVisible(true)
        let again = AppModel(modelDirectory: directory,
                             settingsPersistenceEnabled: true)
        #expect(again.isSidebarVisible)
    }

    /// The Inspector toggle is the sidebar toggle's mirror and is persisted
    /// separately: hiding one panel must not move the other, or the window
    /// comes back a shape nobody asked for.
    @MainActor
    @Test func theInspectorTogglePersistsIndependentlyOfTheSidebar() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("inspector-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel(modelDirectory: directory,
                             settingsPersistenceEnabled: true)
        model.modelPathText = directory.path
        #expect(model.isInspectorVisible, "the Inspector is shown by default")

        model.toggleInspector()
        #expect(!model.isInspectorVisible)
        #expect(model.isSidebarVisible, "hiding one panel moved the other")

        let reopened = AppModel(modelDirectory: directory,
                                settingsPersistenceEnabled: true)
        #expect(!reopened.isInspectorVisible)
        #expect(reopened.isSidebarVisible)
    }

    /// New Chat is guarded rather than silently doing nothing, so the button
    /// and the menu item can be disabled instead of looking broken.
    @MainActor
    @Test func newChatIsRefusedWhileRunningAndBeforeAModelExists() async throws {
        let client = FakeInferenceClient(eventDelay: .milliseconds(20))
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path

        // No model installed yet: there is no chat to start.
        #expect(model.requiresModelInstallation)
        #expect(!model.canStartNewChat)

        try await client.ensureLoaded(
            modelDirectory: directory,
            maxContextTokens: model.maxContextTokens,
            options: model.runtimeOptions,
            forceLogitsHead: true) { _ in }
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1)
        model.installationStatus = .complete
        #expect(model.canStartNewChat)

        model.promptText = "hello"
        model.send()
        await SendWaiting.generationStarts(model)
        #expect(model.isRunning)
        #expect(!model.canStartNewChat)
        let epoch = model.conversation.epoch
        // And the guard is real, not only advisory: calling it anyway must not
        // pull the lineage out from under a running turn.
        model.newChat()
        #expect(model.conversation.epoch == epoch)

        await SendWaiting.turnEnds(model)
        #expect(model.canStartNewChat)
    }
}
