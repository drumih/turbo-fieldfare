import AppKit
import TurboFieldfareAppCore

@MainActor
enum ModelLocationPicker {
    static func choose(for model: AppModel) {
        let currentURL = URL(fileURLWithPath: model.modelPathText, isDirectory: true)
        let panel = NSOpenPanel()
        panel.title = "Choose Model Folder"
        panel.message = "Choose the .gturbo folder that contains manifest.json."
        panel.prompt = "Choose Model"
        panel.directoryURL = currentURL.deletingLastPathComponent()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.resolvesAliases = true

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        let selectedPath = selectedURL.standardizedFileURL.path
        model.setModelURL(selectedURL)
        guard model.modelPathText == selectedPath else { return }
        AppModelLocation.remember(selectedURL)
    }
}
