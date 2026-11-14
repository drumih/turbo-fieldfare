import AppKit
import SwiftUI
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

struct ModelLocationView: View {
    let model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Download location")
                    .font(.callout.weight(.medium))
                Spacer()
                Button("Change…") {
                    ModelLocationPanels.choose(model: model, intent: .download)
                }
                .buttonStyle(.borderless)
                .help("Choose download location")
                .disabled(!model.canChangeModelLocation)
                .accessibilityIdentifier(.modelChooseDownload)
            }
            Text(model.modelPathText)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(2, reservesSpace: true)
                .truncationMode(.middle)
                .help(model.modelPathText)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct ModelLocationMenu: View {
    let model: AppModel

    var body: some View {
        Menu("Location") {
            Button("Change Model Location…") {
                ModelLocationPanels.choose(model: model, intent: .existing, title: "Change Model Location")
            }
            .disabled(!model.canChangeModelLocation)
            .accessibilityIdentifier(.modelChangeLocation)
            Divider()
            Button("Reveal in Finder") { ModelLocationPanels.reveal(model) }
                .disabled(ModelLocationPanels.revealTarget(model) == .unavailable)
                .accessibilityIdentifier(.modelReveal)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Model location actions")
        .accessibilityIdentifier(.modelLocationMenu)
    }
}

struct ModelLocationNotices: View {
    let model: AppModel
    var showsVolumeWarning = true

    var body: some View {
        Group {
            if model.isInspectingModelLocation {
                Text("Checking models in the selected folder…")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if model.isModelChoiceRequired {
                Text("Multiple existing installations were found. Choose which model to use.")
                    .font(.caption).foregroundStyle(.orange)
            }
            if model.isModelLocationOverridden {
                Text("Model location is overridden for this launch. Relaunch without the override to save a different location.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if showsVolumeWarning && !model.isModelLocationAvailable {
                Text("The selected volume is unavailable. Reconnect it and choose Check Again, or choose another location.")
                    .font(.caption).foregroundStyle(.orange)
            }
            if let message = model.modelLocationDiagnostic {
                Text(message).font(.caption).foregroundStyle(.red)
            }
            if let message = model.settingsDiagnostic {
                Text(message).font(.caption).foregroundStyle(.red)
                Button("Reveal Settings") { ModelLocationPanels.revealSettings(model) }
                    .accessibilityIdentifier(.settingsReveal)
                if model.canResetSettings {
                    Button("Back Up and Reset Settings", action: model.resetSettingsAfterBackingUp)
                        .accessibilityIdentifier(.settingsRecover)
                }
            }
        }
    }
}

@MainActor
enum ModelLocationPanels {
    private static var activePanel: NSOpenPanel?
    static func choose(model: AppModel, intent: AppModel.ModelLocationIntent, title: String? = nil) {
        guard model.canChangeModelLocation, model.existingModelReview == nil, activePanel == nil else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = intent == .download
        panel.treatsFilePackagesAsDirectories = true
        panel.title = title ?? (intent == .download ? "Choose Download Location" : "Use Existing Model")
        panel.message = intent == .download
            ? "Choose a parent folder. The model will be installed in gemma4.gturbo inside it."
            : "Choose a folder containing your text and vision model packs, or a text .gturbo folder directly. If multiple usable text models are found, you can choose which to use."
        panel.prompt = intent == .download ? "Choose Destination" : "Select Folder"
        present(panel) { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                if intent == .existing { await model.reviewExistingModelLocation(url) }
                else { await model.selectModelLocation(url, intent: intent) }
            }
        }
    }

    static func importHistory(model: AppModel) {
        guard model.canChangeModelLocation, activePanel == nil else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = true
        panel.title = "Import Legacy Chats"
        panel.message = "Choose the old .gturbo model directory. Chats beside it will be copied and verified; originals stay in place."
        present(panel) { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in await model.importLegacyHistory(from: url) }
        }
    }

    private static func present(_ panel: NSOpenPanel, completion: @escaping @MainActor (NSApplication.ModalResponse) -> Void) {
        guard let window = NSApp.keyWindow else { return }
        activePanel = panel
        panel.beginSheetModal(for: window) { response in
            activePanel = nil
            completion(response)
        }
    }

    static func reveal(_ model: AppModel) {
        switch revealTarget(model) {
        case .selectItem(let url): NSWorkspace.shared.activateFileViewerSelecting([url])
        case .openContainer(let url): NSWorkspace.shared.open(url)
        case .unavailable: break
        }
    }

    static func revealTarget(_ model: AppModel) -> ModelRevealTarget {
        ModelRevealPolicy.target(forModelPath: model.modelPathText,
                                fileExists: FileManager.default.fileExists(atPath:))
    }

    static func revealSettings(_ model: AppModel) {
        NSWorkspace.shared.activateFileViewerSelecting([
            model.appDataDirectory.appendingPathComponent("mac-app-settings.json")])
    }
}
