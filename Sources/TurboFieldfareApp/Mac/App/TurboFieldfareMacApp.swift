import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let icon = MacAppIcon.load() {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct TurboFieldfareMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var model: AppModel

    init() {
        _model = State(initialValue: AppModel(
            client: DecodeServiceInferenceClient(),
            settingsPersistenceEnabled: true))
    }

    var body: some Scene {
        Window("TurboFieldfare", id: "main") {
            RootView(model: model)
                .frame(minWidth: 1040, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1040, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About TurboFieldfare") {
                    NSApp.orderFrontStandardAboutPanel(
                        options: AboutPanelPresentation.options(
                            infoDictionary: Bundle.main.infoDictionary,
                            icon: MacAppIcon.load()))
                }
            }
            CommandMenu("Generation") {
                Button("Cancel Generation") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)
                Button("Cancel Model Installation") { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }
            CommandMenu("Model") {
                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)
                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)
                Button("Unload Model", action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
                Divider()
                Button("Reveal Model in Finder", action: revealModel)
                    .disabled(modelRevealTarget == .unavailable)
            }
            CommandMenu("Settings") {
                Picker("Send with", selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("Prompt examples", selection: showPromptExamplesBinding) {
                    Text("Show").tag(true)
                    Text("Hide").tag(false)
                }
                Picker("After sending", selection: sentPromptBehaviorBinding) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
            }
        }
    }

    private var modelRevealTarget: ModelRevealTarget {
        ModelRevealPolicy.target(
            forModelPath: model.modelPathText,
            fileExists: FileManager.default.fileExists(atPath:))
    }

    private func revealModel() {
        switch modelRevealTarget {
        case .selectItem(let url):
            NSWorkspace.shared.activateFileViewerSelecting([url])
        case .openContainer(let url):
            NSWorkspace.shared.open(url)
        case .unavailable:
            break
        }
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding {
            model.newlineShortcut
        } set: { shortcut in
            model.setNewlineShortcut(shortcut)
        }
    }

    private var showPromptExamplesBinding: Binding<Bool> {
        Binding {
            model.showPromptExamples
        } set: { show in
            model.setShowPromptExamples(show)
        }
    }

    private var sentPromptBehaviorBinding: Binding<AppSentPromptBehavior> {
        Binding {
            model.sentPromptBehavior
        } set: { behavior in
            model.setSentPromptBehavior(behavior)
        }
    }
}
