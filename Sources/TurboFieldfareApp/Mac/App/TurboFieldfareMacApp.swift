import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the scene so quitting can release this session's staged images.
    @MainActor static var model: AppModel?

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { Self.model?.releaseAllAttachments() }
    }

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
        let model = AppModel(
            client: DecodeServiceInferenceClient(),
            visionRuntimeSupported: AppModel.currentDeviceSupportsVisionRuntime,
            settingsPersistenceEnabled: true)
        _model = State(initialValue: model)
        MainActor.assumeIsolated { ForegroundAppDelegate.model = model }
    }

    var body: some Scene {
        Window("TurboFieldfare", id: "main") {
            RootView(model: model)
                .frame(minWidth: 1040, minHeight: 560)
                // Once, when the window first appears: the setting is read
                // from disk in init, and loadModelAtLaunchIfEnabled ignores a
                // model that is missing or already busy.
                .task { model.loadModelAtLaunchIfEnabled() }
                // On the window, not in the Inspector: the menu item works
                // whether or not the Inspector is open.
                .confirmationDialog(
                    "Remove downloaded image support?",
                    isPresented: Bindable(model).isConfirmingVisionPackRemoval,
                    titleVisibility: .visible
                ) {
                    Button("Remove Image Support", role: .destructive) {
                        model.removeVisionPack()
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Text generation will continue to work. "
                        + "Getting image support back means downloading the "
                        + "pack again.")
                }
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
                // The Inspector shows image support only while there is
                // something to decide, so reclaiming the pack lives here:
                // rare, deliberate, and destructive. Which is why it asks
                // first — this is the only reachable way to delete the pack,
                // and it used to call straight through.
                Button("Remove Image Support", action: model.requestVisionPackRemoval)
                    .disabled(!model.canRemoveVisionPack)
            }
            CommandMenu("Settings") {
                Picker("Send Message With", selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("Prompt Examples", selection: showPromptExamplesBinding) {
                    Text("Show").tag(true)
                    Text("Hide").tag(false)
                }
                Picker("After Sending", selection: sentPromptBehaviorBinding) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
                Picker("Load Model At Launch", selection: loadModelOnLaunchBinding) {
                    Text("Off").tag(false)
                    Text("On").tag(true)
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

    private var loadModelOnLaunchBinding: Binding<Bool> {
        Binding {
            model.loadModelOnLaunch
        } set: { enabled in
            model.setLoadModelOnLaunch(enabled)
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
