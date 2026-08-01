import AppKit
import TurboFieldfareAppCore
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let iconURL = Bundle.module.url(
            forResource: "turbofieldfare-app-icon",
            withExtension: "png"
        ), let icon = NSImage(contentsOf: iconURL) {
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
            settingsPersistenceEnabled: true,
            conversationsPersistenceEnabled: true))
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
            CommandGroup(replacing: .newItem) {
                Button("New Chat", action: model.newConversation)
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(model.isRunning)
            }
            CommandGroup(after: .toolbar) {
                Button("Toggle Sidebar") {
                    NotificationCenter.default.post(
                        name: .turboFieldfareToggleSidebar, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.command, .option])
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
            }
        }
    }
}

extension Notification.Name {
    /// The sidebar toggle lives in the menu bar, but the visibility state
    /// belongs to the window's root view; a notification avoids threading a
    /// binding through the Scene.
    static let turboFieldfareToggleSidebar = Notification.Name(
        "com.turbofieldfare.toggleSidebar")
}
