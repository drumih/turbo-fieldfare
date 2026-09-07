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

    /// The last exchange's write finishes before the process goes.
    ///
    /// It starts when the reply lands and, with pictures, runs for seconds
    /// after the send has returned; quitting under it lost the exchange and
    /// released the staged files it was still reading. Bounded, so a write
    /// that hangs cannot keep the app alive.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model = MainActor.assumeIsolated({ Self.model }) else {
            return .terminateNow
        }
        Task { @MainActor in
            if await !model.awaitPendingPersistence(timeout: .seconds(30)) {
                FileHandle.standardError.write(Data(
                    "Conversation persistence did not finish before the quit deadline; the latest exchange may not be saved.\n".utf8))
            }
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated { Self.model?.shutdownForTermination() }
    }

    /// A second window that opened while another held the store's lock can take
    /// it once that one quits.
    ///
    /// Two triggers, because one of them is not enough. Coming forward is what
    /// the user does after closing the other copy — but if this window is
    /// already frontmost when the other quits, that never fires, and the notice
    /// stayed until the next time they switched away and back. Watching for a
    /// peer of this app terminating covers exactly that case.
    func applicationDidBecomeActive(_ notification: Notification) {
        MainActor.assumeIsolated {
            Self.model?.reacquireStoreIfPossible()
            Self.model?.recheckVisionPackAtCurrentLocation()
        }
    }

    private var peerObserver: (any NSObjectProtocol)?

    private func watchForPeersQuitting() {
        peerObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didTerminateApplicationNotification,
            object: nil, queue: .main
        ) { notification in
            let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
                as? NSRunningApplication
            // Another copy of this same executable, not any app that quit.
            guard app?.bundleIdentifier == Bundle.main.bundleIdentifier
                    || app?.executableURL == Bundle.main.executableURL else {
                return
            }
            MainActor.assumeIsolated { Self.model?.reacquireStoreIfPossible() }
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        watchForPeersQuitting()
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
                // The three columns at their minimums, plus their dividers.
                .frame(minWidth: 1112, minHeight: 560)
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
        // No toolbar at all. The window's controls live in the status strip
        // beside the model name, so a title bar here would be an empty strip
        // above the content with nothing in it.
        .windowStyle(.hiddenTitleBar)
        // Above the minimum the three columns impose (260 + 530 + 320 plus
        // dividers), so the window never opens already clamped.
        .defaultSize(width: 1200, height: 780)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { model.newChat() }
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(!model.canStartNewChat)
            }
            // `.sidebar` is the placement for commands that control the app's
            // panels, and replacing it is what makes these items ours: the
            // system's would route through the split view's own toggle, which
            // was removed so the button, the menu item and the shortcut are one
            // action against one persisted setting.
            CommandGroup(replacing: .sidebar) {
                Button(WindowControlsPresentation.sidebarToggleHelp(
                    isVisible: model.isSidebarVisible)) {
                    withAnimation(RootView.panelSlide) { model.toggleSidebar() }
                }
                .keyboardShortcut("s", modifiers: [.command, .control])
                Button(WindowControlsPresentation.inspectorToggleHelp(
                    isVisible: model.isInspectorVisible)) {
                    withAnimation(RootView.panelSlide) { model.toggleInspector() }
                }
                .keyboardShortcut("i", modifiers: [.command, .control])
            }
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

}
