import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@Suite struct AppChromePresentationTests {
    @Test func appearanceOptionsHaveStablePersistenceAndPresentation() {
        #expect(AppAppearance.storageKey == "TurboFieldfare.appearance")
        #expect(AppAppearance.allCases.map(\.rawValue)
            == ["system", "light", "dark"])
        #expect(AppAppearance.allCases.map(\.label)
            == ["System", "Light", "Dark"])
        #expect(AppAppearance.allCases.map(\.systemImage)
            == ["circle.lefthalf.filled", "sun.max", "moon"])
        #expect(AppAppearance.system.preferredColorScheme == nil)
        #expect(AppAppearance.light.preferredColorScheme == .light)
        #expect(AppAppearance.dark.preferredColorScheme == .dark)
    }

    @Test func invalidPersistedAppearanceFallsBackToSystem() {
        #expect(AppAppearance.resolve("dark") == .dark)
        #expect(AppAppearance.resolve("unexpected") == .system)
        #expect(AppAppearance.resolve("") == .system)
    }

    @Test func sidebarControlsAlwaysRemainInTheHeader() {
        for sidebar in AppSidebarKind.allCases {
            let visible = AppSidebarControlPresentation(
                sidebar: sidebar,
                isVisible: true)
            let hidden = AppSidebarControlPresentation(
                sidebar: sidebar,
                isVisible: false)

            #expect(visible.placement == .header)
            #expect(hidden.placement == .header)
            #expect(visible.systemImage == hidden.systemImage)
        }
    }

    @Test func chatSidebarControlChangesStateWithoutChangingIdentity() {
        let visible = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: true)
        let hidden = AppSidebarControlPresentation(
            sidebar: .chats,
            isVisible: false)

        #expect(visible.systemImage == "sidebar.left")
        #expect(visible.title == "Hide chats")
        #expect(visible.help == "Hide chats (⌃⌘S)")
        #expect(visible.accessibilityValue == "Visible")
        #expect(hidden.systemImage == "sidebar.left")
        #expect(hidden.title == "Show chats")
        #expect(hidden.help == "Show chats (⌃⌘S)")
        #expect(hidden.accessibilityValue == "Hidden")
    }

    @Test func inspectorControlUsesTheSameHeaderContract() {
        let visible = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: true)
        let hidden = AppSidebarControlPresentation(
            sidebar: .inspector,
            isVisible: false)

        #expect(visible.systemImage == "sidebar.right")
        #expect(visible.title == "Hide settings")
        #expect(visible.help == "Hide settings (⇧⌘I)")
        #expect(hidden.systemImage == "sidebar.right")
        #expect(hidden.title == "Show settings")
        #expect(hidden.help == "Show settings (⇧⌘I)")
    }

    @Test func minimumWindowWidthTracksVisibleSidebars() {
        #expect(AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: false,
            isInspectorVisible: false) == 680)
        #expect(AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: true,
            isInspectorVisible: false) == 953)
        #expect(AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: false,
            isInspectorVisible: true) == 1_001)
        #expect(AppChromeLayout.minimumWindowWidth(
            isChatSidebarVisible: true,
            isInspectorVisible: true) == 1_274)
    }

    @Test func headerUsesSymmetricHorizontalPadding() {
        #expect(AppChromeLayout.headerHorizontalPadding == 20)
    }

    @Test(
        arguments: [
            (AppModelAction.install, "Install", "arrow.down.circle",
             "Install the local model", false),
            (.cancelInstall, "Cancel", "xmark",
             "Cancel model installation", true),
            (.load, "Load Model", "play.fill",
             "Load the model into memory", false),
            (.retryLoad, "Retry Load", "play.fill",
             "Retry loading the model", false),
            (.cancelLoad, "Cancel", "xmark",
             "Cancel model loading", true),
            (.reload, "Reload Model", "arrow.clockwise",
             "Reload the model with the selected settings", false),
            (.unload, "Unload", "eject",
             "Unload the model from memory", false),
        ])
    func modelHeaderActionsHaveStableCopyAndSymbols(
        action: AppModelAction,
        title: String,
        systemImage: String,
        help: String,
        isCancellation: Bool
    ) {
        let presentation = AppModelActionPresentation(action: action)

        #expect(presentation.title == title)
        #expect(presentation.systemImage == systemImage)
        #expect(presentation.help == help)
        #expect(presentation.isCancellation == isCancellation)
    }
}
