import Foundation
import TurboFieldfareAppCore

public enum AppChromeControlPlacement: Equatable, Sendable {
    case header
}

public enum AppSidebarKind: CaseIterable, Equatable, Sendable {
    case chats
    case inspector
}

public struct AppSidebarControlPresentation: Equatable, Sendable {
    public let placement: AppChromeControlPlacement
    public let systemImage: String
    public let title: String
    public let help: String
    public let accessibilityValue: String

    public init(
        sidebar: AppSidebarKind,
        isVisible: Bool
    ) {
        placement = .header
        accessibilityValue = isVisible ? "Visible" : "Hidden"
        switch sidebar {
        case .chats:
            systemImage = "sidebar.left"
            title = isVisible ? "Hide chats" : "Show chats"
            help = "\(title) (⌃⌘S)"
        case .inspector:
            systemImage = "sidebar.right"
            title = isVisible ? "Hide settings" : "Show settings"
            help = "\(title) (⇧⌘I)"
        }
    }
}

public enum AppChromeLayout {
    public static let primaryMinimumWidth = 680
    public static let chatSidebarWidth = 272
    public static let inspectorWidth = 320
    public static let dividerWidth = 1
    public static let minimumHeight = 560
    public static let regularHeaderLeadingPadding = 20
    public static let trafficLightHeaderLeadingPadding = 84

    public static func minimumWindowWidth(
        isChatSidebarVisible: Bool,
        isInspectorVisible: Bool
    ) -> Int {
        primaryMinimumWidth
            + (isChatSidebarVisible ? chatSidebarWidth + dividerWidth : 0)
            + (isInspectorVisible ? inspectorWidth + dividerWidth : 0)
    }

    public static func headerLeadingPadding(
        isChatSidebarVisible: Bool
    ) -> Int {
        isChatSidebarVisible
            ? regularHeaderLeadingPadding
            : trafficLightHeaderLeadingPadding
    }
}

public struct AppModelActionPresentation: Equatable, Sendable {
    public let title: String
    public let systemImage: String
    public let help: String
    public let isCancellation: Bool

    public init(action: AppModelAction) {
        switch action {
        case .install:
            title = "Install"
            systemImage = "arrow.down.circle"
            help = "Install the local model"
            isCancellation = false
        case .cancelInstall:
            title = "Cancel"
            systemImage = "xmark"
            help = "Cancel model installation"
            isCancellation = true
        case .load:
            title = "Load Model"
            systemImage = "play.fill"
            help = "Load the model into memory"
            isCancellation = false
        case .retryLoad:
            title = "Retry Load"
            systemImage = "play.fill"
            help = "Retry loading the model"
            isCancellation = false
        case .cancelLoad:
            title = "Cancel"
            systemImage = "xmark"
            help = "Cancel model loading"
            isCancellation = true
        case .reload:
            title = "Reload Model"
            systemImage = "arrow.clockwise"
            help = "Reload the model with the selected settings"
            isCancellation = false
        case .unload:
            title = "Unload"
            systemImage = "eject"
            help = "Unload the model from memory"
            isCancellation = false
        }
    }
}
