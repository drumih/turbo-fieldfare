import AppKit
import SwiftUI

public enum AppAppearance: String, CaseIterable, Identifiable, Sendable {
    case system
    case light
    case dark

    public static let storageKey = "TurboFieldfare.appearance"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    public var systemImage: String {
        switch self {
        case .system: "circle.lefthalf.filled"
        case .light: "sun.max"
        case .dark: "moon"
        }
    }

    public var preferredColorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }

    public static func resolve(_ storedValue: String) -> Self {
        Self(rawValue: storedValue) ?? .system
    }
}

public enum TurboFieldfareMacTheme {
    public static let accentNSColor = NSColor(
        srgbRed: 106.0 / 255.0,
        green: 186.0 / 255.0,
        blue: 113.0 / 255.0,
        alpha: 1)

    public static let accentColor = Color(nsColor: accentNSColor)

    public static var sidebarBackgroundColor: Color {
        Color(nsColor: .windowBackgroundColor)
            .mix(with: accentColor, by: 0.025)
    }
}
