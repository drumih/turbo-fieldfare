import AppKit
import SwiftUI

public enum TurboFieldfareMacTheme {
    public static let accentNSColor = NSColor(
        srgbRed: 106.0 / 255.0,
        green: 186.0 / 255.0,
        blue: 113.0 / 255.0,
        alpha: 1)

    public static let accentColor = Color(nsColor: accentNSColor)

    public static var accentGradient: LinearGradient {
        LinearGradient(
            colors: [accentColor, accentColor.opacity(0.68)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing)
    }

    /// A small set of semantic surfaces keeps the app visually coherent while
    /// still respecting the user's macOS appearance and accessibility settings.
    public static var appBackground: Color {
        Color(nsColor: .windowBackgroundColor)
    }

    public static var sidebarBackground: Color {
        Color(nsColor: .underPageBackgroundColor)
    }

    public static var surface: Color {
        Color(nsColor: .controlBackgroundColor)
    }

    public static var elevatedSurface: Color {
        Color(nsColor: .textBackgroundColor)
    }

    public static var mutedSurface: Color {
        Color(nsColor: .underPageBackgroundColor)
    }

    public static var accentSurface: Color {
        accentColor.opacity(0.12)
    }

    public static var hoverSurface: Color {
        accentColor.opacity(0.075)
    }

    public static var border: Color {
        Color.primary.opacity(0.10)
    }

    public static var cardBorder: Color {
        Color.primary.opacity(0.13)
    }
}
