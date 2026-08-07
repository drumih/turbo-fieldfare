import AppKit
import SwiftUI

public enum TurboFieldfareMacTheme {
    public static let accentNSColor = NSColor(
        srgbRed: 106.0 / 255.0,
        green: 186.0 / 255.0,
        blue: 113.0 / 255.0,
        alpha: 1)

    public static let accentColor = Color(nsColor: accentNSColor)

    /// Raised card fill (composer shell, examples panel, HUD, banners).
    /// Uses text-background so surfaces separate from the window chrome —
    /// `controlBackgroundColor` often matches `windowBackgroundColor`.
    public static var elevatedSurface: Color {
        Color(nsColor: .textBackgroundColor)
    }

    /// Nested input field fill, slightly inset from the card.
    public static var fieldSurface: Color {
        Color(nsColor: .controlBackgroundColor).opacity(0.55)
            .mix(with: Color(nsColor: .textBackgroundColor), by: 0.65)
    }

    /// Resting border for cards and fields (stronger than `.separator` 0.5).
    public static var cardBorder: Color {
        Color.primary.opacity(0.14)
    }

    public static var fieldBorder: Color {
        Color.primary.opacity(0.22)
    }

    public static var fieldBorderFocused: Color {
        accentColor.opacity(0.9)
    }

    public static var fieldPlaceholder: Color {
        Color.secondary
    }

    public static var hairlineDivider: Color {
        Color.primary.opacity(0.12)
    }

    public static var warningEmphasis: Color {
        Color.orange
    }
}

public extension View {
    /// Shared elevated card chrome used by composer, examples, install, banners.
    func turboElevatedCard(cornerRadius: CGFloat, border: Color = TurboFieldfareMacTheme.cardBorder) -> some View {
        background {
            RoundedRectangle(cornerRadius: cornerRadius)
                .fill(TurboFieldfareMacTheme.elevatedSurface)
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .stroke(border, lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
        }
    }

    /// Label metrics for compact titled secondary controls (Tips, Clear, etc.).
    func turboQuietControlLabel() -> some View {
        labelStyle(.titleAndIcon)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 10)
            .frame(height: 28)
            .contentShape(Capsule())
    }

    /// Chrome for compact secondary controls next to primary actions.
    func turboQuietControlChrome() -> some View {
        buttonStyle(.borderless)
            .foregroundStyle(.primary)
            .background {
                Capsule()
                    .fill(Color.primary.opacity(0.06))
                    .overlay {
                        Capsule().stroke(TurboFieldfareMacTheme.cardBorder, lineWidth: 1)
                    }
            }
    }
}
