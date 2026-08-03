import SwiftUI

/// Design system for TurboFieldfare's dark UI.
public enum TurboFieldfareAppearance: Sendable {

    /// Accent color: a fresh green that pops against the dark background.
    public static let accent = Color(
        red: 0.30, green: 0.78, blue: 0.38)

    /// Secondary accent for destructive or high-attention states.
    public static let danger = Color(
        red: 0.85, green: 0.30, blue: 0.30)

    /// Primary text color (slightly warm white).
    public static let textPrimary = Color(
        red: 0.93, green: 0.93, blue: 0.90)

    /// Secondary text color (reduced contrast for metadata).
    public static let textSecondary = Color(
        red: 0.58, green: 0.58, blue: 0.55)

    /// Subtle text for captions and timestamps.
    public static let textTertiary = Color(
        red: 0.40, green: 0.40, blue: 0.38)

    /// Background: near-black with a warm undertone.
    public static let background = Color(
        red: 0.07, green: 0.07, blue: 0.09)

    /// Elevated surface (cards, panels).
    public static let surface = Color(
        red: 0.10, green: 0.10, blue: 0.13)

    /// Animated corner radii.
    public static let cornerRadius: CGFloat = 12
    public static let cornerRadiusLarge: CGFloat = 18

    /// Icon size for metrics.
    public static let metricIconSize: CGFloat = 16
}
