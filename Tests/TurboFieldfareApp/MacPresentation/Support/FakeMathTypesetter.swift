import AppKit
import Foundation
@testable import TurboFieldfareMacPresentation

/// Deterministic stand-in for the typesetter. Renderer tests assert what the
/// pipeline hands over — byte-identical LaTeX, the run's font size and colour,
/// the chosen mode — without depending on any glyph the real library draws.
@MainActor
final class FakeMathTypesetter: MathTypesetting {
    struct Call: Equatable {
        let latex: String
        let fontSize: CGFloat
        let tint: NSColor
        let mode: MathRenderMode
    }

    static let ascent: CGFloat = 9
    static let descent: CGFloat = 3

    private(set) var calls: [Call] = []
    /// LaTeX that must come back as a typeset failure, keyed on what the
    /// renderer passes in.
    var failing: Set<String> = []

    func render(
        latex: String,
        fontSize: CGFloat,
        tint: NSColor,
        mode: MathRenderMode
    ) -> MathRender? {
        calls.append(Call(latex: latex, fontSize: fontSize, tint: tint, mode: mode))
        guard !failing.contains(latex) else { return nil }
        return MathRender(
            image: NSImage(size: NSSize(width: 20, height: Self.ascent + Self.descent)),
            ascent: Self.ascent,
            descent: Self.descent)
    }
}
