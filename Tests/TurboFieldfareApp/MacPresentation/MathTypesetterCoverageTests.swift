import AppKit
import Foundation
import SwiftMath
import Testing
@testable import TurboFieldfareMacPresentation

/// The one place that reaches the typesetting library directly rather than
/// through the production conformer. `PinnedMathCoverage` is a model of what
/// the pinned build draws, and a model is only worth having if something
/// measures it against the build; the conformer now refuses what the model
/// says is dropped, so asking it would be asking the model about itself.
@MainActor
@Suite struct MathTypesetterCoverageTests {
    /// The range the probe walks: ASCII through Sinhala, which covers every
    /// alphabet and every mathematical block a model writes into an equation.
    static let probed = 0x21...0x0DFF

    static func recorded() throws -> [String] {
        let url = try #require(Bundle.module.url(
            forResource: "unicode-math-mode",
            withExtension: "txt",
            subdirectory: "Fixtures/math-coverage"))
        return try String(contentsOf: url, encoding: .utf8)
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty }
    }

    /// Draws the equation the way the conformer does, minus the refusal.
    static func pixels(_ latex: String) -> Data? {
        var image = MathImage(latex: latex, fontSize: 13, textColor: .black, labelMode: .text)
        let (error, typeset, _) = image.asImage()
        guard error == nil, let typeset else { return nil }
        let size = typeset.size
        guard size.width >= 1, size.height >= 1, let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(ceil(size.width * 3)),
            pixelsHigh: Int(ceil(size.height * 3)),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0) else {
            return nil
        }
        rep.size = size
        guard let context = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        typeset.draw(in: NSRect(origin: .zero, size: size))
        NSGraphicsContext.restoreGraphicsState()
        return rep.representation(using: .png, properties: [:])
    }

    /// `x<s>x` against `xx`: a scalar the build draws changes the pixels, and
    /// one it drops leaves them identical. That silence is the whole defect —
    /// the image comes back looking finished with the character gone.
    @Test func pinnedMathModeCoverageMatchesTheBuild() throws {
        let baseline = try #require(Self.pixels("xx"))
        var drawn: [String] = []
        for value in Self.probed {
            guard let scalar = Unicode.Scalar(value) else { continue }
            let character = Character(scalar)
            guard !PinnedMathCoverage.structural.contains(character) else { continue }
            let probe = "x\(character)x"
            // A combining mark is never its own character in real input: it
            // joins whatever precedes it, and the build is asked about the
            // grapheme rather than the mark.
            guard probe.count == 3 else { continue }

            let differs = Self.pixels(probe).map { $0 != baseline } ?? false
            #expect(
                differs == PinnedMathCoverage.renders(character),
                "U+\(String(format: "%04X", value)) drawn=\(differs)")
            if differs { drawn.append(String(format: "%04X", value)) }
        }
        #expect(drawn == (try Self.recorded()))
    }

    /// `\text{}` is the one mode that accepts anything, which is what carries
    /// the two characters the normalizer rewrites through it.
    @Test(arguments: ["\u{4E2D}", "\u{2234}", "\u{2235}", "\u{221A}", "\u{00B0}"])
    func unicodeInsideTextIsAccepted(_ character: String) {
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: "\\text{\(character)}") == nil)
        #expect(SwiftMathTypesetter().render(
            latex: "\\text{\(character)}",
            fontSize: 13,
            tint: .black,
            mode: .inline) != nil)
    }

    @Test func commandNamesAreNotLiteralText() {
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: #"\alpha + \beta"#) == nil)
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: #"\frac{a}{b}^2_1"#) == nil)
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: #"\begin{matrix} a & b \end{matrix}"#)
            == nil)
        // An escaped reserved character is a command, not the character.
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: #"5\% of \$3"#) == nil)
        #expect(PinnedMathCoverage.firstDroppedCharacter(in: "5% of $3") == "%")
    }
}
