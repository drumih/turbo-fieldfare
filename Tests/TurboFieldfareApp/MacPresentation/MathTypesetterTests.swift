import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// Exercises the typesetting library only through the production conformer, so
/// what these tests pin is the behaviour the transcript actually gets. A pin
/// change that breaks a command shows up here rather than as a blank equation.
@MainActor
@Suite struct MathTypesetterTests {
    static let fontSize: CGFloat = 13

    static func render(
        _ latex: String,
        mode: MathRenderMode = .inline,
        tint: NSColor = .black
    ) -> MathRender? {
        SwiftMathTypesetter().render(
            latex: latex,
            fontSize: fontSize,
            tint: tint,
            mode: mode)
    }

    /// Draws the attachment image the way the text view does and returns its
    /// pixels, so a tint difference is a measured difference and not an
    /// identity comparison on two `NSImage` objects.
    static func pixels(_ image: NSImage) -> [UInt8]? {
        let width = Int(ceil(image.size.width))
        let height = Int(ceil(image.size.height))
        guard width > 0, height > 0, let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0) else {
            return nil
        }
        rep.size = image.size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(origin: .zero, size: image.size))
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.bitmapData else { return nil }
        return Array(UnsafeBufferPointer(
            start: data,
            count: rep.bytesPerRow * rep.pixelsHigh))
    }

    @Test(arguments: [
        #"\frac{a}{b}"#,
        #"\sqrt{b^2 - 4ac}"#,
        #"\sum_{i=1}^{n} i"#,
        #"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"#,
        #"\alpha + \beta = \gamma"#,
        // The pin probe: tag 1.7.3 duplicates the fenced content here, the
        // pinned revision does not.
        #"\left( \frac{a}{b} \right)"#,
    ])
    func typesetsTheCommandsGemmaEmits(_ latex: String) throws {
        let render = try #require(Self.render(latex))
        #expect(render.ascent > 0)
        #expect(render.descent > 0)
        #expect(render.image.size.width > 0)
        #expect(render.image.size.height > 0)
    }

    @Test(arguments: [
        (#"\begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}"#, true),
        (#"\begin{aligned} a &= b \\ c &= d \end{aligned}"#, true),
        (#"\begin{matrix} a & b \end{matrix}"#, true),
        // Rejected by the pinned revision; the normalizer rewrites these
        // before the transcript reaches the typesetter, and an unrewritten
        // one falls back to raw text rather than disappearing.
        (#"\begin{align*} a &= b \end{align*}"#, false),
        (#"\begin{array}{cc} 1 & 2 \end{array}"#, false),
    ])
    func environmentCoverageIsPinnedInBothDirections(_ probe: (String, Bool)) {
        #expect((Self.render(probe.0) != nil) == probe.1, "\(probe.0)")
    }

    @Test(arguments: [#"\frac{"#, #"\unknowncmd"#, #"\left( x"#])
    func invalidLatexProducesNoImage(_ latex: String) {
        #expect(Self.render(latex) == nil)
    }

    /// The attachment bounds are `-descent` tall below the baseline and
    /// `ascent` above it, so the image must be exactly that tall or the
    /// equation sits off the line. SwiftMath rounds the height up to a whole
    /// point, which is the only slack allowed here.
    @Test func imageHeightIsTheCeilingOfAscentPlusDescent() throws {
        for latex in [#"\frac{a}{b}"#, #"x"#, #"\sum_{i=1}^{n} i"#, #"\sqrt{x}"#] {
            let render = try #require(Self.render(latex))
            #expect(render.image.size.height == ceil(render.ascent + render.descent),
                    "\(latex)")
        }
    }

    @Test func maskTintProducesVisibleGlyphsInTheRequestedColor() throws {
        let render = try #require(Self.render(#"x = \frac{a}{b}"#, tint: .red))
        let pixels = try #require(Self.pixels(render.image))
        let opaqueRed = stride(from: 0, to: pixels.count, by: 4).contains { index in
            pixels[index] > 200 && pixels[index + 1] < 80 && pixels[index + 3] > 200
        }
        #expect(opaqueRed, "the tint must paint the glyph coverage, not the background")
    }

    @Test func twoTintsProduceDifferentPixels() throws {
        let latex = #"E = mc^2"#
        let redRender = try #require(Self.render(latex, tint: .red))
        let blueRender = try #require(Self.render(latex, tint: .blue))
        let red = try #require(Self.pixels(redRender.image))
        let blue = try #require(Self.pixels(blueRender.image))
        #expect(red.count == blue.count)
        #expect(red != blue)
    }

    /// Finalising a fifty-equation answer costs 36.7 ms, and a rebuild while
    /// the answer is terminal re-renders all of it, so the same equation at the
    /// same size and colour must come back as the same image.
    @Test func repeatedRendersOfTheSameEquationReuseOneImage() throws {
        let cache = MathRenderCache()
        let typesetter = SwiftMathTypesetter(cache: cache)
        let first = try #require(typesetter.render(
            latex: #"\frac{a}{b}"#,
            fontSize: 13,
            tint: .labelColor,
            mode: .inline))
        let second = try #require(typesetter.render(
            latex: #"\frac{a}{b}"#,
            fontSize: 13,
            tint: .labelColor,
            mode: .inline))
        #expect(first.image === second.image)
        #expect(cache.count == 1)

        // Anything that changes the pixels is a different entry.
        _ = typesetter.render(latex: #"\frac{a}{b}"#, fontSize: 22, tint: .labelColor, mode: .inline)
        _ = typesetter.render(latex: #"\frac{a}{b}"#, fontSize: 13, tint: .red, mode: .inline)
        _ = typesetter.render(latex: #"\frac{a}{b}"#, fontSize: 13, tint: .labelColor, mode: .display)
        #expect(cache.count == 4)
    }

    /// Measured on the pinned revision: `.text` mode shrinks a fraction (19 pt
    /// against 24 pt at 13 pt type) but still stacks big-operator limits, so
    /// `\sum_{i=1}^n i` is 34 pt tall in both modes and is not a valid pin for
    /// the mode mapping.
    @Test func inlineAndDisplayModesDifferInHeight() throws {
        let inline = try #require(Self.render(#"\frac{a}{b}"#, mode: .inline))
        let display = try #require(Self.render(#"\frac{a}{b}"#, mode: .display))
        #expect(display.image.size.height > inline.image.size.height)

        let inlineSum = try #require(Self.render(#"\sum_{i=1}^n i"#, mode: .inline))
        let displaySum = try #require(Self.render(#"\sum_{i=1}^n i"#, mode: .display))
        #expect(displaySum.image.size.height == inlineSum.image.size.height)
    }
}
