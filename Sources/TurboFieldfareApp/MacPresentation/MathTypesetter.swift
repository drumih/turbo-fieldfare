import AppKit
import Foundation
import SwiftMath
import os

public enum MathRenderMode: Sendable {
    case inline
    case display
}

/// A typeset equation plus the metrics an `NSTextAttachment` needs to sit on
/// the surrounding text baseline. Project-owned so the transcript API never
/// exposes the typesetting library, and so tests can inject a deterministic
/// stand-in without importing it either.
public struct MathRender {
    public let image: NSImage
    public let ascent: CGFloat
    public let descent: CGFloat

    public init(image: NSImage, ascent: CGFloat, descent: CGFloat) {
        self.image = image
        self.ascent = ascent
        self.descent = descent
    }
}

@MainActor
public protocol MathTypesetting {
    func render(
        latex: String,
        fontSize: CGFloat,
        tint: NSColor,
        mode: MathRenderMode
    ) -> MathRender?
}

/// Carries the exact source span so `plainText` and the accessibility layer can
/// recover the LaTeX a reader can no longer select out of the image.
public final class MathAttachment: NSTextAttachment {
    public let latexSource: String

    public init(latexSource: String) {
        self.latexSource = latexSource
        super.init(data: nil, ofType: nil)
    }

    /// Nothing archives the transcript. The pasteboard flavours write the
    /// LaTeX itself rather than the attachment, so an archived one could only
    /// come back as an object-replacement character where an equation was —
    /// which is the failure the attachment exists to prevent. Decoding refuses
    /// instead of returning a source-less attachment.
    public required init?(coder: NSCoder) { nil }
}

@MainActor
public struct SwiftMathTypesetter: MathTypesetting {
    /// SwiftMath bakes the glyph color into the CoreText run at typeset time,
    /// so a single image cannot follow a light/dark switch. Typesetting once in
    /// an opaque color and keeping only the coverage lets the attachment fill
    /// with the run's dynamic color at draw time instead.
    private static let maskScale: CGFloat = 3

    private static let log = Logger(
        subsystem: "TurboFieldfare",
        category: "math-typesetting")

    private let cache: MathRenderCache

    public init(cache: MathRenderCache = .shared) {
        self.cache = cache
    }

    public func render(
        latex: String,
        fontSize: CGFloat,
        tint: NSColor,
        mode: MathRenderMode
    ) -> MathRender? {
        let key = MathRenderCache.Key(
            latex: latex,
            fontSize: fontSize,
            tint: tint.description,
            mode: mode)
        if let cached = cache.value(for: key) { return cached }
        guard let mask = Self.mask(latex: latex, fontSize: fontSize, mode: mode) else {
            return nil
        }
        let render = MathRender(
            image: Self.tinted(mask.image, size: mask.size, tint: tint),
            ascent: mask.ascent,
            descent: mask.descent)
        cache.store(render, cost: mask.bytes, for: key)
        return render
    }

    private struct Mask {
        let image: CGImage
        let size: CGSize
        let ascent: CGFloat
        let descent: CGFloat
        /// What the tinted image retains: the drawing handler clips to this
        /// bitmap, so the mask is the memory the cache is spending.
        let bytes: Int
    }

    private static func mask(
        latex: String,
        fontSize: CGFloat,
        mode: MathRenderMode
    ) -> Mask? {
        // The builder skips a character it has no atom for and reports nothing,
        // so an equation can come back looking finished with an operator
        // missing from it. Refusing here restores the span's own source, which
        // the reader can at least see.
        if let dropped = PinnedMathCoverage.firstDroppedCharacter(in: latex) {
            let code = dropped.unicodeScalars
                .map { String(format: "%04X", $0.value) }
                .joined(separator: "+")
            log.error("""
                math typeset refused: pinned build drops U+\(code, privacy: .public) \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        var image = MathImage(
            latex: latex,
            fontSize: fontSize,
            textColor: .black,
            labelMode: mode == .display ? .display : .text)
        let (error, typeset, layout) = image.asImage()
        if let error {
            log.error("""
                math typeset failed: domain=\(error.domain, privacy: .public) \
                code=\(error.code, privacy: .public) \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        guard let typeset, let layout else {
            log.error("""
                math typeset produced no image without an error: \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        let size = typeset.size
        // A zero-width or zero-height bitmap rep cannot be allocated, and an
        // empty attachment would be an invisible hole where an equation was.
        guard size.width >= 1, size.height >= 1 else {
            log.error("""
                math typeset produced an empty image: \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(ceil(size.width * maskScale)),
            pixelsHigh: Int(ceil(size.height * maskScale)),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0) else {
            log.error("""
                math mask allocation failed: \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        rep.size = size
        // Without this the equation draws into no context at all and the
        // blank rep still yields a `cgImage`, so a transparent attachment
        // would be cached in place of the equation with nothing logged.
        guard let context = NSGraphicsContext(bitmapImageRep: rep) else {
            log.error("""
                math mask has no drawing context: \
                length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        typeset.draw(in: NSRect(origin: .zero, size: size))
        NSGraphicsContext.restoreGraphicsState()
        guard let mask = rep.cgImage else {
            log.error("""
                math mask has no bitmap: length=\(latex.count, privacy: .public)
                """)
            return nil
        }
        return Mask(
            image: mask,
            size: size,
            ascent: layout.ascent,
            descent: layout.descent,
            bytes: rep.bytesPerRow * rep.pixelsHigh)
    }

    private static func tinted(
        _ mask: CGImage,
        size: CGSize,
        tint: NSColor
    ) -> NSImage {
        NSImage(size: size, flipped: false) { rect in
            guard let context = NSGraphicsContext.current?.cgContext else { return false }
            context.saveGState()
            context.clip(to: rect, mask: mask)
            tint.setFill()
            context.fill(rect)
            context.restoreGState()
            return true
        }
    }
}

/// Memo for typeset equations. Measured on the fifty-equation corpus fixture:
/// a finalize pass costs 36.7 ms, well past a 16 ms frame, and a rebuild while
/// the answer is terminal re-renders the whole transcript. The memo removes the
/// repeat cost; the first pass over an answer still pays for every equation.
@MainActor
public final class MathRenderCache {
    public static let shared = MathRenderCache()

    struct Key: Hashable {
        let latex: String
        let fontSize: CGFloat
        let tint: String
        let mode: MathRenderMode
    }

    /// Bounded by what the images cost rather than by how many there are. A
    /// display equation's 3x mask runs to hundreds of kilobytes and an inline
    /// one to tens, so a count limit is either far too small for a page of
    /// prose or far too large for a page of derivations. 16 MiB is roughly a
    /// hundred display equations at 13 pt, or a thousand inline ones.
    private let byteLimit: Int
    private var entries: [Key: MathRender] = [:]
    private var bytes = 0

    public init(byteLimit: Int = 16 << 20) {
        self.byteLimit = byteLimit
    }

    var count: Int { entries.count }
    var byteCount: Int { bytes }

    func value(for key: Key) -> MathRender? { entries[key] }

    func store(_ render: MathRender, cost: Int, for key: Key) {
        // One equation bigger than the whole budget is not worth emptying the
        // table for; it is typeset again if it comes back.
        guard cost <= byteLimit else { return }
        if bytes + cost > byteLimit {
            // Equations are cheap to typeset again, so dropping the table beats
            // tracking per-entry age.
            entries.removeAll(keepingCapacity: true)
            bytes = 0
        }
        if entries.updateValue(render, forKey: key) == nil { bytes += cost }
    }
}

extension MathRenderMode: Hashable {}
