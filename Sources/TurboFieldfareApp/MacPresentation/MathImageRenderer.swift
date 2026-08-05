import AppKit
import SwiftMath

@MainActor
public enum MathImageRenderer {
    private static var cache: [CacheKey: NSImage] = [:]
    private static var cacheOrder: [CacheKey] = []
    private static let cacheLimit = 256

    private struct CacheKey: Hashable {
        let latex: String
        let isDisplay: Bool
        let fontSize: CGFloat
        let appearanceName: String
    }

    public static func attachment(
        latex: String,
        isDisplay: Bool,
        fontSize: CGFloat,
        rawSource: String
    ) -> MathTextAttachment? {
        let appearanceName = NSApp?.effectiveAppearance.name.rawValue ?? "unknown"
        let key = CacheKey(
            latex: latex,
            isDisplay: isDisplay,
            fontSize: fontSize,
            appearanceName: appearanceName)
        let image: NSImage?
        if let cached = cache[key] {
            image = cached
        } else if let rendered = render(latex: latex, isDisplay: isDisplay, fontSize: fontSize) {
            cache[key] = rendered
            cacheOrder.append(key)
            if cacheOrder.count > cacheLimit {
                let removed = cacheOrder.removeFirst()
                cache.removeValue(forKey: removed)
            }
            image = rendered
        } else {
            image = nil
        }
        guard let image else { return nil }
        return makeAttachment(
            image: image,
            isDisplay: isDisplay,
            fontSize: fontSize,
            rawSource: rawSource)
    }

    private static func makeAttachment(
        image: NSImage,
        isDisplay: Bool,
        fontSize: CGFloat,
        rawSource: String
    ) -> MathTextAttachment? {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return nil }
        let baselineOffset = isDisplay ? 0 : -round(fontSize * 0.22)
        let bounds = CGRect(x: 0, y: baselineOffset, width: size.width, height: size.height)
        return MathTextAttachment(
            image: image,
            bounds: bounds,
            latexSource: rawSource,
            isDisplay: isDisplay)
    }

    private static func render(
        latex: String,
        isDisplay: Bool,
        fontSize: CGFloat
    ) -> NSImage? {
        let appearance = NSApp?.effectiveAppearance ?? .current
        let color = NSColor.labelColor.resolvedColor(with: appearance)
        let mode: MTMathUILabelMode = isDisplay ? .display : .text
        let mathImage = MTMathImage(
            latex: latex,
            fontSize: fontSize,
            textColor: color,
            labelMode: mode,
            textAlignment: .center)
        let (_, image) = mathImage.asImage()
        return image
    }
}
