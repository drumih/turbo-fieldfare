import AppKit

public final class MathTextAttachment: NSTextAttachment {
    public let latexSource: String
    public let isDisplay: Bool

    public init(image: NSImage, bounds: CGRect, latexSource: String, isDisplay: Bool) {
        self.latexSource = latexSource
        self.isDisplay = isDisplay
        super.init(data: nil, ofType: nil)
        self.image = image
        self.bounds = bounds
    }

    public required init?(coder: NSCoder) {
        self.latexSource = ""
        self.isDisplay = false
        super.init(coder: coder)
    }
}
