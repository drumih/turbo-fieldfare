import AppKit
import TurboFieldfareAppCore

public struct ConversationTypography: Equatable, Sendable {
    public let textSize: AppTextSize

    public init(_ textSize: AppTextSize = .standard) {
        self.textSize = textSize
    }

    public var scale: CGFloat { CGFloat(textSize.scale) }
    public func scaled(_ baseline: CGFloat) -> CGFloat { baseline * scale }
    public var bodySize: CGFloat { scaled(NSFont.systemFontSize) }
    public var labelSize: CGFloat { scaled(NSFont.smallSystemFontSize) }
    public var bodyFont: NSFont { .systemFont(ofSize: bodySize) }

    public func editorHeight(showsExamples: Bool, availableHeight: CGFloat) -> CGFloat {
        // At large text sizes a fixed line count can consume the conversation
        // viewport. Keep two readable lines and let the editor scroll instead.
        min(scaled(showsExamples ? 46 : 84), max(scaled(40), availableHeight / 5))
    }
}
