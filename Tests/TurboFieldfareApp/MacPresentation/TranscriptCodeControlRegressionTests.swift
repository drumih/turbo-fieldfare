import AppKit
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptCodeControlRegressionTests {
    // A hosted NSTextView had no per-block control at all before 0.9.4.
    // This uses only entry points present before the feature.
    @Test func aRenderedCodeCellExposesAnAccessibleCopyButton() async throws {
        let view = TranscriptTextView(frame: NSRect(x: 0, y: 0, width: 560, height: 300))
        let window = NSWindow(contentRect: view.bounds, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = view
        view.textStorage?.setAttributedString(ResponseMarkdownRenderer().render("```swift\nlet x = 1\n```").attributedString)
        view.layoutManager?.ensureLayout(for: try #require(view.textContainer))
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(50))
        let buttons = view.subviews.compactMap { $0 as? NSButton }
        #expect(buttons.count == 1)
        #expect(buttons.first?.accessibilityLabel() == "Copy code")
        #expect(buttons.first?.accessibilityIdentifier().hasPrefix("transcript.copyCode.") == true)
    }
}
