import AppKit
import Foundation
@testable import TurboFieldfareMacPresentation

/// Records what the transcript controller asks the renderer for. "Each
/// completed block is rendered once" is otherwise invisible from the outside:
/// the document looks the same whether a block was rendered once or on every
/// tick, and only the clock would tell the difference.
@MainActor
final class RecordingBlockRenderer: TranscriptBlockRendering {
    struct Call: Equatable {
        let source: String
        let typesetsMath: Bool
    }

    private let base: ResponseMarkdownRenderer
    private(set) var calls: [Call] = []

    init(base: ResponseMarkdownRenderer = ResponseMarkdownRenderer()) {
        self.base = base
    }

    /// Blocks that were rendered as finished, in order.
    var completedSources: [String] {
        calls.filter(\.typesetsMath).map(\.source)
    }

    /// Renders of the block that was still being written.
    var tailSources: [String] {
        calls.filter { !$0.typesetsMath }.map(\.source)
    }

    func render(_ source: String, typesetsMath: Bool) -> ResponseMarkdownRenderer.Result {
        calls.append(Call(source: source, typesetsMath: typesetsMath))
        return base.render(source, typesetsMath: typesetsMath)
    }

    func blockSeparator(trailingNewlines: Int) -> NSAttributedString {
        base.blockSeparator(trailingNewlines: trailingNewlines)
    }

    func streamingCodeAttributes() -> [NSAttributedString.Key: Any] {
        base.streamingCodeAttributes()
    }
}
