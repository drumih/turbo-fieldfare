import Foundation

/// The rendering corpus: three verbatim Gemma 4 IT answers plus the synthetic
/// shapes that historically broke the renderer. The list is explicit so a new
/// fixture cannot land without an expectation entry.
enum TranscriptCorpus {
    static let fixtures: [String] = [
        "backtick-info-string",
        "arithmetic-prose",
        "blank-line-fence",
        "br-only-paragraph",
        "cjk",
        "crlf",
        "currency",
        "display-math-blank-line",
        "fence-in-list-item",
        "fence-in-quote",
        "fifty-equations",
        "gemma-comparison-table",
        "gemma-derivation-table",
        "gemma-inline-environments",
        "heading-with-math",
        "html-details",
        "large-code-block",
        "latex-paren-bracket",
        "link-dollar-url",
        "list-nested-fence",
        "list-then-fence",
        "lone-display-dollars",
        "long-prose-line",
        "nested-lists",
        "prose-display-dollars",
        "prose-fence",
        "quote-with-math",
        "reference-links",
        "shell-dollars",
        "table-empty-cells",
        "table-in-list",
        "table-html-bold",
        "tab-indented-fence",
        "task-lists",
        "unclosed-display-math",
        "unclosed-fence",
    ]

    /// Fixtures the whole-document render sends to raw source, where the
    /// per-block finalize keeps everything but the block that trips the gate.
    /// This is the difference the per-block finalize exists to make: an
    /// unclosed fence anywhere, a table in a list item, a block-level HTML tag
    /// used to take the whole answer down with them at the moment it finished.
    static let wholeRenderFallbacks: Set<String> = [
        "br-only-paragraph",
        "html-details",
        "table-in-list",
        "unclosed-fence",
    ]

    /// What the per-block render cannot do. A reference definition lives in a
    /// block of its own, so the block that uses the label cannot resolve it.
    /// Scanning the whole response per tick is the shape the streaming timing
    /// gate forbids, and resolving it only at finalize would reintroduce the
    /// flip the per-block pass removes.
    static let perBlockLimitations: Set<String> = ["reference-links"]

    /// Fixtures whose completed blocks end inside a display opener the model
    /// never closed. Completing it is exactly what the pass is for, so these
    /// are only pinned through the call the tail render actually makes, which
    /// keeps the LaTeX as source and leaves the dollars alone.
    static let unclosedDisplayBlocks: Set<String> = ["display-math-blank-line"]

    static func source(_ fixture: String) throws -> String {
        guard let url = Bundle.module.url(
            forResource: fixture,
            withExtension: "md",
            subdirectory: "Fixtures/transcript-corpus") else {
            throw TranscriptFrameError(
                description: "corpus fixture \(fixture).md is not in the test bundle")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}
