import Foundation

/// The rendering corpus: three verbatim Gemma 4 IT answers plus the synthetic
/// shapes that historically broke the renderer. The list is explicit so a new
/// fixture cannot land without an expectation entry.
enum TranscriptCorpus {
    static let fixtures: [String] = [
        "cjk",
        "crlf",
        "currency",
        "fifty-equations",
        "gemma-comparison-table",
        "gemma-derivation-table",
        "gemma-inline-environments",
        "heading-with-math",
        "large-code-block",
        "latex-paren-bracket",
        "link-dollar-url",
        "lone-display-dollars",
        "nested-lists",
        "prose-fence",
        "quote-with-math",
        "shell-dollars",
        "table-html-bold",
        "task-lists",
        "unclosed-display-math",
        "unclosed-fence",
    ]

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
