import AppKit
import Foundation

/// Colors and attribute keys shared by the Markdown renderer and the transcript
/// layout manager. Every color resolves per light/dark appearance so the
/// transcript reads correctly in both.
public enum TranscriptCodeStyle {
    /// Marks the character range of a fenced code block so the layout manager can
    /// draw a single rounded container behind it. The value is the language hint.
    public nonisolated static let codeBlockAttribute = NSAttributedString.Key("TurboFieldfareCodeBlock")

    nonisolated static func dynamic(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        }
    }

    public nonisolated static let codeBackground = dynamic(
        light: NSColor(srgbRed: 0.96, green: 0.96, blue: 0.97, alpha: 1),
        dark: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.055))
    public nonisolated static let codeBorder = dynamic(
        light: NSColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.08),
        dark: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.10))
    public nonisolated static let inlineCodeBackground = dynamic(
        light: NSColor(srgbRed: 0.5, green: 0.5, blue: 0.55, alpha: 0.14),
        dark: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.11))
    public nonisolated static let tableBorder = dynamic(
        light: NSColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.14),
        dark: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.16))
    public nonisolated static let tableHeaderBackground = dynamic(
        light: NSColor(srgbRed: 0.5, green: 0.5, blue: 0.55, alpha: 0.10),
        dark: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.07))

    nonisolated static let codeText = dynamic(
        light: NSColor(srgbRed: 0.13, green: 0.14, blue: 0.16, alpha: 1),
        dark: NSColor(srgbRed: 0.90, green: 0.91, blue: 0.94, alpha: 1))
    nonisolated static let keyword = dynamic(
        light: NSColor(srgbRed: 0.68, green: 0.14, blue: 0.55, alpha: 1),
        dark: NSColor(srgbRed: 1.00, green: 0.48, blue: 0.70, alpha: 1))
    nonisolated static let string = dynamic(
        light: NSColor(srgbRed: 0.77, green: 0.10, blue: 0.09, alpha: 1),
        dark: NSColor(srgbRed: 0.99, green: 0.42, blue: 0.36, alpha: 1))
    nonisolated static let comment = dynamic(
        light: NSColor(srgbRed: 0.42, green: 0.47, blue: 0.53, alpha: 1),
        dark: NSColor(srgbRed: 0.50, green: 0.55, blue: 0.60, alpha: 1))
    nonisolated static let number = dynamic(
        light: NSColor(srgbRed: 0.11, green: 0.00, blue: 0.81, alpha: 1),
        dark: NSColor(srgbRed: 0.82, green: 0.75, blue: 0.41, alpha: 1))
    nonisolated static let type = dynamic(
        light: NSColor(srgbRed: 0.04, green: 0.31, blue: 0.48, alpha: 1),
        dark: NSColor(srgbRed: 0.42, green: 0.87, blue: 1.00, alpha: 1))
}

/// Lightweight, language-aware syntax highlighter. It is intentionally
/// approximate: LLM code answers span many languages, so it colors the features
/// that read the same almost everywhere — comments, strings, numbers, a broad
/// keyword union, and capitalized type names — rather than parsing any single
/// grammar precisely.
public struct CodeSyntaxHighlighter {
    public init() {}

    private static let keywords: Set<String> = [
        "func", "function", "fn", "def", "lambda", "return", "yield",
        "if", "else", "elif", "elseif", "then", "for", "while", "do", "loop",
        "switch", "case", "default", "match", "when", "break", "continue", "goto",
        "let", "var", "const", "val", "mut", "static", "final", "readonly",
        "class", "struct", "enum", "interface", "protocol", "extension", "trait",
        "impl", "namespace", "module", "package", "type", "typedef", "typealias",
        "public", "private", "protected", "internal", "export", "import", "from",
        "using", "include", "require", "use", "pub", "open", "override", "virtual",
        "self", "this", "super", "new", "delete", "nil", "null", "none", "undefined",
        "true", "false", "void", "async", "await", "throw", "throws", "try", "catch",
        "finally", "defer", "guard", "where", "in", "is", "as", "and", "or", "not",
        "with", "begin", "end", "unsafe", "extern", "inline", "operator", "sizeof",
    ]

    private static let cLike = "(?<comment>//[^\\n]*|/\\*[\\s\\S]*?\\*/)"
    private static let hashLike = "(?<comment>#[^\\n]*)"
    private static let dashLike = "(?<comment>--[^\\n]*)"

    private func commentPattern(for language: String) -> String {
        switch language.lowercased() {
        case "python", "py", "ruby", "rb", "bash", "sh", "shell", "zsh", "yaml",
             "yml", "toml", "r", "perl", "makefile", "make", "dockerfile", "conf",
             "ini", "elixir", "ex", "julia", "jl", "coffee":
            return Self.hashLike
        case "sql", "lua", "haskell", "hs", "ada", "vhdl":
            return Self.dashLike
        default:
            return Self.cLike
        }
    }

    /// Apply syntax colors in place over `range` of `storage`. Named capture
    /// groups keep the token categories stable even though the comment pattern
    /// (and therefore its group count) varies by language.
    public func highlight(
        _ storage: NSMutableAttributedString,
        range: NSRange,
        language: String
    ) {
        let text = storage.string as NSString
        let comment = commentPattern(for: language)
        let string = "(?<str>\"\"\"[\\s\\S]*?\"\"\"|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*')"
        let number = "(?<num>\\b(?:0[xX][0-9A-Fa-f_]+|\\d[\\d_]*\\.?[\\d_]*(?:[eE][+-]?\\d+)?)\\b)"
        let ident = "(?<ident>\\b[A-Za-z_][A-Za-z0-9_]*\\b)"
        let pattern = "\(comment)|\(string)|\(number)|\(ident)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return }

        func matched(_ match: NSTextCheckingResult, _ name: String) -> Bool {
            match.range(withName: name).location != NSNotFound
        }

        regex.enumerateMatches(in: storage.string, range: range) { match, _, _ in
            guard let match else { return }
            let color: NSColor
            if matched(match, "comment") {
                color = TranscriptCodeStyle.comment
            } else if matched(match, "str") {
                color = TranscriptCodeStyle.string
            } else if matched(match, "num") {
                color = TranscriptCodeStyle.number
            } else {
                let word = text.substring(with: match.range)
                if Self.keywords.contains(word) {
                    color = TranscriptCodeStyle.keyword
                } else if let first = word.first, first.isUppercase {
                    color = TranscriptCodeStyle.type
                } else {
                    return
                }
            }
            storage.addAttribute(.foregroundColor, value: color, range: match.range)
        }
    }
}
