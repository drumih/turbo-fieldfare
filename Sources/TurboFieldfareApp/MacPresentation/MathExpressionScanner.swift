import Foundation

public struct MathExpression: Equatable {
    public let latex: String
    public let isDisplay: Bool
    public let range: Range<String.Index>

    public init(latex: String, isDisplay: Bool, range: Range<String.Index>) {
        self.latex = latex
        self.isDisplay = isDisplay
        self.range = range
    }
}

public enum MathExpressionScanner {
    public static func scan(_ source: String) -> [MathExpression] {
        var expressions: [MathExpression] = []
        var index = source.startIndex
        while index < source.endIndex {
            guard source[index] == "$", !isEscaped(source, at: index) else {
                index = source.index(after: index)
                continue
            }

            let isDisplay = isDoubleDollar(at: index, in: source)
            let contentStart = isDisplay
                ? source.index(index, offsetBy: 2, limitedBy: source.endIndex)
                    ?? source.endIndex
                : source.index(after: index)

            if !isDisplay, !isInlineDelimiter(at: index, in: source) {
                index = source.index(after: index)
                continue
            }

            guard let closing = closingDelimiter(
                in: source,
                from: contentStart,
                display: isDisplay),
                contentStart < closing,
                let content = validContent(
                    in: source,
                    from: contentStart,
                    to: closing,
                    display: isDisplay) else {
                index = source.index(after: index)
                continue
            }

            let upperBound = source.index(
                closing,
                offsetBy: isDisplay ? 2 : 1,
                limitedBy: source.endIndex) ?? source.endIndex
            expressions.append(MathExpression(
                latex: content,
                isDisplay: isDisplay,
                range: index..<upperBound))
            index = upperBound
        }
        return expressions
    }

    private static func isDoubleDollar(at index: String.Index, in source: String) -> Bool {
        guard let next = source.index(index, offsetBy: 1, limitedBy: source.endIndex),
              next < source.endIndex else { return false }
        return source[next] == "$"
    }

    private static func isInlineDelimiter(at index: String.Index, in source: String) -> Bool {
        let next = source.index(after: index)
        guard next < source.endIndex else { return false }
        let nextCharacter = source[next]
        if nextCharacter == "$" || nextCharacter == " " || nextCharacter.isNumber {
            return false
        }
        if index > source.startIndex {
            let previous = source[source.index(before: index)]
            if previous.isLetter || previous.isNumber {
                return false
            }
        }
        return true
    }

    private static func closingDelimiter(
        in source: String,
        from start: String.Index,
        display: Bool
    ) -> String.Index? {
        var index = start
        while index < source.endIndex {
            if source[index] == "$", !isEscaped(source, at: index) {
                if display {
                    if let next = source.index(index, offsetBy: 1, limitedBy: source.endIndex),
                       next < source.endIndex, source[next] == "$" {
                        return index
                    }
                } else {
                    let next = source.index(after: index)
                    if !(next < source.endIndex && source[next].isNumber) {
                        return index
                    }
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func validContent(
        in source: String,
        from start: String.Index,
        to end: String.Index,
        display: Bool
    ) -> String? {
        let content = String(source[start..<end])
        if display {
            if content.isEmpty || content.contains("\n\n") { return nil }
        } else if content.isEmpty || content.contains("\n") {
            return nil
        }
        guard bracesAreBalanced(content) else { return nil }
        guard looksLikeMath(content) else { return nil }
        return content
    }

    private static func bracesAreBalanced(_ content: String) -> Bool {
        var depth = 0
        var escaped = false
        for character in content {
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth < 0 { return false }
            }
        }
        return depth == 0
    }

    private static func looksLikeMath(_ content: String) -> Bool {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let hasSymbol = trimmed.unicodeScalars.contains { scalar in
            if scalar == "\\" { return true }
            let isLetter = CharacterSet.letters.contains(scalar)
            let isDigit = CharacterSet.decimalDigits.contains(scalar)
            let isWhitespace = CharacterSet.whitespacesAndNewlines.contains(scalar)
            return !isLetter && !isDigit && !isWhitespace
        }
        if hasSymbol { return true }
        return trimmed.count <= 2
    }

    private static func isEscaped(_ source: String, at index: String.Index) -> Bool {
        guard index > source.startIndex else { return false }
        return source[source.index(before: index)] == "\\"
    }
}
