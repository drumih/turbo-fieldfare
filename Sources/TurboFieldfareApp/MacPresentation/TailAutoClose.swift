import Foundation

/// Closes the syntax the model has opened but not yet finished, so the block
/// being written can go through the markdown pass instead of sitting there as
/// raw source until its closing marker arrives.
///
/// Display only. The response string itself is never touched: copy actions,
/// `plainText`, and the finalize render all keep reading the real bytes.
///
/// Two rules decide every case here. A marker that has content after it is
/// closed, because the reader is already looking at the styled text and only
/// the closer is missing. A marker with nothing after it is removed instead,
/// because closing `**` with `**` writes `****`, which is new structure rather
/// than the emphasis the model is about to open.
enum TailAutoClose {
    /// U+200B. Breaks the "only `-` or `=`" rule that would otherwise turn the
    /// line above a half-typed rule into a setext heading for one tick.
    static let setextGuard: Character = "\u{200B}"

    /// - Parameter typesetsMath: whether the caller is going to typeset the
    ///   result. An unfinished `$$` is only worth closing for a pass that turns
    ///   it into an equation; a pass that keeps the LaTeX as source would just
    ///   show two dollars the model never wrote.
    static func close(_ tail: String, typesetsMath: Bool = true) -> String {
        guard !tail.isEmpty else { return tail }
        if let fence = unclosedFence(tail) {
            var closed = tail
            if !closed.hasSuffix("\n") { closed.append("\n") }
            closed.append(String(repeating: fence.marker, count: fence.length))
            return closed
        }
        var text = strippingIncompleteTag(tail)
        text = flatteningIncompleteLink(text)
        text = balancingInlineMarkers(text, typesetsMath: typesetsMath)
        return guardingSetextRule(text)
    }

    // MARK: - Fences

    private struct Fence {
        let marker: Character
        let length: Int
    }

    /// The fence a tail ends inside, if any. Nothing inside a fenced block is
    /// markdown, so this case returns before every other rule.
    private static func unclosedFence(_ text: String) -> Fence? {
        var open: Fence?
        for line in text.components(separatedBy: "\n") {
            guard let run = fenceRun(line) else { continue }
            if let current = open {
                if run.marker == current.marker, run.length >= current.length, run.isBare {
                    open = nil
                }
                continue
            }
            open = Fence(marker: run.marker, length: run.length)
        }
        return open
    }

    private static func fenceRun(
        _ line: String
    ) -> (marker: Character, length: Int, isBare: Bool)? {
        var characters = Substring(line)
        var indent = 0
        while let first = characters.first, first == " " || first == "\t" {
            indent += first == "\t" ? 4 : 1
            characters = characters.dropFirst()
        }
        guard indent <= 3, let marker = characters.first,
              marker == "`" || marker == "~" else {
            return nil
        }
        let run = characters.prefix { $0 == marker }
        guard run.count >= 3 else { return nil }
        let rest = characters.dropFirst(run.count)
        return (marker, run.count, rest.allSatisfy { $0 == " " || $0 == "\t" || $0 == "\r" })
    }

    // MARK: - Trailing fragments

    /// A tag the model is still typing is markup the reader should not see.
    /// The remainder has to look like a bare tag name — `a < b` in prose ends
    /// in an unclosed `<` too, and cutting the sentence there would be worse
    /// than showing it.
    private static func strippingIncompleteTag(_ text: String) -> String {
        guard let open = text.lastIndex(of: "<") else { return text }
        var rest = text[text.index(after: open)...]
        if rest.first == "/" { rest = rest.dropFirst() }
        guard let first = rest.first, first.isLetter else { return text }
        guard rest.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" }) else { return text }
        return String(text[text.startIndex..<open])
    }

    /// `[label](https://exa` renders as its label until the destination is
    /// complete. Leaving it alone shows the raw destination mid-sentence.
    private static func flatteningIncompleteLink(_ text: String) -> String {
        guard let paren = text.range(of: "](", options: .backwards) else { return text }
        guard !text[paren.upperBound...].contains(")") else { return text }
        guard !text[paren.upperBound...].contains("\n") else { return text }
        guard let bracket = text.range(of: "[", options: .backwards, range: text.startIndex..<paren.lowerBound)
        else {
            return text
        }
        let label = String(text[bracket.upperBound..<paren.lowerBound])
        var head = String(text[text.startIndex..<bracket.lowerBound])
        if head.hasSuffix("!") {
            head.removeLast()
        }
        return head + label
    }

    /// A trailing `---` or `===` line promotes the line above it to a setext
    /// heading the moment it appears, and demotes it again when the next
    /// character lands. The guard character keeps the paragraph a paragraph.
    private static func guardingSetextRule(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")
        guard lines.count >= 2, let last = lines.last, !last.isEmpty else { return text }
        let trimmed = last.trimmingCharacters(in: CharacterSet(charactersIn: " \t\r"))
        guard !trimmed.isEmpty else { return text }
        let marker = trimmed.first
        guard marker == "-" || marker == "=", trimmed.allSatisfy({ $0 == marker }) else {
            return text
        }
        let previous = lines[lines.count - 2].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !previous.isEmpty else { return text }
        return text + String(setextGuard)
    }

    // MARK: - Inline markers

    private enum Marker {
        case code(length: Int)
        case displayMath
        case strong(Character)
        case emphasis(Character)

        var closer: String {
            switch self {
            case .code(let length): String(repeating: "`", count: length)
            case .displayMath: "$$"
            case .strong(let character): String(repeating: character, count: 2)
            case .emphasis(let character): String(character)
            }
        }
    }

    private struct Opening {
        let marker: Marker
        let start: String.Index
        let end: String.Index
    }

    /// Walks the tail once, keeping a stack of what is open. Order matters:
    /// text after an unterminated backtick is code, and text after an
    /// unterminated `$$` is math, so neither can hold live emphasis. Closing
    /// the stack in reverse produces the right nesting.
    private static func balancingInlineMarkers(
        _ text: String,
        typesetsMath: Bool
    ) -> String {
        var stack: [Opening] = []
        var index = text.startIndex
        var lineStart = text.startIndex

        while index < text.endIndex {
            let character = text[index]
            if character == "\n" {
                index = text.index(after: index)
                lineStart = index
                continue
            }
            if character == "\\", case .code? = stack.last?.marker {
                // A backslash is literal inside code.
                index = text.index(after: index)
                continue
            }
            if character == "\\" {
                index = text.index(after: index)
                if index < text.endIndex { index = text.index(after: index) }
                continue
            }
            if character == "`" {
                let run = text[index...].prefix { $0 == "`" }
                let end = text.index(index, offsetBy: run.count)
                if case .code(let length)? = stack.last?.marker {
                    if run.count == length { stack.removeLast() }
                } else if end == text.endIndex || !text[end].isWhitespace {
                    // "Press the ` key" is prose: a run followed by whitespace
                    // is not the start of a code span, and closing it would
                    // style the rest of the sentence as code while it is the
                    // open tail. Closers stay unconditional above.
                    stack.append(Opening(
                        marker: .code(length: run.count),
                        start: index,
                        end: end))
                }
                index = end
                continue
            }
            if case .code? = stack.last?.marker {
                index = text.index(after: index)
                continue
            }
            if character == "$", text.index(after: index) < text.endIndex,
               text[text.index(after: index)] == "$" {
                let end = text.index(index, offsetBy: 2)
                if case .displayMath? = stack.last?.marker {
                    stack.removeLast()
                } else if typesetsMath,
                          isStandaloneDisplayOpener(text, index: index, lineStart: lineStart) {
                    stack.append(Opening(marker: .displayMath, start: index, end: end))
                }
                index = end
                continue
            }
            if case .displayMath? = stack.last?.marker {
                index = text.index(after: index)
                continue
            }
            if character == "*" || character == "_" {
                index = consumeEmphasis(
                    text,
                    at: index,
                    lineStart: lineStart,
                    marker: character,
                    stack: &stack)
                continue
            }
            index = text.index(after: index)
        }

        // A marker with nothing after it is the model mid-keystroke, not an
        // opener the reader has already seen content for.
        var cut = text.endIndex
        while let last = stack.last, last.end == text.endIndex, last.start < cut {
            cut = last.start
            stack.removeLast()
        }
        let body = cut == text.endIndex ? text : String(text[text.startIndex..<cut])
        guard let innermost = stack.last else { return body }
        var closers = stack.dropLast().reversed().map(\.marker.closer).joined()
        // The first half of a `$$` closer is already on screen; completing it
        // with another pair would leave a stray dollar behind.
        if case .displayMath = innermost.marker, body.hasSuffix("$"), !body.hasSuffix("$$") {
            closers = "$" + closers
        } else {
            closers = innermost.marker.closer + closers
        }
        return body + closers
    }

    /// Single `$` is never closed — the ambiguity with currency is exactly the
    /// bug every surveyed chat UI reports — so only a `$$` that starts its own
    /// line counts as an opener.
    private static func isStandaloneDisplayOpener(
        _ text: String,
        index: String.Index,
        lineStart: String.Index
    ) -> Bool {
        text[lineStart..<index].allSatisfy { $0 == " " || $0 == "\t" || $0 == "\r" }
    }

    private static func consumeEmphasis(
        _ text: String,
        at index: String.Index,
        lineStart: String.Index,
        marker: Character,
        stack: inout [Opening]
    ) -> String.Index {
        let run = text[index...].prefix { $0 == marker }
        let end = text.index(index, offsetBy: run.count)
        // A bullet and a thematic break are block structure; closing them
        // would turn a list into bold text.
        if isListBullet(text, index: index, lineStart: lineStart, runLength: run.count)
            || isThematicBreakLine(text, lineStart: lineStart) {
            return end
        }

        let before = index == text.startIndex ? nil : text[text.index(before: index)]
        let after = end == text.endIndex ? nil : text[end]
        let canOpen = after.map { !$0.isWhitespace } ?? false
        let canClose = before.map { !$0.isWhitespace } ?? false
        // `snake_case` is not emphasis: an intraword underscore neither opens
        // nor closes.
        if marker == "_", isWord(before), isWord(after) { return end }

        var remaining = run.count
        var cursor = index
        while remaining > 0 {
            let width = remaining >= 2 ? 2 : 1
            let step = text.index(cursor, offsetBy: width)
            let wanted: Marker = width == 2 ? .strong(marker) : .emphasis(marker)
            if canClose, matches(stack.last?.marker, wanted) {
                stack.removeLast()
            } else if canOpen || end == text.endIndex {
                // A run that ends the tail has nothing to emphasise yet. It is
                // pushed so the trailing-marker rule can drop it instead of
                // leaving the asterisks on screen.
                stack.append(Opening(marker: wanted, start: cursor, end: end))
            }
            cursor = step
            remaining -= width
        }
        return end
    }

    private static func isWord(_ character: Character?) -> Bool {
        guard let character else { return false }
        return character.isLetter || character.isNumber
    }

    private static func matches(_ open: Marker?, _ wanted: Marker) -> Bool {
        switch (open, wanted) {
        case (.strong(let a), .strong(let b)): a == b
        case (.emphasis(let a), .emphasis(let b)): a == b
        default: false
        }
    }

    private static func isListBullet(
        _ text: String,
        index: String.Index,
        lineStart: String.Index,
        runLength: Int
    ) -> Bool {
        guard runLength == 1, text[lineStart..<index].allSatisfy({ $0 == " " || $0 == "\t" })
        else {
            return false
        }
        let next = text.index(after: index)
        return next >= text.endIndex || text[next] == " " || text[next] == "\t"
    }

    private static func isThematicBreakLine(_ text: String, lineStart: String.Index) -> Bool {
        let line = text[lineStart...].prefix { $0 != "\n" }
        let trimmed = line.filter { $0 != " " && $0 != "\t" && $0 != "\r" }
        guard trimmed.count >= 3, let first = trimmed.first,
              first == "*" || first == "-" || first == "_" else {
            return false
        }
        return trimmed.allSatisfy { $0 == first }
    }
}
