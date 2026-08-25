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
            closed.append(fence.closerPrefix)
            closed.append(String(repeating: fence.run.character, count: fence.run.length))
            return closed
        }
        var text = strippingIncompleteTag(tail)
        text = flatteningIncompleteLink(text)
        text = balancingInlineMarkers(text, typesetsMath: typesetsMath)
        return guardingSetextRule(text)
    }

    // MARK: - Fences

    private struct OpenFence {
        let run: FenceLine.Run
        /// Content column of the item the fence was opened in.
        let containerIndent: Int
        /// What a closing line needs in front of the marker to be read as this
        /// fence's closer. Written flush left instead, the parser reads it as
        /// prose after the item and the listing keeps its markers on screen.
        let closerPrefix: String
    }

    /// The fence a tail ends inside, if any. Nothing inside a fenced block is
    /// markdown, so this case returns before every other rule.
    private static func unclosedFence(_ text: String) -> OpenFence? {
        var open: OpenFence?
        var column = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let quote = ContainerPrefix.strippingQuoteMarkers(line)
            if let current = open {
                if let run = FenceLine.run(quote.rest, containerIndent: current.containerIndent),
                   run.closes(marker: current.run.marker, length: current.run.length) {
                    open = nil
                }
                continue
            }
            if let item = ContainerPrefix.listItem(quote.rest),
               item.markerIndent < column || column == 0 {
                column = item.contentColumn
            }
            guard let run = FenceLine.run(quote.rest, containerIndent: column) else { continue }
            open = OpenFence(
                run: run,
                containerIndent: column,
                closerPrefix: String(repeating: "> ", count: quote.depth)
                    + String(repeating: " ", count: run.indent))
        }
        return open
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
        let start: Int
        let end: Int
    }

    /// Walks the tail once, keeping a stack of what is open. Order matters:
    /// text after an unterminated backtick is code, and text after an
    /// unterminated `$$` is math, so neither can hold live emphasis. Closing
    /// the stack in reverse produces the right nesting.
    ///
    /// The walk is over the detector's own units, with its code and equation
    /// masks: `$x_{1}$` in a sentence is an equation, and the `_` inside it is
    /// a subscript. Per-line state is carried rather than rescanned — the old
    /// pass re-read the line from its start for every marker on it, which is
    /// quadratic in the line and measured 164 ms a tick on an 8 KB paragraph.
    private static func balancingInlineMarkers(
        _ text: String,
        typesetsMath: Bool
    ) -> String {
        let protection = MathSpanDetector.tailProtection(text)
        let units = protection.units
        var stack: [Opening] = []
        var index = 0
        var lineHasContent = false
        var lineIsThematicBreak = isThematicBreakLine(units, from: 0)

        while index < units.count {
            let character = units[index]
            if character == "\n" {
                index += 1
                lineHasContent = false
                lineIsThematicBreak = isThematicBreakLine(units, from: index)
                continue
            }
            let hadContent = lineHasContent
            if character != " ", character != "\t", character != "\r" {
                lineHasContent = true
            }
            if protection.masked[index] {
                index += 1
                continue
            }
            // Everything from an unclosed inline opener on is the equation the
            // model is still writing.
            if let dollar = protection.openInlineDollar, index >= dollar {
                index += 1
                continue
            }
            if character == "\\" {
                // A backslash is literal inside code; outside it, it escapes
                // the character after it.
                if case .code? = stack.last?.marker {
                    index += 1
                } else {
                    index += min(2, units.count - index)
                }
                continue
            }
            if character == "`" {
                var end = index
                while end < units.count, units[end] == "`" { end += 1 }
                if case .code(let length)? = stack.last?.marker {
                    if end - index == length { stack.removeLast() }
                } else if end >= units.count || !units[end].isWhitespace {
                    // "Press the ` key" is prose: a run followed by whitespace
                    // is not the start of a code span, and closing it would
                    // style the rest of the sentence as code while it is the
                    // open tail. Closers stay unconditional above.
                    stack.append(Opening(
                        marker: .code(length: end - index),
                        start: index,
                        end: end))
                }
                index = end
                continue
            }
            if case .code? = stack.last?.marker {
                index += 1
                continue
            }
            if character == "$", index + 1 < units.count, units[index + 1] == "$" {
                let end = index + 2
                if case .displayMath? = stack.last?.marker {
                    stack.removeLast()
                } else if typesetsMath, !hadContent {
                    // Single `$` is never closed — the ambiguity with currency
                    // is exactly the bug every surveyed chat UI reports — so
                    // only a `$$` that starts its own line counts as an opener.
                    stack.append(Opening(marker: .displayMath, start: index, end: end))
                }
                index = end
                continue
            }
            if case .displayMath? = stack.last?.marker {
                index += 1
                continue
            }
            if character == "*" || character == "_" {
                index = consumeEmphasis(
                    units,
                    at: index,
                    marker: character,
                    lineHasContent: hadContent,
                    lineIsThematicBreak: lineIsThematicBreak,
                    stack: &stack)
                continue
            }
            index += 1
        }

        // A marker with nothing after it is the model mid-keystroke, not an
        // opener the reader has already seen content for.
        var cut = units.count
        while let last = stack.last, last.end == units.count, last.start < cut {
            cut = last.start
            stack.removeLast()
        }
        let body = cut == units.count ? text : String(units[0..<cut])
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

    private static func consumeEmphasis(
        _ units: [Character],
        at index: Int,
        marker: Character,
        lineHasContent: Bool,
        lineIsThematicBreak: Bool,
        stack: inout [Opening]
    ) -> Int {
        var end = index
        while end < units.count, units[end] == marker { end += 1 }
        // A bullet and a thematic break are block structure; closing them
        // would turn a list into bold text.
        if lineIsThematicBreak { return end }
        if end - index == 1, !lineHasContent,
           end >= units.count || units[end] == " " || units[end] == "\t" {
            return end
        }

        let before: Character? = index > 0 ? units[index - 1] : nil
        let after: Character? = end < units.count ? units[end] : nil
        // CommonMark 0.31 flanking, with two deliberate deviations. A `*`
        // between word characters is multiplication or a glob, never emphasis,
        // so `2*3` and `x**2` neither open nor close. And `x *= 2` is an
        // operator that reads as left-flanking only because `=` is punctuation;
        // closing it would italicise the rest of the line.
        if marker == "*", isWord(before), isWord(after) { return end }
        let beforeWhitespace = before?.isWhitespace ?? true
        let afterWhitespace = after?.isWhitespace ?? true
        let beforePunctuation = before.map(isPunctuation) ?? false
        let afterPunctuation = after.map(isPunctuation) ?? false
        let leftFlanking = !afterWhitespace
            && (!afterPunctuation || beforeWhitespace || beforePunctuation)
        let rightFlanking = !beforeWhitespace
            && (!beforePunctuation || afterWhitespace || afterPunctuation)
        var canOpen = leftFlanking && after != "="
        var canClose = rightFlanking
        if marker == "_" {
            // `snake_case` is not emphasis: an intraword underscore neither
            // opens nor closes.
            canOpen = leftFlanking && (!rightFlanking || beforePunctuation)
            canClose = rightFlanking && (!leftFlanking || afterPunctuation)
        }

        var remaining = end - index
        var cursor = index
        while remaining > 0 {
            let width = remaining >= 2 ? 2 : 1
            let wanted: Marker = width == 2 ? .strong(marker) : .emphasis(marker)
            if canClose, matches(stack.last?.marker, wanted) {
                stack.removeLast()
            } else if canOpen || end >= units.count {
                // A run that ends the tail has nothing to emphasise yet. It is
                // pushed so the trailing-marker rule can drop it instead of
                // leaving the asterisks on screen.
                stack.append(Opening(marker: wanted, start: cursor, end: end))
            }
            cursor += width
            remaining -= width
        }
        return end
    }

    private static func isPunctuation(_ character: Character) -> Bool {
        character.isPunctuation || character.isSymbol
    }

    /// Computed once per line, and it stops at the first character that is not
    /// the rule's own marker.
    private static func isThematicBreakLine(_ units: [Character], from start: Int) -> Bool {
        var marker: Character?
        var count = 0
        var index = start
        while index < units.count, units[index] != "\n" {
            let character = units[index]
            if character == " " || character == "\t" || character == "\r" {
                index += 1
                continue
            }
            if let marker {
                guard character == marker else { return false }
            } else {
                guard character == "*" || character == "-" || character == "_" else {
                    return false
                }
                marker = character
            }
            count += 1
            index += 1
        }
        return count >= 3
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
}
