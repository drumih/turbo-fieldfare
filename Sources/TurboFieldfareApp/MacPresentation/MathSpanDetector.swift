import Foundation

struct MathSpan: Equatable {
    enum Mode: Equatable {
        case inline
        case display
    }

    /// Character offsets into the string the span was found in.
    let range: Range<Int>
    let mode: Mode
    /// The exact source text, delimiters included.
    let source: String
    /// What goes to the typesetter: the text between the delimiters.
    let latex: String
    /// A partial equation from an interrupted generation. It is substituted
    /// like any other span so the markdown pass cannot eat its backslashes,
    /// then restored verbatim instead of typeset.
    let isLiteralProtect: Bool
}

/// Finds the math in a model answer before the markdown parser gets to mangle
/// it. Every rule here exists because a real answer broke without it: currency
/// pairs, shell variables in prose, `$` inside link destinations, `\$` inside
/// an equation, and code regions that must never be searched at all.
enum MathSpanDetector {
    static let openSentinel: Character = "\u{E000}"
    static let closeSentinel: Character = "\u{E001}"

    struct Substitution: Equatable {
        /// The source with each span replaced by its indexed sentinel.
        let working: String
        let spans: [MathSpan]
    }

    /// Environments KaTeX auto-renders at block level. Anything outside this
    /// set is left to the markdown pass, which is the current behaviour.
    private static let displayEnvironments: Set<String> = [
        "equation", "equation*",
        "align", "align*",
        "gather", "gather*",
        "alignat", "alignat*",
        "CD",
    ]

    /// Model output cannot be allowed to alias a placeholder, so the private
    /// use characters the substitution reserves are removed up front.
    static func strippingSentinels(_ source: String) -> String {
        guard source.contains(openSentinel) || source.contains(closeSentinel) else {
            return source
        }
        return String(source.filter { $0 != openSentinel && $0 != closeSentinel })
    }

    static func substitute(_ source: String) -> Substitution {
        let stripped = strippingSentinels(source)
        let spans = self.spans(in: stripped)
        guard !spans.isEmpty else { return Substitution(working: stripped, spans: spans) }

        let chars = scanUnits(stripped)
        var working = ""
        var cursor = 0
        for (index, span) in spans.enumerated() {
            working.append(contentsOf: chars[cursor..<span.range.lowerBound])
            let sentinel = "\(openSentinel)\(index)\(closeSentinel)"
            // Two display blocks separated by a single newline parse into one
            // paragraph, with the break collapsing to a space. Promoting a
            // standalone block to its own paragraph is what keeps consecutive
            // equations from sharing a line.
            if span.mode == .display, isAloneOnItsLine(span, chars: chars) {
                working += "\n\n" + sentinel + "\n\n"
            } else {
                working += sentinel
            }
            cursor = span.range.upperBound
        }
        working.append(contentsOf: chars[cursor...])
        return Substitution(working: working, spans: spans)
    }

    static func spans(in source: String) -> [MathSpan] {
        let chars = scanUnits(source)
        let masked = codeMask(chars)
        var found: [MathSpan] = []
        var index = 0
        while index < chars.count {
            guard !masked[index] else {
                index += 1
                continue
            }
            if chars[index] == "$", !isEscaped(chars, at: index) {
                index = scanDollar(chars, masked: masked, at: index, into: &found)
                continue
            }
            if chars[index] == "\\", !isEscaped(chars, at: index) {
                index = scanBackslash(chars, masked: masked, at: index, into: &found)
                continue
            }
            index += 1
        }
        return found
    }

    // MARK: - Dollar forms

    private static func scanDollar(
        _ chars: [Character],
        masked: [Bool],
        at start: Int,
        into found: inout [MathSpan]
    ) -> Int {
        let isDisplay = start + 1 < chars.count
            && chars[start + 1] == "$"
            && !masked[start + 1]
        guard isDisplay else {
            switch inlineDollarClose(chars, masked: masked, opener: start) {
            case .closed(let close):
                found.append(span(chars, from: start, to: close + 1, delimiter: 1, mode: .inline))
                return close + 1
            case .exhausted(let resume):
                return resume
            }
        }

        let contentStart = start + 2
        guard let close = displayDollarClose(
            chars,
            masked: masked,
            from: contentStart) else {
            guard let end = literalProtectEnd(chars, from: contentStart) else {
                return contentStart
            }
            found.append(span(
                chars,
                from: start,
                to: end,
                delimiter: 2,
                mode: .display,
                literalProtect: true))
            return end
        }
        // A blank line ends a display block (pandoc), and a candidate that
        // spans one is not math. Scanning resumes immediately after this
        // opener rather than after the rejected region, so the rest of the
        // message keeps its own chances.
        guard close > contentStart,
              !containsBlankLine(chars, in: contentStart..<close) else {
            return contentStart
        }
        found.append(span(chars, from: start, to: close + 2, delimiter: 2, mode: .display))
        return close + 2
    }

    /// What the inline pass found, and where the next opener may start.
    ///
    /// Reporting how far a failed scan looked is what keeps the pass linear.
    /// The closer test depends only on the candidate's own neighbours, so a
    /// `$` that cannot close for this opener cannot close for any opener
    /// between them either, and the region is walked once per message instead
    /// of once per dollar. 180 KB of `$a ` took 16.1 s the other way.
    private enum InlineClose {
        case closed(Int)
        case exhausted(resumeAt: Int)
    }

    /// The pandoc `tex_math_dollars` rules plus the adjacency extensions that
    /// reject `x=$a$b`, `$FOO/bar$BAZ`, and `https://ex.com/$a/$b`. Pandoc
    /// forbids only a digit after the closer; letters are included here
    /// because shell and URL text is the common false positive, not prose.
    private static func inlineDollarClose(
        _ chars: [Character],
        masked: [Bool],
        opener: Int
    ) -> InlineClose {
        let after = opener + 1
        guard after < chars.count, !isSpace(chars[after]), chars[after] != "$" else {
            return .exhausted(resumeAt: after)
        }
        if opener > 0 {
            let before = chars[opener - 1]
            guard !before.isLetter, !before.isNumber, before != "/" else {
                return .exhausted(resumeAt: after)
            }
        }
        var index = after
        while index < chars.count {
            let character = chars[index]
            // Both stops end the region a closer could live in, so the next
            // opener starts here rather than one character past this one.
            if character == "\n" || masked[index] { return .exhausted(resumeAt: index) }
            guard character == "$", !isEscaped(chars, at: index) else {
                index += 1
                continue
            }
            let previous = chars[index - 1]
            let next = index + 1 < chars.count ? chars[index + 1] : nil
            let closes = !isSpace(previous)
                && !(next?.isLetter ?? false)
                && !(next?.isNumber ?? false)
            if closes { return .closed(index) }
            index += 1
        }
        return .exhausted(resumeAt: chars.count)
    }

    private static func displayDollarClose(
        _ chars: [Character],
        masked: [Bool],
        from start: Int
    ) -> Int? {
        var index = start
        while index + 1 < chars.count {
            if chars[index] == "$", chars[index + 1] == "$",
               !masked[index], !masked[index + 1],
               !isEscaped(chars, at: index) {
                return index
            }
            index += 1
        }
        return nil
    }

    // MARK: - Backslash forms

    private static func scanBackslash(
        _ chars: [Character],
        masked: [Bool],
        at start: Int,
        into found: inout [MathSpan]
    ) -> Int {
        guard start + 1 < chars.count else { return start + 1 }
        switch chars[start + 1] {
        case "[":
            guard let close = closingPair(
                chars,
                masked: masked,
                from: start + 2,
                closer: "]") else {
                guard let end = literalProtectEnd(chars, from: start + 2) else {
                    return start + 2
                }
                found.append(span(
                    chars,
                    from: start,
                    to: end,
                    delimiter: 2,
                    mode: .display,
                    literalProtect: true))
                return end
            }
            guard !containsBlankLine(chars, in: (start + 2)..<close) else { return start + 2 }
            found.append(span(chars, from: start, to: close + 2, delimiter: 2, mode: .display))
            return close + 2
        case "(":
            guard let close = closingPair(
                chars,
                masked: masked,
                from: start + 2,
                closer: ")") else {
                return start + 2
            }
            let content = String(chars[(start + 2)..<close])
            // Prose that happens to bracket a word — a regex group, a Swift
            // escape, `\(count\)` — is not math. Requiring a command or an
            // operator is what separates the two.
            guard !containsBlankLine(chars, in: (start + 2)..<close),
                  looksLikeMath(content) else {
                return start + 2
            }
            found.append(span(chars, from: start, to: close + 2, delimiter: 2, mode: .inline))
            return close + 2
        default:
            guard isLineStart(chars, before: start),
                  let environment = environmentName(chars, at: start, keyword: "begin"),
                  displayEnvironments.contains(environment.name),
                  let close = environmentEnd(
                    chars,
                    masked: masked,
                    from: environment.end,
                    name: environment.name) else {
                return start + 1
            }
            let source = String(chars[start..<close])
            found.append(MathSpan(
                range: start..<close,
                mode: .display,
                source: source,
                latex: source,
                isLiteralProtect: false))
            return close
        }
    }

    private static func closingPair(
        _ chars: [Character],
        masked: [Bool],
        from start: Int,
        closer: Character
    ) -> Int? {
        var index = start
        while index + 1 < chars.count {
            if chars[index] == "\\", chars[index + 1] == closer,
               !masked[index], !isEscaped(chars, at: index) {
                return index
            }
            index += 1
        }
        return nil
    }

    private struct EnvironmentKeyword {
        let name: String
        /// Index just past `\begin{name}`.
        let end: Int
    }

    private static func environmentName(
        _ chars: [Character],
        at start: Int,
        keyword: String
    ) -> EnvironmentKeyword? {
        var index = start + 1
        for character in keyword {
            guard index < chars.count, chars[index] == character else { return nil }
            index += 1
        }
        guard index < chars.count, chars[index] == "{" else { return nil }
        index += 1
        var name = ""
        while index < chars.count, chars[index] != "}" {
            name.append(chars[index])
            index += 1
        }
        guard index < chars.count else { return nil }
        return EnvironmentKeyword(name: name, end: index + 1)
    }

    private static func environmentEnd(
        _ chars: [Character],
        masked: [Bool],
        from start: Int,
        name: String
    ) -> Int? {
        var index = start
        while index < chars.count {
            if chars[index] == "\\", !masked[index],
               let keyword = environmentName(chars, at: index, keyword: "end"),
               keyword.name == name {
                return keyword.end
            }
            index += 1
        }
        return nil
    }

    // MARK: - Shared helpers

    /// Swift reads CRLF as a single `Character`, which is neither `\n` nor
    /// `\r` to any rule below: a CRLF answer never ended a line, so its
    /// fences went unmasked, its blank lines went unseen, and a currency
    /// dollar paired with one on the next line. CRLF is the only grapheme
    /// cluster that can contain a line break, so splitting it is enough;
    /// every other cluster stays whole and the offsets still land on
    /// boundaries of the source.
    private static func scanUnits(_ source: String) -> [Character] {
        guard source.contains("\r\n") else { return Array(source) }
        var units: [Character] = []
        units.reserveCapacity(source.unicodeScalars.count)
        for character in source {
            guard character == "\r\n" else {
                units.append(character)
                continue
            }
            units.append("\r")
            units.append("\n")
        }
        return units
    }

    private static func span(
        _ chars: [Character],
        from start: Int,
        to end: Int,
        delimiter: Int,
        mode: MathSpan.Mode,
        literalProtect: Bool = false
    ) -> MathSpan {
        let source = String(chars[start..<end])
        let latex = literalProtect
            ? source
            : String(chars[(start + delimiter)..<(end - delimiter)])
        return MathSpan(
            range: start..<end,
            mode: mode,
            source: source,
            latex: latex,
            isLiteralProtect: literalProtect)
    }

    /// An unterminated display opener protects everything to the end of the
    /// message, minus trailing newlines so the shielded text does not swallow
    /// the paragraph break after it.
    ///
    /// Only a *trailing* opener earns that. A display block may not cross a
    /// blank line, so an opener with one after it is a generation that already
    /// moved on — `Rated $$ on the price scale.` — not one cut off
    /// mid-equation, and protecting from there shows the rest of the answer as
    /// raw source.
    private static func literalProtectEnd(_ chars: [Character], from start: Int) -> Int? {
        var end = chars.count
        while end > start, chars[end - 1] == "\n" || chars[end - 1] == "\r" { end -= 1 }
        guard end > start, chars[start..<end].contains(where: { !isSpace($0) }) else {
            return nil
        }
        guard !containsBlankLine(chars, in: start..<end) else { return nil }
        return end
    }

    private static func looksLikeMath(_ content: String) -> Bool {
        var previous: Character?
        for character in content {
            if previous == "\\", character.isLetter { return true }
            if "^_=<>+-/*".contains(character) { return true }
            previous = character
        }
        return false
    }

    private static func isEscaped(_ chars: [Character], at index: Int) -> Bool {
        var backslashes = 0
        var cursor = index - 1
        while cursor >= 0, chars[cursor] == "\\" {
            backslashes += 1
            cursor -= 1
        }
        return !backslashes.isMultiple(of: 2)
    }

    private static func isSpace(_ character: Character) -> Bool {
        character == " " || character == "\t" || character == "\n" || character == "\r"
    }

    private static func containsBlankLine(_ chars: [Character], in range: Range<Int>) -> Bool {
        var sawNewline = false
        for index in range {
            let character = chars[index]
            if character == "\n" {
                if sawNewline { return true }
                sawNewline = true
                continue
            }
            if character == " " || character == "\t" || character == "\r" { continue }
            sawNewline = false
        }
        return false
    }

    private static func isLineStart(_ chars: [Character], before index: Int) -> Bool {
        var cursor = index - 1
        while cursor >= 0 {
            if chars[cursor] == "\n" { return true }
            guard chars[cursor] == " " || chars[cursor] == "\t" || chars[cursor] == "\r" else {
                return false
            }
            cursor -= 1
        }
        return true
    }

    private static func isAloneOnItsLine(_ span: MathSpan, chars: [Character]) -> Bool {
        let start = span.range.lowerBound
        // Column 0 only: an indented, quoted, or listed block keeps the
        // paragraph style of whatever encloses it.
        guard start == 0 || chars[start - 1] == "\n" else { return false }
        var cursor = span.range.upperBound
        while cursor < chars.count, chars[cursor] != "\n" {
            guard isSpace(chars[cursor]) else { return false }
            cursor += 1
        }
        return true
    }

    // MARK: - Code masking

    /// Fenced blocks, indented code, and backtick spans are masked before any
    /// dollar is looked at. Without this a `bash` fence containing
    /// `"$HOME/$PATH"` renders as an equation.
    static func codeMask(_ chars: [Character]) -> [Bool] {
        var masked = [Bool](repeating: false, count: chars.count)
        maskBlocks(chars, into: &masked)
        maskBacktickSpans(chars, into: &masked)
        return masked
    }

    private struct Line {
        let start: Int
        /// Index just past the last content character, excluding the newline.
        let end: Int
        /// Index just past the newline, or the end of input.
        let next: Int
    }

    private static func lines(_ chars: [Character]) -> [Line] {
        var result: [Line] = []
        var start = 0
        while start <= chars.count {
            var end = start
            while end < chars.count, chars[end] != "\n" { end += 1 }
            let next = end < chars.count ? end + 1 : end
            var content = end
            if content > start, chars[content - 1] == "\r" { content -= 1 }
            result.append(Line(start: start, end: content, next: next))
            if end >= chars.count { break }
            start = next
        }
        return result
    }

    private static func maskBlocks(_ chars: [Character], into masked: inout [Bool]) {
        var fence: (marker: Character, length: Int)?
        var listActive = false
        var previousBlank = true
        var previousIndented = false
        for line in lines(chars) {
            let indent = indentWidth(chars, line: line)
            let trimmed = trimmedStart(chars, line: line)
            let blank = trimmed >= line.end
            if let open = fence {
                mask(line, into: &masked)
                if indent <= 3, closesFence(chars, from: trimmed, to: line.end, open: open) {
                    fence = nil
                }
                previousBlank = false
                previousIndented = false
                continue
            }
            if indent <= 3, let opened = openingFence(chars, from: trimmed, to: line.end) {
                mask(line, into: &masked)
                fence = opened
                previousBlank = false
                previousIndented = false
                continue
            }
            if blank {
                previousBlank = true
                continue
            }
            if indent >= 4 {
                // Four-space text inside a list is item continuation, not a
                // code block; only a fresh indented block after a blank line
                // outside any list is code.
                if !listActive, previousBlank || previousIndented {
                    mask(line, into: &masked)
                    previousIndented = true
                } else {
                    previousIndented = false
                }
                previousBlank = false
                continue
            }
            if isListMarker(chars, from: trimmed, to: line.end) {
                listActive = true
            } else if previousBlank {
                listActive = false
            }
            previousBlank = false
            previousIndented = false
        }
    }

    private static func mask(_ line: Line, into masked: inout [Bool]) {
        for index in line.start..<line.next where index < masked.count {
            masked[index] = true
        }
    }

    private static func indentWidth(_ chars: [Character], line: Line) -> Int {
        var width = 0
        var index = line.start
        while index < line.end {
            if chars[index] == " " { width += 1 } else if chars[index] == "\t" { width += 4 } else { break }
            index += 1
        }
        return width
    }

    private static func trimmedStart(_ chars: [Character], line: Line) -> Int {
        var index = line.start
        while index < line.end, chars[index] == " " || chars[index] == "\t" { index += 1 }
        return index
    }

    private static func openingFence(
        _ chars: [Character],
        from start: Int,
        to end: Int
    ) -> (marker: Character, length: Int)? {
        guard start < end, chars[start] == "`" || chars[start] == "~" else { return nil }
        let marker = chars[start]
        var length = 0
        var index = start
        while index < end, chars[index] == marker {
            length += 1
            index += 1
        }
        guard length >= 3 else { return nil }
        return (marker, length)
    }

    private static func closesFence(
        _ chars: [Character],
        from start: Int,
        to end: Int,
        open: (marker: Character, length: Int)
    ) -> Bool {
        var length = 0
        var index = start
        while index < end, chars[index] == open.marker {
            length += 1
            index += 1
        }
        guard length >= open.length else { return false }
        while index < end {
            guard chars[index] == " " || chars[index] == "\t" else { return false }
            index += 1
        }
        return true
    }

    private static func isListMarker(_ chars: [Character], from start: Int, to end: Int) -> Bool {
        var index = start
        if index < end, chars[index] == "-" || chars[index] == "*" || chars[index] == "+" {
            index += 1
        } else {
            var digits = 0
            while index < end, chars[index].isNumber {
                digits += 1
                index += 1
            }
            guard digits > 0, index < end, chars[index] == "." || chars[index] == ")" else {
                return false
            }
            index += 1
        }
        return index >= end || chars[index] == " " || chars[index] == "\t"
    }

    /// CommonMark code spans: a run of N backticks closes on the next run of
    /// exactly N, within one paragraph. An unmatched run is literal text and
    /// cannot hide anything.
    ///
    /// The paragraph bound is load-bearing. Two prose sentences a few
    /// paragraphs apart — "press the ` key", "use the ` character" — paired
    /// across everything between them, so an equation in the middle was masked
    /// at finalize and shown as raw source, while the streaming pass rendered
    /// each paragraph on its own and typeset it.
    private static func maskBacktickSpans(_ chars: [Character], into masked: inout [Bool]) {
        var index = 0
        while index < chars.count {
            guard chars[index] == "`", !masked[index] else {
                index += 1
                continue
            }
            let openStart = index
            while index < chars.count, chars[index] == "`", !masked[index] { index += 1 }
            let length = index - openStart
            var cursor = index
            var closeStart: Int?
            var sawNewline = false
            while cursor < chars.count {
                let character = chars[cursor]
                if character == "\n" {
                    if sawNewline { break }
                    sawNewline = true
                    cursor += 1
                    continue
                }
                if character == " " || character == "\t" || character == "\r" {
                    cursor += 1
                    continue
                }
                sawNewline = false
                guard character == "`", !masked[cursor] else {
                    cursor += 1
                    continue
                }
                let runStart = cursor
                while cursor < chars.count, chars[cursor] == "`", !masked[cursor] { cursor += 1 }
                if cursor - runStart == length {
                    closeStart = runStart
                    break
                }
            }
            guard let closeStart else { continue }
            for position in openStart..<(closeStart + length) { masked[position] = true }
            index = closeStart + length
        }
    }
}
