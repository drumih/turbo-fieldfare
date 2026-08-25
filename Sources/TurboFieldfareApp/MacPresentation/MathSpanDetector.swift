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
        return spans(chars, masked: codeMask(chars), escaped: escapeMask(chars))
    }

    /// What the streaming tail pass must not treat as markdown.
    ///
    /// The tail is auto-closed with the math kept as source, so nothing masked
    /// `$...$` and every `_` in a subscript and `*` in an equation read as an
    /// emphasis marker: `The value $x_{1}$ is small` came back as
    /// `The value $x_{1}$ is small_`.
    struct TailProtection {
        /// The tail split the way the detector scans it, CRLF apart.
        let units: [Character]
        /// Code regions and completed equations. A span the model was cut off
        /// in the middle of is deliberately live: completing it is the tail
        /// pass's own job.
        let masked: [Bool]
        /// The last inline opener on the last line that nothing closes.
        /// Everything from there on is inside an equation still being written.
        let openInlineDollar: Int?
    }

    static func tailProtection(_ source: String) -> TailProtection {
        let units = scanUnits(source)
        var masked = codeMask(units)
        let escaped = escapeMask(units)
        for span in spans(units, masked: masked, escaped: escaped)
        where !span.isLiteralProtect {
            for index in span.range { masked[index] = true }
        }
        return TailProtection(
            units: units,
            masked: masked,
            openInlineDollar: openInlineDollar(units, masked: masked, escaped: escaped))
    }

    /// A digit-led opener is currency — `costs $20 and *very goo` still has
    /// live emphasis after it — and a single-dollar span cannot cross a line,
    /// so only the last line can hold one that is still open.
    private static func openInlineDollar(
        _ units: [Character],
        masked: [Bool],
        escaped: [Bool]
    ) -> Int? {
        var lineStart = units.count
        while lineStart > 0, units[lineStart - 1] != "\n" { lineStart -= 1 }
        var opener: Int?
        for index in lineStart..<units.count {
            guard units[index] == "$", !masked[index], !escaped[index] else { continue }
            let after = index + 1
            guard after < units.count, !isSpace(units[after]), units[after] != "$",
                  !isASCIIDigit(units[after]) else {
                continue
            }
            if index > 0 {
                let before = units[index - 1]
                guard !isASCIIAlphanumeric(before), before != "/" else { continue }
            }
            opener = index
        }
        return opener
    }

    private static func spans(
        _ chars: [Character],
        masked: [Bool],
        escaped: [Bool]
    ) -> [MathSpan] {
        var found: [MathSpan] = []
        var index = 0
        while index < chars.count {
            guard !masked[index] else {
                index += 1
                continue
            }
            if chars[index] == "$", !escaped[index] {
                index = scanDollar(
                    chars, masked: masked, escaped: escaped, at: index, into: &found)
                continue
            }
            if chars[index] == "\\", !escaped[index] {
                index = scanBackslash(
                    chars, masked: masked, escaped: escaped, at: index, into: &found)
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
        escaped: [Bool],
        at start: Int,
        into found: inout [MathSpan]
    ) -> Int {
        var run = 0
        while start + run < chars.count, chars[start + run] == "$", !masked[start + run] {
            run += 1
        }
        // `$$$` is emphasis on a price, a rating, or a typo — never a
        // delimiter. Accepting it made `It costs $$$ a lot.` an opener and
        // shielded everything under it as raw source.
        guard run <= 2 else { return start + run }
        guard run == 2 else {
            switch inlineDollarClose(
                chars, masked: masked, escaped: escaped, opener: start) {
            case .closed(let close):
                found.append(span(chars, from: start, to: close + 1, delimiter: 1, mode: .inline))
                return close + 1
            case .exhausted(let resume):
                return resume
            }
        }

        let contentStart = start + 2
        // A blank line ends a display block (pandoc). It bounds both passes:
        // nothing past one can close this opener, and an opener nothing closes
        // shields only as far as the boundary.
        let blockEnd = displayBlockEnd(chars, from: contentStart)
        guard let close = displayDollarClose(
            chars,
            masked: masked,
            escaped: escaped,
            from: contentStart,
            to: blockEnd) else {
            guard let end = literalProtectEnd(
                chars,
                opener: start,
                from: contentStart,
                to: blockEnd) else {
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
        guard close > contentStart else { return contentStart }
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
    /// forbids only a digit after the closer; ASCII letters are included here
    /// because shell and URL text is the common false positive, not prose.
    /// `isLetter` and `isNumber` are Unicode-wide, so `质量为$m$的物体` read as
    /// one word and its equation stayed raw.
    private static func inlineDollarClose(
        _ chars: [Character],
        masked: [Bool],
        escaped: [Bool],
        opener: Int
    ) -> InlineClose {
        let after = opener + 1
        guard after < chars.count, !isSpace(chars[after]), chars[after] != "$" else {
            return .exhausted(resumeAt: after)
        }
        if opener > 0 {
            let before = chars[opener - 1]
            guard !isASCIIAlphanumeric(before), before != "/" else {
                return .exhausted(resumeAt: after)
            }
        }
        var index = after
        while index < chars.count {
            let character = chars[index]
            // Both stops end the region a closer could live in, so the next
            // opener starts here rather than one character past this one.
            if character == "\n" || masked[index] { return .exhausted(resumeAt: index) }
            guard character == "$", !escaped[index] else {
                index += 1
                continue
            }
            // A `$` after whitespace cannot close anything, and walking past it
            // let a candidate swallow the closer of the span after it:
            // `It costs $5 (or $10 for two). The formula $E=mc^2$ applies.`
            // came back as one span from the `$5` to the equation's closer.
            // The refused closer is where the next candidate starts.
            if isSpace(chars[index - 1]) { return .exhausted(resumeAt: index) }
            let next = index + 1 < chars.count ? chars[index + 1] : nil
            guard !isASCIIAlphanumeric(next) else {
                index += 1
                continue
            }
            guard closesAsMath(chars, opener: opener, closer: index) else {
                return .exhausted(resumeAt: after)
            }
            return .closed(index)
        }
        return .exhausted(resumeAt: chars.count)
    }

    /// A candidate opened on a digit and spanning whitespace is two amounts in
    /// a sentence — `$50 item$` — unless its content actually reads as math.
    private static func closesAsMath(
        _ chars: [Character],
        opener: Int,
        closer: Int
    ) -> Bool {
        let content = (opener + 1)..<closer
        guard isASCIIDigit(chars[opener + 1]),
              content.contains(where: { chars[$0] == " " || chars[$0] == "\t" }) else {
            return true
        }
        return looksLikeMath(String(chars[content]))
    }

    private static func displayDollarClose(
        _ chars: [Character],
        masked: [Bool],
        escaped: [Bool],
        from start: Int,
        to limit: Int
    ) -> Int? {
        var index = start
        while index + 1 < limit {
            if chars[index] == "$", chars[index + 1] == "$",
               !masked[index], !masked[index + 1],
               !escaped[index] {
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
        escaped: [Bool],
        at start: Int,
        into found: inout [MathSpan]
    ) -> Int {
        guard start + 1 < chars.count else { return start + 1 }
        switch chars[start + 1] {
        case "[":
            let blockEnd = displayBlockEnd(chars, from: start + 2)
            guard let close = closingPair(
                chars,
                masked: masked,
                escaped: escaped,
                from: start + 2,
                to: blockEnd,
                closer: "]") else {
                guard let end = literalProtectEnd(
                    chars,
                    opener: start,
                    from: start + 2,
                    to: blockEnd) else {
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
            // Prose that happens to bracket a list — `\[1, 2, 3\]`, an
            // escaped `\[optional\]` flag — is not math. The `\(` branch has
            // required a command or an operator all along; this one accepted
            // anything and fed the words to a typesetter that failed on them.
            guard looksLikeMath(String(chars[(start + 2)..<close]))
                || ownsItsLine(chars, from: start, to: close + 2) else {
                return start + 2
            }
            found.append(span(chars, from: start, to: close + 2, delimiter: 2, mode: .display))
            return close + 2
        case "(":
            guard let close = closingPair(
                chars,
                masked: masked,
                escaped: escaped,
                from: start + 2,
                to: displayBlockEnd(chars, from: start + 2),
                closer: ")") else {
                return start + 2
            }
            // Prose that happens to bracket a word — a regex group, a Swift
            // escape, `\(count\)` — is not math. Requiring a command or an
            // operator is what separates the two.
            guard looksLikeMath(String(chars[(start + 2)..<close])) else {
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
        escaped: [Bool],
        from start: Int,
        to limit: Int,
        closer: Character
    ) -> Int? {
        var index = start
        while index + 1 < limit {
            if chars[index] == "\\", chars[index + 1] == closer,
               !masked[index], !escaped[index] {
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

    /// A blank line is the display block's boundary here too. Without it a
    /// `\begin{align}` in one block paired with an `\end{align}` in another,
    /// which the whole-message render resolved and the per-block finalize
    /// cannot.
    private static func environmentEnd(
        _ chars: [Character],
        masked: [Bool],
        from start: Int,
        name: String
    ) -> Int? {
        let limit = displayBlockEnd(chars, from: start)
        var index = start
        while index < limit {
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

    /// An unterminated display opener shields everything up to the end of its
    /// block, minus trailing newlines so the shielded text does not swallow
    /// the paragraph break after it.
    ///
    /// Two conditions earn that. The opener starts its own line, because a
    /// mid-sentence `$$` is a rating or a price — `Rated $$ on the price
    /// scale.` — not a generation cut off mid-equation. And what follows reads
    /// as TeX, because `$$x = 1` in prose is a sentence and shielding it shows
    /// every heading, list and table under it as raw source.
    private static func literalProtectEnd(
        _ chars: [Character],
        opener: Int,
        from start: Int,
        to limit: Int
    ) -> Int? {
        var end = limit
        while end > start, chars[end - 1] == "\n" || chars[end - 1] == "\r" { end -= 1 }
        guard end > start, chars[start..<end].contains(where: { !isSpace($0) }) else {
            return nil
        }
        guard isLineStart(chars, before: opener), hasTeXContent(chars, in: start..<end) else {
            return nil
        }
        return end
    }

    /// Where the display block that starts at `start` ends: the first blank
    /// line, or the end of the message.
    private static func displayBlockEnd(_ chars: [Character], from start: Int) -> Int {
        var newline: Int?
        for index in start..<chars.count {
            let character = chars[index]
            if character == "\n" {
                if let first = newline { return first }
                newline = index
                continue
            }
            if character == " " || character == "\t" || character == "\r" { continue }
            newline = nil
        }
        return chars.count
    }

    private static func hasTeXContent(_ chars: [Character], in range: Range<Int>) -> Bool {
        var previous: Character?
        for index in range {
            let character = chars[index]
            if previous == "\\", character.isLetter || character == "{" || character == "\\" {
                return true
            }
            if character == "^" || character == "_" || character == "{" { return true }
            previous = character
        }
        return false
    }

    /// The span is the whole line, indentation aside: what makes a delimiter
    /// pair a display block even when its content is a bare list of numbers.
    private static func ownsItsLine(_ chars: [Character], from start: Int, to end: Int) -> Bool {
        guard isLineStart(chars, before: start) else { return false }
        var cursor = end
        while cursor < chars.count, chars[cursor] != "\n" {
            guard isSpace(chars[cursor]) else { return false }
            cursor += 1
        }
        return true
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

    /// Which characters a backslash run has escaped, in one forward pass.
    /// Counting the run backwards per candidate is quadratic in a run of
    /// backslashes: 80 KB of them measured 86.5 s, re-run on the open block on
    /// every streaming tick.
    private static func escapeMask(_ chars: [Character]) -> [Bool] {
        var escaped = [Bool](repeating: false, count: chars.count)
        var backslashes = 0
        for index in chars.indices {
            escaped[index] = !backslashes.isMultiple(of: 2)
            backslashes = chars[index] == "\\" ? backslashes + 1 : 0
        }
        return escaped
    }

    private static func isASCIIAlphanumeric(_ character: Character?) -> Bool {
        guard let value = character?.asciiValue else { return false }
        return (value >= UInt8(ascii: "0") && value <= UInt8(ascii: "9"))
            || (value >= UInt8(ascii: "A") && value <= UInt8(ascii: "Z"))
            || (value >= UInt8(ascii: "a") && value <= UInt8(ascii: "z"))
    }

    private static func isASCIIDigit(_ character: Character) -> Bool {
        guard let value = character.asciiValue else { return false }
        return value >= UInt8(ascii: "0") && value <= UInt8(ascii: "9")
    }

    private static func isSpace(_ character: Character) -> Bool {
        character == " " || character == "\t" || character == "\n" || character == "\r"
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

    /// What kind of line the previous one was. An indented chunk is code only
    /// where a new block can start; after a paragraph line it is a lazy
    /// continuation of that paragraph.
    private enum LineKind {
        case blank
        case fence
        case thematicBreak
        case heading
        case listItem
        case code
        case paragraph

        var startsAChunk: Bool {
            switch self {
            case .blank, .fence, .thematicBreak, .heading, .code: true
            case .listItem, .paragraph: false
            }
        }
    }

    /// Classifies each line in the splitter's order — fence, thematic break,
    /// heading, list, paragraph — against the container it sits in.
    ///
    /// Reading every fence at column zero meant a listing indented into a
    /// nested item was not a fence at all, so its shell variables typeset as
    /// equations. Reading `- - -` as a bullet kept a list open across the rule,
    /// so the indented block under it was item continuation instead of code.
    /// And an indented block after a heading was neither, because the old scan
    /// only started one after a blank line.
    private static func maskBlocks(_ chars: [Character], into masked: inout [Bool]) {
        var fence: (run: FenceLine.Run, containerIndent: Int)?
        /// Content column of the innermost open list item.
        var column: Int?
        var previous = LineKind.blank
        for line in lines(chars) {
            let body = chars[line.start..<line.end]
            let indent = indentWidth(chars, line: line)
            let trimmed = trimmedStart(chars, line: line)

            if let open = fence {
                mask(line, into: &masked)
                if let run = FenceLine.run(body, containerIndent: open.containerIndent),
                   run.closes(marker: open.run.marker, length: open.run.length) {
                    fence = nil
                }
                previous = .fence
                continue
            }
            if trimmed >= line.end {
                previous = .blank
                continue
            }
            // Code comes first: inside a fenced or indented block a rule, a
            // heading and a bullet are all literal text.
            if indent >= (column ?? 0) + 4, previous.startsAChunk {
                mask(line, into: &masked)
                previous = .code
                continue
            }
            if let run = FenceLine.run(body, containerIndent: column ?? 0) {
                mask(line, into: &masked)
                fence = (run, column ?? 0)
                previous = .fence
                continue
            }
            // A fence left of the item's content column is a block of its own,
            // and the list ends where it starts.
            if column != nil, let run = FenceLine.run(body) {
                mask(line, into: &masked)
                fence = (run, 0)
                column = nil
                previous = .fence
                continue
            }
            if isThematicBreak(chars, from: trimmed, to: line.end) {
                column = nil
                previous = .thematicBreak
                continue
            }
            if let item = ContainerPrefix.listItem(body) {
                column = item.contentColumn
                previous = .listItem
                continue
            }
            if isATXHeading(chars, from: trimmed, to: line.end) {
                if indent < (column ?? 0) { column = nil }
                previous = .heading
                continue
            }
            // A line back at the left margin after a blank one has left the
            // list; one that is still indented is item continuation.
            if previous == .blank, indent < 2 { column = nil }
            previous = .paragraph
        }
    }

    private static func isThematicBreak(
        _ chars: [Character],
        from start: Int,
        to end: Int
    ) -> Bool {
        guard start < end else { return false }
        let marker = chars[start]
        guard marker == "-" || marker == "*" || marker == "_" else { return false }
        var count = 0
        for index in start..<end {
            let character = chars[index]
            if character == marker {
                count += 1
            } else if character != " " && character != "\t" && character != "\r" {
                return false
            }
        }
        return count >= 3
    }

    private static func isATXHeading(
        _ chars: [Character],
        from start: Int,
        to end: Int
    ) -> Bool {
        var index = start
        var hashes = 0
        while index < end, chars[index] == "#" {
            hashes += 1
            index += 1
        }
        guard hashes >= 1, hashes <= 6 else { return false }
        return index >= end || chars[index] == " " || chars[index] == "\t"
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
