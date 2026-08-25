import AppKit
import Foundation

/// The renderer surface the transcript controller depends on. Narrow on
/// purpose: it is also the seam a test uses to count how many times a block is
/// rendered while an answer streams.
@MainActor
public protocol TranscriptBlockRendering {
    func render(_ source: String, typesetsMath: Bool) -> ResponseMarkdownRenderer.Result
    func blockSeparator(trailingNewlines: Int) -> NSAttributedString
    func streamingCodeAttributes() -> [NSAttributedString.Key: Any]
}

@MainActor
public struct ResponseMarkdownRenderer: TranscriptBlockRendering {
    public struct Result {
        public let attributedString: NSAttributedString
        public let usedFallback: Bool

        public init(attributedString: NSAttributedString, usedFallback: Bool) {
            self.attributedString = attributedString
            self.usedFallback = usedFallback
        }
    }

    private struct TableCell: Equatable {
        let table: Int
        let columns: Int
        let row: Int
        let column: Int
        let isHeader: Bool
        let alignment: NSTextAlignment
    }

    private enum BlockKind: Equatable {
        case paragraph
        case heading(Int)
        case code
        case quote
        case unorderedList(indent: Int)
        case orderedList(ordinal: Int, indent: Int)
        case thematicBreak
        case tableCell(TableCell)

        var isList: Bool {
            switch self {
            case .unorderedList, .orderedList: true
            default: false
            }
        }

        /// A header row reads as a header from its weight, not only from its
        /// fill, which is barely visible in dark mode.
        var isHeaderCell: Bool {
            switch self {
            case .tableCell(let cell): cell.isHeader
            default: false
            }
        }

        var tableIdentity: Int? {
            switch self {
            case .tableCell(let cell): cell.table
            default: nil
            }
        }
    }

    private struct Block: Equatable {
        let identity: Int
        let kind: BlockKind
    }

    private struct Item {
        let text: String
        let inlineIntent: InlinePresentationIntent?
        let link: URL?
        let imageURL: URL?
        let block: Block
    }

    private let mathEnabled: Bool
    private let typesetter: any MathTypesetting

    public init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        typesetter: any MathTypesetting = SwiftMathTypesetter()
    ) {
        mathEnabled = environment["TURBO_FIELDFARE_DISABLE_MATH"] != "1"
        self.typesetter = typesetter
    }

    /// `typesetsMath: false` still lifts every equation out of the markdown
    /// pass — which would eat its backslashes — but puts the source back
    /// instead of an image. The block being streamed uses it: a half-typed
    /// equation cannot typeset, and asking a failing parse per tick buys a
    /// log line and nothing else.
    public func render(_ source: String, typesetsMath: Bool = true) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        guard !requiresRawFallback(source) else { return fallback(source) }
        // Math is lifted out before the markdown pass, which eats backslashes,
        // unescapes braces, and drops `\[` delimiters outright. Every fallback
        // below returns the untouched `source`, never this working copy.
        let substitution = mathEnabled ? MathSpanDetector.substitute(source) : nil
        let presentationSource = Self.promotingBoldHeadings(substitution?.working ?? source)

        do {
            let parsed = try AttributedString(
                markdown: presentationSource,
                options: .init(
                    interpretedSyntax: .full,
                    failurePolicy: .returnPartiallyParsedIfPossible))
            guard !containsUnsupportedBlock(in: parsed) else { return fallback(source) }

            let items = Self.completingTables(parsed.runs.map { run in
                let text = String(parsed[run.range].characters)
                return Item(
                    // An image is its alt text, or its destination when the
                    // model wrote none: the parser hands back a bare
                    // object-replacement character, which draws as nothing at
                    // all where the reader expected a picture.
                    text: Self.imageText(text, url: run.imageURL) ?? text,
                    inlineIntent: run.inlinePresentationIntent,
                    link: Self.openable(run.link),
                    imageURL: Self.openable(run.imageURL),
                    block: block(for: run.presentationIntent))
            })

            let output = NSMutableAttributedString()
            let decoration = BlockDecoration()
            var previousBlock: Block?
            var index = 0
            while index < items.count {
                let block = items[index].block
                var end = index
                while end < items.count, items[end].block == block { end += 1 }
                appendSeparator(to: output, previous: previousBlock, next: block)
                append(
                    Array(items[index..<end]),
                    block: block,
                    to: output,
                    decoration: decoration)
                previousBlock = block
                index = end
            }

            guard output.length > 0 else { return fallback(source) }
            if let substitution, !substitution.spans.isEmpty {
                guard resolveMath(
                    substitution.spans,
                    in: output,
                    typesetsMath: typesetsMath) else {
                    return fallback(source)
                }
            }
            return Result(attributedString: output, usedFallback: false)
        } catch {
            return fallback(source)
        }
    }

    /// Copy actions read the raw response, but this is the transcript's own
    /// text projection and must never start returning the object-replacement
    /// character where an equation used to be.
    public func plainText(_ source: String) -> String {
        TranscriptPlainText.string(of: render(source).attributedString)
    }

    /// Replaces every sentinel with its attachment, or with the raw span text
    /// when the equation does not typeset. Returns false when an emitted index
    /// never turns up: a sentinel inside a link destination is percent-encoded
    /// into the `.link` attribute and disappears from the character stream, and
    /// a whole-message raw fallback beats a silently deleted equation.
    private func resolveMath(
        _ spans: [MathSpan],
        in output: NSMutableAttributedString,
        typesetsMath: Bool
    ) -> Bool {
        let open = String(MathSpanDetector.openSentinel)
        let close = String(MathSpanDetector.closeSentinel)
        var consumed: Set<Int> = []
        var searchStart = 0
        while true {
            let text = output.string as NSString
            let tail = NSRange(
                location: searchStart,
                length: max(0, text.length - searchStart))
            let opened = text.range(of: open, range: tail)
            guard opened.location != NSNotFound else { break }
            let afterOpen = NSRange(
                location: opened.upperBound,
                length: text.length - opened.upperBound)
            let closed = text.range(of: close, range: afterOpen)
            guard closed.location != NSNotFound else { return false }
            let digits = text.substring(with: NSRange(
                location: opened.upperBound,
                length: closed.location - opened.upperBound))
            guard let index = Int(digits), spans.indices.contains(index),
                  consumed.insert(index).inserted else {
                return false
            }
            let sentinel = NSRange(
                location: opened.location,
                length: closed.upperBound - opened.location)
            let resolved = resolution(
                for: spans[index],
                sentinel: sentinel,
                in: output,
                typesetsMath: typesetsMath)
            output.replaceCharacters(in: sentinel, with: resolved.text)
            if resolved.centered {
                let paragraph = (output.string as NSString).paragraphRange(
                    for: NSRange(location: sentinel.location, length: resolved.text.length))
                center(paragraph, in: output)
            }
            searchStart = sentinel.location + resolved.text.length
        }
        return consumed.count == spans.count
    }

    private struct MathResolution {
        let text: NSAttributedString
        let centered: Bool
    }

    private func resolution(
        for span: MathSpan,
        sentinel: NSRange,
        in output: NSMutableAttributedString,
        typesetsMath: Bool
    ) -> MathResolution {
        // `NSAttributedString(attachment:)` carries no attributes at all, so
        // the sentinel run's font, colour, and paragraph style are copied
        // first; without them math inside a list or quote loses its indent.
        let attributes = output.attributes(at: sentinel.location, effectiveRange: nil)
        guard typesetsMath, !span.isLiteralProtect else {
            return MathResolution(
                text: NSAttributedString(string: span.source, attributes: attributes),
                centered: false)
        }
        let blockShaped = span.mode == .display || Self.isBlockShaped(span.latex)
        let alone = blockShaped && Self.isAloneInParagraph(output, sentinel: sentinel)
        let mode: MathRenderMode = span.mode == .display || alone ? .display : .inline
        let font = attributes[.font] as? NSFont
            ?? NSFont.systemFont(ofSize: NSFont.systemFontSize)
        let tint = attributes[.foregroundColor] as? NSColor ?? .labelColor
        guard let render = typesetter.render(
            latex: MathCommandNormalizer.normalize(span.latex),
            fontSize: font.pointSize,
            tint: tint,
            mode: mode) else {
            return MathResolution(
                text: NSAttributedString(string: span.source, attributes: attributes),
                centered: false)
        }
        let attachment = MathAttachment(latexSource: span.source)
        // A stored property is not in the accessibility tree; the image is.
        // The typesetter memoises by latex, size, and tint, so two spans
        // written with different delimiters share one image and the second
        // description overwrote the first for both of them. The attachment
        // takes its own handle on the same drawing instead.
        let image = render.image.copy() as? NSImage ?? render.image
        image.accessibilityDescription = span.source
        attachment.image = image
        attachment.bounds = CGRect(
            x: 0,
            y: -render.descent,
            width: image.size.width,
            height: image.size.height)
        let replacement = NSMutableAttributedString(
            string: "\u{FFFC}",
            attributes: attributes)
        replacement.addAttribute(
            .attachment,
            value: attachment,
            range: NSRange(location: 0, length: replacement.length))
        return MathResolution(text: replacement, centered: alone)
    }

    /// Inline environments and hard row breaks are block notation wearing
    /// inline delimiters: Gemma writes `$\begin{cases} ... \end{cases}$`.
    private static func isBlockShaped(_ latex: String) -> Bool {
        latex.contains("\\begin{aligned}")
            || latex.contains("\\begin{cases}")
            || latex.contains("\\\\")
    }

    /// Paragraph styles are per-paragraph in AppKit, so centring is only safe
    /// when the equation is the whole paragraph. A cell or code container owns
    /// its own style through `textBlocks` and keeps it.
    private static func isAloneInParagraph(
        _ output: NSAttributedString,
        sentinel: NSRange
    ) -> Bool {
        let text = output.string as NSString
        let paragraph = text.paragraphRange(for: sentinel)
        let before = text.substring(with: NSRange(
            location: paragraph.location,
            length: sentinel.location - paragraph.location))
        let after = text.substring(with: NSRange(
            location: sentinel.upperBound,
            length: max(0, paragraph.upperBound - sentinel.upperBound)))
        guard before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              after.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        let style = output.attribute(
            .paragraphStyle,
            at: sentinel.location,
            effectiveRange: nil) as? NSParagraphStyle
        return style?.textBlocks.isEmpty ?? true
    }

    private func center(_ range: NSRange, in output: NSMutableAttributedString) {
        guard range.length > 0 else { return }
        let existing = output.attribute(
            .paragraphStyle,
            at: range.location,
            effectiveRange: nil) as? NSParagraphStyle
        let style = (existing?.mutableCopy() as? NSMutableParagraphStyle)
            ?? NSMutableParagraphStyle()
        style.alignment = .center
        output.addAttribute(.paragraphStyle, value: style, range: range)
    }

    /// True when the answer ends inside a fenced block the model never closed.
    /// Counting ```` ``` ```` delimiters instead read a single mid-line run as
    /// an unclosed fence and sent a whole styled answer — heading, table and
    /// all — to raw source.
    private func requiresRawFallback(_ source: String) -> Bool {
        var open: FenceLine.Run?
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            guard let run = FenceLine.run(line) else { continue }
            guard let current = open else {
                open = run
                continue
            }
            if run.closes(marker: current.marker, length: current.length) { open = nil }
        }
        return open != nil
    }

    private func containsUnsupportedBlock(in parsed: AttributedString) -> Bool {
        parsed.runs.contains { run in
            run.inlinePresentationIntent?.contains(.blockHTML) == true
                || Self.isEnclosedTable(run.presentationIntent)
        }
    }

    /// A table nested in a quote or a list item. The cell walk owns the
    /// paragraph style through `textBlocks`, so the enclosing indent, bar, and
    /// marker are all lost and the rows draw as a bare table hanging in the
    /// message. Raw fallback keeps the content with its markers instead;
    /// rendering the nesting properly is separate work.
    private static func isEnclosedTable(_ intent: PresentationIntent?) -> Bool {
        guard let components = intent?.components else { return false }
        var table = false
        var enclosing = false
        for component in components {
            switch component.kind {
            case .table: table = true
            case .blockQuote, .orderedList, .unorderedList, .listItem: enclosing = true
            default: break
            }
        }
        return table && enclosing
    }

    /// A bold-only line is how models write a heading they did not mark up.
    /// Giving it its own paragraph is what keeps the line under it from
    /// joining it, but a fenced listing is the one block whose bytes have to
    /// survive this pass exactly: promoting a `**Note**` line inside code
    /// inserted a blank line the model never wrote, so the finalize render
    /// stopped matching the bytes the stream had already drawn.
    private static func promotingBoldHeadings(_ source: String) -> String {
        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        var segments: [(lines: [Substring], isCode: Bool)] = []
        var fence: FenceLine.Run?
        for line in lines {
            let run = FenceLine.run(line)
            let isCode: Bool
            if let open = fence {
                isCode = true
                if let run, run.closes(marker: open.marker, length: open.length) {
                    fence = nil
                }
            } else if let run {
                isCode = true
                fence = run
            } else {
                isCode = false
            }
            if segments.last?.isCode == isCode {
                segments[segments.count - 1].lines.append(line)
            } else {
                segments.append(([line], isCode))
            }
        }
        return segments.map { segment in
            let text = segment.lines.joined(separator: "\n")
            guard !segment.isCode else { return text }
            return text.replacingOccurrences(
                of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
                with: "$1\n\n",
                options: .regularExpression)
        }.joined(separator: "\n")
    }

    private func block(for intent: PresentationIntent?) -> Block {
        guard let components = intent?.components, let leaf = components.first else {
            return Block(identity: 0, kind: .paragraph)
        }

        var headingLevel: Int?
        var code = false
        var quote = false
        var thematicBreak = false
        var listDepth = 0
        var tableIdentity: Int?
        var tableColumns: [PresentationIntent.TableColumn] = []
        var tableRow: Int?
        var tableColumn: Int?
        var isHeaderRow = false

        for component in components {
            switch component.kind {
            case .header(let level): headingLevel = level
            case .codeBlock: code = true
            case .blockQuote: quote = true
            case .thematicBreak: thematicBreak = true
            case .orderedList, .unorderedList:
                listDepth += 1
            case .table(let columns):
                tableIdentity = component.identity
                tableColumns = columns
            case .tableHeaderRow:
                isHeaderRow = true
                tableRow = 0
            case .tableRow(let index):
                tableRow = index
            case .tableCell(let index):
                tableColumn = index
            default: break
            }
        }

        if let tableIdentity, let tableRow, let tableColumn {
            let alignment = tableColumn < tableColumns.count
                ? Self.alignment(for: tableColumns[tableColumn].alignment)
                : NSTextAlignment.natural
            return Block(
                identity: leaf.identity,
                kind: .tableCell(TableCell(
                    table: tableIdentity,
                    columns: tableColumns.count,
                    row: tableRow,
                    column: tableColumn,
                    isHeader: isHeaderRow,
                    alignment: alignment)))
        }

        let marker = Self.innermostListMarker(components)
        let kind: BlockKind
        if thematicBreak {
            kind = .thematicBreak
        } else if let headingLevel {
            kind = .heading(headingLevel)
        } else if code {
            kind = .code
        } else if let marker, marker.isOrdered {
            kind = .orderedList(ordinal: marker.ordinal, indent: max(0, listDepth - 1))
        } else if marker != nil {
            kind = .unorderedList(indent: max(0, listDepth - 1))
        } else if quote {
            kind = .quote
        } else {
            kind = .paragraph
        }
        return Block(identity: leaf.identity, kind: kind)
    }

    /// The marker a nested item actually wears. `components` runs
    /// innermost-first, so the item is the first `listItem` in the array and
    /// its list is the first list container after it. Reading the whole array
    /// last-write-wins handed every item the *outermost* list's kind and
    /// ordinal, so a bullet under an ordered item rendered as a number and
    /// every nested ordered item repeated its parent's ordinal as "1. 1.".
    private static func innermostListMarker(
        _ components: [PresentationIntent.IntentType]
    ) -> (ordinal: Int, isOrdered: Bool)? {
        guard let item = components.firstIndex(where: {
            if case .listItem = $0.kind { return true }
            return false
        }) else {
            return nil
        }
        guard case .listItem(let ordinal) = components[item].kind else { return nil }
        for component in components[(item + 1)...] {
            switch component.kind {
            case .orderedList: return (ordinal, true)
            case .unorderedList: return (ordinal, false)
            default: continue
            }
        }
        return nil
    }

    private static func alignment(
        for column: PresentationIntent.TableColumn.Alignment?
    ) -> NSTextAlignment {
        switch column {
        case .left: .left
        case .center: .center
        case .right: .right
        default: .natural
        }
    }

    private func appendSeparator(
        to output: NSMutableAttributedString,
        previous: Block?,
        next: Block
    ) {
        guard let previous else { return }
        let adjacentListItems = previous.kind.isList && next.kind.isList
        let sameTable = previous.kind.tableIdentity != nil
            && previous.kind.tableIdentity == next.kind.tableIdentity
        let requiredNewlines = adjacentListItems || sameTable ? 1 : 2
        let trailingNewlines = output.string.reversed().prefix { $0 == "\n" }.count
        output.append(separator(
            trailingNewlines: trailingNewlines,
            required: requiredNewlines))
    }

    /// What the whole-document walk puts between two top-level blocks. The
    /// progressive render appends one block at a time and has to reproduce it
    /// exactly, or a streamed answer would not match its own finalize pass.
    public func blockSeparator(trailingNewlines: Int) -> NSAttributedString {
        separator(trailingNewlines: trailingNewlines, required: 2)
    }

    private func separator(trailingNewlines: Int, required: Int) -> NSAttributedString {
        let missing = required - max(0, trailingNewlines)
        guard missing > 0 else { return NSAttributedString() }
        return NSAttributedString(
            string: String(repeating: "\n", count: missing),
            attributes: baseAttributes())
    }

    private func append(
        _ items: [Item],
        block: Block,
        to output: NSMutableAttributedString,
        decoration: BlockDecoration
    ) {
        if block.kind == .thematicBreak {
            output.append(NSAttributedString(
                string: "────────────────",
                attributes: attributes(block: block)))
            return
        }
        if block.kind == .code {
            appendCode(
                items.map(\.text).joined(),
                block: block,
                to: output,
                decoration: decoration)
            return
        }
        var cellStyle: NSParagraphStyle?
        if case .tableCell(let cell) = block.kind {
            cellStyle = tableCellStyle(cell, decoration: decoration)
        }

        var texts = items.map(\.text)
        var taskGlyph: String?
        if block.kind.isList,
           let first = texts.first,
           let glyph = Self.taskMarker(first) {
            taskGlyph = glyph
            texts[0] = String(first.dropFirst(Self.taskMarkerLength))
        }
        let prefix = self.prefix(for: block.kind, taskGlyph: taskGlyph)
        if !prefix.isEmpty {
            output.append(NSAttributedString(
                string: prefix,
                attributes: attributes(block: block)))
        }

        var html = InlineHTMLState()
        for (item, text) in zip(items, texts) {
            if item.inlineIntent?.contains(.inlineHTML) == true {
                appendInlineHTML(
                    text,
                    state: &html,
                    block: block,
                    paragraphStyle: cellStyle,
                    to: output)
                continue
            }
            guard !text.isEmpty else { continue }
            var values = attributes(
                inlineIntent: item.inlineIntent,
                link: item.link ?? item.imageURL,
                html: html.style,
                block: block)
            if let cellStyle { values[.paragraphStyle] = cellStyle }
            output.append(NSAttributedString(string: text, attributes: values))
        }
        // A markdown cell is one paragraph terminated by a newline, so math
        // sentinels, inline HTML, and wrapping all behave as they do elsewhere.
        // An empty cell is that terminator and nothing else.
        guard let cellStyle else { return }
        var terminator = attributes(block: block)
        terminator[.paragraphStyle] = cellStyle
        output.append(NSAttributedString(string: "\n", attributes: terminator))
    }

    /// Inline HTML runs carry the tag text itself. A recognised tag becomes an
    /// attribute on the runs it encloses and its text is dropped; anything
    /// else is shown as written, because the parser reports `Vec<String>` and
    /// `x < y` as inline HTML too and deleting them lost the reader's words.
    private func appendInlineHTML(
        _ text: String,
        state: inout InlineHTMLState,
        block: Block,
        paragraphStyle: NSParagraphStyle? = nil,
        to output: NSMutableAttributedString
    ) {
        if let tokens = Self.tagTokens(text) {
            for token in tokens {
                guard let tag = InlineHTMLTag(rawValue: token.name) else {
                    var values = attributes(html: state.style, block: block)
                    if let paragraphStyle { values[.paragraphStyle] = paragraphStyle }
                    output.append(NSAttributedString(string: "\u{2028}", attributes: values))
                    continue
                }
                if token.isClosing {
                    state.close(tag)
                } else {
                    state.open(tag)
                }
            }
            return
        }
        guard !text.isEmpty else { return }
        var values = attributes(html: state.style, block: block)
        if let paragraphStyle { values[.paragraphStyle] = paragraphStyle }
        output.append(NSAttributedString(string: text, attributes: values))
    }

    private struct TableShape {
        var columns = 0
        var lastRow = 0
        /// Taken from the cells that do exist. A table's header row is always
        /// complete, so every column is covered.
        var alignments: [Int: NSTextAlignment] = [:]
    }

    /// Fills in the cells the parser never reported.
    ///
    /// Foundation emits no run for an empty cell, and none at all for the
    /// cells a short row never wrote, so no `NSTextTableBlock` was created for
    /// those positions: a three-column table came back with blocks at
    /// (0,0..2), (1,0), (1,1), (2,0), (2,2), (3,0), (3,1) and TextKit laid
    /// every body row out with the wrong column count. An all-empty last row
    /// still cannot be recovered — the parser reports nothing for it at all.
    private static func completingTables(_ items: [Item]) -> [Item] {
        var shapes: [Int: TableShape] = [:]
        for item in items {
            guard case .tableCell(let cell) = item.block.kind else { continue }
            var shape = shapes[cell.table] ?? TableShape()
            shape.columns = max(shape.columns, cell.columns, cell.column + 1)
            shape.lastRow = max(shape.lastRow, cell.row)
            shape.alignments[cell.column] = cell.alignment
            shapes[cell.table] = shape
        }
        guard !shapes.isEmpty else { return items }

        var completed: [Item] = []
        completed.reserveCapacity(items.count)
        var openTable: Int?
        var cursor = (row: 0, column: 0)
        var drawn: (row: Int, column: Int)?

        func fill(to target: (row: Int, column: Int)) {
            guard let table = openTable, let shape = shapes[table] else { return }
            while cursor.row < target.row
                || (cursor.row == target.row && cursor.column < target.column) {
                completed.append(Item(
                    text: "",
                    inlineIntent: nil,
                    link: nil,
                    imageURL: nil,
                    block: Block(identity: -1, kind: .tableCell(TableCell(
                        table: table,
                        columns: shape.columns,
                        row: cursor.row,
                        column: cursor.column,
                        isHeader: cursor.row == 0,
                        alignment: shape.alignments[cursor.column] ?? .natural)))))
                cursor = advanced(cursor, columns: shape.columns)
            }
        }

        func closeTable() {
            guard let table = openTable, let shape = shapes[table] else { return }
            fill(to: (row: shape.lastRow, column: shape.columns))
            openTable = nil
            drawn = nil
        }

        for item in items {
            guard case .tableCell(let cell) = item.block.kind else {
                closeTable()
                completed.append(item)
                continue
            }
            if openTable != cell.table {
                closeTable()
                openTable = cell.table
                cursor = (row: 0, column: 0)
            }
            if drawn?.row != cell.row || drawn?.column != cell.column {
                fill(to: (row: cell.row, column: cell.column))
                drawn = (row: cell.row, column: cell.column)
                cursor = advanced(cursor, columns: shapes[cell.table]?.columns ?? cell.columns)
            }
            completed.append(item)
        }
        closeTable()
        return completed
    }

    private static func advanced(
        _ position: (row: Int, column: Int),
        columns: Int
    ) -> (row: Int, column: Int) {
        position.column + 1 < max(columns, 1)
            ? (row: position.row, column: position.column + 1)
            : (row: position.row + 1, column: 0)
    }

    private func tableCellStyle(
        _ cell: TableCell,
        decoration: BlockDecoration
    ) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 0
        style.alignment = cell.alignment
        style.textBlocks = [decoration.tableCell(
            table: cell.table,
            columns: cell.columns,
            row: cell.row,
            column: cell.column,
            isHeader: cell.isHeader)]
        return style
    }

    private func appendCode(
        _ text: String,
        block: Block,
        to output: NSMutableAttributedString,
        decoration: BlockDecoration
    ) {
        var body = text
        if body.hasSuffix("\n") { body.removeLast() }
        let attributes = codeAttributes(cell: decoration.codeCell(for: block.identity))
        for line in body.components(separatedBy: "\n") {
            output.append(NSAttributedString(string: line + "\n", attributes: attributes))
        }
    }

    /// Attributes for a fenced block that is still arriving. The caller keeps
    /// the dictionary for the life of that block: the container is an
    /// `NSTextTableBlock`, and TextKit only joins consecutive paragraphs into
    /// one cell when they carry the same block object, so a fresh dictionary
    /// per tick would draw a new box per line.
    public func streamingCodeAttributes() -> [NSAttributedString.Key: Any] {
        codeAttributes(cell: BlockDecoration.codeCell())
    }

    private func codeAttributes(cell: NSTextTableBlock) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        style.paragraphSpacing = 0
        style.textBlocks = [cell]
        return [
            .font: font(for: .code, inlineIntent: nil, html: InlineHTMLStyle()),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static let taskMarkerLength = 4

    private static func taskMarker(_ text: String) -> String? {
        if text.hasPrefix("[ ] ") { return "\u{2610}" }
        if text.hasPrefix("[x] ") || text.hasPrefix("[X] ") { return "\u{2611}" }
        return nil
    }

    private func prefix(for block: BlockKind, taskGlyph: String?) -> String {
        switch block {
        case .unorderedList:
            return (taskGlyph ?? "\u{2022}") + "\t"
        case .orderedList(let ordinal, _):
            guard let taskGlyph else { return "\(ordinal).\t" }
            return "\(ordinal).\t\(taskGlyph) "
        case .quote:
            return "\u{2502}\t"
        default:
            return ""
        }
    }

    private func attributes(
        inlineIntent: InlinePresentationIntent? = nil,
        link: URL? = nil,
        html: InlineHTMLStyle = InlineHTMLStyle(),
        block: Block
    ) -> [NSAttributedString.Key: Any] {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: block.kind)
        let font = self.font(for: block.kind, inlineIntent: inlineIntent, html: html)
        values[.font] = font

        if block.kind == .quote {
            values[.foregroundColor] = NSColor.secondaryLabelColor
        }
        if inlineIntent?.contains(.code) == true || html.monospaced {
            values[.backgroundColor] = NSColor.quaternarySystemFill
        }
        if inlineIntent?.contains(.strikethrough) == true {
            values[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if html.scriptDepth > 0 {
            values[.baselineOffset] = font.pointSize * 0.34 * CGFloat(html.scriptDirection)
        }
        if let link {
            values[.link] = link
            values[.foregroundColor] = NSColor.linkColor
            values[.underlineStyle] = NSUnderlineStyle.single.rawValue
        }
        return values
    }

    private static func imageText(_ text: String, url: URL?) -> String? {
        guard let url, text.allSatisfy({ $0 == "\u{FFFC}" }) else { return nil }
        return url.absoluteString
    }

    /// The destination reaches the text view as a real `.link` now, so a scheme
    /// this transcript is not willing to open has to arrive as plain text
    /// instead. `javascript:` and `data:` are the reason; `file:` is the one a
    /// local model reaches for on its own.
    private static let openableSchemes: Set<String> = ["http", "https", "mailto"]

    private static func openable(_ url: URL?) -> URL? {
        guard let url, let scheme = url.scheme?.lowercased(),
              openableSchemes.contains(scheme) else {
            return nil
        }
        return url
    }

    private func baseAttributes() -> [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle(for: .paragraph),
        ]
    }

    private func font(
        for block: BlockKind,
        inlineIntent: InlinePresentationIntent?,
        html: InlineHTMLStyle
    ) -> NSFont {
        let bold = inlineIntent?.contains(.stronglyEmphasized) == true || html.bold
            || block.isHeaderCell
        let italic = inlineIntent?.contains(.emphasized) == true || html.italic
        let monospaced = block == .code
            || inlineIntent?.contains(.code) == true
            || html.monospaced

        var size: CGFloat
        var baseWeight: NSFont.Weight
        if monospaced {
            size = NSFont.systemFontSize - 0.5
            baseWeight = .regular
        } else if case .heading(let level) = block {
            size = max(NSFont.systemFontSize + 1, 22 - CGFloat(level - 1) * 2)
            baseWeight = .semibold
        } else {
            size = NSFont.systemFontSize
            baseWeight = .regular
        }
        if html.scriptDepth > 0 {
            size *= pow(0.72, CGFloat(min(html.scriptDepth, 2)))
        }

        let weight: NSFont.Weight = bold ? .semibold : baseWeight
        var font = monospaced
            ? NSFont.monospacedSystemFont(ofSize: size, weight: weight)
            : NSFont.systemFont(ofSize: size, weight: weight)
        if italic {
            let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
            font = NSFont(descriptor: descriptor, size: size) ?? font
        }
        return font
    }

    private func paragraphStyle(for block: BlockKind) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6

        switch block {
        case .heading:
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 4
        case .quote:
            style.firstLineHeadIndent = 4
            style.headIndent = 20
            style.tailIndent = -8
            style.tabStops = [NSTextTab(textAlignment: .left, location: 16)]
        case .unorderedList(let indent), .orderedList(_, let indent):
            let base = CGFloat(22 + indent * 18)
            style.firstLineHeadIndent = CGFloat(indent * 18)
            style.headIndent = base
            style.tabStops = [NSTextTab(textAlignment: .left, location: base)]
            style.paragraphSpacing = 2
        case .thematicBreak:
            style.alignment = .center
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 8
        // A table cell's style comes from `tableCellStyle`, which owns the
        // text block the cell is drawn in.
        case .paragraph, .code, .tableCell:
            break
        }
        return style
    }

    private func fallback(_ source: String) -> Result {
        Result(
            attributedString: NSAttributedString(
                string: source,
                attributes: baseAttributes()),
            usedFallback: true)
    }
}

private struct InlineHTMLStyle {
    var bold = false
    var italic = false
    var monospaced = false
    var scriptDepth = 0
    /// Superscripts minus subscripts, so nested pairs cancel their offset.
    var scriptDirection = 0
}

private enum InlineHTMLTag: String {
    case b, strong, i, em, code, kbd, sub, sup
}

private struct InlineHTMLState {
    private var stack: [InlineHTMLTag] = []

    mutating func open(_ tag: InlineHTMLTag) {
        stack.append(tag)
    }

    /// Closing an unbalanced tag drops everything opened after it rather than
    /// leaving a style that never ends.
    mutating func close(_ tag: InlineHTMLTag) {
        guard let index = stack.lastIndex(of: tag) else { return }
        stack.removeSubrange(index...)
    }

    var style: InlineHTMLStyle {
        var style = InlineHTMLStyle()
        for tag in stack {
            switch tag {
            case .b, .strong: style.bold = true
            case .i, .em: style.italic = true
            case .code, .kbd: style.monospaced = true
            case .sub:
                style.scriptDepth += 1
                style.scriptDirection -= 1
            case .sup:
                style.scriptDepth += 1
                style.scriptDirection += 1
            }
        }
        return style
    }
}

extension ResponseMarkdownRenderer {
    fileprivate struct HTMLTagToken {
        let name: String
        let isClosing: Bool
    }

    /// Tags this renderer can honour. `br` is the void element; the rest map
    /// onto `InlineHTMLTag`.
    fileprivate static let handledTags: Set<String> = [
        "b", "strong", "i", "em", "code", "kbd", "sub", "sup", "br",
    ]

    /// The whole run read as complete tags, or nil when any of it is the
    /// reader's own words.
    ///
    /// Reading the leading letters alone made `if a<b then c>d` an opening
    /// `<b>` tag — which it is, to CommonMark, with two bare attributes — so
    /// the run was dropped and everything after it went bold. Requiring every
    /// attribute to carry a value is the one tightening that separates markup
    /// from prose. A sequence is accepted because Foundation coalesces
    /// adjacent inline-HTML runs into one.
    fileprivate static func tagTokens(_ text: String) -> [HTMLTagToken]? {
        var rest = Substring(text)
        var tokens: [HTMLTagToken] = []
        while !rest.isEmpty {
            guard let token = tagToken(&rest) else { return nil }
            tokens.append(token)
        }
        return tokens.isEmpty ? nil : tokens
    }

    private static func tagToken(_ rest: inout Substring) -> HTMLTagToken? {
        var scan = rest
        guard scan.first == "<" else { return nil }
        scan = scan.dropFirst()
        let isClosing = scan.first == "/"
        if isClosing { scan = scan.dropFirst() }
        let name = scan.prefix { $0.isLetter || $0.isNumber }.lowercased()
        guard handledTags.contains(name) else { return nil }
        // A closing `</br>` is not a tag this renderer can honour; the reader
        // sees what the model wrote.
        guard !isClosing || name != "br" else { return nil }
        scan = scan.dropFirst(name.count)

        var selfClosing = false
        while true {
            let spaces = scan.prefix(while: isTagSpace)
            scan = scan.dropFirst(spaces.count)
            guard let next = scan.first else { return nil }
            if next == ">" {
                scan = scan.dropFirst()
                break
            }
            if next == "/" {
                scan = scan.dropFirst()
                guard scan.first == ">" else { return nil }
                scan = scan.dropFirst()
                selfClosing = true
                break
            }
            guard !isClosing, !spaces.isEmpty, consumeAttribute(&scan) else { return nil }
        }
        // `<br/>` is a void element. `<b/>` is a formatting tag that closes
        // nothing, so it is shown rather than silently dropped.
        guard !selfClosing || name == "br" else { return nil }
        rest = scan
        return HTMLTagToken(name: name, isClosing: isClosing)
    }

    private static func consumeAttribute(_ scan: inout Substring) -> Bool {
        let name = scan.prefix {
            $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == ":"
        }
        guard !name.isEmpty else { return false }
        var rest = scan.dropFirst(name.count).drop(while: isTagSpace)
        guard rest.first == "=" else { return false }
        rest = rest.dropFirst().drop(while: isTagSpace)
        guard let quote = rest.first else { return false }
        if quote == "\"" || quote == "'" {
            rest = rest.dropFirst()
            guard let end = rest.firstIndex(of: quote) else { return false }
            rest = rest[rest.index(after: end)...]
        } else {
            let value = rest.prefix { !isTagSpace($0) && $0 != ">" && $0 != "/" }
            guard !value.isEmpty else { return false }
            rest = rest.dropFirst(value.count)
        }
        scan = rest
        return true
    }

    private static func isTagSpace(_ character: Character) -> Bool {
        character == " " || character == "\t" || character == "\n" || character == "\r"
    }
}

/// Owns the `NSTextTable` instances a render pass creates. TextKit 1 groups
/// consecutive paragraphs into one cell only when they share the same
/// `NSTextBlock` object, so the table must outlive the individual paragraph
/// styles that reference it.
@MainActor
private final class BlockDecoration {
    private var codeCells: [Int: NSTextTableBlock] = [:]
    private var tables: [Int: NSTextTable] = [:]
    private var tableCells: [TableCellKey: NSTextTableBlock] = [:]

    struct TableCellKey: Hashable {
        let table: Int
        let row: Int
        let column: Int
    }

    static func codeCell() -> NSTextTableBlock {
        let table = NSTextTable()
        table.numberOfColumns = 1
        table.collapsesBorders = true
        let cell = NSTextTableBlock(
            table: table,
            startingRow: 0,
            rowSpan: 1,
            startingColumn: 0,
            columnSpan: 1)
        cell.setContentWidth(100, type: .percentageValueType)
        cell.setWidth(8, type: .absoluteValueType, for: .padding)
        cell.setWidth(1, type: .absoluteValueType, for: .border)
        cell.setBorderColor(.separatorColor)
        cell.backgroundColor = .quaternarySystemFill
        return cell
    }

    func codeCell(for identity: Int) -> NSTextTableBlock {
        if let existing = codeCells[identity] { return existing }
        let cell = Self.codeCell()
        codeCells[identity] = cell
        return cell
    }

    func tableCell(
        table identity: Int,
        columns: Int,
        row: Int,
        column: Int,
        isHeader: Bool
    ) -> NSTextTableBlock {
        let key = TableCellKey(table: identity, row: row, column: column)
        if let existing = tableCells[key] { return existing }
        let table = self.table(for: identity, columns: columns)
        let cell = NSTextTableBlock(
            table: table,
            startingRow: row,
            rowSpan: 1,
            startingColumn: column,
            columnSpan: 1)
        cell.setWidth(6, type: .absoluteValueType, for: .padding)
        cell.setWidth(1, type: .absoluteValueType, for: .border)
        cell.setBorderColor(.separatorColor)
        if isHeader { cell.backgroundColor = .quaternarySystemFill }
        tableCells[key] = cell
        return cell
    }

    private func table(for identity: Int, columns: Int) -> NSTextTable {
        if let existing = tables[identity] { return existing }
        let table = NSTextTable()
        table.numberOfColumns = columns
        table.collapsesBorders = true
        tables[identity] = table
        return table
    }
}
