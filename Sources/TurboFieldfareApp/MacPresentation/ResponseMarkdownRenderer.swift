import AppKit
import Foundation

@MainActor
public struct ResponseMarkdownRenderer {
    public struct Result {
        public let attributedString: NSAttributedString
        public let usedFallback: Bool

        public init(attributedString: NSAttributedString, usedFallback: Bool) {
            self.attributedString = attributedString
            self.usedFallback = usedFallback
        }
    }

    private enum BlockKind: Equatable {
        case paragraph
        case heading(Int)
        case code(language: String?)
        case quote
        case unorderedList(indent: Int)
        case orderedList(ordinal: Int, indent: Int)
        case thematicBreak

        var isList: Bool {
            switch self {
            case .unorderedList, .orderedList: true
            default: false
            }
        }

        var isCode: Bool {
            if case .code = self { true } else { false }
        }
    }

    private struct Block: Equatable {
        let identity: Int
        let kind: BlockKind
    }

    public init() {}

    /// Render Markdown to an attributed string.
    ///
    /// Pass `streaming: true` while a response is still arriving. In that mode an
    /// unterminated trailing code fence is treated as closed for display, so an
    /// in-progress code block renders as a code block instead of dragging the
    /// whole message into the raw-text fallback on every token. On the final,
    /// terminal render use the default strict mode so a genuinely unclosed
    /// fence stays readable as exact raw text.
    public func render(_ source: String, streaming: Bool = false) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        let normalized = streaming ? closingOpenCodeFence(source) : source
        guard !requiresRawFallback(normalized) else { return fallback(source) }
        let presentationSource = normalized.replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)

        do {
            let parsed = try AttributedString(
                markdown: presentationSource,
                options: .init(
                    interpretedSyntax: .full,
                    failurePolicy: .returnPartiallyParsedIfPossible))

            let output = NSMutableAttributedString()
            var previousBlock: Block?
            let runs = Array(parsed.runs)
            var index = 0

            while index < runs.count {
                // A table is a group of consecutive cell runs sharing one table
                // identity; render the whole group as an NSTextTable at once.
                if let info = tableCellInfo(for: runs[index].presentationIntent) {
                    let next = appendTable(
                        from: runs, at: index, parsed: parsed, info: info,
                        previous: previousBlock, to: output)
                    previousBlock = Block(identity: info.identity, kind: .paragraph)
                    index = next
                    continue
                }

                let run = runs[index]
                let block = block(for: run.presentationIntent)
                if block != previousBlock {
                    appendSeparator(to: output, previous: previousBlock, next: block)
                    appendPrefix(for: block.kind, to: output)
                    if block.kind == .thematicBreak {
                        output.append(NSAttributedString(
                            string: "────────────────",
                            attributes: attributes(
                                inlineIntent: nil,
                                link: nil,
                                block: block.kind)))
                    }
                    previousBlock = block
                }

                index += 1
                guard block.kind != .thematicBreak else { continue }
                let text = String(parsed[run.range].characters)
                guard !text.isEmpty else { continue }
                output.append(NSAttributedString(
                    string: text,
                    attributes: attributes(
                        inlineIntent: run.inlinePresentationIntent,
                        link: run.link,
                        block: block.kind)))
            }

            guard output.length > 0 else { return fallback(source) }
            applySyntaxHighlighting(to: output)
            return Result(attributedString: output, usedFallback: false)
        } catch {
            return fallback(source)
        }
    }

    /// Color the token ranges of every fenced code block. The block's language
    /// hint is stored as the value of `codeBlockAttribute` by `attributes(...)`.
    private func applySyntaxHighlighting(to output: NSMutableAttributedString) {
        let highlighter = CodeSyntaxHighlighter()
        output.enumerateAttribute(
            TranscriptCodeStyle.codeBlockAttribute,
            in: NSRange(location: 0, length: output.length),
            options: []
        ) { value, range, _ in
            guard let language = value as? String else { return }
            highlighter.highlight(output, range: range, language: language)
        }
    }

    public func plainText(_ source: String) -> String {
        render(source).attributedString.string
    }

    /// When a response ends inside an open ``` fence, append a synthetic closing
    /// fence so the in-progress block parses as a code block. Leaves already
    /// balanced sources untouched. Used only for streaming renders.
    private func closingOpenCodeFence(_ source: String) -> String {
        let fenceCount = source.components(separatedBy: "```").count - 1
        guard !fenceCount.isMultiple(of: 2) else { return source }
        let separator = source.hasSuffix("\n") ? "" : "\n"
        return source + separator + "```"
    }

    private func requiresRawFallback(_ source: String) -> Bool {
        let fenceCount = source.components(separatedBy: "```").count - 1
        if !fenceCount.isMultiple(of: 2) { return true }
        // Check for raw HTML / images only in prose. Code routinely contains
        // angle brackets (generics like `Array<Int>`, `#include <stdio.h>`,
        // `2..<n`) that would otherwise look like HTML tags and drag the whole
        // message into the raw-text fallback.
        let prose = strippingCodeRegions(source)
        if prose.range(
            of: #"</?[A-Za-z][^>]*>"#,
            options: .regularExpression) != nil {
            return true
        }
        return prose.range(
            of: #"!\[[^\]]*\]\([^\)]*\)"#,
            options: .regularExpression) != nil
    }

    /// Replace fenced code blocks and inline code spans with spaces so the raw
    /// HTML / image heuristics never inspect code content.
    private func strippingCodeRegions(_ source: String) -> String {
        source
            .replacingOccurrences(
                of: #"(?s)```.*?```"#, with: " ", options: .regularExpression)
            .replacingOccurrences(
                of: #"`[^`\n]*`"#, with: " ", options: .regularExpression)
    }

    private struct TableCellInfo {
        let column: Int
        let row: Int
        let isHeader: Bool
        let columns: Int
        let identity: Int
        let alignments: [NSTextAlignment]
    }

    /// Table geometry for a cell run, or nil if the run is not part of a table.
    /// Header cells map to row 0; body cells keep their `tableRow` index (which
    /// the parser starts at 1), so a header + two rows becomes grid rows 0, 1, 2.
    private func tableCellInfo(for intent: PresentationIntent?) -> TableCellInfo? {
        guard let components = intent?.components else { return nil }
        var column: Int?
        var row = 0
        var isHeader = false
        var columns = 0
        var identity = 0
        var alignments: [NSTextAlignment] = []
        for component in components {
            switch component.kind {
            case .tableCell(let columnIndex): column = columnIndex
            case .tableHeaderRow: isHeader = true
            case .tableRow(let rowIndex): row = rowIndex
            case .table(let tableColumns):
                columns = tableColumns.count
                identity = component.identity
                alignments = tableColumns.map { column in
                    switch column.alignment {
                    case .left: .left
                    case .center: .center
                    case .right: .right
                    @unknown default: .left
                    }
                }
            default: break
            }
        }
        guard let column else { return nil }
        return TableCellInfo(
            column: column, row: isHeader ? 0 : row, isHeader: isHeader,
            columns: columns, identity: identity, alignments: alignments)
    }

    /// Consume every cell run of the table starting at `start` and append one
    /// NSTextTable to `output`. Returns the index of the first run after it.
    private func appendTable(
        from runs: [AttributedString.Runs.Run],
        at start: Int,
        parsed: AttributedString,
        info: TableCellInfo,
        previous: Block?,
        to output: NSMutableAttributedString
    ) -> Int {
        var cells: [Int: [Int: NSMutableAttributedString]] = [:]
        var headerRows: Set<Int> = []
        var columns = info.columns
        var alignments = info.alignments

        var index = start
        while index < runs.count,
              let cell = tableCellInfo(for: runs[index].presentationIntent),
              cell.identity == info.identity {
            let text = String(parsed[runs[index].range].characters)
            let attributes = cellInlineAttributes(
                inlineIntent: runs[index].inlinePresentationIntent,
                link: runs[index].link,
                isHeader: cell.isHeader)
            if cells[cell.row]?[cell.column] == nil {
                cells[cell.row, default: [:]][cell.column] = NSMutableAttributedString()
            }
            cells[cell.row]![cell.column]!.append(
                NSAttributedString(string: text, attributes: attributes))
            if cell.isHeader { headerRows.insert(cell.row) }
            columns = max(columns, cell.column + 1)
            if !cell.alignments.isEmpty { alignments = cell.alignments }
            index += 1
        }

        appendSeparator(
            to: output, previous: previous,
            next: Block(identity: info.identity, kind: .paragraph))

        let table = NSTextTable()
        table.numberOfColumns = columns
        table.setContentWidth(100, type: .percentageValueType)
        let baseAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize),
            .foregroundColor: NSColor.labelColor,
        ]

        for (gridRow, rowKey) in cells.keys.sorted().enumerated() {
            let isHeader = headerRows.contains(rowKey)
            for column in 0..<columns {
                let content = cells[rowKey]?[column].map {
                    NSMutableAttributedString(attributedString: $0)
                } ?? NSMutableAttributedString()
                if content.length == 0 {
                    content.append(NSAttributedString(string: " ", attributes: baseAttributes))
                }

                let block = NSTextTableBlock(
                    table: table, startingRow: gridRow, rowSpan: 1,
                    startingColumn: column, columnSpan: 1)
                block.setWidth(1, type: .absoluteValueType, for: .border)
                block.setWidth(7, type: .absoluteValueType, for: .padding)
                block.setBorderColor(TranscriptCodeStyle.tableBorder)
                if isHeader { block.backgroundColor = TranscriptCodeStyle.tableHeaderBackground }

                let paragraph = NSMutableParagraphStyle()
                paragraph.textBlocks = [block]
                paragraph.alignment = column < alignments.count ? alignments[column] : .left

                content.addAttribute(
                    .paragraphStyle, value: paragraph,
                    range: NSRange(location: 0, length: content.length))
                output.append(content)
                output.append(NSAttributedString(
                    string: "\n",
                    attributes: baseAttributes.merging(
                        [.paragraphStyle: paragraph]) { _, new in new }))
            }
        }
        return index
    }

    private func cellInlineAttributes(
        inlineIntent: InlinePresentationIntent?,
        link: URL?,
        isHeader: Bool
    ) -> [NSAttributedString.Key: Any] {
        let size = NSFont.systemFontSize
        var values: [NSAttributedString.Key: Any] = [.foregroundColor: NSColor.labelColor]
        let bold = isHeader || inlineIntent?.contains(.stronglyEmphasized) == true

        if inlineIntent?.contains(.code) == true {
            values[.font] = NSFont.monospacedSystemFont(ofSize: size - 0.5, weight: .regular)
            values[.foregroundColor] = TranscriptCodeStyle.codeText
            values[.backgroundColor] = TranscriptCodeStyle.inlineCodeBackground
        } else {
            var font = NSFont.systemFont(ofSize: size, weight: bold ? .semibold : .regular)
            if inlineIntent?.contains(.emphasized) == true {
                let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
                font = NSFont(descriptor: descriptor, size: size) ?? font
            }
            values[.font] = font
        }
        if inlineIntent?.contains(.strikethrough) == true {
            values[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if link != nil {
            values[.foregroundColor] = NSColor.linkColor
            values[.underlineStyle] = NSUnderlineStyle.single.rawValue
        }
        return values
    }

    private func block(for intent: PresentationIntent?) -> Block {
        guard let components = intent?.components, let leaf = components.first else {
            return Block(identity: 0, kind: .paragraph)
        }

        var headingLevel: Int?
        var code = false
        var codeLanguage: String?
        var quote = false
        var thematicBreak = false
        var ordinal: Int?
        var ordered = false
        var unordered = false
        var listDepth = 0

        for component in components {
            switch component.kind {
            case .header(let level): headingLevel = level
            case .codeBlock(let languageHint):
                code = true
                codeLanguage = languageHint
            case .blockQuote: quote = true
            case .thematicBreak: thematicBreak = true
            case .listItem(let itemOrdinal): ordinal = itemOrdinal
            case .orderedList:
                ordered = true
                listDepth += 1
            case .unorderedList:
                unordered = true
                listDepth += 1
            default: break
            }
        }

        let kind: BlockKind
        if thematicBreak {
            kind = .thematicBreak
        } else if let headingLevel {
            kind = .heading(headingLevel)
        } else if code {
            kind = .code(language: codeLanguage)
        } else if ordered, let ordinal {
            kind = .orderedList(ordinal: ordinal, indent: max(0, listDepth - 1))
        } else if unordered {
            kind = .unorderedList(indent: max(0, listDepth - 1))
        } else if quote {
            kind = .quote
        } else {
            kind = .paragraph
        }
        return Block(identity: leaf.identity, kind: kind)
    }

    private func appendSeparator(
        to output: NSMutableAttributedString,
        previous: Block?,
        next: Block
    ) {
        guard let previous else { return }
        // Lines inside one code block, and adjacent list items, join with a
        // single newline; every other block boundary gets a blank line.
        let tight = (previous.kind.isList && next.kind.isList)
            || (previous.kind.isCode && next.kind.isCode)
        let requiredNewlines = tight ? 1 : 2
        let trailingNewlines = output.string.reversed().prefix { $0 == "\n" }.count
        guard trailingNewlines < requiredNewlines else { return }
        output.append(NSAttributedString(
            string: String(repeating: "\n", count: requiredNewlines - trailingNewlines),
            attributes: baseAttributes()))
    }

    private func appendPrefix(for block: BlockKind, to output: NSMutableAttributedString) {
        let prefix: String
        switch block {
        case .unorderedList:
            prefix = "•\t"
        case .orderedList(let ordinal, _):
            prefix = "\(ordinal).\t"
        case .quote:
            prefix = "│\t"
        default:
            return
        }
        output.append(NSAttributedString(
            string: prefix,
            attributes: attributes(inlineIntent: nil, link: nil, block: block)))
    }

    private func attributes(
        inlineIntent: InlinePresentationIntent?,
        link: URL?,
        block: BlockKind
    ) -> [NSAttributedString.Key: Any] {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: block)
        values[.font] = font(for: block, inlineIntent: inlineIntent)

        if case .code(let language) = block {
            values[.foregroundColor] = TranscriptCodeStyle.codeText
            values[TranscriptCodeStyle.codeBlockAttribute] = language ?? ""
        } else if block == .quote {
            values[.foregroundColor] = NSColor.secondaryLabelColor
        } else if inlineIntent?.contains(.code) == true {
            values[.foregroundColor] = TranscriptCodeStyle.codeText
            values[.backgroundColor] = TranscriptCodeStyle.inlineCodeBackground
        }
        if inlineIntent?.contains(.strikethrough) == true {
            values[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if link != nil {
            values[.foregroundColor] = NSColor.linkColor
            values[.underlineStyle] = NSUnderlineStyle.single.rawValue
        }
        return values
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
        inlineIntent: InlinePresentationIntent?
    ) -> NSFont {
        let isCodeBlock: Bool = if case .code = block { true } else { false }
        if isCodeBlock || inlineIntent?.contains(.code) == true {
            return NSFont.monospacedSystemFont(
                ofSize: NSFont.systemFontSize - 0.5,
                weight: .regular)
        }

        let size: CGFloat
        let baseWeight: NSFont.Weight
        switch block {
        case .heading(let level):
            size = max(NSFont.systemFontSize + 1, 22 - CGFloat(level - 1) * 2)
            baseWeight = .semibold
        default:
            size = NSFont.systemFontSize
            baseWeight = .regular
        }

        let stronglyEmphasized = inlineIntent?.contains(.stronglyEmphasized) == true
        let emphasized = inlineIntent?.contains(.emphasized) == true
        var font = NSFont.systemFont(
            ofSize: size,
            weight: stronglyEmphasized ? .semibold : baseWeight)
        if emphasized {
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
        case .code:
            style.firstLineHeadIndent = 12
            style.headIndent = 12
            style.tailIndent = -12
            style.paragraphSpacingBefore = 0
            style.paragraphSpacing = 2
            style.lineSpacing = 3
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
        case .paragraph:
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
