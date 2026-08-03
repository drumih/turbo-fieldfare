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
        case code
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
    }

    private struct Block: Equatable {
        let identity: Int
        let kind: BlockKind
    }

    public init() {}

    public func render(_ source: String) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        let normalizedSource = normalizeTables(in: source)
        guard !requiresRawFallback(normalizedSource) else {
            return fallback(normalizedSource)
        }
        let presentationSource = normalizedSource.replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)

        do {
            let parsed = try AttributedString(
                markdown: presentationSource,
                options: .init(
                    interpretedSyntax: .full,
                    failurePolicy: .returnPartiallyParsedIfPossible))
            guard !containsUnsupportedBlock(in: parsed) else {
                return fallback(normalizedSource)
            }

            let output = NSMutableAttributedString()
            var previousBlock: Block?

            for run in parsed.runs {
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

            guard output.length > 0 else { return fallback(normalizedSource) }
            // A partial streamed emphasis marker is not useful to display;
            // keep the text readable until the closing marker arrives.
            if output.string.contains("**")
                || output.string.contains("__")
                || output.string.contains("~~") {
                return fallback(normalizedSource)
            }
            return Result(attributedString: output, usedFallback: false)
        } catch {
            return fallback(normalizedSource)
        }
    }

    public func plainText(_ source: String) -> String {
        render(source).attributedString.string
    }

    /// A lightweight, marker-free representation for a response that is
    /// still arriving. It deliberately avoids reparsing and reflowing the
    /// whole growing Markdown document on every transport snapshot; full
    /// Markdown rendering happens when the response is terminal.
    public func streamingText(_ source: String) -> String {
        fallback(normalizeTables(in: source)).attributedString.string
    }

    private func requiresRawFallback(_ source: String) -> Bool {
        let fenceCount = source.components(separatedBy: "```").count - 1
        if !fenceCount.isMultiple(of: 2) { return true }
        if source.range(
            of: #"</?[A-Za-z][^>]*>"#,
            options: .regularExpression) != nil {
            return true
        }
        return source.range(
            of: #"!\[[^\]]*\]\([^\)]*\)"#,
            options: .regularExpression) != nil
    }

    private func containsUnsupportedBlock(in parsed: AttributedString) -> Bool {
        parsed.runs.contains { run in
            run.presentationIntent?.components.contains { component in
                switch component.kind {
                case .table, .tableHeaderRow, .tableRow, .tableCell:
                    return true
                default:
                    return false
                }
            } == true
        }
    }

    /// Foundation's Markdown parser intentionally does not interpret tables.
    /// Convert GitHub-style tables to readable labelled rows before parsing so
    /// one table cannot force the entire response down the raw-text path.
    private func normalizeTables(in source: String) -> String {
        let lines = source.components(separatedBy: .newlines)
        var normalized: [String] = []
        var index = 0
        var insideFence = false

        while index < lines.count {
            let line = lines[index]
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                insideFence.toggle()
                normalized.append(line)
                index += 1
                continue
            }

            guard !insideFence,
                  index + 1 < lines.count,
                  let headers = tableCells(from: line),
                  let separator = tableCells(from: lines[index + 1]),
                  isTableSeparator(separator) else {
                normalized.append(line)
                index += 1
                continue
            }

            var rows: [[String]] = []
            var next = index + 2
            while next < lines.count,
                  let cells = tableCells(from: lines[next]),
                  !cells.isEmpty {
                rows.append(cells)
                next += 1
            }

            if rows.isEmpty {
                normalized.append(headers.map(cleanTableLabel).joined(separator: "  "))
            } else {
                for (rowIndex, row) in rows.enumerated() {
                    if rowIndex > 0 { normalized.append("") }
                    for column in 0..<min(headers.count, row.count) {
                        let label = cleanTableLabel(headers[column])
                        let value = row[column].trimmingCharacters(in: .whitespaces)
                        guard !label.isEmpty, !value.isEmpty else { continue }
                        normalized.append("**\(label):** \(value)")
                    }
                }
            }
            index = next
        }

        return normalized.joined(separator: "\n")
    }

    private func tableCells(from line: String) -> [String]? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("|") else { return nil }

        var cells: [String] = []
        var current = ""
        var escaped = false
        for character in trimmed {
            if character == "|" && !escaped {
                cells.append(current)
                current = ""
            } else {
                current.append(character)
            }
            escaped = character == "\\" && !escaped
            if character != "\\" { escaped = false }
        }
        cells.append(current)

        if trimmed.first == "|" { cells.removeFirst() }
        if trimmed.last == "|", !cells.isEmpty { cells.removeLast() }
        return cells.map {
            $0.replacingOccurrences(of: #"\\\|"#, with: "|",
                                    options: .regularExpression)
                .trimmingCharacters(in: .whitespaces)
        }
    }

    private func isTableSeparator(_ cells: [String]) -> Bool {
        !cells.isEmpty && cells.allSatisfy {
            $0.range(of: #"^:?-{3,}:?$"#, options: .regularExpression) != nil
        }
    }

    private func cleanTableLabel(_ label: String) -> String {
        label
            .replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "__", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func block(for intent: PresentationIntent?) -> Block {
        guard let components = intent?.components, let leaf = components.first else {
            return Block(identity: 0, kind: .paragraph)
        }

        var headingLevel: Int?
        var code = false
        var quote = false
        var thematicBreak = false
        var ordinal: Int?
        var ordered = false
        var unordered = false
        var listDepth = 0

        for component in components {
            switch component.kind {
            case .header(let level): headingLevel = level
            case .codeBlock: code = true
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
            kind = .code
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
        let adjacentListItems = previous.kind.isList && next.kind.isList
        let requiredNewlines = adjacentListItems ? 1 : 2
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

        if block == .quote {
            values[.foregroundColor] = NSColor.secondaryLabelColor
        }
        if block == .code || inlineIntent?.contains(.code) == true {
            values[.backgroundColor] = NSColor.controlBackgroundColor
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
        if block == .code || inlineIntent?.contains(.code) == true {
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
            style.firstLineHeadIndent = 10
            style.headIndent = 10
            style.tailIndent = -10
            style.paragraphSpacingBefore = 6
            style.paragraphSpacing = 6
            style.lineSpacing = 2
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
        let readable = source
            .replacingOccurrences(
                of: #"```[A-Za-z0-9_+.-]*\n?"#,
                with: "",
                options: .regularExpression)
            .replacingOccurrences(
                of: #"!\[([^\]]*)\]\([^\)]*\)"#,
                with: "$1",
                options: .regularExpression)
            .replacingOccurrences(
                of: #"\[([^\]]+)\]\([^\)]*\)"#,
                with: "$1",
                options: .regularExpression)
            .replacingOccurrences(
                of: #"(?m)^[ \t]{0,3}#{1,6}[ \t]+"#,
                with: "",
                options: .regularExpression)
            .replacingOccurrences(
                of: #"(?m)^([ \t]*)[-+*][ \t]+"#,
                with: "$1• ",
                options: .regularExpression)
            .replacingOccurrences(
                of: #"(?m)^[ \t]*>[ \t]?"#,
                with: "",
                options: .regularExpression)
            .replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "__", with: "")
            .replacingOccurrences(of: "~~", with: "")
            .replacingOccurrences(of: "`", with: "")
        return Result(
            attributedString: NSAttributedString(
                string: readable,
                attributes: baseAttributes()),
            usedFallback: true)
    }
}
