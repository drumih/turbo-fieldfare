import AppKit
import Foundation
import SwiftMath

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

    private struct MathSpan {
        let latex: String
        let displayMode: Bool
    }

    private struct MathExtraction {
        let maskedSource: String
        let spans: [MathSpan]
    }

    private static let mathPlaceholder = "\u{FFFC}"

    private static let displayMathPattern = try! NSRegularExpression(
        pattern: #"\$\$([^\$]+?)\$\$"#,
        options: [.dotMatchesLineSeparators])

    private static let inlineMathPattern = try! NSRegularExpression(
        pattern: #"(?<!\$)\$(?!\$)(?!\s)([^\$\n]+?)(?<!\s)\$(?!\$)"#,
        options: [])

    public init() {}

    public func render(_ source: String) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        guard !requiresRawFallback(source) else { return fallback(source) }
        let mathExtraction = extractMathSpans(from: source)
        let presentationSource = mathExtraction.maskedSource.replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)

        do {
            let parsed = try AttributedString(
                markdown: presentationSource,
                options: .init(
                    interpretedSyntax: .full,
                    failurePolicy: .returnPartiallyParsedIfPossible))
            guard !containsUnsupportedBlock(in: parsed) else { return fallback(source) }

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

            guard output.length > 0 else { return fallback(source) }
            if !mathExtraction.spans.isEmpty {
                substituteMathPlaceholders(in: output, spans: mathExtraction.spans)
            }
            return Result(attributedString: output, usedFallback: false)
        } catch {
            return fallback(source)
        }
    }

    public func plainText(_ source: String) -> String {
        render(source).attributedString.string
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

    private func extractMathSpans(from source: String) -> MathExtraction {
        var spans: [MathSpan] = []
        var working = source as NSString

        func replaceMatches(
            _ regex: NSRegularExpression,
            displayMode: Bool,
            accept: (String) -> Bool = { _ in true }
        ) {
            let matches = regex.matches(
                in: working as String,
                range: NSRange(location: 0, length: working.length))
            guard !matches.isEmpty else { return }

            var result = ""
            var lastEnd = 0
            for match in matches {
                let contentRange = match.range(at: 1)
                guard contentRange.location != NSNotFound else { continue }
                let content = working.substring(with: contentRange)
                guard accept(content) else { continue }

                result += working.substring(with: NSRange(location: lastEnd, length: match.range.location - lastEnd))
                result += Self.mathPlaceholder
                spans.append(MathSpan(latex: content, displayMode: displayMode))
                lastEnd = match.range.location + match.range.length
            }
            result += working.substring(from: lastEnd)
            working = result as NSString
        }

        replaceMatches(Self.displayMathPattern, displayMode: true)
        replaceMatches(Self.inlineMathPattern, displayMode: false, accept: looksLikeMath)

        return MathExtraction(maskedSource: working as String, spans: spans)
    }

    private func looksLikeMath(_ content: String) -> Bool {
        let mathIndicators = CharacterSet(charactersIn: #"\^_{}=<>+"#)
        if content.rangeOfCharacter(from: mathIndicators) != nil { return true }
        if content.contains(" ") { return false }
        return content.rangeOfCharacter(from: .letters) != nil
    }

    private func substituteMathPlaceholders(
        in output: NSMutableAttributedString,
        spans: [MathSpan]
    ) {
        var searchStart = 0
        for span in spans {
            let searchRange = NSRange(location: searchStart, length: output.length - searchStart)
            let found = (output.string as NSString).range(
                of: Self.mathPlaceholder, options: [], range: searchRange)
            guard found.location != NSNotFound else { break }

            let replacement = renderMathAttachment(latex: span.latex, displayMode: span.displayMode)
            output.replaceCharacters(in: found, with: replacement)
            searchStart = found.location + replacement.length
        }
    }

    private func renderMathAttachment(latex: String, displayMode: Bool) -> NSAttributedString {
        let trimmed = latex.trimmingCharacters(in: .whitespacesAndNewlines)
        let rawFallback = displayMode ? "$$\(trimmed)$$" : "$\(trimmed)$"
        guard !trimmed.isEmpty else {
            return NSAttributedString(string: rawFallback, attributes: baseAttributes())
        }

        var mathImage = MathImage(
            latex: trimmed,
            fontSize: NSFont.systemFontSize + (displayMode ? 3 : 0),
            textColor: NSColor.labelColor,
            labelMode: displayMode ? .display : .text,
            textAlignment: .center)
        let (error, image, layout) = mathImage.asImage()
        guard error == nil, let image, let layout, image.size.width > 0, image.size.height > 0 else {
            return NSAttributedString(
                string: rawFallback,
                attributes: attributes(inlineIntent: .code, link: nil, block: .paragraph))
        }

        let attachment = NSTextAttachment()
        attachment.image = image
        attachment.bounds = CGRect(x: 0, y: -layout.descent, width: image.size.width, height: image.size.height)
        return NSAttributedString(attachment: attachment)
    }

    private func fallback(_ source: String) -> Result {
        Result(
            attributedString: NSAttributedString(
                string: source,
                attributes: baseAttributes()),
            usedFallback: true)
    }
}
