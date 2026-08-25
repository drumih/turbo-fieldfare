import AppKit
import Foundation
@testable import TurboFieldfareMacPresentation

/// A comparable description of what an attributed string draws.
///
/// Two renders of the same markdown are never `isEqual` — an `NSTextTable`, an
/// `NSTextTableBlock`, and an `NSTextAttachment` all compare by object
/// identity, and every render pass makes its own. Runs are also cut in
/// different places depending on how a string was assembled, which changes
/// nothing on screen. The digest keeps the values and merges adjacent runs that
/// carry the same ones, so a comparison fails on a real difference and only on
/// a real difference.
struct AttributedStringDigest: Equatable, CustomStringConvertible {
    struct Run: Equatable {
        var text: String
        let font: String
        let foreground: String
        let background: String
        var paragraph: String
        let link: String
        let attachment: String
        let underline: Int
        let strikethrough: Int
        let baselineOffset: Double
    }

    let runs: [Run]

    var text: String { runs.map(\.text).joined() }

    /// One entry per character, so two renders can be compared position by
    /// position even when their runs are cut in different places.
    var perCharacter: [Run] {
        runs.flatMap { run in
            run.text.map { character in
                var single = run
                single.text = String(character)
                return single
            }
        }
    }

    var description: String {
        runs.map { run in
            "\(run.text.debugDescription) font=\(run.font) fg=\(run.foreground) "
                + "bg=\(run.background) para=\(run.paragraph) link=\(run.link) "
                + "att=\(run.attachment) u=\(run.underline) s=\(run.strikethrough) "
                + "base=\(run.baselineOffset)"
        }.joined(separator: "\n")
    }

    @MainActor
    init(_ attributed: NSAttributedString) {
        var runs: [Run] = []
        // Distinct container objects get distinct numbers: two tables that
        // should be separate digest the same as one merged table otherwise,
        // and that is exactly the bug a shared cached render would cause.
        var identities: [ObjectIdentifier: Int] = [:]
        let string = attributed.string as NSString
        attributed.enumerateAttributes(
            in: NSRange(location: 0, length: attributed.length),
            options: []) { attributes, range, _ in
            let run = Run(
                text: string.substring(with: range),
                font: Self.describe(attributes[.font] as? NSFont),
                foreground: Self.describe(attributes[.foregroundColor] as? NSColor),
                background: Self.describe(attributes[.backgroundColor] as? NSColor),
                paragraph: Self.describe(
                    attributes[.paragraphStyle] as? NSParagraphStyle,
                    identities: &identities),
                link: (attributes[.link] as? URL)?.absoluteString
                    ?? attributes[.link] as? String ?? "",
                attachment: Self.describe(attributes[.attachment] as? NSTextAttachment),
                underline: attributes[.underlineStyle] as? Int ?? 0,
                strikethrough: attributes[.strikethroughStyle] as? Int ?? 0,
                baselineOffset: (attributes[.baselineOffset] as? CGFloat).map(Double.init) ?? 0)
            if var last = runs.last, Self.sameAttributes(last, run) {
                last.text += run.text
                runs[runs.count - 1] = last
            } else {
                runs.append(run)
            }
        }
        self.runs = runs
    }

    private static func sameAttributes(_ lhs: Run, _ rhs: Run) -> Bool {
        var probe = lhs
        probe.text = rhs.text
        return probe == rhs
    }

    private static func describe(_ font: NSFont?) -> String {
        guard let font else { return "-" }
        let traits = font.fontDescriptor.symbolicTraits
        return "\(font.fontName)@\(String(format: "%.2f", font.pointSize))/\(traits.rawValue)"
    }

    private static func describe(_ color: NSColor?) -> String {
        color?.description ?? "-"
    }

    private static func describe(_ attachment: NSTextAttachment?) -> String {
        guard let attachment else { return "-" }
        guard let math = attachment as? MathAttachment else { return "attachment" }
        let bounds = attachment.bounds
        return "math(\(math.latexSource))@"
            + "\(String(format: "%.1fx%.1f", bounds.width, bounds.height))"
    }

    private static func describe(
        _ style: NSParagraphStyle?,
        identities: inout [ObjectIdentifier: Int]
    ) -> String {
        guard let style else { return "-" }
        var identifiers = identities
        let blocks = style.textBlocks
            .map { describe($0, identities: &identifiers) }
            .joined(separator: ",")
        identities = identifiers
        return [
            "align=\(style.alignment.rawValue)",
            "line=\(style.lineSpacing)",
            "space=\(style.paragraphSpacing)",
            "before=\(style.paragraphSpacingBefore)",
            "first=\(style.firstLineHeadIndent)",
            "head=\(style.headIndent)",
            "tail=\(style.tailIndent)",
            "tabs=\(style.tabStops.map(\.location))",
            "blocks=[\(blocks)]",
        ].joined(separator: " ")
    }

    private static func describe(
        _ block: NSTextBlock,
        identities: inout [ObjectIdentifier: Int]
    ) -> String {
        let identity = identities[ObjectIdentifier(block)]
            ?? { let next = identities.count; identities[ObjectIdentifier(block)] = next; return next }()
        var description = "block#\(identity)(bg=\(block.backgroundColor?.description ?? "-")"
        description += " pad=\(block.width(for: .padding, edge: .minX))"
        description += " border=\(block.width(for: .border, edge: .minX))"
        if let cell = block as? NSTextTableBlock {
            description += " cell=\(cell.startingRow),\(cell.startingColumn)"
            description += " cols=\(cell.table.numberOfColumns)"
        }
        return description + ")"
    }
}
