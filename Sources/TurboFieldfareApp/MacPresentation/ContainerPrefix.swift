import Foundation

/// The two container prefixes a line can wear, read the way CommonMark reads
/// them. A list item's content column decides what belongs to the item and what
/// interrupts it, and both the splitter and the tail auto-close need the same
/// answer: a fence at the column is the item's own listing, one before it ends
/// the list.
enum ContainerPrefix {
    static func markdownLines(_ source: String) -> [Substring] {
        // CommonMark line endings exclude Unicode separators such as U+2028,
        // which remain literal characters in code even though isNewline is true.
        source.split(omittingEmptySubsequences: false) { $0 == "\n" || $0 == "\r" || $0 == "\r\n" }
    }

    struct ListItem: Equatable {
        /// Columns of indentation before the marker.
        let markerIndent: Int
        /// Column the item's content starts at. CommonMark 5.2: the marker's
        /// indent plus its width plus the spaces after it, except that a marker
        /// which ends the line or is followed by five or more spaces starts its
        /// content one column past itself.
        let contentColumn: Int
    }

    static func listItem(
        _ bytes: UnsafeBufferPointer<UInt8>,
        in content: Range<Int>
    ) -> ListItem? {
        var index = content.lowerBound
        var indent = 0
        while index < content.upperBound {
            if bytes[index] == UInt8(ascii: " ") {
                indent += 1
            } else if bytes[index] == UInt8(ascii: "\t") {
                indent += 4
            } else {
                break
            }
            index += 1
        }
        guard index < content.upperBound else { return nil }
        var width = 0
        let first = bytes[index]
        if first == UInt8(ascii: "-") || first == UInt8(ascii: "*")
            || first == UInt8(ascii: "+") {
            width = 1
        } else {
            var digits = 0
            var cursor = index
            while cursor < content.upperBound, bytes[cursor] >= UInt8(ascii: "0"),
                  bytes[cursor] <= UInt8(ascii: "9") {
                digits += 1
                cursor += 1
            }
            guard digits > 0, digits <= 9, cursor < content.upperBound,
                  bytes[cursor] == UInt8(ascii: ".") || bytes[cursor] == UInt8(ascii: ")") else {
                return nil
            }
            width = digits + 1
        }
        var spaces = 0
        var cursor = index + width
        let markerEnd = cursor
        while cursor < content.upperBound {
            if bytes[cursor] == UInt8(ascii: " ") {
                spaces += 1
            } else if bytes[cursor] == UInt8(ascii: "\t") {
                spaces += 4
            } else {
                break
            }
            cursor += 1
        }
        guard markerEnd >= content.upperBound || spaces > 0 else { return nil }
        return item(indent: indent, width: width, spaces: spaces, endsLine: markerEnd >= content.upperBound)
    }

    static func listItem(_ line: some Collection<Character>) -> ListItem? {
        var rest = line[...]
        var indent = 0
        while let first = rest.first, first == " " || first == "\t" {
            indent += first == "\t" ? 4 : 1
            rest = rest.dropFirst()
        }
        guard let first = rest.first else { return nil }
        var width = 0
        if first == "-" || first == "*" || first == "+" {
            width = 1
        } else {
            let digits = rest.prefix { $0.isASCII && $0.isNumber }
            let after = rest.dropFirst(digits.count).first
            guard !digits.isEmpty, digits.count <= 9, after == "." || after == ")" else {
                return nil
            }
            width = digits.count + 1
        }
        let afterMarker = rest.dropFirst(width)
        var spaces = 0
        for character in afterMarker {
            guard character == " " || character == "\t" else { break }
            spaces += character == "\t" ? 4 : 1
        }
        guard afterMarker.isEmpty || spaces > 0 else { return nil }
        return item(indent: indent, width: width, spaces: spaces, endsLine: afterMarker.isEmpty)
    }

    private static func item(
        indent: Int,
        width: Int,
        spaces: Int,
        endsLine: Bool
    ) -> ListItem {
        let column = endsLine || spaces >= 5 ? indent + width + 1 : indent + width + spaces
        return ListItem(markerIndent: indent, contentColumn: column)
    }

    /// The line with its blockquote markers removed, and how many there were.
    /// A closer written back into the tail has to carry them again or the
    /// parser reads it as prose after the quote.
    static func strippingQuoteMarkers(_ line: Substring, limit: Int = .max) -> (depth: Int, rest: Substring) {
        var rest = line
        var depth = 0
        var column = 0
        while depth < limit {
            var probe = rest
            var probeColumn = column
            while let first = probe.first, first == " " || first == "\t", probeColumn - column < 3 {
                probeColumn += first == "\t" ? 4 - probeColumn % 4 : 1
                probe = probe.dropFirst()
            }
            guard probeColumn - column <= 3, probe.first == ">" else { return (depth, rest) }
            column = probeColumn + 1
            probe = probe.dropFirst()
            if probe.first == " " {
                probe = probe.dropFirst()
                column += 1
            }
            depth += 1
            rest = probe
        }
        return (depth, rest)
    }
}
