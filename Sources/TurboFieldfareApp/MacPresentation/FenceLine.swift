import Foundation

/// What a fenced-code fence line is, in one place.
///
/// Four scanners carried their own copy of this grammar and disagreed with
/// CommonMark and with each other. A backtick fence's info string may not
/// contain a backtick, so ```` ```bash``` ```` in a sentence is a code span:
/// the splitter, the bold-heading promotion, the tail auto-close and the code
/// mask all read it as an opener and boxed the rest of the answer as code,
/// while the renderer's raw-fallback gate counted delimiters instead of lines
/// and sent an answer with one mid-line ```` ``` ```` to raw text entirely.
enum FenceLine {
    static let backtick = UInt8(ascii: "`")
    static let tilde = UInt8(ascii: "~")

    struct Run: Equatable {
        /// ASCII `` ` `` or `~`.
        let marker: UInt8
        let length: Int
        /// The line's own indentation in columns, container included.
        let indent: Int
        /// Nothing after the marker run but spaces, which is what a closing
        /// fence is; an opener carries an info string.
        let isBare: Bool

        var character: Character { Character(UnicodeScalar(marker)) }

        func closes(marker: UInt8, length: Int) -> Bool {
            isBare && self.marker == marker && self.length >= length
        }
    }

    /// - Parameter containerIndent: the content column of the block the line
    ///   sits in — a list item's text column, zero at top level. CommonMark
    ///   allows a fence up to three columns past it, and a fence further left
    ///   than the column belongs to whatever encloses the container.
    static func run(
        _ bytes: UnsafeBufferPointer<UInt8>,
        in content: Range<Int>,
        containerIndent: Int = 0
    ) -> Run? {
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
        let marker = bytes[index]
        var length = 0
        while index < content.upperBound, bytes[index] == marker {
            length += 1
            index += 1
        }
        var bare = true
        var infoHasBacktick = false
        while index < content.upperBound {
            let byte = bytes[index]
            if byte != UInt8(ascii: " "), byte != UInt8(ascii: "\t"),
               byte != UInt8(ascii: "\r") {
                bare = false
                if byte == backtick { infoHasBacktick = true }
            }
            index += 1
        }
        return validated(
            marker: marker,
            length: length,
            indent: indent,
            containerIndent: containerIndent,
            isBare: bare,
            infoHasBacktick: infoHasBacktick)
    }

    static func run(
        _ characters: some Collection<Character>,
        containerIndent: Int = 0
    ) -> Run? {
        var indent = 0
        var rest = characters[...]
        while let first = rest.first, first == " " || first == "\t" {
            indent += first == "\t" ? 4 : 1
            rest = rest.dropFirst()
        }
        guard let first = rest.first, let marker = first.asciiValue else { return nil }
        let length = rest.prefix { $0 == first }.count
        var bare = true
        var infoHasBacktick = false
        for character in rest.dropFirst(length)
        where character != " " && character != "\t" && character != "\r" {
            bare = false
            if character == "`" { infoHasBacktick = true }
        }
        return validated(
            marker: marker,
            length: length,
            indent: indent,
            containerIndent: containerIndent,
            isBare: bare,
            infoHasBacktick: infoHasBacktick)
    }

    private static func validated(
        marker: UInt8,
        length: Int,
        indent: Int,
        containerIndent: Int,
        isBare: Bool,
        infoHasBacktick: Bool
    ) -> Run? {
        guard marker == backtick || marker == tilde, length >= 3 else { return nil }
        let relative = indent - containerIndent
        guard relative >= 0, relative <= 3 else { return nil }
        guard marker != backtick || !infoHasBacktick else { return nil }
        return Run(marker: marker, length: length, indent: indent, isBare: isBare)
    }
}
