import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// The grammar four scanners share. Every row runs through both entry points,
/// because the byte reference and the `Character` mirror disagreeing is the
/// class of bug this file exists to remove.
@Suite struct FenceLineTests {
    struct Row: Sendable, CustomStringConvertible {
        let line: String
        let containerIndent: Int
        let marker: Character?
        let length: Int
        let indent: Int
        let isBare: Bool

        init(
            _ line: String,
            containerIndent: Int = 0,
            marker: Character? = nil,
            length: Int = 0,
            indent: Int = 0,
            isBare: Bool = false
        ) {
            self.line = line
            self.containerIndent = containerIndent
            self.marker = marker
            self.length = length
            self.indent = indent
            self.isBare = isBare
        }

        var description: String { "\(line.debugDescription) at \(containerIndent)" }
    }

    static func byteRun(_ row: Row) -> FenceLine.Run? {
        var line = row.line
        return line.withUTF8 { bytes in
            FenceLine.run(bytes, in: 0..<bytes.count, containerIndent: row.containerIndent)
        }
    }

    static func characterRun(_ row: Row) -> FenceLine.Run? {
        FenceLine.run(row.line, containerIndent: row.containerIndent)
    }

    static let rows: [Row] = [
        Row("```", marker: "`", length: 3, isBare: true),
        Row("````", marker: "`", length: 4, isBare: true),
        Row("~~~", marker: "~", length: 3, isBare: true),
        Row("```swift", marker: "`", length: 3),
        Row("```  ", marker: "`", length: 3, isBare: true),
        Row("```\r", marker: "`", length: 3, isBare: true),
        Row("   ```", marker: "`", length: 3, indent: 3, isBare: true),
        // Four columns of indentation is indented code, not a fence.
        Row("    ```"),
        Row("\t```"),
        Row("``"),
        Row("~~"),
        Row(""),
        Row("   "),
        Row("plain text"),
        Row("x```"),
        // CommonMark: a backtick fence's info string may not contain a
        // backtick. A tilde fence's may.
        Row("```bash```"),
        Row("```a`b"),
        Row("~~~bash```", marker: "~", length: 3),
        // A container moves the whole 0...3 window with it.
        Row("    ```", containerIndent: 4, marker: "`", length: 3, indent: 4, isBare: true),
        Row("       ```", containerIndent: 4, marker: "`", length: 3, indent: 7, isBare: true),
        Row("        ```", containerIndent: 4),
        Row("  ```", containerIndent: 4),
    ]

    @Test(arguments: rows)
    func bothEntryPointsReadTheSameGrammar(_ row: Row) throws {
        let bytes = Self.byteRun(row)
        let characters = Self.characterRun(row)
        #expect(bytes == characters, "\(row) read differently by the two entry points")

        guard let marker = row.marker else {
            #expect(bytes == nil, "\(row) is not a fence line")
            return
        }
        let run = try #require(bytes, "\(row) is a fence line")
        #expect(run.character == marker)
        #expect(run.length == row.length)
        #expect(run.indent == row.indent)
        #expect(run.isBare == row.isBare)
    }

    @Test func onlyABareRunOfTheSameMarkerCloses() throws {
        let opener = try #require(FenceLine.run("```swift"))
        #expect(try #require(FenceLine.run("```")).closes(marker: opener.marker, length: 3))
        #expect(try #require(FenceLine.run("````")).closes(marker: opener.marker, length: 3))
        // Shorter, wrong marker, or carrying an info string: not a closer.
        #expect(!(try #require(FenceLine.run("~~~")).closes(marker: opener.marker, length: 3)))
        #expect(!(try #require(FenceLine.run("```text")).closes(marker: opener.marker, length: 3)))
        #expect(try #require(FenceLine.run("```")).closes(marker: opener.marker, length: 4) == false)
    }

    /// Multi-byte text before the marker must not shift the column count: the
    /// byte entry point measures indentation in spaces and tabs, not bytes.
    @Test func nonASCIIContentIsNotAFenceLine() {
        #expect(FenceLine.run("\u{4E2D}\u{6587}```") == nil)
        #expect(FenceLine.run("\u{2018}```") == nil)
    }
}
