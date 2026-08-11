import Testing
import Foundation
@testable import TurboFieldfare

/// `JSONByteAutomaton(dialect: .gemmaToolArguments)`: the argument language the
/// Gemma 4 chat template renders and `GemmaToolCallParser` reads. Every accept
/// case here must also parse; every reject case is either outside the parser's
/// language or inside it but decoded to something other than what was
/// generated.
@Suite struct GemmaArgumentDialectTests {

    private func feedBytes(_ automaton: inout JSONByteAutomaton, _ bytes: [UInt8]) -> Bool {
        for byte in bytes {
            guard automaton.consume(byte) else { return false }
        }
        return true
    }

    private func step(_ automaton: inout JSONByteAutomaton, _ byte: UInt8) -> Bool {
        automaton.consume(byte)
    }

    private func accepts(_ bytes: [UInt8]) -> JSONByteAutomaton? {
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        return feedBytes(&automaton, bytes) ? automaton : nil
    }

    private func completes(_ text: String) -> Bool {
        accepts(Array(text.utf8))?.isComplete == true
    }

    private func rejects(_ text: String) -> Bool {
        accepts(Array(text.utf8)) == nil
    }

    // MARK: - Structure

    @Test func emptyObjectCompletes() {
        #expect(completes("{}"))
    }

    @Test func bareKeysAtEveryDepth() throws {
        let body = "{a:1,b:{c:[1,{d:true}],e:null},f:false}"
        #expect(completes(body))
        let call = try GemmaToolCallParser().parse(
            "call:f\(body)", allowedTools: ["f"], id: "x")
        #expect(call.arguments.objectValue?.count == 3)
    }

    @Test func exoticKeyAlphabetAccepted() {
        #expect(completes("{a_b-c.d$e:1}"))
        #expect(completes("{0:1}"))
        #expect(completes("{$:1}"))
        #expect(completes("{-:1}"))
    }

    @Test func whitespaceAtJunctionsAccepted() throws {
        let body = "{ a : 1 , b : [ 1 , 2 ] }"
        #expect(completes(body))
        let call = try GemmaToolCallParser().parse(
            "call:f\(body)", allowedTools: ["f"], id: "x")
        #expect(call.arguments == .object(["a": .integer(1),
                                           "b": .array([.integer(1), .integer(2)])]))
    }

    @Test func quotedKeyRejected() {
        // Failure cause #1 in the upstream report: the model drifts to
        // canonical JSON and the parser rejects the quoted key.
        #expect(rejects(#"{"a":1}"#))
    }

    @Test func quotedStringValueRejected() {
        // Accepted by the parser but silently whitespace-mangled by it.
        #expect(rejects(#"{a:"x"}"#))
    }

    @Test func arrayRootRejected() {
        #expect(rejects("[1]"))
        #expect(rejects("1"))
        #expect(rejects(#"<|"|>x<|"|>"#))
    }

    @Test func trailingCommaAndEmptyMemberRejected() {
        #expect(rejects("{a:1,}"))
        #expect(rejects("{,"))
        #expect(rejects("{:1}"))
    }

    @Test func depthIsBounded() {
        let opens = String(repeating: "[", count: JSONByteAutomaton.maxDepth)
        #expect(accepts(Array("{a:\(opens)".utf8)) == nil)
        let fits = String(repeating: "[", count: JSONByteAutomaton.maxDepth - 1)
        #expect(accepts(Array("{a:\(fits)".utf8)) != nil)
    }

    // MARK: - Numbers

    @Test func boundedNumbersAccepted() throws {
        let body = "{a:0.000000000000001,b:999999999999999.999999999999999,c:-0,d:0}"
        #expect(completes(body))
        let call = try GemmaToolCallParser().parse(
            "call:f\(body)", allowedTools: ["f"], id: "x")
        #expect(call.arguments.objectValue?["c"] == .integer(0))
    }

    @Test func exponentsAndMalformedNumbersRejected() {
        #expect(rejects("{a:1e5}"))
        #expect(rejects("{a:1E5}"))
        #expect(rejects("{a:1.5e-3}"))
        #expect(rejects("{a:01}"))
        #expect(rejects("{a:.5}"))
        #expect(rejects("{a:1.}"))
        #expect(rejects("{a:+1}"))
        #expect(rejects("{a:NaN}"))
    }

    @Test func digitCountIsCapped() {
        #expect(completes("{a:999999999999999}"))          // 15 integer digits
        #expect(rejects("{a:9999999999999999}"))           // 16
        #expect(completes("{a:0.999999999999999}"))        // 15 fraction digits
        #expect(rejects("{a:0.9999999999999999}"))         // 16
    }

    // MARK: - Strings

    @Test func gemmaStringsAcceptContentAndEscapes() throws {
        // Raw quotes, a raw tab and a raw newline are legal content; the escape
        // forms are the ones the parser's `escapedFragment` reads.
        let raw = "{a:<|\"|>he said \"hi\"\t\nx<|\"|>,"
        let escaped = #"b:<|"|>\"\\\/\b\f\n\r\t<|"|>,c:<|"|>\ud83d\ude00<|"|>}"#
        let body = raw + escaped
        #expect(completes(body))
        let call = try GemmaToolCallParser().parse(
            "call:f\(body)", allowedTools: ["f"], id: "x")
        #expect(call.arguments.objectValue?["a"] == .string("he said \"hi\"\t\nx"))
        #expect(call.arguments.objectValue?["b"] == .string("\"\\/\u{8}\u{c}\n\r\t"))
        #expect(call.arguments.objectValue?["c"] == .string("\u{1F600}"))
    }

    @Test func delimiterMaySpellOutOneByteAtATime() {
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        #expect(feedBytes(&automaton, Array("{a:".utf8)))
        for byte in Array(#"<|"|>"#.utf8) { #expect(step(&automaton, byte)) }
        #expect(step(&automaton, UInt8(ascii: "x")))
        for byte in Array(#"<|"|>"#.utf8) { #expect(step(&automaton, byte)) }
        #expect(step(&automaton, UInt8(ascii: "}")))
        #expect(automaton.isComplete)
    }

    @Test func multibyteScalarMaySplitAcrossBytes() {
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        #expect(feedBytes(&automaton, Array(#"{a:<|"|>"#.utf8)))
        for byte in Array("ñ😀".utf8) { #expect(step(&automaton, byte)) }
        // Mid-sequence the delimiter cannot start.
        #expect(feedBytes(&automaton, [0xF0]))
        #expect(!step(&automaton, UInt8(ascii: "<")))
        #expect(feedBytes(&automaton, [0x9F, 0x98, 0x81]))
        #expect(feedBytes(&automaton, Array(#"<|"|>}"#.utf8)))
        #expect(automaton.isComplete)
    }

    @Test func invalidUTF8InsideStringRejected() {
        for bad: [UInt8] in [[0xC0], [0xF5], [0x80], [0xE0, 0x80], [0xF4, 0x90], [0xED, 0xA0]] {
            var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
            #expect(feedBytes(&automaton, Array(#"{a:<|"|>"#.utf8)))
            #expect(!feedBytes(&automaton, bad))
        }
    }

    @Test func controlBytesInsideStringRejected() {
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        #expect(feedBytes(&automaton, Array(#"{a:<|"|>"#.utf8)))
        #expect(!step(&automaton, 0x07))
        for byte: UInt8 in [0x09, 0x0A, 0x0D] { #expect(step(&automaton, byte)) }
    }

    @Test func loneSurrogateEscapesRejected() {
        #expect(rejects(#"{a:<|"|>\ud83d<|"|>}"#))
        #expect(rejects(#"{a:<|"|>\udc00x<|"|>}"#))
        #expect(rejects(#"{a:<|"|>\ud83dx<|"|>}"#))
    }

    // MARK: - Delimiter scanning matches the parser's leftmost scan

    @Test func partialDelimitersStayContent() throws {
        for content in ["<|", #"<|"|"#, "<<", "|>", #""|>"#, #"<|"|<"#] {
            let body = #"{a:<|"|>"# + content + #"<|"|>}"#
            #expect(completes(body), "not accepted: \(content)")
            let call = try GemmaToolCallParser().parse(
                "call:f\(body)", allowedTools: ["f"], id: "x")
            #expect(call.arguments.objectValue?["a"] == .string(content))
        }
    }

    /// An embedded delimiter closes the string at the same byte the parser
    /// closes it: the pattern has no proper border, so resetting to zero and
    /// re-dispatching is the leftmost match.
    @Test func embeddedDelimiterClosesWhereTheParserCloses() throws {
        let body = #"{a:<|"|><|<|"|>}"#
        #expect(completes(body))
        let call = try GemmaToolCallParser().parse(
            "call:f\(body)", allowedTools: ["f"], id: "x")
        #expect(call.arguments.objectValue?["a"] == .string("<|"))
    }

    // MARK: - Grapheme-merge guard (Prepend scalars)

    @Test func prependScalarBlocksDelimiterAndEscape() {
        // U+0600 is Grapheme_Cluster_Break=Prepend: in the parser's
        // `[Character]` lexer it fuses with the next `<`, hiding the closing
        // delimiter, so the string never terminates.
        for next: UInt8 in [UInt8(ascii: "<"), UInt8(ascii: "\\")] {
            var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
            #expect(feedBytes(&automaton, Array(#"{a:<|"|>"#.utf8)))
            #expect(feedBytes(&automaton, [0xD8, 0x80]))
            #expect(!step(&automaton, next))
            // Any other content clears the guard.
            #expect(step(&automaton, UInt8(ascii: "x")))
            #expect(step(&automaton, next))
        }
    }

    @Test func prependGuardSurvivesTheRealParser() throws {
        // The rejected form is exactly the one the parser cannot decode.
        #expect(throws: GemmaToolCallParserError.malformed) {
            try GemmaToolCallParser().parse(
                "call:f{a:<|\"|>x\u{0600}<|\"|>}", allowedTools: ["f"], id: "x")
        }
        let call = try GemmaToolCallParser().parse(
            "call:f{a:<|\"|>x\u{0600}y<|\"|>}", allowedTools: ["f"], id: "x")
        #expect(call.arguments.objectValue?["a"] == .string("x\u{0600}y"))
    }

    /// A combining scalar joins onto whatever precedes it, so at the two
    /// positions where the parser compares a literal ASCII character — the
    /// opening delimiter's `>` and the last character of an escape — it would
    /// hide that character and the parse falls apart.
    /// A rejection can leave a partially consumed sequence behind, so each case
    /// starts from a fresh automaton parked at the guarded position.
    private func atGuardedPosition(_ prefix: String) -> JSONByteAutomaton {
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        #expect(feedBytes(&automaton, Array(prefix.utf8)))
        return automaton
    }

    @Test func combiningScalarCannotHideTheOpeningDelimiter() throws {
        for mark in ["\u{0618}", "\u{0301}", "\u{200D}"] {
            var automaton = atGuardedPosition(#"{a:<|"|>"#)
            #expect(!feedBytes(&automaton, Array(mark.utf8)), "accepted \(mark)")
        }
        // A base scalar is fine, and clears the guard for a following mark.
        var automaton = atGuardedPosition(#"{a:<|"|>"#)
        #expect(feedBytes(&automaton, Array("é\u{0301}".utf8)))
        #expect(feedBytes(&automaton, Array(#"<|"|>}"#.utf8)))
        #expect(automaton.isComplete)

        #expect(throws: GemmaToolCallParserError.malformed) {
            try GemmaToolCallParser().parse(
                "call:f{a:<|\"|>\u{0618}x<|\"|>}", allowedTools: ["f"], id: "x")
        }
    }

    @Test func combiningScalarCannotHideAnEscape() throws {
        for prefix in [#"{a:<|"|>\n"#, #"{a:<|"|>\\"#, #"{a:<|"|>\u0041"#] {
            var automaton = atGuardedPosition(prefix)
            #expect(!feedBytes(&automaton, Array("\u{0301}".utf8)), "accepted after \(prefix)")
        }
        // Ordinary content is not a guarded position: a mark joining onto a
        // base character changes nothing the parser reads.
        var plain = atGuardedPosition(#"{a:<|"|>A"#)
        #expect(feedBytes(&plain, Array("\u{0301}".utf8)))
        var automaton = atGuardedPosition(#"{a:<|"|>\n"#)
        #expect(feedBytes(&automaton, Array(#"x<|"|>}"#.utf8)))
        #expect(automaton.isComplete)

        #expect(throws: GemmaToolCallParserError.malformed) {
            try GemmaToolCallParser().parse(
                "call:f{a:<|\"|>\\n\u{0301}<|\"|>}", allowedTools: ["f"], id: "x")
        }
    }

    /// A sequence prefix whose every completion is a combining mark must be
    /// refused at the prefix, not at the last byte — otherwise the fallback
    /// scan is stranded mid-codepoint with the whole vocabulary vetoed.
    @Test func allCombiningSequencePrefixesAreRefusedEarly() {
        // 0xCC covers U+0300...U+033F, combining diacriticals throughout.
        var dead = atGuardedPosition(#"{a:<|"|>"#)
        #expect(!feedBytes(&dead, [0xCC]))
        // 0xC3 covers U+00C0...U+00FF, which has bases, so the lead passes and
        // the verdict lands on the completed scalar instead.
        var live = atGuardedPosition(#"{a:<|"|>"#)
        #expect(feedBytes(&live, [0xC3, 0xA9]))
        #expect(feedBytes(&live, Array(#"<|"|>}"#.utf8)))
        #expect(live.isComplete)
    }

    /// Every state the guard can reach must still have a legal next byte.
    @Test func guardedPositionsAreNeverDeadEnds() {
        for prefix in [#"{a:<|"|>"#, #"{a:<|"|>\n"#, #"{a:<|"|>A"#] {
            var frontier = [atGuardedPosition(prefix)]
            for _ in 0..<3 {
                var next: [JSONByteAutomaton] = []
                for state in frontier {
                    var reachable = false
                    for byte in UInt8.min...UInt8.max {
                        var candidate = state
                        guard candidate.consume(byte) else { continue }
                        reachable = true
                        // Only unfinished sequences can still be guarded.
                        if byte >= 0x80 { next.append(candidate) }
                    }
                    #expect(reachable, "dead end under the guard after \(prefix)")
                }
                frontier = next
            }
        }
    }

    @Test func escapedPrependAndZeroWidthJoinerDoNotGuard() {
        // `\u0600` is ASCII in the raw text, and ZWJ merges backwards.
        #expect(completes(#"{a:<|"|>\u0600<|"|>}"#))
        var automaton = JSONByteAutomaton(dialect: .gemmaToolArguments)
        #expect(feedBytes(&automaton, Array(#"{a:<|"|>"#.utf8)))
        #expect(feedBytes(&automaton, Array("x\u{200D}".utf8)))
        #expect(feedBytes(&automaton, Array(#"<|"|>}"#.utf8)))
        #expect(automaton.isComplete)
    }

    // MARK: - Strict dialect is untouched

    @Test func strictDialectStillAcceptsCanonicalJSON() {
        var automaton = JSONByteAutomaton()
        #expect(feedBytes(&automaton, Array(#"{"a":1e5,"b":[0.5]}"#.utf8)))
        #expect(automaton.isComplete)
        #expect(automaton.dialect == .strict)
    }
}
