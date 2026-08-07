import Testing
import Foundation
@testable import TurboFieldfare

/// Byte-level automaton coverage: acceptance criteria cases from the Tikum
/// handoff — nesting, escapes, unicode split across tokens, numbers,
/// truncation — plus the structural rejections that make forced JSON safe.
@Suite struct JSONByteAutomatonTests {

    private func feed(_ automaton: inout JSONByteAutomaton, _ text: String) -> Bool {
        feedBytes(&automaton, Array(text.utf8))
    }

    private func feedBytes(_ automaton: inout JSONByteAutomaton, _ bytes: [UInt8]) -> Bool {
        for byte in bytes {
            guard automaton.consume(byte) else { return false }
        }
        return true
    }

    /// `#expect` cannot call mutating members inline; route single bytes here.
    private func step(_ automaton: inout JSONByteAutomaton, _ byte: UInt8) -> Bool {
        automaton.consume(byte)
    }

    private func accepts(_ text: String) -> JSONByteAutomaton? {
        var automaton = JSONByteAutomaton()
        return feed(&automaton, text) ? automaton : nil
    }

    @Test func flatObjectCompletes() {
        let automaton = accepts(#"{"urgencia": "alta", "derivar": true, "score": 0.87}"#)
        #expect(automaton?.isComplete == true)
    }

    @Test func nestedContainersComplete() {
        let automaton = accepts(
            #"{"a": [1, {"b": [true, null]}, -2.5e+3], "c": {"d": {}}}"#)
        #expect(automaton?.isComplete == true)
    }

    @Test func leadingWhitespaceAllowed() {
        let automaton = accepts(" \n\t {\"a\": 1}")
        #expect(automaton?.isComplete == true)
    }

    @Test func trailingWhitespaceAfterCompleteAllowed() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, "{\"a\": 1} \n"))
        #expect(automaton.isComplete)
        #expect(!step(&automaton, UInt8(ascii: "x")))
    }

    @Test func proseIsRejectedAtRoot() {
        var automaton = JSONByteAutomaton()
        #expect(!step(&automaton, UInt8(ascii: "H")))
        // The rejection must not corrupt state: a valid document still passes.
        #expect(feed(&automaton, "{}"))
        #expect(automaton.isComplete)
    }

    @Test func stringEscapes() {
        let automaton = accepts(#"{"s": "a\"b\\c\/d\n\té"}"#)
        #expect(automaton?.isComplete == true)
    }

    @Test func invalidEscapeRejected() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": "a\"#))
        #expect(!step(&automaton, UInt8(ascii: "x")))
    }

    @Test func incompleteUnicodeEscapeRejectsNonHex() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": "\u00"#))
        #expect(!step(&automaton, UInt8(ascii: "g")))
        #expect(step(&automaton, UInt8(ascii: "e")))
    }

    /// A multi-byte codepoint split across tokens arrives as bare
    /// continuation bytes; inside a string each byte must be accepted.
    @Test func splitUTF8InsideString() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": ""#))
        let emoji = Array("🇦🇷ñ".utf8)
        let first = emoji[..<3]
        let second = emoji[3...]
        for byte in first { #expect(step(&automaton, byte)) }
        for byte in second { #expect(step(&automaton, byte)) }
        #expect(feed(&automaton, "\"}"))
        #expect(automaton.isComplete)
    }

    @Test func controlByteInsideStringRejected() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": ""#))
        #expect(!step(&automaton, 0x07))
    }

    @Test func numberForms() {
        for doc in ["[0]", "[-0]", "[42]", "[0.5]", "[-12.25]",
                    "[1e9]", "[1E-9]", "[0.5e+10]", "[123, -4.5]"] {
            #expect(accepts(doc)?.isComplete == true, "expected \(doc) to parse")
        }
    }

    @Test func malformedNumbersRejected() {
        for doc in ["[01", "[1.e", "[--1", "[.5", "[+1", "[1..", "[0x"] {
            #expect(accepts(doc) == nil, "expected \(doc) to be rejected")
        }
        // "[1e+" is a legal prefix (extendable to "[1e+5]") — but it can
        // neither stop nor close the array here.
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, "[1e+"))
        #expect(!automaton.canStop)
        #expect(!step(&automaton, UInt8(ascii: "]")))
    }

    @Test func literals() {
        #expect(accepts("[true, false, null]")?.isComplete == true)
        #expect(accepts("[tru]") == nil)
        #expect(accepts("[nulll") == nil)
    }

    @Test func structuralMistakesRejected() {
        for doc in ["{\"a\" 1", "{\"a\": 1,}", "{,", "[1,]", "{\"a\": 1}}",
                    "{]", "[}", "{\"a\":: 1", "{1: 2"] {
            #expect(accepts(doc) == nil || accepts(doc)?.isComplete == false,
                    "expected \(doc) to be rejected or incomplete")
        }
        #expect(accepts("{,") == nil)
        #expect(accepts("{\"a\" 1") == nil)
    }

    /// Truncation by max-new: a prefix of a valid document is accepted but
    /// never complete, and mid-string it cannot stop.
    @Test func truncatedDocumentIsIncomplete() {
        let automaton = accepts(#"{"urgencia": "al"#)
        #expect(automaton != nil)
        #expect(automaton?.isComplete == false)
        #expect(automaton?.canStop == false)
    }

    @Test func rootBareNumberCanStopButIsNotComplete() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, "12"))
        #expect(!automaton.isComplete)
        #expect(automaton.canStop)
        // A digit may still extend it…
        #expect(step(&automaton, UInt8(ascii: "3")))
        // …and a terminator closes it.
        #expect(step(&automaton, UInt8(ascii: " ")))
        #expect(automaton.isComplete)
    }

    @Test func depthLimitEnforced() {
        var automaton = JSONByteAutomaton()
        for _ in 0..<JSONByteAutomaton.maxDepth {
            #expect(step(&automaton, UInt8(ascii: "[")))
        }
        #expect(!step(&automaton, UInt8(ascii: "[")))
        #expect(step(&automaton, UInt8(ascii: "]")))
    }

    @Test func emptyContainers() {
        #expect(accepts("{}")?.isComplete == true)
        #expect(accepts("[]")?.isComplete == true)
        #expect(accepts("[[], {}]")?.isComplete == true)
        #expect(accepts("[,") == nil)
    }
}

/// Token-level filter: SentencePiece piece mapping (▁, byte-fallback),
/// special-token blocking, stop-token gating on `canStop`, and state commit
/// on acceptance.
@Suite struct JSONTokenFilterTests {

    private static let eos: Int32 = 1
    private static let endOfTurn: Int32 = 106

    /// Tiny fake vocabulary exercising every piece shape the filter handles.
    private static let pieces: [Int32: String] = [
        eos: "<eos>",
        endOfTurn: "<turn|>",
        10: "{\"",
        11: "urgencia",
        12: "\":",
        13: "▁\"alta\"",
        14: "}",
        15: "Hola",
        16: "▁no",
        17: "<0xF0>",       // 4-byte emoji, one byte-fallback token per byte
        18: "<0x9F>",
        23: "<0x87>",
        24: "<0xA6>",
        19: "<unused12>",
        20: "",
        21: "▁",
        22: "123",
    ]

    private func makeFilter() -> JSONTokenFilter {
        JSONTokenFilter(pieceLookup: { Self.pieces[$0] },
                        blockedIDs: [],
                        stopIDs: [Self.eos, Self.endOfTurn])
    }

    @Test func buildsObjectAndForcesCompletion() {
        let filter = makeFilter()
        // Prose openers and stop tokens are vetoed before the root opens.
        #expect(!filter.tryAccept(15))
        #expect(!filter.tryAccept(Self.eos))
        #expect(filter.tryAccept(10))   // {"
        #expect(filter.tryAccept(11))   // urgencia
        #expect(filter.tryAccept(12))   // ":
        #expect(filter.tryAccept(13))   // ▁"alta"
        #expect(!filter.isComplete)
        #expect(filter.tryAccept(14))   // }
        #expect(filter.isComplete)
        // Once complete, EOS is legal and content is not.
        #expect(filter.tryAccept(Self.eos))
        #expect(!filter.tryAccept(15))
    }

    @Test func spaceMarkerMapsToSpace() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))   // {"
        #expect(filter.tryAccept(11))   // key
        #expect(filter.tryAccept(12))   // ":
        #expect(filter.tryAccept(21))   // "▁" alone → structural space
        #expect(filter.tryAccept(22))   // 123
        #expect(filter.tryAccept(14))   // }
        #expect(filter.isComplete)
    }

    @Test func byteFallbackSequenceInsideString() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))   // {"
        // A 4-byte emoji arriving one byte-fallback token at a time.
        #expect(filter.tryAccept(17))   // F0 lead
        // Mid-sequence, closing the key is forbidden (would leave broken UTF-8)…
        #expect(!filter.tryAccept(12))  // ":
        #expect(filter.tryAccept(18))   // 9F
        #expect(filter.tryAccept(23))   // 87
        #expect(filter.tryAccept(24))   // A6 — sequence complete
        // …now it isn't.
        #expect(filter.tryAccept(12))   // ":
        // A raw lead byte outside a string is never valid JSON.
        #expect(!filter.tryAccept(17))
    }

    @Test func lastAcceptedBytesTracksCommittedToken() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))               // {"
        #expect(filter.lastAcceptedBytes == Array("{\"".utf8))
        #expect(filter.tryAccept(11))               // urgencia (key content)
        #expect(filter.lastAcceptedBytes == Array("urgencia".utf8))
        #expect(filter.tryAccept(21))               // "▁" → mapped to a space
        #expect(filter.lastAcceptedBytes == Array(" ".utf8))
    }

    @Test func markerAndEmptyPiecesBlocked() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))
        // Inside a string "<unused12>" bytes WOULD be legal — the filter must
        // still block marker-shaped pieces (the detokenizer strips them).
        #expect(!filter.tryAccept(19))
        #expect(!filter.tryAccept(20))  // empty piece can't make progress
        #expect(!filter.tryAccept(99))  // unknown id
    }

    @Test func rejectionDoesNotAdvanceState() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))   // {"
        #expect(filter.tryAccept(11))   // urgencia
        #expect(filter.tryAccept(12))   // ":
        #expect(!filter.tryAccept(11))  // a bare identifier is not a value
        #expect(filter.tryAccept(13))   // ▁"alta" still accepted from same state
        #expect(filter.tryAccept(14))   // }
        #expect(filter.isComplete)
    }
}

/// UTF-8 structural validation inside strings: the automaton must reject any
/// byte sequence that would leave the emitted document undecodable.
@Suite struct JSONAutomatonUTF8Tests {

    private func stringOpen() -> JSONByteAutomaton {
        var automaton = JSONByteAutomaton()
        for byte in Array("{\"k\": \"".utf8) {
            precondition(automaton.consume(byte))
        }
        return automaton
    }

    private func step(_ automaton: inout JSONByteAutomaton, _ byte: UInt8) -> Bool {
        automaton.consume(byte)
    }

    @Test func validSequencesAccepted() {
        for text in ["é", "ñandú", "€", "🇦🇷", "\u{10FFFF}", "日本語"] {
            var automaton = stringOpen()
            for byte in Array(text.utf8) {
                #expect(step(&automaton, byte), "byte of \(text) rejected")
            }
            #expect(step(&automaton, UInt8(ascii: "\"")))
        }
    }

    @Test func bareContinuationRejected() {
        var automaton = stringOpen()
        #expect(!step(&automaton, 0x80))
        #expect(!step(&automaton, 0xBF))
    }

    @Test func leadFollowedByNonContinuationRejected() {
        var automaton = stringOpen()
        #expect(step(&automaton, 0xC3))
        #expect(!step(&automaton, UInt8(ascii: "a")))
        #expect(!step(&automaton, 0xC3))          // another lead: also invalid
    }

    @Test func quoteAndEscapeMidSequenceRejected() {
        var automaton = stringOpen()
        #expect(step(&automaton, 0xF0))
        #expect(step(&automaton, 0x9F))
        #expect(!step(&automaton, UInt8(ascii: "\"")))
        #expect(!step(&automaton, UInt8(ascii: "\\")))
        // Completing the sequence re-enables both.
        #expect(step(&automaton, 0x87))
        #expect(step(&automaton, 0xA6))
        #expect(step(&automaton, UInt8(ascii: "\"")))
    }

    @Test func invalidLeadBytesRejected() {
        for lead: UInt8 in [0xC0, 0xC1, 0xF5, 0xFE, 0xFF] {
            var automaton = stringOpen()
            #expect(!step(&automaton, lead), "lead \(lead) must be rejected")
        }
    }

    @Test func surrogateAndRangeBoundaries() {
        // ED 9F BF (U+D7FF, last before surrogates) is legal; ED A0 is not.
        var legal = stringOpen()
        #expect(step(&legal, 0xED))
        #expect(step(&legal, 0x9F))
        #expect(step(&legal, 0xBF))
        var surrogate = stringOpen()
        #expect(step(&surrogate, 0xED))
        #expect(!step(&surrogate, 0xA0))
        // F4 8F is legal (up to U+10FFFF); F4 90 overflows.
        var top = stringOpen()
        #expect(step(&top, 0xF4))
        #expect(step(&top, 0x8F))
        var beyond = stringOpen()
        #expect(step(&beyond, 0xF4))
        #expect(!step(&beyond, 0x90))
        // E0 80 would be overlong.
        var overlong = stringOpen()
        #expect(step(&overlong, 0xE0))
        #expect(!step(&overlong, 0x80))
    }

    @Test func multibyteOutsideStringsRejected() {
        var automaton = JSONByteAutomaton()
        #expect(!step(&automaton, 0xC3))
    }
}

/// The byte-stream assembler that replaces detokenizer text under forceJSON.
@Suite struct UTF8StreamAssemblerTests {

    @Test func asciiPassesThrough() {
        var assembler = UTF8StreamAssembler()
        #expect(assembler.push(Array("{\"a\": 1".utf8)) == "{\"a\": 1")
        #expect(assembler.push(Array("}".utf8)) == "}")
        #expect(assembler.flush() == "")
    }

    @Test func splitCodepointHeldUntilComplete() {
        var assembler = UTF8StreamAssembler()
        #expect(assembler.push([0xF0, 0x9F]) == "")
        #expect(assembler.push([0x87]) == "")
        #expect(assembler.push([0xA6, 0x21]) == "\u{1F1E6}!")
        #expect(assembler.flush() == "")
    }

    @Test func completeTailEmittedOnPush() {
        var assembler = UTF8StreamAssembler()
        #expect(assembler.push(Array("é".utf8)) == "é")
    }

    @Test func mixedAsciiAndPendingSequence() {
        var assembler = UTF8StreamAssembler()
        #expect(assembler.push([UInt8(ascii: "\""), 0xC3]) == "\"")
        #expect(assembler.push([0xA9, UInt8(ascii: "\"")]) == "é\"")
    }

    @Test func flushDropsIncompleteTail() {
        var assembler = UTF8StreamAssembler()
        #expect(assembler.push([UInt8(ascii: "a"), 0xE2, 0x82]) == "a")
        #expect(assembler.flush() == "")
        // After flush the pending tail is gone for good.
        #expect(assembler.push([UInt8(ascii: "b")]) == "b")
    }
}

/// Surrogate pairing in \uXXXX escapes: Foundation's JSON parsers reject
/// lone surrogates, so the automaton must too.
@Suite struct JSONAutomatonSurrogateTests {

    private func feed(_ automaton: inout JSONByteAutomaton, _ text: String) -> Bool {
        for byte in Array(text.utf8) {
            guard automaton.consume(byte) else { return false }
        }
        return true
    }

    private func step(_ automaton: inout JSONByteAutomaton, _ byte: UInt8) -> Bool {
        automaton.consume(byte)
    }

    @Test func pairedSurrogatesAccepted() {
        var automaton = JSONByteAutomaton()
        // Escaped pair for U+1F600 (😀): high \ud83d + low \ude00.
        #expect(feed(&automaton, "{\"s\": \"\\ud83d\\ude00\"}"))
        #expect(automaton.isComplete)
    }

    @Test func loneHighSurrogateCannotCloseString() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": "\ud800"#))
        // After a high surrogate only `\` is legal — not a quote, not text.
        #expect(!step(&automaton, UInt8(ascii: "\"")))
        #expect(!step(&automaton, UInt8(ascii: "a")))
        #expect(step(&automaton, UInt8(ascii: "\\")))
        // …and after the backslash, only `u`.
        #expect(!step(&automaton, UInt8(ascii: "n")))
        #expect(step(&automaton, UInt8(ascii: "u")))
    }

    @Test func loneLowSurrogateRejectedAtFinalDigit() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": "\udc0"#))
        #expect(!step(&automaton, UInt8(ascii: "0")))
        // A non-surrogate completion from the same prefix stays legal:
        // \udc0 can't be saved, but the rejection must not corrupt state —
        // hex digits beyond the range are equally rejected…
        #expect(!step(&automaton, UInt8(ascii: "f")))
    }

    @Test func highSurrogateFollowedByNonLowEscapeRejected() {
        var automaton = JSONByteAutomaton()
        #expect(feed(&automaton, #"{"s": "\ud800\u004"#))
        #expect(!step(&automaton, UInt8(ascii: "1")))   // A is not low
        #expect(!step(&automaton, UInt8(ascii: "0")))
    }

    @Test func highSurrogateFollowedByHighRejected() {
        var automaton = JSONByteAutomaton()
        // The verdict lands on the fourth hex digit: \ud80_ can only be a
        // high surrogate, which is not a valid pair completion.
        #expect(feed(&automaton, #"{"s": "\ud800\ud80"#))
        #expect(!step(&automaton, UInt8(ascii: "0")))
        // A proper pair from scratch works: \ud800 + \udc00 = U+10000.
        var paired = JSONByteAutomaton()
        #expect(feed(&paired, "{\"s\": \"\\ud800\\udc00\"}"))
        #expect(paired.isComplete)
    }

    @Test func nonSurrogateEscapesUnaffected() {
        var automaton = JSONByteAutomaton()
        // \u00e9 (é) and \ud7ff (last scalar before the surrogate range).
        #expect(feed(&automaton, "{\"s\": \"\\u00e9\\ud7ff\"}"))
        #expect(automaton.isComplete)
    }
}
