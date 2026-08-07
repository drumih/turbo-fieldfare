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
        17: "<0xF0>",       // first byte of a 4-byte emoji
        18: "<0x9F>",
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

    @Test func byteFallbackPairInsideString() {
        let filter = makeFilter()
        #expect(filter.tryAccept(10))   // {"
        // Raw UTF-8 lead + continuation bytes are valid string content.
        #expect(filter.tryAccept(17))
        #expect(filter.tryAccept(18))
        // …but not outside a string: close key/value, then reject at "}".
        #expect(filter.tryAccept(12))   // ":
        #expect(!filter.tryAccept(17))
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
