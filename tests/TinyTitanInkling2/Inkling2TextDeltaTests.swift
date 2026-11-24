import Testing

@testable import TinyTitanInkling2

/// TT-128: the raw-checkpoint route streams; its deltas concatenate to the
/// answer and never carry a replacement character a later token completes.
@Suite struct Inkling2TextDeltaTests {
    @Test func deltasConcatenateToTheAnswer() {
        var stream = Inkling2TextDelta()
        let prefixes = ["Swift", "Swift and", "Swift and C++", "Swift and C++."]
        var out: [String] = []
        for text in prefixes { out.append(stream.advance(to: text)) }
        out.append(stream.finish("Swift and C++."))
        #expect(out == ["Swift", " and", " C++", ".", ""])
        #expect(out.joined() == "Swift and C++.")
    }

    @Test func holdsBackAnIncompleteCharacter() {
        var stream = Inkling2TextDelta()
        #expect(stream.advance(to: "caf") == "caf")
        // The token ended inside "é": the tokenizer shows U+FFFD.
        #expect(stream.advance(to: "caf\u{FFFD}") == "")
        #expect(stream.advance(to: "café") == "é")
        #expect(stream.finish("café") == "")
    }

    @Test func finishReleasesAReplacementNothingCompletes() {
        var stream = Inkling2TextDelta()
        #expect(stream.advance(to: "ok\u{FFFD}") == "ok")
        #expect(stream.finish("ok\u{FFFD}") == "\u{FFFD}")
    }
}
