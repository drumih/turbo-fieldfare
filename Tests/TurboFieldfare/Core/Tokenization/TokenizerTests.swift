import Foundation
import Hub
import Testing
@testable import TurboFieldfare

/// First run downloads the Gemma 4 IT tokenizer (~32 MB) from Hugging Face
/// Hub to `~/.cache/huggingface/`. Subsequent runs are offline.
@Suite("Tokenizer")
struct TokenizerTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load()
    }

    // MARK: - Special tokens

    @Test("Special token IDs are distinct and within vocab")
    func specialTokensDistinct() {
        let ids: [Int32] = [tok.bosID, tok.eosID, tok.padID, tok.endOfTurnID]
        #expect(Set(ids).count == 4, "expected 4 distinct special token IDs, got \(ids)")
        for id in ids {
            #expect(id >= 0)
            #expect(id < Int32(tok.vocabSize))
        }
    }

    @Test("Stop-token set covers EOS, end-of-turn, and tool response")
    func stopTokens() {
        #expect(tok.stopTokenIDs.contains(tok.eosID))
        #expect(tok.stopTokenIDs.contains(tok.endOfTurnID))
        #expect(tok.stopTokenIDs.contains(tok.toolResponseID))
        #expect(tok.stopTokenIDs.count == 3)
    }

    @Test("convertTokenToId substitutes unk for absent tokens")
    func convertTokenToIdSubstitutesUnk() {
        // Pins the library behavior `requireTokenID` exists for: BPE resolves an
        // absent token to the unknown-token ID, never nil, so a plain `guard
        // let` cannot detect a missing special marker.
        let id = tok.tokenizer.convertTokenToId("<no-such-token-xyzzy>")
        #expect(id != nil)
        #expect(id == tok.tokenizer.unknownTokenId)
    }

    @Test("Marker resolution rejects unk substitution")
    func markerResolutionRejectsUnk() throws {
        #expect(throws: GFTokenizerError.self) {
            _ = try GFTokenizer.requireTokenID(tok.tokenizer, "<no-such-token-xyzzy>")
        }
        let pad = try GFTokenizer.requireTokenID(tok.tokenizer, "<pad>")
        #expect(pad == tok.padID)
    }

    @Test("Special-token ID set matches the library's skip filter")
    func specialTokenSetMatchesLibrary() {
        // The config-derived set must reproduce the semantics the old runtime
        // probe encoded: a special token decodes to nothing when skipped and to
        // its literal text when kept.
        var probes = randomIDs(count: 64, seed: 0xDEAD_BEEF_CAFE_F00D)
        probes += [tok.bosID, tok.eosID, tok.padID, tok.endOfTurnID,
                   tok.toolCallStartID, tok.toolCallEndID, tok.toolResponseID,
                   tok.toolResponseEndID, tok.channelStartID, tok.channelEndID]
        for id in probes {
            let skipped = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: true)
            let kept = tok.tokenizer.decode(tokens: [Int(id)], skipSpecialTokens: false)
            let librarySpecial = skipped.isEmpty && !kept.isEmpty
            #expect(tok.specialTokenIDs.contains(id) == librarySpecial,
                    "id \(id) special-set membership diverged from library semantics")
        }
    }

    @Test("Out-of-range added-token IDs are rejected, not trapped")
    func outOfRangeAddedTokenIDRejected() {
        // Config.integer() returns full-range Int; a trapping Int32(_:) on a
        // corrupt tokenizer.json would crash at load instead of throwing.
        let data: Config = [
            "decoder": [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"],
                    ["type": "Fuse"],
                ],
            ],
            "added_tokens": [
                ["id": 3_000_000_000, "content": "<huge>", "special": true],
            ],
        ]
        #expect(throws: GFTokenizerError.self) {
            _ = try GFTokenizer(tokenizer: tok.tokenizer, tokenizerData: data)
        }
    }

    @Test("Library fuses a split codepoint across byte-fallback tokens")
    func libraryFusesSplitCodepoint() throws {
        // The behavioral half of the old init-time decoder probe, kept as a
        // test: the library remains the oracle for the pinned chain's
        // ByteFallback/Fuse behavior. The run needs a trailing non-byte token
        // because the library drops trailing byte runs (issue #58).
        let ids = try ["<0xC3>", "<0xA9>", "the"].map {
            Int(try GFTokenizer.requireTokenID(tok.tokenizer, $0))
        }
        #expect(tok.tokenizer.decode(tokens: ids, skipSpecialTokens: false) == "éthe")
    }

    // MARK: - Encode / decode

    @Test("Round-trip ASCII", arguments: [
        "Hello, world.",
        "The quick brown fox jumps over the lazy dog.",
        "code:  let x = 42;  // comment",
        "numbers 0 1 2 3 4 5 6 7 8 9",
    ])
    func roundTripASCII(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("Decode keeps spaces before punctuation", arguments: [
        "step 1 . done",
        "x , y",
        "he said ' ok ' now",
        "a    . b",
        "wait ! really ?",
    ])
    func decodePreservesSpacingBeforePunctuation(_ text: String) {
        // HF's clean_up_tokenization_spaces collapsed these ("he said ' ok ' now"
        // became "he said'ok'now"), which broke decode(encode(x)) == x for a
        // tokenizer that is lossless by construction.
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("Round-trip multi-byte UTF-8", arguments: [
        "你好，世界。",
        "漢字",
        "🦝 raccoon emoji",
        "mixed 漢 and 🦝 and a",
        "Здравствуй",
    ])
    func roundTripMultibyte(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("BOS is added when requested and absent otherwise")
    func bosPrepend() {
        let withBOS = tok.encode("x", addBOS: true)
        let without = tok.encode("x", addBOS: false)
        #expect(withBOS.first == tok.bosID)
        #expect(without.first != tok.bosID)
        #expect(withBOS.count == without.count + 1)
        #expect(Array(withBOS.dropFirst()) == without)
    }

    @Test("Empty string encodes to BOS-only or empty")
    func encodeEmpty() {
        let none = tok.encode("", addBOS: false)
        #expect(none.isEmpty)
        let bos = tok.encode("", addBOS: true)
        #expect(bos == [tok.bosID])
    }

    @Test("Decoding strips special tokens when requested")
    func decodeStripsSpecial() {
        let ids = tok.encode("hi", addBOS: true)
        #expect(ids.first == tok.bosID)
        let withoutSpecial = tok.decode(ids, skipSpecialTokens: true)
        #expect(withoutSpecial == "hi")
    }

    // MARK: - Streaming detokenizer

    @Test("Streaming detokenizer reassembles ASCII")
    func streamingASCII() {
        let target = "Hello, world."
        assertStreams(target)
    }

    @Test("Streaming detokenizer reassembles multi-byte UTF-8", arguments: [
        "漢字",
        "你好",
        "🦝🦝🦝",
        "mixed 漢 and 🦝",
        "Здравствуй мир",
        "ห้ามใช้ในสัปดาห์นี้",
        "नमस्ते दुनिया",
        "مَرْحَبًا",
        "\u{1F469}\u{200D}\u{1F4BB} works with \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}",
        "ends with emoji 🦝",
        "🦝 starts with emoji",
        "🦝 middle 漢 end",
    ])
    func streamingMultibyte(_ target: String) {
        assertStreams(target)
    }

    @Test("Streaming detokenizer never emits replacement chars")
    func streamingNoMojibake() {
        let ids = tok.encode("mixed 漢 and 🦝 text", addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        for id in ids {
            let delta = detok.push(id)
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"),
                    "delta contained replacement char: '\(delta)'")
        }
        let tail = detok.flush()
        #expect(!tail.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Streaming reassembles graphemes extended by later scalars", arguments: [
        "ห้าม",
        "e\u{301}",
        "ا\u{64E}",
        "ש\u{5B8}",
        "क्ष",
        "\u{1100}\u{1161}",
        "\u{263A}\u{FE0F}",
        "\u{1F44D}\u{1F3FD}",
        "1\u{FE0F}\u{20E3}",
        "\u{1F1EC}\u{1F1E7}",
        "\u{1F469}\u{200D}\u{1F4BB}",
    ])
    func streamingExtendedGraphemes(_ text: String) {
        assertStreams(text)
    }

    @Test("Whitespace run before punctuation survives streaming")
    func streamingWhitespaceThenPunctuation() {
        // Encoder-unreachable but generator-reachable: the normalizer maps " " to
        // "▁", so no encode() call produces a bare whitespace run followed by
        // punctuation. A generating model emits it freely — in indented markdown
        // and in code. HF's clean_up_tokenization_spaces rewrote " ." to "."
        // across that boundary, which made the streaming delta drop the
        // punctuation token entirely and silently.
        for (run, mark) in [("▁▁▁▁", "."), ("▁▁", ","), ("▁▁▁▁", "?"), ("▁▁", "!")] {
            guard let runID = tok.tokenizer.convertTokenToId(run),
                  let markID = tok.tokenizer.convertTokenToId(mark) else {
                Issue.record("vocabulary is missing \(run) or \(mark)")
                continue
            }
            let ids = [Int32(runID), Int32(markID)]
            var detok = GFDetokenizer(tokenizer: tok)
            var assembled = ""
            for id in ids { assembled += detok.push(id) }
            assembled += detok.flush()

            #expect(assembled == tok.decode(ids))
            #expect(assembled == String(repeating: " ", count: run.count) + mark,
                    "expected the whitespace run and '\(mark)' intact, got '\(assembled)'")
        }
    }

    @Test("Decode differs from the library only by its cleanup pass")
    func decodeDiffersFromLibraryOnlyByCleanup() {
        // The streaming test below compares against our own batch decode, so it
        // cannot catch a fault shared by both. This one keeps swift-transformers
        // as an independent oracle: our decode, run through the cleanup pass we
        // deliberately dropped, must reproduce the library byte for byte. That
        // pins the intended differences and covers special-token filtering,
        // byte-fallback grouping, and the "▁" mapping. Besides the cleanup
        // pass, we intentionally diverge from the Swift port on byte-fallback
        // runs — invalid runs decode to one U+FFFD per byte and trailing runs
        // are flushed, both per the reference Rust decoder, where the port
        // decodes leniently and drops trailing runs (issue #58). The fixed
        // seeds produce no multi-byte-token runs, so those divergences do not
        // reach this comparison; the byte-soup test covers them against a
        // reference-semantics oracle.
        var ids = randomIDs(count: 256, seed: 0x2545_F491_4F6C_DD1D)
        ids += [tok.bosID, tok.eosID, tok.endOfTurnID, tok.channelStartID, tok.padID]
        // The library drops byte-fallback tokens that end a sequence (issue #58)
        // while we assemble them, so the two only agree when a regular token
        // closes the stream. End on one rather than depend on where the random
        // IDs happen to land.
        if let trailing = tok.tokenizer.convertTokenToId("the") {
            ids.append(Int32(trailing))
        }
        for skipSpecialTokens in [true, false] {
            let mine = tok.decode(ids, skipSpecialTokens: skipSpecialTokens)
            let library = tok.tokenizer.decode(tokens: ids.map(Int.init),
                                               skipSpecialTokens: skipSpecialTokens)
            #expect(Self.huggingFaceCleanUp(mine) == library,
                    "decode diverged from the library beyond the cleanup pass (skipSpecialTokens: \(skipSpecialTokens))")
        }
    }

    // MARK: - Byte-fallback runs

    @Test("Byte run fuses across a skipped special token")
    func byteRunFusesAcrossSkippedSpecial() throws {
        // The library filters skipped special IDs before its decoder chain, so
        // a byte run must fuse ACROSS a skipped special. In keep mode the
        // special is an ordinary token and closes the run instead, leaving two
        // single-byte runs that are invalid on their own - one U+FFFD each.
        let ids: [Int32] = try [
            tokenID("<0xC3>"), tok.eosID, tokenID("<0xA9>"), tokenID("the"),
        ]
        #expect(tok.decode(ids, skipSpecialTokens: true) == "éthe")
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids { assembled += detok.push(id) }
        assembled += detok.flush()
        #expect(assembled == "éthe")
        #expect(tok.decode(ids, skipSpecialTokens: false) == "\u{FFFD}<eos>\u{FFFD}the")
    }

    @Test("Invalid byte runs stream replacement characters immediately")
    func invalidByteRunsStreamImmediately() throws {
        // Once a run can no longer become valid UTF-8, every byte - past and
        // future - decodes to exactly one U+FFFD, so the stream emits them
        // live instead of freezing and buffering until the next regular token.
        var detok = GFDetokenizer(tokenizer: tok)
        let stray = try tokenID("<0x80>")
        for _ in 0..<64 {
            #expect(detok.push(stray) == "\u{FFFD}")
        }
        #expect(detok.flush() == "")

        detok = GFDetokenizer(tokenizer: tok)
        #expect(try detok.push(tokenID("<0xC3>")) == "")
        #expect(try detok.push(tokenID("<0xFF>")) == "\u{FFFD}\u{FFFD}")

        // E0's first continuation is constrained to A0-BF, so 0x80 poisons the
        // run even though it is a continuation byte.
        detok = GFDetokenizer(tokenizer: tok)
        #expect(try detok.push(tokenID("<0xE0>")) == "")
        #expect(try detok.push(tokenID("<0x80>")) == "\u{FFFD}\u{FFFD}")
    }

    @Test("An incomplete trailing sequence invalidates the whole run")
    func incompleteTailInvalidatesWholeRun() throws {
        // The reference decoder validates the run as a whole, so an incomplete
        // tail turns even a valid prefix into per-byte replacement characters.
        var detok = GFDetokenizer(tokenizer: tok)
        #expect(try detok.push(tokenID("<0xE2>")) == "")
        #expect(try detok.push(tokenID("<0x82>")) == "")
        #expect(detok.flush() == "\u{FFFD}\u{FFFD}")

        detok = GFDetokenizer(tokenizer: tok)
        for token in ["<0xC3>", "<0xA9>", "<0xE2>", "<0x82>"] {
            #expect(try detok.push(tokenID(token)) == "")
        }
        #expect(detok.flush() == String(repeating: "\u{FFFD}", count: 4))
    }

    @Test("A valid byte run commits with the token that closes it")
    func validByteRunCommitsAtBoundary() throws {
        var detok = GFDetokenizer(tokenizer: tok)
        for token in ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"] {
            #expect(try detok.push(tokenID(token)) == "")
        }
        #expect(try detok.push(tokenID("the")) == "😀the")
    }

    @Test("A barrier marker commits the run as its own delta")
    func barrierMarkerCommitsHeldRun() throws {
        // The generation pipeline must not let a byte run span a channel/tool
        // marker: the held text surfaces as the marker's own delta, so the
        // structured decoder can route it under the pre-switch channel state.
        var detok = GFDetokenizer(tokenizer: tok,
                                  barrierTokenIDs: tok.structuralMarkerIDs)
        for token in ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"] {
            #expect(try detok.push(tokenID(token)) == "")
        }
        #expect(detok.push(tok.channelEndID) == "😀")
        // Non-barrier specials still fuse: a codepoint split around eos stays
        // whole, preserving decode parity for everything but the markers.
        #expect(try detok.push(tokenID("<0xC3>")) == "")
        #expect(detok.push(tok.eosID) == "")
        #expect(try detok.push(tokenID("<0xA9>")) == "")
        #expect(try detok.push(tokenID("the")) == "éthe")
    }

    @Test("Byte-token soup matches batch decode and reference run semantics")
    func byteSoupMatchesReference() throws {
        // Adversarial byte-run coverage the encoder can never produce: mostly
        // byte tokens with occasional regular tokens, checked against a direct
        // reimplementation of the reference decoder's run semantics
        // (from_utf8, else one U+FFFD per byte) and against our own streaming.
        let byteIDs: [Int32] = try (0...255).map {
            try tokenID(String(format: "<0x%02X>", $0))
        }
        let theID = try tokenID("the")
        var state: UInt64 = 0xA076_1D64_78BD_642F
        func next() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        for round in 0..<16 {
            var ids: [Int32] = []
            var expected = ""
            var run: [UInt8] = []
            func closeRun() {
                guard !run.isEmpty else { return }
                expected += String(bytes: run, encoding: .utf8)
                    ?? String(repeating: "\u{FFFD}", count: run.count)
                run.removeAll()
            }
            for _ in 0..<256 {
                let r = next()
                if r % 8 == 0 {
                    closeRun()
                    ids.append(theID)
                    expected += "the"
                } else {
                    let byte = UInt8(truncatingIfNeeded: r >> 8)
                    ids.append(byteIDs[Int(byte)])
                    run.append(byte)
                }
            }
            closeRun()

            #expect(tok.decode(ids) == expected, "round \(round) batch diverged")
            var detok = GFDetokenizer(tokenizer: tok)
            var assembled = ""
            for id in ids { assembled += detok.push(id) }
            assembled += detok.flush()
            #expect(assembled == expected, "round \(round) streaming diverged")
        }
    }

    @Test("Streaming matches batch decode for arbitrary token streams")
    func streamingMatchesBatchForArbitraryIDs() {
        // Every other streaming test round-trips encode() output, which can only
        // produce token sequences the encoder itself emits. A generating model is
        // under no such constraint, so drive the detokenizer with raw IDs.
        for round in 0..<16 {
            let ids = randomIDs(count: 256, seed: 0x9E37_79B9_7F4A_7C15 &+ UInt64(round))
            var detok = GFDetokenizer(tokenizer: tok)
            var assembled = ""
            for id in ids { assembled += detok.push(id) }
            assembled += detok.flush()
            #expect(assembled == tok.decode(ids), "round \(round) diverged from batch decode")
        }
    }

    @Test("Streaming detokenizer matches full decode for complex Unicode", arguments: [
        "English العَرَبِيَّةُ ١٢٣ ثم English",
        "left \u{2067}العَرَبِيَّةُ ١٢٣\u{2069} right",
        "English עברית מְנֻּקֶדֶת 42",
        "فارسی می\u{200C}خواهم English",
        "क्\u{200D}ष क्\u{200C}ष English",
        "ພາສາລາວ English",
        "ខ្ញុំសរសេរភាសាខ្មែរ English",
        "မြန်မာစာ English",
        "বাংলা ভাষা English",
        "සිංහල භාෂාව English",
        "བོད་ཡིག English",
        "\u{1100}\u{1161}\u{11A8} 한글 English",
        "a\u{301}\u{323}\u{304} e\u{308}\u{301}",
        "\u{1F44D}\u{1F3FD} \u{263A}\u{FE0F} 1\u{FE0F}\u{20E3} \u{1F1EC}\u{1F1E7} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}",
    ])
    func streamingComplexUnicode(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        let expected = tok.decode(ids)
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == expected)
        #expect(!assembled.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Streaming detokenizer preserves long mixed output", arguments: [
        "The quick brown fox jumps over the lazy dog. 0123456789\n",
        "\u{E000}\u{E001}\u{E002}\u{E003}\u{E004}\u{E005}\u{E006}\u{E007}",
        "TurboFieldfare 漢字 Здравствуй 🦝 café Ελληνικά \u{E000}\n",
    ])
    func streamingLongOutput(_ seed: String) {
        var text = seed
        var ids = tok.encode(text, addBOS: false)
        while ids.count < 256 {
            text += seed
            ids = tok.encode(text, addBOS: false)
        }

        let prefix = Array(ids.prefix(256))
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in prefix {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == tok.decode(prefix))
        #expect(!assembled.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Flush with no tokens yields empty string")
    func streamingEmpty() {
        var detok = GFDetokenizer(tokenizer: tok)
        #expect(detok.flush() == "")
    }

    @Test("Single-token push works for in-vocab characters")
    func streamingSingleToken() {
        // "漢字" is a single in-vocab token (verified empirically).
        var detok = GFDetokenizer(tokenizer: tok)
        let ids = tok.encode("漢字", addBOS: false)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == "漢字")
    }

    // MARK: - Helpers

    private func tokenID(_ token: String) throws -> Int32 {
        try GFTokenizer.requireTokenID(tok.tokenizer, token)
    }

    /// Deterministic xorshift64 IDs — token streams the encoder cannot produce
    /// but a generating model can.
    private func randomIDs(count: Int, seed: UInt64) -> [Int32] {
        var state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        return (0..<count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Int32(state % UInt64(tok.vocabSize))
        }
    }

    /// swift-transformers' `clean_up_tokenization_spaces` pass, reproduced here
    /// so the differential test can pin our only intended difference from the
    /// library to exactly this pass. Not used in production — see `GemmaDecoding`.
    private static func huggingFaceCleanUp(_ text: String) -> String {
        let replacements = [
            (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
            (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
        ]
        return replacements.reduce(text) { partial, rule in
            partial.replacingOccurrences(of: rule.0, with: rule.1)
        }
    }

    private func assertStreams(_ target: String) {
        let ids = tok.encode(target, addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == target, "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }
}
