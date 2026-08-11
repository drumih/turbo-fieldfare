import Testing
import Foundation
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The property that makes the tool-call constraint a fix rather than a filter:
/// **every region the grammar closes parses**. The walk emulates
/// `sampleOnce` + `grammarFallbackToken` — a random "GPU" proposal, a capped
/// random retry standing in for the top-256 probability scan, then an
/// exhaustive vocabulary sweep — over a vocabulary built to spell every known
/// way the model derails.
///
/// No Metal, no model, no network. Seeds are derived from the walk index, so
/// any failure prints a seed and a token trace that reproduce it exactly.
@Suite struct ToolCallGrammarFuzzTests {

    // MARK: - Adversarial vocabulary

    private static let markerIDs = ToolCallMarkerIDs(
        toolCallStart: 48, toolCallEnd: 49, toolResponse: 50, toolResponseEnd: 51,
        channelStart: 100, channelEnd: 101, escape: 52, endOfTurn: 106,
        eos: 1, bos: 2, pad: 0)

    private static let declaredNames: Set<String> =
        ["get-weather", "get", "get_all", "read", "read-file"]

    private static let byteFallbackBytes: [UInt8] = {
        var bytes = Set(Array(#"{}[],:"\<|>aclerdfginostuw0129.-+$_ "#.utf8))
        // The real vocabulary has a byte-fallback piece for all 256 bytes, so
        // every declared name is spellable one byte at a time; a trie node with
        // no reachable continuation would be a false deadlock.
        bytes.formUnion(Array(declaredNames.joined().utf8))
        // Leads and continuations of U+0600 (Prepend), U+0D4E (Prepend),
        // U+1F601, U+00F1 — plus bytes that can never start or continue a
        // sequence.
        bytes.formUnion([0xD8, 0x80, 0xE0, 0xB5, 0x8E, 0xF0, 0x9F, 0x98, 0x81,
                         0xC3, 0xB1, 0xC0, 0xF5, 0xBF, 0xFF, 0x07, 0x09, 0x0A])
        return bytes.sorted()
    }()

    private static let pieces: [Int32: String] = {
        var pieces: [Int32: String] = [
            0: "<pad>", 1: "<eos>", 2: "<bos>",
            48: "<|tool_call>", 49: "<tool_call|>", 50: "<|tool_response>",
            51: "<tool_response|>", 52: "<|\"|>", 100: "<|channel>",
            101: "<channel|>", 106: "<turn|>",
        ]
        var next: Int32 = 200
        func add(_ text: String) {
            pieces[next] = text
            next += 1
        }
        for byte in byteFallbackBytes { add(String(format: "<0x%02X>", byte)) }
        // The canonical-JSON drift that causes the upstream failures.
        for text in ["{\"", "\":", "\", \"", "\"", "\":\"", "\"}"] { add(text) }
        // Grammar symbols, including the delimiter spelled in fragments.
        for text in ["call", "call:", ":", "{", "}", "[", "]", ",",
                     "<", "<|", "<|\"", "|>", "\"|>", "<|\"|>"] { add(text) }
        // Numbers and number traps.
        for text in ["0", "01", "1", "9", ".", "-", "+", "e", "E",
                     "1e999", "123456789012345678", "0.5"] { add(text) }
        // Literals and literal traps.
        for text in ["true", "truex", "false", "null", "nulll", "NaN", "True"] { add(text) }
        // Escapes and escape traps.
        for text in ["\\", "\\u", "\\n", "d83d", "dc00", "d800", "ffff", "zz",
                     "0600", "u0600"] { add(text) }
        // Whitespace, including a multi-byte one the parser skips but the
        // dialect does not accept.
        for text in ["\u{2581}", "\n", "\t", "\u{00A0}"] { add(text) }
        // Declared names and near misses.
        for text in ["get-weather", "get", "get_all", "read", "read-file",
                     "weather", "getweather", "reader", "file", "-weather",
                     "_all", "city", "path", "units"] { add(text) }
        // Junk markers the detokenizer would strip.
        for text in ["<unused12>", "<td>", "<|tool>", "<start_of_image>"] { add(text) }
        // Ordinary prose.
        for text in ["Hola", " I will call", "\u{1F601}", "ñ"] { add(text) }
        return pieces
    }()

    private static let ids: [Int32] = pieces.keys.sorted()

    /// The byte mapping `TokenByteTable` performs, mirrored here so the walk can
    /// keep an independent copy of the region and cross-check it against
    /// `closedCallBody`.
    private static func contentBytes(_ id: Int32) -> [UInt8]? {
        guard let piece = pieces[id] else { return nil }
        if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
           let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16) {
            return [byte]
        }
        if id != markerIDs.escape, piece.count > 2,
           piece.hasPrefix("<"), piece.hasSuffix(">"),
           !piece.dropFirst().dropLast().contains(where: { $0 == "<" || $0 == ">" }) {
            return nil
        }
        return Array(piece.replacingOccurrences(of: "\u{2581}", with: " ").utf8)
    }

    /// True when the region ends inside an unterminated `<|"|>` string. Outside
    /// a string the dialect admits `<` only as a delimiter opener and `\` only
    /// inside one, so this left-to-right scan is exact.
    private static func endsInsideGemmaString(_ region: [UInt8]) -> Bool {
        let delimiter = Array(#"<|"|>"#.utf8)
        var index = 0
        var inside = false
        while index < region.count {
            if inside, region[index] == UInt8(ascii: "\\") {
                index += 2
                continue
            }
            if region[index...].starts(with: delimiter) {
                inside.toggle()
                index += delimiter.count
                continue
            }
            index += 1
        }
        return inside
    }

    // MARK: - Walk

    private struct Coverage {
        var walks = 0
        var walksWithCall = 0
        var closedCalls = 0
        var nameVetoes = 0
        var argumentVetoes = 0
        var vetoesInsideGemmaString = 0
        var exhaustiveSweeps = 0
        var bodiesWithGemmaString = 0
        var bodiesWithNesting = 0
        var bodiesWithNumber = 0
        var bodiesWithMultibyteScalar = 0
    }

    private static let maximumSteps = 96
    private static let randomRetries = 32

    private func walk(index: Int, coverage: inout Coverage) {
        let seed = 0x9E37_79B9_7F4A_7C15 &* UInt64(index) &+ 1
        var rng = SplitMix64(seed: seed)
        // The real fallback walks the vocabulary in probability order; a
        // per-walk shuffle stands in for it, so the sweep does not always land
        // on the same lowest id and the traces stay diverse.
        let order = Self.ids.shuffled(using: &rng)
        let filter = ToolCallTokenFilter(pieceLookup: { Self.pieces[$0] },
                                         markers: Self.markerIDs,
                                         allowedNames: Self.declaredNames)
        var trace: [Int32] = []
        var region: [UInt8] = []
        var openRegion = false
        var justClosed = false
        var closedCalls = 0

        func fail(_ message: String) {
            Issue.record("\(message) seed=\(seed) trace=\(trace)")
        }

        // Half the walks open with the marker so the region grammar — not the
        // free-prose passthrough — is what the budget is spent on.
        var forcedOpener: Int32? = index.isMultiple(of: 2) ? Self.markerIDs.toolCallStart : nil

        for _ in 0..<Self.maximumSteps {
            var chosen: Int32?
            if let forced = forcedOpener {
                forcedOpener = nil
                guard filter.tryAccept(forced) else {
                    fail("the opening marker was vetoed in free prose")
                    return
                }
                chosen = forced
            } else {
                let proposal = order.randomElement(using: &rng)!
                if filter.tryAccept(proposal) {
                    chosen = proposal
                } else {
                    let namesBefore = filter.nameVetoCount
                    let insideString = openRegion && Self.endsInsideGemmaString(region)
                    filter.noteVeto()
                    if filter.nameVetoCount > namesBefore {
                        coverage.nameVetoes += 1
                    } else if openRegion {
                        coverage.argumentVetoes += 1
                        if insideString { coverage.vetoesInsideGemmaString += 1 }
                    }
                }
            }
            if chosen == nil {
                for _ in 0..<Self.randomRetries {
                    let candidate = order.randomElement(using: &rng)!
                    if filter.tryAccept(candidate) {
                        chosen = candidate
                        break
                    }
                }
            }
            if chosen == nil {
                coverage.exhaustiveSweeps += 1
                for candidate in order where filter.tryAccept(candidate) {
                    chosen = candidate
                    break
                }
            }
            // Property 2: the fallback scan can never come back empty, which is
            // what keeps `grammarFallbackToken` from throwing.
            guard let token = chosen else {
                fail("no token in the vocabulary can extend the region")
                return
            }
            trace.append(token)

            // Property 4: marker invariants.
            if justClosed, token != Self.markerIDs.toolCallStart,
               token != Self.markerIDs.toolResponse {
                fail("free content followed <tool_call|>")
                return
            }
            justClosed = false

            if token == Self.markerIDs.toolCallStart {
                if openRegion {
                    fail("nested <|tool_call>")
                    return
                }
                openRegion = true
                region = []
            } else if token == Self.markerIDs.toolCallEnd {
                guard openRegion else {
                    fail("<tool_call|> with no region open")
                    return
                }
                guard let body = filter.closedCallBody else {
                    fail("the closing token produced no validated body")
                    return
                }
                if body != region {
                    fail("validated body diverged from the mirrored region")
                    return
                }
                let text = String(decoding: body, as: UTF8.self)
                // Property 3: the bytes are valid UTF-8, so no U+FFFD was spliced in.
                if Array(text.utf8) != body {
                    fail("the region is not valid UTF-8: \(body)")
                    return
                }
                // Property 1: the grammar is a subset of the parser.
                do {
                    let call = try GemmaToolCallParser().parse(
                        text, allowedTools: Self.declaredNames, id: "fuzz")
                    if !Self.declaredNames.contains(call.name) {
                        fail("undeclared tool \(call.name)")
                        return
                    }
                } catch {
                    fail("parse failed on \(String(reflecting: text)): \(error)")
                    return
                }
                openRegion = false
                justClosed = true
                closedCalls += 1
                coverage.closedCalls += 1
                if body.contains(UInt8(ascii: "<")) { coverage.bodiesWithGemmaString += 1 }
                if body.filter({ $0 == UInt8(ascii: "{") }).count > 1
                    || body.contains(UInt8(ascii: "[")) {
                    coverage.bodiesWithNesting += 1
                }
                if body.contains(where: { (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0) }) {
                    coverage.bodiesWithNumber += 1
                }
                if body.contains(where: { $0 >= 0x80 }) { coverage.bodiesWithMultibyteScalar += 1 }
                region = []
                if closedCalls == 2 { break }
            } else if token == Self.markerIDs.toolResponse {
                if closedCalls == 0 {
                    fail("<|tool_response> before any call closed")
                    return
                }
                break
            } else if token == Self.markerIDs.eos || token == Self.markerIDs.endOfTurn {
                break
            } else if openRegion {
                guard let bytes = Self.contentBytes(token) else {
                    fail("a marker-shaped piece entered the region: \(token)")
                    return
                }
                region.append(contentsOf: bytes)
            }
        }
        coverage.walks += 1
        if closedCalls > 0 { coverage.walksWithCall += 1 }
    }

    @Test(.timeLimit(.minutes(3)))
    func grammarIsASubsetOfTheParser() {
        let requested = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_FUZZ_WALKS"]
            .flatMap(Int.init)
        let walks = max(requested ?? 5_000, 5_000)
        var coverage = Coverage()
        for index in 0..<walks {
            walk(index: index, coverage: &coverage)
        }
        #expect(coverage.walks == walks)
        // Anti-vacuity: a fuzz that never reaches the interesting states proves
        // nothing, so the shape of the exploration is asserted too.
        #expect(coverage.walksWithCall >= 500)
        #expect(coverage.closedCalls >= 500)
        #expect(coverage.nameVetoes >= 1)
        #expect(coverage.argumentVetoes >= 1)
        #expect(coverage.exhaustiveSweeps >= 1)
        #expect(coverage.vetoesInsideGemmaString >= 1)
        #expect(coverage.bodiesWithGemmaString >= 1)
        #expect(coverage.bodiesWithNesting >= 1)
        #expect(coverage.bodiesWithNumber >= 1)
        #expect(coverage.bodiesWithMultibyteScalar >= 1)
    }

    /// The walks stop at 96 steps, so they never come within three orders of
    /// magnitude of the 256 KiB region cap — which is why property 2 above
    /// reads as "the scan is never empty" when the truth is "never empty below
    /// the cap". At the cap it IS empty, unavoidably: every token overflows it
    /// and `<tool_call|>` cannot close a string that is still open. That state
    /// is the one dead end the grammar has, and it is admissible only because
    /// `runRawCompletion` reads it as the budget running out inside a call —
    /// hence the `isInsideCall` assertion, which is what routes it to the
    /// `.maxTokens` stop instead of a failed request.
    @Test func theRegionByteCapIsTheGrammarsOnlyDeadEnd() {
        let filler: Int32 = 9_000
        let opening = #"call:read{path:<|"|>"#
        let chunk = (GemmaToolCallParser.maximumBytes - opening.utf8.count) / 4
        #expect(opening.utf8.count + 4 * chunk == GemmaToolCallParser.maximumBytes)

        func fill(_ count: Int) -> ToolCallTokenFilter {
            let filter = ToolCallTokenFilter(
                pieceLookup: { $0 == filler ? String(repeating: "a", count: chunk)
                                            : Self.pieces[$0] },
                markers: Self.markerIDs,
                allowedNames: Self.declaredNames)
            for id in [Self.markerIDs.toolCallStart, Self.id("call:"), Self.id("read"),
                       Self.id("{"), Self.id("path"), Self.id(":"), Self.markerIDs.escape] {
                #expect(filter.tryAccept(id))
            }
            for step in 0..<count {
                #expect(filter.tryAccept(filler), "filler \(step) did not fit under the cap")
            }
            return filter
        }

        // One filler short of the cap the sweep still finds a move, so the cap
        // is what closes the door and not the argument shape.
        let roomy = fill(3)
        #expect(Self.ids.contains { roomy.tryAccept($0) })

        let stuck = fill(4)
        #expect(!Self.ids.contains { stuck.tryAccept($0) })
        #expect(!stuck.tryAccept(filler))
        #expect(stuck.isInsideCall)
    }

    private static func id(_ piece: String) -> Int32 {
        pieces.first { $0.value == piece }!.key
    }

    /// Seeds that previously exposed a bug stay pinned as ordinary cases.
    @Test(arguments: [0, 1, 2, 3, 7, 42, 4_999])
    func pinnedSeedsReplayCleanly(index: Int) {
        var coverage = Coverage()
        walk(index: index, coverage: &coverage)
        #expect(coverage.walks == 1)
    }
}
