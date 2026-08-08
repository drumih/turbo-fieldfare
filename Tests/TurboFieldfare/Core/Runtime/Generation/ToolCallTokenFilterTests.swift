import Testing
import Foundation
@testable import TurboFieldfare

/// Token-level tool-call grammar over a fake vocabulary: marker discipline,
/// name forcing through the trie, the argument dialect, and the byte handoff to
/// the decoder. Same fake-piece pattern as `JSONTokenFilterTests`.
@Suite struct ToolCallTokenFilterTests {

    private enum ID {
        static let bos: Int32 = 2
        static let eos: Int32 = 1
        static let pad: Int32 = 0
        static let toolCallStart: Int32 = 48
        static let toolCallEnd: Int32 = 49
        static let toolResponse: Int32 = 50
        static let toolResponseEnd: Int32 = 51
        static let escape: Int32 = 52
        static let channelStart: Int32 = 100
        static let channelEnd: Int32 = 101
        static let endOfTurn: Int32 = 106
    }

    private static let markers = ToolCallMarkerIDs(
        toolCallStart: ID.toolCallStart,
        toolCallEnd: ID.toolCallEnd,
        toolResponse: ID.toolResponse,
        toolResponseEnd: ID.toolResponseEnd,
        channelStart: ID.channelStart,
        channelEnd: ID.channelEnd,
        escape: ID.escape,
        endOfTurn: ID.endOfTurn,
        eos: ID.eos,
        bos: ID.bos,
        pad: ID.pad)

    private static let pieces: [Int32: String] = [
        ID.pad: "<pad>",
        ID.eos: "<eos>",
        ID.bos: "<bos>",
        ID.toolCallStart: "<|tool_call>",
        ID.toolCallEnd: "<tool_call|>",
        ID.toolResponse: "<|tool_response>",
        ID.toolResponseEnd: "<tool_response|>",
        ID.escape: "<|\"|>",
        ID.channelStart: "<|channel>",
        ID.channelEnd: "<channel|>",
        ID.endOfTurn: "<turn|>",
        200: "call:",
        201: "call",
        202: ":",
        203: "get-weather",
        204: "get",
        205: "get_all",
        206: "getweather",
        207: "weather{",
        208: "{",
        209: "}",
        210: "city",
        211: "[",
        212: "]",
        213: ",",
        214: "1",
        215: "0",
        216: "01",
        217: "1e5",
        218: "Rosario",
        219: "\"",
        220: "{\"",
        221: "\":",
        222: "l:",
        223: "<unused12>",
        224: "Hola",
        225: "<0xD8>",
        226: "<0x80>",
        227: "x",
        228: "true",
        229: "<0xF0>",
        230: "<0x9F>",
        231: "<0x98>",
        232: "<0x81>",
        233: "<",
        234: "|",
        235: "\"|>",
        236: "9999999999999999",
        237: "▁",
        238: "call:get-weather{",
        239: "}\u{2581}",
    ]

    private func makeFilter(
        names: Set<String> = ["get-weather", "get", "get_all"]
    ) -> ToolCallTokenFilter {
        ToolCallTokenFilter(pieceLookup: { Self.pieces[$0] },
                            markers: Self.markers,
                            allowedNames: names)
    }

    private func accept(_ filter: ToolCallTokenFilter, _ ids: [Int32]) {
        for id in ids {
            #expect(filter.tryAccept(id), "rejected \(id) (\(Self.pieces[id] ?? "?"))")
        }
    }

    // MARK: - Free prose

    @Test func proseIsUnconstrained() {
        let filter = makeFilter()
        accept(filter, [224, 237, 223, ID.channelStart, 224, ID.channelEnd, ID.escape])
        #expect(filter.tryAccept(ID.endOfTurn))
        #expect(filter.tryAccept(ID.eos))
        #expect(!filter.isInsideCall)
    }

    @Test func strayClosingMarkersAreVetoedInProse() {
        let filter = makeFilter()
        #expect(!filter.tryAccept(ID.toolCallEnd))
        #expect(!filter.tryAccept(ID.toolResponseEnd))
        // The orphan tool response the server currently turns into a 500.
        #expect(!filter.tryAccept(ID.toolResponse))
    }

    @Test func noDeclaredToolsBlocksTheBlockEntirely() {
        let filter = makeFilter(names: [])
        #expect(!filter.tryAccept(ID.toolCallStart))
        #expect(filter.tryAccept(224))
    }

    @Test func nonASCIINamesAreDroppedRatherThanForced() {
        let filter = makeFilter(names: ["hérramienta", "bad name", ""])
        #expect(!filter.tryAccept(ID.toolCallStart))
    }

    // MARK: - Name forcing

    @Test func undeclaredNameIsVetoedAndCountedSeparately() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200])
        #expect(!filter.tryAccept(206))          // getweather
        filter.noteVeto()
        #expect(filter.nameVetoCount == 1)
        #expect(filter.vetoCount == 1)
        accept(filter, [203, 208, 209])          // get-weather{}
        #expect(filter.tryAccept(ID.toolCallEnd))
        #expect(filter.emittedCalls == 1)
    }

    @Test func sharedPrefixNamesAreBothReachable() {
        for name in ["get", "get_all"] {
            let filter = makeFilter()
            accept(filter, [ID.toolCallStart, 200])
            let id: Int32 = name == "get" ? 204 : 205
            accept(filter, [id, 208, 209])
            #expect(filter.tryAccept(ID.toolCallEnd))
            #expect(String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
                        == "call:\(name){}")
        }
    }

    @Test func piecesMayStraddlePhaseBoundaries() throws {
        // "call" + ":" splits the prefix; one piece can also carry the prefix,
        // the whole name and the opening brace; and `}` + the space marker
        // crosses out of the argument object into trailing whitespace.
        let spellings: [[Int32]] = [
            [201, 202, 203, 208, 209],
            [238, 209],
            [200, 203, 208, 239],
        ]
        for spelling in spellings {
            let filter = makeFilter()
            accept(filter, [ID.toolCallStart])
            accept(filter, spelling)
            #expect(filter.tryAccept(ID.toolCallEnd))
            let body = String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
            let call = try GemmaToolCallParser().parse(
                body, allowedTools: ["get-weather"], id: "x")
            #expect(call.name == "get-weather")
        }
    }

    @Test func namePieceMayNotOvershootTheDeclaredSet() {
        let filter = makeFilter(names: ["weather"])
        accept(filter, [ID.toolCallStart, 200])
        #expect(filter.tryAccept(207))           // "weather{" — terminal then `{`
        accept(filter, [209])
        #expect(filter.tryAccept(ID.toolCallEnd))
    }

    // MARK: - Argument dialect at token level

    @Test func canonicalJSONDerailIsVetoed() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203])
        #expect(!filter.tryAccept(220))          // `{"` — quoted key
        accept(filter, [208])
        #expect(!filter.tryAccept(219))          // bare `"` in key position
        accept(filter, [210, 202])               // city:
        #expect(!filter.tryAccept(219))          // `"` in value position
        #expect(!filter.tryAccept(216))          // 01
        #expect(!filter.tryAccept(217))          // 1e5
        #expect(!filter.tryAccept(236))          // 16 integer digits
        accept(filter, [214, 213])               // 1,
        #expect(!filter.tryAccept(209))          // trailing comma before `}`
        #expect(!filter.tryAccept(219))          // and still no quoted key
        accept(filter, [210, 202, 214, 209])
        #expect(filter.tryAccept(ID.toolCallEnd))
    }

    @Test func delimiterIsAcceptedAsOneTokenOrFivePieces() {
        for spelling: [Int32] in [[ID.escape], [233, 234, 235]] {
            let filter = makeFilter()
            accept(filter, [ID.toolCallStart, 200, 203, 208, 210, 202])
            accept(filter, spelling)
            accept(filter, [218])
            accept(filter, spelling)
            accept(filter, [209])
            #expect(filter.tryAccept(ID.toolCallEnd))
            #expect(String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
                        == #"call:get-weather{city:<|"|>Rosario<|"|>}"#)
        }
    }

    @Test func byteFallbackSequenceCrossesTheDelimiter() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 210, 202, ID.escape])
        accept(filter, [229, 230, 231, 232])     // 😀 one byte-fallback at a time
        // Mid-sequence the delimiter cannot start.
        accept(filter, [229])
        #expect(!filter.tryAccept(ID.escape))
        accept(filter, [230, 231, 232, ID.escape, 209])
        #expect(filter.tryAccept(ID.toolCallEnd))
        #expect(String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
                    == "call:get-weather{city:<|\"|>\u{1F601}\u{1F601}<|\"|>}")
    }

    @Test func prependScalarCannotHideTheClosingDelimiter() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 210, 202, ID.escape])
        accept(filter, [225, 226])               // U+0600, Prepend
        #expect(!filter.tryAccept(ID.escape))
        #expect(!filter.tryAccept(233))          // bare `<`
        accept(filter, [227, ID.escape, 209])    // any content clears the guard
        #expect(filter.tryAccept(ID.toolCallEnd))
    }

    // MARK: - Marker discipline inside a region

    @Test func markersInsideARegionAreVetoed() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208])
        for marker in [ID.toolCallStart, ID.toolResponse, ID.toolResponseEnd,
                       ID.channelStart, ID.channelEnd, ID.endOfTurn,
                       ID.eos, ID.bos, ID.pad] {
            #expect(!filter.tryAccept(marker), "marker \(marker) leaked into a region")
        }
        // `<tool_call|>` too, until the object closes.
        #expect(!filter.tryAccept(ID.toolCallEnd))
        accept(filter, [209])
        #expect(filter.tryAccept(ID.toolCallEnd))
    }

    @Test func afterACallOnlyTheTwoContinuationsPass() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 209, ID.toolCallEnd])
        for rejected in [ID.toolCallEnd, ID.toolResponseEnd, ID.channelStart,
                         ID.channelEnd, ID.endOfTurn, ID.eos, 224] {
            #expect(!filter.tryAccept(rejected))
        }
        #expect(filter.tryAccept(ID.toolResponse))
        #expect(filter.tryAccept(ID.toolCallStart))
    }

    @Test func toolResponseIsLegalOnceACallExists() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 209, ID.toolCallEnd,
                        ID.toolResponse])
        #expect(filter.emittedCalls == 1)
    }

    // MARK: - Handoff and bookkeeping

    @Test func closedCallBodyOnlyLandsOnTheClosingToken() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 209])
        #expect(filter.closedCallBody == nil)
        #expect(filter.tryAccept(ID.toolCallEnd))
        #expect(String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
                    == "call:get-weather{}")
        #expect(filter.tryAccept(ID.toolResponse))
        #expect(filter.closedCallBody == nil)
    }

    @Test func closedBodyParsesAsTheDeclaredTool() throws {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 210, 202, ID.escape,
                        218, ID.escape, 213, 210, 202, 228, 209])
        #expect(filter.tryAccept(ID.toolCallEnd))
        let body = String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
        let call = try GemmaToolCallParser().parse(
            body, allowedTools: ["get-weather", "get", "get_all"], id: "x")
        #expect(call.name == "get-weather")
    }

    @Test func rejectionDoesNotAdvanceState() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203, 208, 210])
        #expect(!filter.tryAccept(220))
        #expect(!filter.tryAccept(219))
        // The key is still open, so the colon still lands.
        accept(filter, [202, 214, 209])
        #expect(filter.tryAccept(ID.toolCallEnd))
        #expect(String(decoding: filter.closedCallBody ?? [], as: UTF8.self)
                    == "call:get-weather{city:1}")
    }

    /// A token that clears the trie and then breaks the argument grammar says
    /// nothing about the tool name, even though the committed phase is `.name`.
    @Test func aVetoPastTheNameIsNotANameVeto() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart, 200, 203])
        #expect(!filter.tryAccept(220))          // `{"` — enters, then dies
        filter.noteVeto()
        #expect(filter.vetoCount == 1)
        #expect(filter.nameVetoCount == 0)
        #expect(!filter.tryAccept(206))          // getweather — dies in the trie
        filter.noteVeto()
        #expect(filter.nameVetoCount == 1)
    }

    @Test func vetoCountsAttributePhaseCorrectly() {
        let filter = makeFilter()
        accept(filter, [ID.toolCallStart])
        filter.noteVeto()                        // prefix
        accept(filter, [200, 203, 208])
        filter.noteVeto()                        // arguments
        #expect(filter.vetoCount == 2)
        #expect(filter.nameVetoCount == 1)
    }

    @Test func isInsideCallTracksTheRegion() {
        let filter = makeFilter()
        #expect(!filter.isInsideCall)
        accept(filter, [ID.toolCallStart])
        #expect(filter.isInsideCall)
        accept(filter, [200, 203, 208, 209])
        #expect(filter.isInsideCall)
        #expect(filter.tryAccept(ID.toolCallEnd))
        #expect(!filter.isInsideCall)
    }

    /// The region body is exactly what the parser will see, so capping it here
    /// makes `GemmaToolCallParserError.oversized` unreachable.
    @Test func regionBytesAreCapped() {
        let filler = String(repeating: "a", count: 1024)
        let big: Int32 = 900
        let filter = ToolCallTokenFilter(
            pieceLookup: { id in id == big ? filler : Self.pieces[id] },
            markers: Self.markers,
            allowedNames: ["get-weather"])
        accept(filter, [ID.toolCallStart, 200, 203, 208, 210, 202, ID.escape])
        var accepted = 0
        while filter.tryAccept(big) { accepted += 1 }
        #expect(accepted * filler.utf8.count <= GemmaToolCallParser.maximumBytes)
        #expect(accepted > 200)
    }
}
