import Foundation
import NIOCore
import Testing
@testable import TurboFieldfareServerCore

@Suite struct StreamingChatRequestBodyTests {
    @Test func extractsImageBase64AcrossArbitraryChunks() throws {
        let source = Data(#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj","detail":"auto"}},{"type":"text","text":"describe"}]}]}"#.utf8)
        let parser = StreamingChatRequestBody()
        var offset = 0
        for size in [1, 2, 3, 5, 8, 13, 21] {
            while offset < source.count {
                let end = min(source.count, offset + size)
                var buffer = ByteBuffer(bytes: source[offset..<end])
                try parser.feed(&buffer)
                offset = end
                if end == source.count { break }
            }
            if offset == source.count { break }
        }
        let parsed = try parser.finish()
        #expect(!String(decoding: parsed.json, as: UTF8.self).contains("YWJj"))
        let staged = try #require(parsed.stagedImages.values.first)
        #expect(try Data(contentsOf: staged.fileURL) == Data("abc".utf8))

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: parsed.json)
        let validated = try OpenAIRequestValidator.validate(
            request,
            modelID: "m",
            preStagedImages: parsed.stagedImages,
            attachmentLease: parsed.lease)
        #expect(validated.multimodalMessages?.count == 1)
        #expect(validated.imageFiles.count == 1)
    }

    /// A spec-valid serializer may escape the solidus (PHP's json_encode does
    /// by default) or write payload characters as \u escapes; both must stage
    /// the same bytes as the unescaped form.
    @Test func decodesJSONEscapesInsideImageDataURLs() throws {
        let source = Data(#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image\/png;base64,YWJj"}}]}]}"#.utf8)
        let parser = StreamingChatRequestBody()
        var buffer = ByteBuffer(bytes: source)
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        let staged = try #require(parsed.stagedImages.values.first)
        #expect(try Data(contentsOf: staged.fileURL) == Data("abc".utf8))

        let unicodeJSON = #"{"model":"m","messages":[{"role":"user","content":"#
            + #"[{"type":"image_url","image_url":{"url":"data:image/png;base64,\u0059WJj"}}]}]}"#
        let unicodeSource = Data(unicodeJSON.utf8)
        let unicodeParser = StreamingChatRequestBody()
        var unicodeBuffer = ByteBuffer(bytes: unicodeSource)
        try unicodeParser.feed(&unicodeBuffer)
        let unicodeParsed = try unicodeParser.finish()
        let unicodeStaged = try #require(unicodeParsed.stagedImages.values.first)
        #expect(try Data(contentsOf: unicodeStaged.fileURL) == Data("abc".utf8))
    }

    /// Padding must terminate the payload, and the answer must not depend on
    /// how the wire happened to chunk the bytes: mid-stream padding is always
    /// rejected, terminal padding always accepted.
    @Test func base64PaddingHandlingIsChunkingIndependent() throws {
        func parse(_ payload: String, chunk: Int) throws -> ParsedChatRequestBody {
            let source = Data(#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,\#(payload)"}}]}]}"#.utf8)
            let parser = StreamingChatRequestBody()
            var offset = 0
            while offset < source.count {
                let end = min(source.count, offset + chunk)
                var buffer = ByteBuffer(bytes: source[offset..<end])
                try parser.feed(&buffer)
                offset = end
            }
            return try parser.finish()
        }
        for chunk in [1, 3, 7, 64, 4_096] {
            #expect(throws: ServerRequestError.self, "chunk \(chunk)") {
                _ = try parse("AA==AAAA", chunk: chunk)
            }
            let parsed = try parse("QUJDRA==", chunk: chunk)
            let staged = try #require(parsed.stagedImages.values.first)
            #expect(try Data(contentsOf: staged.fileURL) == Data("ABCD".utf8))
        }
    }

    /// A non-data URL with escapes is not staged and must pass through with
    /// its original bytes, not the decoded ones.
    @Test func escapedNonDataURLsPassThroughVerbatim() throws {
        let source = Data(#"{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https:\/\/example.com\/a.png"}}]}]}"#.utf8)
        let parser = StreamingChatRequestBody()
        var buffer = ByteBuffer(bytes: source)
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        #expect(parsed.json == source)
        #expect(parsed.stagedImages.isEmpty)
    }

    /// An image_url-shaped object outside messages[].content[] — here inside a
    /// tool schema — is someone else's data and must not be staged, rewritten,
    /// or rejected.
    @Test func imageURLShapesOutsideMessageContentAreUntouched() throws {
        let source = Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","example":{"image_url":{"url":"data:image/png;base64,YWJj"}}}}}]}"#.utf8)
        let parser = StreamingChatRequestBody()
        var buffer = ByteBuffer(bytes: source)
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        #expect(parsed.json == source)
        #expect(parsed.stagedImages.isEmpty)
    }

    @Test func leavesDataURLTextUntouched() throws {
        let source = Data(#"{"model":"m","messages":[{"role":"user","content":"data:image/png;base64,YWJj"}]}"#.utf8)
        let parser = StreamingChatRequestBody()
        var buffer = ByteBuffer(bytes: source)
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        #expect(parsed.json == source)
        #expect(parsed.stagedImages.isEmpty)
    }
}
