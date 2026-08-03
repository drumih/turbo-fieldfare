import Foundation
import Testing
@testable import TurboFieldfareServerCore

@Suite struct ServerDocumentExtractorTests {
    @Test func inlineTextIsReturned() throws {
        let text = try ServerDocumentExtractor.text(
            for: ServerDocumentReference(text: "  hello world  "))
        #expect(text == "hello world")
    }

    @Test func base64TextIsDecoded() throws {
        let base64 = Data("encoded content".utf8).base64EncodedString()
        let text = try ServerDocumentExtractor.text(
            for: ServerDocumentReference(base64: base64))
        #expect(text == "encoded content")
    }

    @Test func pathReadsTextFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("notes.txt")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("file content".utf8).write(to: url)

        let text = try ServerDocumentExtractor.text(
            for: ServerDocumentReference(path: url.path))
        #expect(text == "file content")
    }

    @Test func binaryFormatsAreRefused() {
        #expect(throws: ServerDocumentExtractor.ExtractionError.self) {
            _ = try ServerDocumentExtractor.text(
                for: ServerDocumentReference(path: "/tmp/report.pdf"))
        }
    }

    @Test func missingSourcesAreRejected() {
        #expect(throws: ServerDocumentExtractor.ExtractionError.self) {
            _ = try ServerDocumentExtractor.text(for: ServerDocumentReference())
        }
    }

    @Test func multipleSourcesAreRejected() {
        #expect(throws: ServerDocumentExtractor.ExtractionError.self) {
            _ = try ServerDocumentExtractor.text(
                for: ServerDocumentReference(path: "/tmp/x.txt", text: "a"))
        }
    }

    @Test func emptyTextIsRejected() {
        #expect(throws: ServerDocumentExtractor.ExtractionError.self) {
            _ = try ServerDocumentExtractor.text(
                for: ServerDocumentReference(text: "   "))
        }
    }
}
