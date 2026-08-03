import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite @MainActor struct DocumentLibraryTests {
    private func makeLibrary() -> (DocumentLibrary, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("documents-\(UUID().uuidString).json")
        return (DocumentLibrary(storageURL: url), url)
    }

    private func attachment(filename: String, text: String = "body") -> DocumentAttachment {
        DocumentAttachment(filename: filename,
                           type: .txt,
                           fileSize: 10,
                           extractedText: text)
    }

    @Test func upsertReplacesByFilename() {
        let (library, _) = makeLibrary()
        library.upsert(attachment(filename: "a.txt"))
        library.upsert(attachment(filename: "b.txt"))

        #expect(library.documents.count == 2)

        library.upsert(attachment(filename: "a.txt", text: "updated"))
        #expect(library.documents.count == 2)
        #expect(library.document(named: "a.txt")?.extractedText == "updated")
    }

    @Test func lookupByName() {
        let (library, _) = makeLibrary()
        library.upsert(attachment(filename: "report.pdf"))

        #expect(library.document(named: "report.pdf") != nil)
        #expect(library.document(named: "missing.pdf") == nil)
    }

    @Test func persistsAcrossInstances() {
        let (library, url) = makeLibrary()
        library.upsert(attachment(filename: "notes.txt"))

        let reloaded = DocumentLibrary(storageURL: url)
        #expect(reloaded.documents.count == 1)
        #expect(reloaded.document(named: "notes.txt")?.extractedText == "body")
    }

    @Test func removeAndClear() {
        let (library, _) = makeLibrary()
        let first = attachment(filename: "a.txt")
        library.upsert(first)
        library.upsert(attachment(filename: "b.txt"))

        library.remove(first)
        #expect(library.documents.count == 1)

        library.clear()
        #expect(library.documents.isEmpty)
    }

    @Test func attachDocumentsCachesInLibrary() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("cached.txt")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("cache me".utf8).write(to: url)

        let model = AppModel(conversationStore: ConversationStore(
            storageURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("doc-\(UUID().uuidString).json")))
        model.attachDocuments(from: [url])
        // Wait for the detached extraction to finish.
        var waited = 0
        while model.isExtractingAttachment && waited < 5_000 {
            try? await Task.sleep(for: .milliseconds(20))
            waited += 20
        }

        #expect(model.documentLibrary.document(named: "cached.txt")?.extractedText == "cache me")
    }
}
