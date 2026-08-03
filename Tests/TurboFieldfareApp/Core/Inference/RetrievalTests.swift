import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct RetrievalTests {
    private let documentID = UUID()

    private func words(_ count: Int, seed: String = "word") -> String {
        (0..<count).map { "\(seed)\($0)" }.joined(separator: " ")
    }

    @Test func shortTextProducesSingleChunk() {
        let text = "A short document with just a few words to keep it small."
        let chunks = DocumentChunker.chunk(text, documentID: documentID)

        #expect(chunks.count == 1)
        #expect(chunks[0].text == text)
    }

    @Test func longTextProducesOverlappingChunks() {
        // 1500 words with a 600-word target produces 3 chunks.
        let text = words(1_500)
        let chunks = DocumentChunker.chunk(text, documentID: documentID)

        #expect(chunks.count == 3)
        // Overlap: the second chunk starts 540 words in, not at 600.
        #expect(chunks[1].text.hasPrefix("word540 "))
        // Every chunk belongs to the source document.
        #expect(chunks.allSatisfy { $0.sourceDocumentID == documentID })
    }

    @Test func retrieverRanksRelevantChunkFirst() {
        let text = """
        \(words(600, seed: "irrelevant"))
        metal compute pipeline shaders kernels
        \(words(600, seed: "noise"))
        """
        let chunks = DocumentChunker.chunk(text, documentID: documentID)

        let top = TfIdfRetriever.topChunks(chunks, query: "metal compute shaders", limit: 1)

        #expect(top.count == 1)
        #expect(top[0].text.contains("metal compute pipeline"))
    }

    @Test func retrieverFallsBackWhenQueryEmpty() {
        let chunks = [
            DocumentChunk(sourceDocumentID: documentID, text: "first"),
            DocumentChunk(sourceDocumentID: documentID, text: "second"),
        ]

        let top = TfIdfRetriever.topChunks(chunks, query: "   ", limit: 1)

        #expect(top.count == 1)
    }

    @Test func retrieverIgnoresEmptyChunks() {
        let top = TfIdfRetriever.topChunks([], query: "anything", limit: 5)
        #expect(top.isEmpty)
    }
}
