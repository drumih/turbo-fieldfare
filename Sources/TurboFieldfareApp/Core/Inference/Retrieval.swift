import Foundation

/// A slice of an extracted document used for retrieval.
public struct DocumentChunk: Equatable, Sendable, Identifiable {
    public let id = UUID()
    public let sourceDocumentID: UUID
    public let text: String

    public init(sourceDocumentID: UUID, text: String) {
        self.sourceDocumentID = sourceDocumentID
        self.text = text
    }
}

/// Splits extracted document text into overlapping word chunks, so long
/// documents can be trimmed to the relevant parts instead of overflowing
/// the context window.
public enum DocumentChunker {
    /// Target words per chunk.
    public static let targetWordCount = 600
    /// Words of overlap between consecutive chunks.
    public static let overlapWordCount = 60

    public static func chunk(_ text: String,
                             documentID: UUID,
                             targetWordCount: Int = targetWordCount,
                             overlapWordCount: Int = overlapWordCount) -> [DocumentChunk] {
        let words = text.split(whereSeparator: { $0.isWhitespace })
        guard !words.isEmpty else { return [] }
        guard words.count > targetWordCount else {
            return [DocumentChunk(sourceDocumentID: documentID, text: text)]
        }

        let step = max(1, targetWordCount - overlapWordCount)
        var chunks: [DocumentChunk] = []
        var start = 0
        while start < words.count {
            let end = min(start + targetWordCount, words.count)
            let chunkWords = words[start..<end]
            chunks.append(DocumentChunk(
                sourceDocumentID: documentID,
                text: chunkWords.joined(separator: " ")))
            if end == words.count { break }
            start += step
        }
        return chunks
    }
}

/// Lightweight TF-IDF retriever over document chunks. Pure Swift, no
/// dependencies: enough to pick the chunks most relevant to a prompt when
/// the full documents would not fit in the context window.
public enum TfIdfRetriever {
    /// Returns the `limit` most relevant chunks for `query`, best first.
    public static func topChunks(_ chunks: [DocumentChunk],
                                 query: String,
                                 limit: Int) -> [DocumentChunk] {
        guard !chunks.isEmpty else { return [] }
        let chunkWordLists = chunks.map { tokenize($0.text) }
        let queryTerms = Set(tokenize(query))
        guard !queryTerms.isEmpty else { return Array(chunks.prefix(limit)) }

        // Inverse document frequency over the chunk collection.
        var documentFrequencies: [String: Int] = [:]
        for words in chunkWordLists {
            for term in Set(words) {
                documentFrequencies[term, default: 0] += 1
            }
        }
        let chunkCount = chunks.count
        func idf(_ term: String) -> Double {
            log(Double(chunkCount) / (1 + Double(documentFrequencies[term] ?? 0)))
        }

        // Score each chunk by the summed tf-idf of the query terms.
        var scored: [(index: Int, score: Double)] = []
        for (index, words) in chunkWordLists.enumerated() {
            let total = Double(max(1, words.count))
            let termFrequency = Dictionary(grouping: words, by: { $0 })
                .mapValues { Double($0.count) / total }
            var score = 0.0
            for term in queryTerms {
                score += (termFrequency[term] ?? 0) * idf(term)
            }
            if score > 0 {
                scored.append((index, score))
            }
        }
        scored.sort { $0.score > $1.score }
        return scored.prefix(limit).map { chunks[$0.index] }
    }

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 1 }
    }
}
