import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Tokenizer load coordinator")
struct TokenizerLoadCoordinatorTests {
    @Test("Concurrent loads return equivalent tokenizer metadata")
    func concurrentLoadsShareLoadedTokenizer() async throws {
        let tokenizers = try await withThrowingTaskGroup(of: GFTokenizer.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    try await GFTokenizer.load()
                }
            }

            var loaded: [GFTokenizer] = []
            loaded.reserveCapacity(8)
            for try await tokenizer in group {
                loaded.append(tokenizer)
            }
            return loaded
        }

        #expect(tokenizers.count == 8)
        let first = try #require(tokenizers.first)
        for tokenizer in tokenizers {
            #expect(tokenizer.bosID == first.bosID)
            #expect(tokenizer.eosID == first.eosID)
            #expect(tokenizer.padID == first.padID)
            #expect(tokenizer.endOfTurnID == first.endOfTurnID)
            #expect(tokenizer.encode("The capital of France is", addBOS: true)
                    == first.encode("The capital of France is", addBOS: true))
        }
    }

    @Test("Consecutive loads reuse the completed process cache")
    func consecutiveLoadsReuseCompletedTask() async throws {
        let first = try await GFTokenizer.load()
        let second = try await GFTokenizer.load()

        #expect(second.bosID == first.bosID)
        #expect(second.eosID == first.eosID)
        #expect(second.endOfTurnID == first.endOfTurnID)
        #expect(second.decode(first.encode("cache check", addBOS: false)) == "cache check")
    }

    @Test("Model tokenizer sidecar is discovered")
    func modelTokenizerSidecarIsDiscovered() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let modelTokenizer = model.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: modelTokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: modelTokenizer.appendingPathComponent("tokenizer.json"))

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == modelTokenizer.standardizedFileURL)
    }

    @Test("Missing model tokenizer sidecar returns nil")
    func missingModelTokenizerSidecarReturnsNil() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == nil)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
