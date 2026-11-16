import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite("Tokenizer load coordinator")
struct TokenizerLoadCoordinatorTests {
    @Test("Concurrent loads return equivalent tokenizer metadata")
    func concurrentLoadsShareLoadedTokenizer() async throws {
        let tokenizers = try await withThrowingTaskGroup(of: GFTokenizer.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
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
            #expect(
                tokenizer.encode("The capital of France is", addBOS: true)
                    == first.encode("The capital of France is", addBOS: true))
        }
    }

    @Test("Consecutive loads reuse the completed process cache")
    func consecutiveLoadsReuseCompletedTask() async throws {
        let first = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let second = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())

        #expect(second.bosID == first.bosID)
        #expect(second.eosID == first.eosID)
        #expect(second.endOfTurnID == first.endOfTurnID)
        #expect(second.decode(first.encode("cache check", addBOS: false)) == "cache check")
    }

    @Test("Model tokenizer sidecar is discovered")
    func modelTokenizerSidecarIsDiscovered() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.ssdai", isDirectory: true)
        let modelTokenizer = model.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelTokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: modelTokenizer.appendingPathComponent("tokenizer.json"))

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == modelTokenizer.standardizedFileURL)
    }

    @Test("Missing model tokenizer sidecar returns nil")
    func missingModelTokenizerSidecarReturnsNil() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.ssdai", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == nil)
    }

    @Test("A missing model directory names the path, not the tokenizer template")
    func missingModelDirectoryNamesThePath() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-such-model-\(UUID().uuidString)", isDirectory: true)

        do {
            _ = try await GFTokenizer.load(forModelDirectory: missing)
            Issue.record("loading a model directory that does not exist should fail")
        } catch let error as GFTokenizerError {
            #expect(
                error.description
                    == "model directory not found: \(missing.standardizedFileURL.path)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("A model path that is a regular file is refused by name")
    func modelPathThatIsAFileIsRefused() async throws {
        let root = try temporaryDirectory()
        let file = root.appendingPathComponent("not-a-directory.ssdai")
        try Data("{}".utf8).write(to: file)

        do {
            _ = try await GFTokenizer.load(forModelDirectory: file)
            Issue.record("loading a regular file as a model directory should fail")
        } catch let error as GFTokenizerError {
            #expect(
                error.description
                    == "model directory not found: \(file.standardizedFileURL.path)")
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
