import Foundation
import Testing

@testable import TinyTitan

/// The GGUF carries its tokenizer as metadata; this checks the sidecar the
/// exporter writes is one the runtime can actually load, and that it tokenizes
/// the same text the way llama.cpp does.
///
/// The reference ids come from `llama-tokenize --no-bos --ids` at the ggml
/// commit named in the fixture. Like the dequant vectors, this is an external
/// oracle: a tokenizer compared only against itself would agree with itself and
/// prove nothing.
@Suite struct GGUFTokenizerTests {
    private struct ReferenceCase: Decodable {
        let text: String
        let ids: [Int]
    }

    private struct ReferenceDocument: Decodable {
        let reference: String
        let cases: [ReferenceCase]
    }

    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func extractedSidecarLoadsAndMatchesLlamaTokenize() async throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let directory = temporary.appendingPathComponent("tokenizer")
        try GGUFTokenizerSidecar.write(from: model, to: directory)

        let templateURL = directory.appendingPathComponent("chat_template.jinja")
        let written = try String(contentsOf: templateURL, encoding: .utf8)
        #expect(written == (try GGUFTokenizerSidecar.chatTemplate(from: model)))

        let raw = try Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))
        let object = try JSONSerialization.jsonObject(with: raw) as? [String: Any]
        let added = try #require(object?["added_tokens"] as? [[String: Any]])
        func entry(_ content: String) throws -> [String: Any] {
            try #require(added.first { ($0["content"] as? String) == content })
        }
        // The runtime's streaming decoder needs the ChatML barriers literal and
        // the stop tokens special; both come from GGML's token types.
        #expect(try entry("<tool_call>")["special"] as? Bool == false)
        #expect(try entry("<tool_response>")["special"] as? Bool == false)
        #expect(try entry("<think>")["special"] as? Bool == false)
        #expect(try entry("<|im_end|>")["special"] as? Bool == true)
        #expect(try entry("<|endoftext|>")["special"] as? Bool == true)

        let fixture = try #require(
            Bundle.module.url(forResource: "GGUFTokenizerVectors", withExtension: "json"))
        let document = try JSONDecoder().decode(
            ReferenceDocument.self, from: try Data(contentsOf: fixture))
        #expect(document.cases.count >= 8)

        let tokenizer = try await GFTokenizer.load(from: directory)
        for reference in document.cases {
            let ids = tokenizer.encode(reference.text).map(Int.init)
            #expect(
                ids == reference.ids,
                "\(reference.text.debugDescription): got \(ids), expected \(reference.ids)")
        }
    }
}
