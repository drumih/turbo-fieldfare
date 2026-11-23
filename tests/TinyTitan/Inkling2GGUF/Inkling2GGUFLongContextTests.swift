import Foundation
import Testing

@testable import TinyTitan

/// Long-context greedy parity of the native Inkling2 GGUF decoder against
/// llama.cpp (TT-120): needle prompts of tens of thousands of tokens whose
/// answer is a record near the start, so it depends on the global-attention
/// visits reaching that far back.
///
/// The reference (`q2kxl-long-reference.json`, 27,074 and 56,687 tokens) was
/// captured with `benchmark/inkling2_gguf_reference.py --long` from the Unsloth
/// llama.cpp build at `-c 65536`. llama.cpp cannot run the Q4 file at these
/// lengths on a 24 GB Mac (Metal out of memory even at `-c 28672`), so there is
/// no Q4 reference: a Q4 run checks the prompt ids and that greedy decoding
/// retrieves the right record, and token parity is checked on Q2, whose
/// attention, positions and caches are the same code. It stores the generator's
/// inputs, not the prompt text, so `longPrompt` must stay byte-identical to the
/// script's `long_prompt`; the id count and hash pin that it does.
///
/// Model-gated like the short suite: `TINYTITAN_INKLING2_GGUF` names the
/// `.gguf` (Q2_K_XL for parity, Q4_K_XL for retrieval).
@Suite(.serialized) struct Inkling2GGUFLongContextTests {
    private static var modelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    /// The context the decoder is sized for: the longest case must fit.
    static let maxContext = 65_536

    struct Reference: Decodable {
        struct Step: Decodable {
            let id: Int
            let logprob: Double
            let top: [[Double]]
        }
        struct Case: Decodable {
            let records: Int
            let query: Int
            let prompt_id_count: Int
            let prompt_id_fnv1a: String
            let answer: String
            let text: String
            let steps: [Step]
        }
        let model: String
        let cases: [Case]
    }

    static let words = [
        "amber", "basil", "cedar", "delta", "ember", "fable", "garnet", "harbor",
        "indigo", "juniper", "kestrel", "lantern", "meadow", "nectar", "onyx", "pepper",
        "quartz", "raven", "saffron", "thistle", "umber", "velvet", "willow", "xenon",
        "yarrow", "zephyr", "anchor", "birch", "cobalt", "dune", "falcon", "glacier",
    ]

    /// The script's `long_prompt`, byte for byte.
    static func longPrompt(records: Int, query: Int) -> String {
        var text = "Below is a list of records. Remember them.\n\n"
        for index in 1...records {
            let word = words[(index * 13 + 7) % words.count]
            text += "Record \(padded(index)): the code word is \(word).\n"
        }
        text += "\nQuestion: what is the code word of record \(padded(query))?\n"
        text += "Answer: the code word of record \(padded(query)) is"
        return text
    }

    private static func padded(_ value: Int) -> String {
        let digits = String(value)
        return String(repeating: "0", count: max(0, 5 - digits.count)) + digits
    }

    /// FNV-1a 64 over the ids as little-endian uint32, as the script hashes them.
    static func fnv1a(_ ids: [Int]) -> String {
        var value: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in ids {
            var word = UInt32(truncatingIfNeeded: token).littleEndian
            withUnsafeBytes(of: &word) { bytes in
                for byte in bytes {
                    value ^= UInt64(byte)
                    value = value &* 0x0000_0100_0000_01b3
                }
            }
        }
        let hex = String(value, radix: 16)
        return String(repeating: "0", count: 16 - hex.count) + hex
    }

    private static func reference() throws -> Reference {
        let url = try #require(
            Bundle.module.url(
                forResource: "q2kxl-long-reference", withExtension: "json",
                subdirectory: "Inkling2GGUFReferences"))
        // lint:allow-unbounded-read a fixture this package ships, a few KB.
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    /// The generator and the hash agree with the script on a case small
    /// enough to check by eye; model-free, so it always runs.
    @Test func promptGeneratorMatchesTheScript() {
        let text = Self.longPrompt(records: 2, query: 1)
        #expect(
            text
                == "Below is a list of records. Remember them.\n\n"
                + "Record 00001: the code word is umber.\n"
                + "Record 00002: the code word is basil.\n"
                + "\nQuestion: what is the code word of record 00001?\n"
                + "Answer: the code word of record 00001 is")
        // FNV-1a 64 of no bytes is the offset basis.
        #expect(Self.fnv1a([]) == "cbf29ce484222325")
        // The script's `fnv1a([1, 2, 3])`.
        #expect(Self.fnv1a([1, 2, 3]) == "fd1f0f4381eb0395")
    }

    @Test(.enabled(if: modelPath != nil))
    func longPromptsMatchLlamaCpp() async throws {
        let path = try #require(Self.modelPath)
        let url = URL(fileURLWithPath: path)
        let reference = try Self.reference()
        // Token parity needs llama.cpp's run on the same file; see the type note.
        let parity = path.contains("Q2_K_XL")
        let tokenizer = try await Inkling2GGUFTokenizer.load(
            gguf: url, model: GGUFModelReader.read(modelAt: url))
        let decoder = try Inkling2GGUFDecoder(url: url, maxContext: Self.maxContext)
        decoder.weights.prefault()
        let log = { (line: String) in FileHandle.standardError.write(Data((line + "\n").utf8)) }
        var matched = 0
        var total = 0
        for item in reference.cases {
            let ids = tokenizer.encode(Self.longPrompt(records: item.records, query: item.query))
            #expect(ids.count == item.prompt_id_count, "\(item.records) records: id count")
            #expect(Self.fnv1a(ids) == item.prompt_id_fnv1a, "\(item.records) records: ids")
            decoder.state.reset()
            let started = Date()
            var result = try decoder.prefill(ids, position: 0, returnLogits: true)
            let prefillSeconds = Date().timeIntervalSince(started)
            var position = ids.count
            var produced: [Int] = []
            for (index, expected) in item.steps.enumerated() {
                let logits = try #require(result.logits)
                let ours = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)
                produced.append(result.next)
                guard parity else {
                    if index + 1 < item.steps.count {
                        result = try decoder.step(
                            token: result.next, position: position, returnLogits: true)
                        position += 1
                    }
                    continue
                }
                total += 1
                matched += result.next == expected.id ? 1 : 0
                #expect(
                    ours.contains(expected.id),
                    "\(item.records) records step \(index): llama's \(expected.id) not in \(Array(ours))"
                )
                // Continue on llama's token so one divergence does not cascade.
                if index + 1 < item.steps.count {
                    result = try decoder.step(
                        token: expected.id, position: position, returnLogits: true)
                    position += 1
                }
            }
            let text = tokenizer.decode(produced)
            log(
                String(
                    format: "%d records, %d tokens, prefill %.1f s: %@ (llama %@)",
                    item.records, ids.count, prefillSeconds, text.debugDescription,
                    item.text.debugDescription))
            #expect(text.contains(item.answer), "\(item.records) records: answer \(item.answer)")
        }
        log("long-context greedy agreement \(matched)/\(total)")
        #expect(matched == total, "long-context greedy agreement \(matched)/\(total)")
    }
}
