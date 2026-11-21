import Foundation
import Testing

@testable import TinyTitan

/// Greedy parity of the native Inkling2 GGUF decoder against llama.cpp.
///
/// The references (`Inkling2GGUFReferences/*.json`) were captured from
/// `llama-server` on the same files with `benchmark/inkling2_gguf_reference.py`.
/// Model-gated: `TINYTITAN_INKLING2_GGUF` names the `.gguf`; the reference is
/// chosen by its filename (`Q4_K_XL` or `Q2_K_XL`).
@Suite(.serialized) struct Inkling2GGUFParityTests {
    private static var modelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    struct Reference: Decodable {
        struct Step: Decodable {
            let id: Int
            let logprob: Double
            let top: [[Double]]
        }
        struct Case: Decodable {
            let prompt: String
            let prompt_ids: [Int]
            let text: String
            let steps: [Step]
        }
        let model: String
        let cases: [Case]
    }

    private static func reference(for path: String) throws -> Reference {
        let name = path.contains("Q2_K_XL") ? "q2kxl-reference" : "q4kxl-reference"
        let url = try #require(
            Bundle.module.url(
                forResource: name, withExtension: "json", subdirectory: "Inkling2GGUFReferences"))
        // lint:allow-unbounded-read a fixture this package ships, a few KB.
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    @Test(.enabled(if: modelPath != nil))
    func greedyTokensMatchLlamaCpp() throws {
        let path = try #require(Self.modelPath)
        let reference = try Self.reference(for: path)
        let decoder = try Inkling2GGUFDecoder(url: URL(fileURLWithPath: path), maxContext: 256)
        let started = Date()
        decoder.weights.prefault()
        let log = { (line: String) in FileHandle.standardError.write(Data((line + "\n").utf8)) }
        log(String(format: "prefault %.1f s", Date().timeIntervalSince(started)))
        var matched = 0
        var total = 0
        for item in reference.cases {
            decoder.state.reset()
            var position = 0
            var stepResult: (next: Int, logits: [Float]?) = (0, nil)
            for token in item.prompt_ids {
                stepResult = try decoder.step(token: token, position: position, returnLogits: true)
                position += 1
            }
            var produced: [Int] = []
            var firstDivergence: String?
            for (index, expected) in item.steps.enumerated() {
                let logits = try #require(stepResult.logits)
                let ours = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)
                produced.append(stepResult.next)
                total += 1
                if stepResult.next == expected.id {
                    matched += 1
                } else if firstDivergence == nil {
                    let theirs = expected.top.map { Int($0[0]) }
                    firstDivergence = "step \(index): ours \(Array(ours)) llama \(theirs)"
                }
                #expect(
                    ours.contains(expected.id),
                    "\(item.prompt.debugDescription) step \(index): llama's \(expected.id) not in our top-5 \(Array(ours))"
                )
                // Continue on llama's token so one divergence does not cascade.
                if index + 1 < item.steps.count {
                    stepResult = try decoder.step(
                        token: expected.id, position: position, returnLogits: true)
                    position += 1
                }
            }
            log(
                "\(item.prompt.debugDescription): \(produced == item.steps.map(\.id) ? "identical" : firstDivergence ?? "?")"
            )
        }
        log("greedy agreement \(matched)/\(total)")
        #expect(matched == total, "greedy agreement \(matched)/\(total)")
    }
}

extension Inkling2GGUFParityTests {
    /// Diagnostic: the norm of every intermediate vector of the first token.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_TRACE"] == "1"))
    func traceFirstToken() throws {
        let path = try #require(Self.modelPath)
        let decoder = try Inkling2GGUFDecoder(url: URL(fileURLWithPath: path), maxContext: 64)
        let result = try decoder.step(token: 371, position: 0, returnLogits: true)
        for row in decoder.rowSummaries() {
            let first = row.first.map { String(format: "%.4f", $0) }.joined(separator: " ")
            FileHandle.standardError.write(
                Data(
                    String(
                        format: "%@ norm=%.4f sum=%.6f first=[%@]\n", row.label, row.norm, row.sum,
                        first
                    ).utf8))
        }
        let logits = try #require(result.logits)
        let finite = logits.allSatisfy { $0.isFinite }
        #expect(finite)
    }
}

extension Inkling2GGUFParityTests {
    /// Decode throughput: greedy tokens after a short prompt, wall clock per
    /// token including commit and wait. Opt-in with `TINYTITAN_KERNEL_BENCH=1`.
    @Test(
        .enabled(
            if: modelPath != nil
                && ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_BENCH"] == "1"))
    func decodeThroughput() throws {
        let path = try #require(Self.modelPath)
        let decoder = try Inkling2GGUFDecoder(url: URL(fileURLWithPath: path), maxContext: 512)
        decoder.weights.prefault()
        if let skip = ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_SKIP"] {
            decoder.ops.skipped = Set(skip.split(separator: ",").map(String.init))
        }
        let prompt = [371, 151451, 149879, 151194, 149884]
        var next = 0
        for (position, token) in prompt.enumerated() {
            next = try decoder.step(token: token, position: position).next
        }
        let count = 64
        var samples: [Double] = []
        var encode: [Double] = []
        var gpu: [Double] = []
        for offset in 0..<count {
            let started = Date()
            next = try decoder.step(token: next, position: prompt.count + offset).next
            samples.append(Date().timeIntervalSince(started))
            encode.append(decoder.lastEncodeSeconds)
            gpu.append(decoder.lastGPUSeconds)
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_SERIES"] == "1" {
            let series = gpu.map { String(format: "%.0f", $0 * 1e3) }.joined(separator: " ")
            FileHandle.standardError.write(Data("GPU ms per token: \(series)\n".utf8))
        }
        encode.sort()
        gpu.sort()
        let total = samples.reduce(0, +)
        let sorted = samples.sorted()
        #expect(total > 0)
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "decode %d tokens: %.2f tok/s, median %.1f ms, min %.1f ms; median encode %.1f ms, GPU %.1f ms\n",
                    count, Double(count) / total, sorted[count / 2] * 1e3, sorted[0] * 1e3,
                    encode[count / 2] * 1e3, gpu[count / 2] * 1e3
                ).utf8))
    }
}
