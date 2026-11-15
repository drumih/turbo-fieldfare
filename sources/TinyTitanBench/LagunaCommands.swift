import Foundation
import TinyTitan
import Tokenizers
import TinyTitanCore

/// `TinyTitanBench lagunagpu <install> "<prompt>" [n]`: the Laguna-S-2.1
/// install through the probe graph on Metal, greedy.
///
/// This is the real 118B weights minus the serving runner: one token at a
/// time, per-layer probes with plain K/V caches, and the fetched routed
/// experts. It exists to put a number and a text on the graph before the
/// server is wired to it.
extension TinyTitanBench {
    static func runLagunaGPU(install path: String, prompt: String, limit: Int) throws {
        /// unchecked-invariant: written only inside the Task below and read
        /// only after `semaphore.wait()` returns, which orders the read after
        /// the write; no concurrent access.
        final class Box: @unchecked Sendable {
            var error: Error?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runLagunaGPUAsync(
                    install: URL(fileURLWithPath: path), prompt: prompt,
                    limit: limit)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    private static func runLagunaGPUAsync(
        install: URL, prompt: String, limit: Int
    ) async throws {
        let config = try ArchConfig.resolved(
            forFamily: .laguna, directoryURL: install)
        let context = try MetalContext.runtimeDefault()
        let model = try Model.load(
            directoryURL: install, device: context.device, expecting: config,
            streamingMode: .pread(slotCount: 16))
        guard let tokenizer = try loadK2Tokenizer(install) else {
            FileHandle.standardError.write(Data("no tokenizer in \(install.path)\n".utf8))
            exit(2)
        }
        let chat = ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_CHAT"] == "1"
        let ids: [Int]
        if chat {
            // lint:allow-unbounded-read the checkpoint's own template: 4 KB,
            // written by the converter, and read once per process.
            let template = try String(
                contentsOf: install.appendingPathComponent(
                    "tokenizer/chat_template.jinja"),
                encoding: .utf8)
            let messages: [Message] = [["role": "user", "content": prompt]]
            ids = try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: .literal(template),
                addGenerationPrompt: true, truncation: false,
                maxLength: nil, tools: nil)
        } else {
            ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
        }
        guard !ids.isEmpty else {
            FileHandle.standardError.write(Data("the prompt tokenized to nothing\n".utf8))
            exit(2)
        }
        let debugPrompt = ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_PROMPT"] == "1"
        if debugPrompt {
            print("prompt ids: " + ids.map(String.init).joined(separator: ","))
            print(
                "prompt text: "
                    + tokenizer.decode(tokens: ids, skipSpecialTokens: false))
        }
        let graph = try LagunaGraphProbe(
            model: model, context: context,
            maxSequence: ids.count + limit + 2)

        let start = Date()
        var logits: [Float] = []
        for (position, token) in ids.enumerated() {
            logits = try await graph.forward(
                tokenID: UInt32(token),
                position: UInt32(position))
        }
        let prefill = Date().timeIntervalSince(start)

        var produced: [Int] = []
        let decodeStart = Date()
        var position = ids.count
        for _ in 0..<limit {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] {
                best = index
            }
            let token = best
            if LagunaTensors.eosTokens.contains(token) { break }
            produced.append(token)
            logits = try await graph.forward(
                tokenID: UInt32(token),
                position: UInt32(position))
            position += 1
        }
        let decode = Date().timeIntervalSince(decodeStart)
        print(
            "prompt \(ids.count) tokens: \(String(format: "%.2f", prefill))s "
                + "(\(String(format: "%.2f", Double(ids.count) / prefill)) tok/s)")
        print(
            "decode \(produced.count) tokens: \(String(format: "%.2f", decode))s "
                + "(\(String(format: "%.2f", Double(produced.count) / max(decode, 0.001))) tok/s)")
        print(
            "tokens: "
                + produced.map { String($0) }.joined(separator: ","))
        print("output: \(tokenizer.decode(tokens: produced, skipSpecialTokens: true))")
        print(
            "raw: \(tokenizer.decode(tokens: produced, skipSpecialTokens: false))")
        if ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_LOGITS"] == "1" {
            print(
                "logits: "
                    + logits.prefix(8).map { String(format: "%.5f", $0) }
                    .joined(separator: ","))
            let ranked = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)
            print(
                "top5: "
                    + ranked.map { "\($0):\(String(format: "%.3f", logits[$0]))" }
                    .joined(separator: ","))
            print(
                "top5 text: "
                    + tokenizer.decode(
                        tokens: Array(ranked), skipSpecialTokens: false))
        }
    }
}
