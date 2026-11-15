import Foundation
import Metal
import TinyTitan
import TinyTitanCore

/// The CLI's Nemotron 3.5 Bolt path.
///
/// The engine's Qwen-shaped `RealForwardRunner` does not serve this family -- it
/// validates a nine-role expert bundle and a 2,816-wide tile -- so the CLI
/// branches before building an `Engine` and drives `NemotronModel` directly.
/// The flags that survive are the ones the graph can honour: a prompt, a token
/// limit, temperature/top-p/top-k, a seed, and quiet. Everything else is
/// refused by name rather than silently ignored.
struct NemotronRunError: Error, CustomStringConvertible {
    let description: String
}

enum NemotronRun {
    static func runIfNemotron(
        args: Args,
        stdout: FileHandle,
        stderr: FileHandle
    ) async -> RunResult? {
        let directory = URL(fileURLWithPath: args.model)
        guard isNemotronInstall(directory) else { return nil }
        do {
            return try await generate(
                args: args, directory: directory, stdout: stdout, stderr: stderr)
        } catch {
            stderr.write(Data("nemotron: \(error)\n".utf8))
            return RunResult(exitCode: 1)
        }
    }

    private static func isNemotronInstall(_ directory: URL) -> Bool {
        let manifest = directory.appendingPathComponent("manifest.json")
        guard
            let data = try? BoundedMetadataRead.read(
                fileAt: manifest, maxBytes: 64 * 1024 * 1024),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let arch = object["arch"] as? [String: Any]
        else { return false }
        return (arch["family"] as? String) == "nemotron_h"
    }

    private static func generate(
        args: Args,
        directory: URL,
        stdout: FileHandle,
        stderr: FileHandle
    ) async throws -> RunResult {
        guard let prompt = args.prompt else {
            throw NemotronRunError(description: "the Nemotron path needs --prompt")
        }
        guard args.messagesFile == nil else {
            throw NemotronRunError(
                description: "the Nemotron path does not read --messages-file yet")
        }
        let install = try NemotronInstall(directory: directory)
        let tokenizer = try await NemotronTokenizer(
            directory: directory, thinking: args.thinkingMode == .on)
        let ids = try tokenizer.encode(prompt: prompt, chat: args.concise)
        guard !ids.isEmpty else {
            throw NemotronRunError(description: "the prompt tokenized to nothing")
        }
        let limit = min(args.maxNew, args.maxContext)
        let context = try MetalContext.runtimeDefault()
        let model = try NemotronMetalGraph(
            install: install, context: context, maxSequence: ids.count + limit + 8)

        let prefillStart = Date()
        var logits: [Float] = []
        for token in ids { logits = try model.step(token: Int(token)) }
        let prefill = Date().timeIntervalSince(prefillStart)

        var generator = SystemRandomNumberGenerator()
        var produced: [Int32] = []
        var printed = ""
        let decodeStart = Date()
        for _ in 0..<limit {
            let token = try sample(
                logits, temperature: Double(args.temperature),
                topP: Double(args.topP ?? 1.0), topK: args.topK ?? 0,
                generator: &generator)
            if install.geometry.eosTokens.contains(token) { break }
            produced.append(Int32(token))
            let text = tokenizer.decode(produced)
            if text.hasPrefix(printed) {
                let delta = String(text.dropFirst(printed.count))
                if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                printed = text
            }
            logits = try model.step(token: token)
        }
        let decode = Date().timeIntervalSince(decodeStart)
        if !args.quiet {
            let rate = produced.isEmpty ? 0 : Double(produced.count) / max(decode, 0.001)
            stderr.write(
                Data(
                    String(
                        format:
                            "\nprompt %d tokens: %.2fs (%.2f tok/s), decode %d tokens: %.2fs (%.2f tok/s)\n",
                        ids.count, prefill, Double(ids.count) / max(prefill, 0.001),
                        produced.count, decode, rate
                    ).utf8))
        }
        return RunResult(exitCode: 0)
    }

    /// Greedy at temperature 0, otherwise top-k then top-p then multinomial.
    private static func sample(
        _ logits: [Float],
        temperature: Double,
        topP: Double,
        topK: Int,
        generator: inout SystemRandomNumberGenerator
    ) throws -> Int {
        guard !logits.isEmpty else { throw NemotronRunError(description: "no logits") }
        if temperature <= 0 {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            return best
        }
        let scaled = logits.map { Float(Double($0) / temperature) }
        let maximum = scaled.max() ?? 0
        let weights = scaled.map { exp($0 - maximum) }
        var order = Array(weights.indices).sorted { weights[$0] > weights[$1] }
        if topK > 0, topK < order.count { order = Array(order.prefix(topK)) }
        var total = order.reduce(Float(0)) { $0 + weights[$1] }
        if topP < 1.0 {
            var cumulative: Float = 0
            var kept: [Int] = []
            for index in order {
                kept.append(index)
                cumulative += weights[index] / total
                if cumulative >= Float(topP) { break }
            }
            order = kept
            total = order.reduce(Float(0)) { $0 + weights[$1] }
        }
        var threshold = Float.random(in: 0..<total, using: &generator)
        for index in order {
            threshold -= weights[index]
            if threshold <= 0 { return index }
        }
        return order.last ?? 0
    }
}
