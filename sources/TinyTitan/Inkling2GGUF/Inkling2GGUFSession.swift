import Foundation
import TinyTitanCore
import Tokenizers

/// The tokenizer of an Inkling2 GGUF: the BPE vocabulary, merges and control
/// tokens the GGUF carries, exported once as a Hugging Face `tokenizer.json`
/// (`GGUFTokenizerSidecar`, GPT-4o pre-tokenizer for `inkling2`) and run by
/// swift-transformers. The sidecar lives in the user's cache directory, keyed
/// by the GGUF's name and size, so nothing is written beside the model.
///
/// unchecked-invariant: every stored property is a `let`; the swift-transformers
/// tokenizer is immutable after loading and only read (encode/decode/lookup).
package struct Inkling2GGUFTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    /// End-of-sequence, end-of-turn and `<|end_message|>` ids.
    package let stopIDs: Set<Int>

    package static func load(gguf url: URL, model: GGUFModel) async throws -> Self {
        let size =
            (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        let caches =
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        let folder =
            caches
            .appendingPathComponent("TinyTitan/gguf-tokenizers", isDirectory: true)
            .appendingPathComponent(
                "\(url.deletingPathExtension().lastPathComponent)-\(size)", isDirectory: true)
        if !FileManager.default.fileExists(
            atPath: folder.appendingPathComponent(GGUFTokenizerSidecar.tokenizerFileName).path)
        {
            try GGUFTokenizerSidecar.write(from: model, to: folder)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        var stops = Set<Int>()
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let value = model.metadata[key] {
                for item in value.array ?? [value] {
                    if let id = item.integer { stops.insert(Int(id)) }
                }
            }
        }
        if let end = tokenizer.convertTokenToId("<|end_message|>") { stops.insert(end) }
        return Self(tokenizer: tokenizer, stopIDs: stops)
    }

    package func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    package func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }
}

/// One Inkling2 GGUF model ready to generate: the native decoder plus its
/// tokenizer and chat format. The CLI and the loopback server both generate
/// through this, so there is one prompt renderer and one decode loop.
package final class Inkling2GGUFSession {
    package enum Role: Sendable { case system, user, assistant, tool }

    package struct Generation: Sendable {
        package let tokens: [Int]
        package let text: String
        package let stopReason: String
        package let promptTokens: Int
        package let prefillSeconds: Double
        package let decodeSeconds: Double
    }

    let decoder: Inkling2GGUFDecoder
    package let tokenizer: Inkling2GGUFTokenizer

    package init(gguf url: URL, maxContext: Int, maxChunk: Int = 256) async throws {
        decoder = try Inkling2GGUFDecoder(url: url, maxContext: maxContext, maxChunk: maxChunk)
        tokenizer = try await Inkling2GGUFTokenizer.load(gguf: url, model: decoder.manifest.model)
        decoder.weights.prefault()
    }

    package var maximumContext: Int { decoder.state.maxContext }

    /// True for a `.gguf` file (or a directory holding exactly one model
    /// `.gguf`) whose architecture is `inkling2`.
    package static func modelFile(at url: URL) -> URL? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return nil
        }
        var candidate = url
        if isDirectory.boolValue {
            let names = ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? [])
                .filter { $0.hasSuffix(".gguf") && !$0.hasPrefix("mmproj") }
            guard names.count == 1 else { return nil }
            candidate = url.appendingPathComponent(names[0])
        } else if url.pathExtension != "gguf" {
            return nil
        }
        guard let model = try? GGUFModelReader.read(modelAt: candidate),
            model.architecture == "inkling2"
        else { return nil }
        return candidate
    }

    /// Inkling's message format: each turn is `<|message_<role>|><|content_text|>
    /// … <|end_message|>`, and generation opens a model turn.
    package static func render(_ messages: [(role: Role, content: String)]) -> String {
        var out = ""
        for message in messages {
            let tag: String
            switch message.role {
            case .system: tag = "system"
            case .user: tag = "user"
            case .assistant: tag = "model"
            case .tool: tag = "tool"
            }
            out += "<|message_\(tag)|><|content_text|>\(message.content)<|end_message|>"
        }
        return out + "<|message_model|><|content_text|>"
    }

    /// Prefills `promptIDs` and decodes up to `maxTokens`, streaming text
    /// deltas through `onText`. Greedy sampling uses the GPU argmax; any other
    /// plan reads the logits and samples on the CPU.
    package func generate(
        promptIDs: [Int], maxTokens: Int, sampler: CPUSampler,
        onText: (String) -> Void = { _ in }
    ) throws -> Generation {
        guard !promptIDs.isEmpty else {
            throw MetalError.invalidState("the prompt tokenized to nothing")
        }
        let budget = min(maxTokens, maximumContext - promptIDs.count)
        guard budget > 0 else {
            throw MetalError.invalidState(
                "context overflow: \(promptIDs.count) prompt tokens for \(maximumContext)")
        }
        decoder.state.reset()
        let greedy =
            sampler.isGreedy && sampler.repetitionPenalty == 1
            && sampler.presencePenalty == 0
        let generator = sampler.makeGenerator()
        let prefillStarted = Date()
        var result = try decoder.prefill(promptIDs, position: 0, returnLogits: !greedy)
        let prefillSeconds = Date().timeIntervalSince(prefillStarted)
        var produced: [Int] = []
        var text = ""
        var stopReason = "length"
        let decodeStarted = Date()
        for index in 0..<budget {
            let next =
                greedy
                ? result.next
                : sampler.pick(
                    result.logits ?? [], history: (promptIDs + produced).map(Int32.init),
                    using: generator)
            if tokenizer.stopIDs.contains(next) {
                stopReason = "stop"
                break
            }
            produced.append(next)
            // Whole-sequence decode so multi-token characters never tear.
            let decoded = tokenizer.decode(produced)
            if decoded.hasPrefix(text), decoded.count > text.count {
                onText(String(decoded.dropFirst(text.count)))
                text = decoded
            }
            if index + 1 < budget {
                result = try decoder.step(
                    token: next, position: promptIDs.count + index, returnLogits: !greedy)
            }
        }
        return Generation(
            tokens: produced, text: tokenizer.decode(produced), stopReason: stopReason,
            promptTokens: promptIDs.count, prefillSeconds: prefillSeconds,
            decodeSeconds: Date().timeIntervalSince(decodeStarted))
    }
}
