import Foundation
import Metal
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
    /// End-of-sequence, end-of-turn and `<|end_message|>` ids: where the CLI's
    /// single text message ends.
    package let stopIDs: Set<Int>
    /// End-of-sequence and end-of-turn (`<|content_model_end_sampling|>`) ids
    /// only: where a whole model turn ends, after any thinking, text and tool
    /// messages it holds.
    package let endOfTurnIDs: Set<Int>
    /// Every CONTROL or USER_DEFINED token (GGUF token types 3 and 4): the
    /// chat format's markers, which never belong to decoded text.
    package let controlIDs: Set<Int>
    /// The GGUF's byte-level token strings, by id, for the JSON grammar table.
    private let pieces: [String]

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
        var turnEnds = Set<Int>()
        for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let value = model.metadata[key] {
                for item in value.array ?? [value] {
                    if let id = item.integer { turnEnds.insert(Int(id)) }
                }
            }
        }
        if let end = tokenizer.convertTokenToId("<|content_model_end_sampling|>") {
            turnEnds.insert(end)
        }
        var stops = turnEnds
        if let end = tokenizer.convertTokenToId("<|end_message|>") { stops.insert(end) }
        let pieces = model.metadata["tokenizer.ggml.tokens"]?.strings ?? []
        let types = (model.metadata["tokenizer.ggml.token_type"]?.array ?? []).map {
            $0.integer.map(Int.init) ?? 1
        }
        let control = Set(types.indices.filter { types[$0] == 3 || types[$0] == 4 })
        return Self(
            tokenizer: tokenizer, stopIDs: stops, endOfTurnIDs: turnEnds, controlIDs: control,
            pieces: pieces)
    }

    package func id(of token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }

    package var vocabularySize: Int { pieces.count }

    /// The byte spelling of every ordinary token, for `JSONTokenTable`.
    /// Control tokens and pieces that are not byte-level are left out, so the
    /// grammar can never pick one; `stopTokens` are what may end a document.
    package func jsonTokenTable(stopTokens: Set<Int>) -> JSONTokenTable {
        var entries: [(id: Int32, bytes: [UInt8])] = []
        entries.reserveCapacity(pieces.count)
        for (id, piece) in pieces.enumerated() where !controlIDs.contains(id) {
            var bytes: [UInt8] = []
            bytes.reserveCapacity(piece.utf8.count)
            var literal = true
            for scalar in piece.unicodeScalars {
                guard let byte = GFDetokenizer.byteLevelScalarToByte[scalar.value] else {
                    literal = false
                    break
                }
                bytes.append(byte)
            }
            if literal, !bytes.isEmpty { entries.append((Int32(id), bytes)) }
        }
        return JSONTokenTable(
            vocab: pieces.count, entries: entries, stopTokens: Set(stopTokens.map(Int32.init)))
    }

    package func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    package func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }
}

/// What the session drives: the GGUF route's decoder, or the Inkling 2 Q4
/// engine variant over an `.ssdai` install (TT-121), which runs the same graph
/// with its experts streamed. Greedy `next` is the GPU argmax; `logits` are
/// the softcapped logits of the unpadded vocabulary when asked for.
package protocol Inkling2Decoding: AnyObject {
    var maxContext: Int { get }
    func resetSequence()
    func prefill(_ prompt: [Int], position: Int, returnLogits: Bool) throws
        -> (next: Int, logits: [Float]?)
    func step(token: Int, position: Int, returnLogits: Bool) throws
        -> (next: Int, logits: [Float]?)
}

extension Inkling2GGUFDecoder: Inkling2Decoding {
    package var maxContext: Int { state.maxContext }
    package func resetSequence() { state.reset() }
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

    let decoder: any Inkling2Decoding
    package let tokenizer: Inkling2GGUFTokenizer

    /// `maxContext` is what the front end asks for; the session allocates the
    /// window this device can hold beside the weights
    /// (`Inkling2GGUFContextBudget`), which `maximumContext` reports.
    package init(gguf url: URL, maxContext: Int, maxChunk: Int = 256) async throws {
        let window = try Self.affordableContext(gguf: url, requested: maxContext)
        let decoder = try Inkling2GGUFDecoder(
            url: url, maxContext: window, maxChunk: min(maxChunk, window))
        tokenizer = try await Inkling2GGUFTokenizer.load(gguf: url, model: decoder.manifest.model)
        decoder.weights.prefault()
        self.decoder = decoder
    }

    /// A session over another decoder of the same graph, with the tokenizer
    /// read from `header` (a GGUF header saved without its data section).
    package init(decoder: any Inkling2Decoding, ggufHeader header: URL) async throws {
        self.decoder = decoder
        tokenizer = try await Inkling2GGUFTokenizer.load(
            gguf: header, model: try GGUFModelReader.readHeader(at: header))
    }

    package var maximumContext: Int { decoder.maxContext }

    /// The window the device affords for `gguf`, reported on stderr when it is
    /// below what was asked.
    static func affordableContext(gguf url: URL, requested: Int) throws -> Int {
        let manifest = try Inkling2GGUFManifest(url: url)
        let fileBytes =
            (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        let workingSet = Int(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
        let window = Inkling2GGUFContextBudget.affordable(
            requested: requested, model: modelMaxContext, workingSet: workingSet,
            weightBytes: fileBytes,
            bytesPerPosition: Inkling2GGUFContextBudget.bytesPerPosition(manifest))
        if window < requested {
            FileHandle.standardError.write(
                Data(
                    ("Inkling2 GGUF: context \(window) tokens (asked \(requested)): the GPU working "
                        + "set is \(workingSet >> 20) MiB and the weights take \(fileBytes >> 20) MiB\n")
                        .utf8))
        }
        return window
    }

    /// The longest context a front end may ask for: the model's own declared
    /// window (`model_max_length`, 1M). There is no lower artificial cap; what a
    /// session actually allocates is the part of the request this device's GPU
    /// can hold beside the weights (`Inkling2GGUFContextBudget`, TT-120).
    package static let modelMaxContext = 1_048_576

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
    /// deltas through `onText` and stopping at the end of the first message
    /// (`stopIDs`). The CLI's path: the prompt has already opened a text turn.
    package func generate(
        promptIDs: [Int], maxTokens: Int, sampler: CPUSampler,
        onText: (String) -> Void = { _ in }
    ) throws -> Generation {
        var produced: [Int] = []
        var text = ""
        let run = try decode(
            promptIDs: promptIDs, maxTokens: maxTokens, sampler: sampler,
            stopIDs: tokenizer.stopIDs
        ) { next in
            produced.append(next)
            // Whole-sequence decode so multi-token characters never tear.
            let decoded = tokenizer.decode(produced)
            if decoded.hasPrefix(text), decoded.count > text.count {
                onText(String(decoded.dropFirst(text.count)))
                text = decoded
            }
            return true
        }
        return Generation(
            tokens: run.tokens, text: tokenizer.decode(run.tokens),
            stopReason: run.stopReason, promptTokens: promptIDs.count,
            prefillSeconds: run.prefillSeconds, decodeSeconds: run.decodeSeconds)
    }

    /// What one decode produced and why it ended: `"stop"` for a stop id or a
    /// caller that asked to stop, `"length"` for the token budget.
    package struct DecodeRun: Sendable {
        package let tokens: [Int]
        package let stopReason: String
        package let prefillSeconds: Double
        package let decodeSeconds: Double
    }

    /// The one decode loop. Greedy sampling with no mask reads the GPU argmax;
    /// anything else — temperature, penalties, a grammar, min-p — reads the
    /// logits, lets `shapeLogits` floor what it forbids, and samples on the
    /// CPU. `onToken` sees every kept token and returns false to stop after it.
    package func decode(
        promptIDs: [Int], maxTokens: Int, sampler: CPUSampler, stopIDs: Set<Int>,
        shapeLogits: ((inout [Float]) throws -> Void)? = nil,
        onToken: (Int) throws -> Bool
    ) throws -> DecodeRun {
        guard !promptIDs.isEmpty else {
            throw MetalError.invalidState("the prompt tokenized to nothing")
        }
        let budget = min(maxTokens, maximumContext - promptIDs.count)
        guard budget > 0 else {
            throw MetalError.invalidState(
                "context overflow: \(promptIDs.count) prompt tokens for \(maximumContext)")
        }
        decoder.resetSequence()
        let greedy =
            sampler.isGreedy && sampler.repetitionPenalty == 1
            && sampler.presencePenalty == 0 && shapeLogits == nil
        let generator = sampler.makeGenerator()
        let prefillStarted = Date()
        var result = try decoder.prefill(promptIDs, position: 0, returnLogits: !greedy)
        let prefillSeconds = Date().timeIntervalSince(prefillStarted)
        var produced: [Int] = []
        var stopReason = "length"
        let decodeStarted = Date()
        for index in 0..<budget {
            let next: Int
            if greedy {
                next = result.next
            } else {
                var logits = result.logits ?? []
                try shapeLogits?(&logits)
                next = sampler.pick(
                    logits, history: (promptIDs + produced).map(Int32.init), using: generator)
            }
            if stopIDs.contains(next) {
                stopReason = "stop"
                break
            }
            produced.append(next)
            guard try onToken(next) else {
                stopReason = "stop"
                break
            }
            if index + 1 < budget {
                result = try decoder.step(
                    token: next, position: promptIDs.count + index, returnLogits: !greedy)
            }
        }
        return DecodeRun(
            tokens: produced, stopReason: stopReason, prefillSeconds: prefillSeconds,
            decodeSeconds: Date().timeIntervalSince(decodeStarted))
    }
}
