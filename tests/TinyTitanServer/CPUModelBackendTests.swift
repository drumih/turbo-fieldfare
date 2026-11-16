import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// The CPU serving path's ceilings, which it enforces rather than promises,
/// and its handling of a thinking model's output.
@Suite struct CPUModelBackendTests {

    /// Qwen3.5 claims 262,144 positions and the GPU engine honours it. On
    /// the CPU attention is a loop over the cache and the cache is held per
    /// token: at that length the keys and values alone are over three
    /// gigabytes and every token would walk all of them.
    ///
    /// A ceiling that is quietly enforced beats a promise that is quietly
    /// broken, so the backend clamps rather than repeating the checkpoint.
    @Test func theContextCeilingIsEnforcedNotPromised() {
        #expect(CPUModelBackend.contextCeiling < 262_144)
        #expect(
            CPUModelBackend.contextCeiling >= 8_192,
            "and still enough for the work this engine is for")
    }

    // MARK: - a model that says what it is told

    /// The real engine on a snapshot small enough to write in a test: no
    /// layers, so each next token depends on the current one alone, and a
    /// tied embedding built so that greedy decoding walks `chain`.
    ///
    /// Row `chain[k]` is `3^k * (e_k + e_(k+1))`. Its own logit is then
    /// `2 * 9^k` against `3 * 9^k` for `chain[k + 1]`, and every row outside
    /// the chain is zero -- so the successor wins at every step, with a
    /// margin that BF16 scales cannot close.
    private func writeScriptedModel(
        chain: [Int32], tokenizer: URL,
        sidecar: Bool = false
    ) throws -> URL {
        let hidden = 64
        let rows = 248_320
        precondition(chain.count < hidden)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("scripted-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Eight-bit lanes pack low-first into little-endian words, so the
        // byte stream is simply each row's levels in column order.
        var levels = [UInt8](repeating: 0, count: rows * hidden)
        var scales = [Float](repeating: 0, count: rows)
        for (step, token) in chain.enumerated() {
            let row = Int(token)
            levels[row * hidden + step] = 255
            levels[row * hidden + step + 1] = 255
            scales[row] = Float(pow(3.0, Double(step))) / 255
        }
        let stem = "language_model.model.embed_tokens."
        let shard = try Self.safetensors([
            (stem + "weight", "U32", [rows, hidden / 4], levels),
            (stem + "scales", "BF16", [rows, 1], Self.bf16(scales)),
            (stem + "biases", "BF16", [rows, 1], Self.bf16([Float](repeating: 0, count: rows))),
            (
                "language_model.model.norm.weight", "BF16", [hidden],
                Self.bf16([Float](repeating: 1, count: hidden))
            ),
        ])
        try shard.write(to: directory.appendingPathComponent("model.safetensors"))
        let config: [String: Any] = [
            "model_type": "qwen3_5",
            "hidden_size": hidden, "num_hidden_layers": 0, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": hidden, "full_attention_interval": 4,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": hidden, "linear_value_head_dim": hidden,
            "linear_conv_kernel_dim": 4, "intermediate_size": hidden,
            "vocab_size": rows, "rms_norm_eps": 1e-6,
            "quantization": ["bits": 8, "group_size": 64, "mode": "affine"],
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appendingPathComponent("config.json"))
        let names = [
            stem + "weight", stem + "scales", stem + "biases",
            "language_model.model.norm.weight",
        ]
        try JSONSerialization.data(withJSONObject: [
            "weight_map": Dictionary(uniqueKeysWithValues: names.map { ($0, "model.safetensors") })
        ])
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        // A shipped `.ssdai` install keeps the tokenizer in a `tokenizer/`
        // sidecar; a converter's snapshot keeps it at the root. Both shapes
        // ship, so both are fixtures.
        let tokenizerDirectory =
            sidecar
            ? directory.appendingPathComponent("tokenizer")
            : directory
        try FileManager.default.createDirectory(
            at: tokenizerDirectory,
            withIntermediateDirectories: true)
        for file in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            try FileManager.default.copyItem(
                at: tokenizer.appendingPathComponent(file),
                to: tokenizerDirectory.appendingPathComponent(file))
        }
        return directory
    }

    private static func safetensors(
        _ tensors: [(
            name: String, dtype: String,
            shape: [Int], bytes: [UInt8]
        )]
    ) throws -> Data {
        var header: [String: Any] = [:]
        var payload: [UInt8] = []
        for tensor in tensors {
            header[tensor.name] = [
                "dtype": tensor.dtype, "shape": tensor.shape,
                "data_offsets": [payload.count, payload.count + tensor.bytes.count],
            ]
            payload.append(contentsOf: tensor.bytes)
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(contentsOf: payload)
        return out
    }

    private static func bf16(_ values: [Float]) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(values.count * 2)
        for value in values {
            let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            withUnsafeBytes(of: bits.littleEndian) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    /// unchecked-invariant: every access is under `lock`.
    private final class Sink: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [ServerInferenceEvent] = []
        func append(_ event: ServerInferenceEvent) { lock.withLock { events.append(event) } }
        var all: [ServerInferenceEvent] { lock.withLock { events } }
    }

    private func request(
        stops: [String] = [],
        reasoning: RequestReasoning? = nil
    ) -> ValidatedChatRequest {
        var configuration = GenerationConfig(maxNewTokens: 32, temperature: 0)
        configuration.stopStrings = stops
        return ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [], stream: true, includeUsage: false,
            generationConfig: configuration,
            maximumCompletionTokens: 32,
            reasoning: reasoning)
    }

    /// AUD-176. Every field the validator put on the config has to *arrive* at
    /// the sampler. Each one is read by the shared validator and applied by the
    /// GPU sampler, so a field that stops here makes `--cpu` answer the same
    /// request with a different distribution than the GPU entry -- silently, and
    /// with no golden baseline on the CPU path to catch it. `seed` and both
    /// penalties were exactly such a stop.
    @Test func everySamplingFieldReachesTheCPUSampler() throws {
        let configuration = GenerationConfig(
            maxNewTokens: 8, temperature: 0.8, topK: 40, topP: 0.9,
            presencePenalty: 1.5, minP: 0, repetitionPenalty: 1.1, seed: 1234)
        let sampler = CPUModelBackend.cpuSampler(for: configuration)
        #expect(sampler.temperature == 0.8)
        #expect(sampler.topK == 40)
        #expect(sampler.topP == 0.9)
        #expect(sampler.presencePenalty == 1.5)
        #expect(sampler.repetitionPenalty == 1.1)
        #expect(sampler.seed == 1234)
        // An absent seed stays absent, which is the clock. It is not seed 0: the
        // generator gives 0 its own fixed state, so folding nil into 0 would
        // make every unseeded request draw the same sequence.
        #expect(CPUModelBackend.cpuSampler(for: GenerationConfig(temperature: 1)).seed == nil)
    }

    /// The scripted backend, without running a generation: the tests that ask
    /// the backend about itself need the object, not an answer.
    private func scriptedBackend(thinking: ModelThinkingMode) async throws
        -> (backend: CPUModelBackend, directory: URL, spoken: [Int32])
    {
        let fixture = try TokenizerFixture.folder()
        let tok = try await GFTokenizer.load(from: fixture, thinkingMode: thinking)
        let prompt = tok.encode(try tok.applyChatTemplate(request().messages), addBOS: false)
        func single(_ text: String) throws -> Int32 {
            let ids = tok.encode(text, addBOS: false)
            try #require(ids.count == 1)
            return ids[0]
        }
        let spoken = [
            try single("h"), try single("m"), try #require(tok.thinkEndID),
            try single("o"), try single("k"),
        ]
        let chain = [try #require(prompt.last)] + spoken + [tok.eosID]
        try #require(Set(chain).count == chain.count, "the walk needs distinct tokens")

        let directory = try writeScriptedModel(chain: chain, tokenizer: fixture)
        let backend = try await CPUModelBackend(
            snapshotDirectory: directory, resident: false,
            thinkingMode: thinking)
        return (backend, directory, spoken)
    }

    /// The CPU engine holds no prompt cache at all, so a banner or a residency
    /// line has to say `off` rather than repeat the mode the server was asked
    /// for -- which is what the routing banner did before it asked the backend.
    @Test func reportsNoPromptCache() async throws {
        let (backend, directory, _) = try await scriptedBackend(thinking: .off)
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(backend.promptCacheMode == .off)
        #expect(
            ServerLog.promptCacheField(for: backend) == " prompt_cache=off",
            "the field the residency line prints comes from this backend")
    }

    /// The served engine keeps the width its load chose, token after token.
    ///
    /// `CPUQwen35` can narrow itself per token when it is given a contention
    /// signal, and the side engine is: one thread while a person is waiting,
    /// four in the gaps. The served engine is handed none, and this is the
    /// guard for that being right rather than accidental. A signal read from
    /// `ServerCoordinator.generating` looks like the obvious thing to wire
    /// here and is the wrong one, because that signal is raised by a CPU
    /// client generation too: the served model would answer at one thread
    /// while it was the only thing running. (AUD-259: the setter whose doc
    /// promised the caller arms it had no caller, and no scenario needs one.)
    @Test func theServedEngineKeepsTheWidthItsLoadChose() async throws {
        let (backend, directory, _) = try await scriptedBackend(thinking: .off)
        defer { try? FileManager.default.removeItem(at: directory) }
        let loaded = backend.threads
        _ = try await backend.generate(request()) { _ in }
        let after = await backend.threadWidthInUse
        #expect(
            after == loaded,
            "after a whole generation the width is still what the load picked")
    }

    /// Runs the scripted model through the backend: after the prompt it
    /// says "hm", closes its thought, says "ok", and ends the turn.
    private func generate(thinking: ModelThinkingMode, stops: [String] = []) async throws
        -> (completion: ServerCompletion, events: [ServerInferenceEvent], tokens: [Int32])
    {
        let (backend, directory, spoken) = try await scriptedBackend(thinking: thinking)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sink = Sink()
        let completion = try await backend.generate(request(stops: stops)) { sink.append($0) }
        return (completion, sink.all, spoken)
    }

    /// A client stop string that ends the answer is named on the completion,
    /// as the GPU path names it, so a Messages client is told `stop_sequence`
    /// and which one. The CPU path used to end the turn and forget which.
    @Test func aStopStringThatEndsTheAnswerIsNamed() async throws {
        let run = try await generate(thinking: .on, stops: ["k"])
        #expect(run.completion.content == "o")
        #expect(run.completion.stopSequence == "k")
        #expect(run.completion.finishReason == "stop")
        let unstopped = try await generate(thinking: .on)
        #expect(unstopped.completion.stopSequence == nil)
    }

    /// Thinking on, the rendered prompt ends inside `<think>`: the model's
    /// first token is already a thought and `<think>` is never generated.
    /// The CPU path splits that through the same decoder the GPU path runs.
    @Test func aThinkingModelsThoughtIsReasoningNotTheAnswer() async throws {
        let run = try await generate(thinking: .on)
        #expect(run.events == [.reasoning("h"), .reasoning("m"), .content("o"), .content("k")])
        #expect(run.completion.reasoning == "hm")
        #expect(run.completion.content == "ok")
        #expect(run.completion.finishReason == "stop")
        #expect(
            run.completion.usage.completionTokens == 5,
            "thought tokens are generated tokens, and were always counted")
    }

    /// Thinking off, a model that writes a stray `</think>` but no opening: it
    /// is not reasoning, and the answer is the text the chain spells.
    ///
    /// The per-token decode this used to compare against is gone -- both
    /// engines detokenize through the streaming detokenizer now, so that the
    /// decoder can see the markers -- and the property that mattered is the
    /// text.
    @Test func thinkingOffIgnoresAStrayClosingMarker() async throws {
        let run = try await generate(thinking: .off)
        #expect(run.completion.content == "hmok")
        #expect(run.completion.reasoning.isEmpty)
        #expect(run.completion.unrequestedReasoning == 0)
        #expect(!run.events.contains { if case .reasoning = $0 { true } else { false } })
    }

    /// Thinking off, a model that opens a thought of its own: the thought is
    /// reasoning and the answer is what the client sees.
    ///
    /// The failure this covers is measured. Qwen AgentWorld 35B-A3B 8-bit,
    /// asked `Capital of Paris` with the server at `--reasoning off`, opens a
    /// `<think>` block itself and never leaves it inside the token budget; the
    /// path that existed for thinking off streamed that scaffold as `content`
    /// and left `reasoning_content` empty, so a client that caps tokens got no
    /// answer at all. Qwen 3.5 9B 8-bit does the same on the CPU engine, and
    /// the scripted chain below is its shape.
    @Test func aThoughtTheModelStartsWithThinkingOffIsStillReasoning() async throws {
        let fixture = try TokenizerFixture.folder()
        let tok = try await GFTokenizer.load(from: fixture, thinkingMode: .off)
        func single(_ text: String) throws -> Int32 {
            let ids = tok.encode(text, addBOS: false)
            try #require(ids.count == 1)
            return ids[0]
        }
        let prompt = tok.encode(try tok.applyChatTemplate(request().messages), addBOS: false)
        // The prompt closed the block; the model opens one anyway.
        let spoken = [
            try #require(tok.thinkStartID), try single("m"), try single("h"),
            try #require(tok.thinkEndID), try single("o"), try single("k"),
        ]
        let chain = [try #require(prompt.last)] + spoken + [tok.eosID]
        try #require(Set(chain).count == chain.count, "the walk needs distinct tokens")

        let directory = try writeScriptedModel(chain: chain, tokenizer: fixture)
        defer { try? FileManager.default.removeItem(at: directory) }
        let backend = try await CPUModelBackend(
            snapshotDirectory: directory, resident: false,
            thinkingMode: .off)
        let completion = try await backend.generate(request(stops: [])) { _ in }
        #expect(completion.content == "ok")
        #expect(completion.reasoning == "mh")
        // The fact the server logs, so the log line has teeth: this request
        // rendered with thinking off and the model thought anyway.
        #expect(completion.unrequestedReasoning == 2)
    }

    /// A request that switches thinking mode re-renders through a tokenizer
    /// for the requested mode, and on a `.ssdai` install that tokenizer lives
    /// in a `tokenizer/` sidecar. The re-render used to hand
    /// `GFTokenizer.load(from:)` the *model directory*, which has no
    /// `tokenizer.json` in that layout, so every thinking switch against an
    /// installed model failed: HTTP 500, stderr
    /// `HubClientError.configurationMissing("tokenizer.json")`. The fixture
    /// here is that layout, and both entry points a client can reach are
    /// exercised -- generation and the Messages count.
    @Test func aThinkingSwitchWorksWithASidecarTokenizer() async throws {
        let fixture = try TokenizerFixture.folder()
        let tok = try await GFTokenizer.load(from: fixture, thinkingMode: .off)
        let prompt = tok.encode(try tok.applyChatTemplate(request().messages), addBOS: false)
        func single(_ text: String) throws -> Int32 {
            let ids = tok.encode(text, addBOS: false)
            try #require(ids.count == 1)
            return ids[0]
        }
        let spoken = [
            try single("h"), try single("m"), try #require(tok.thinkEndID),
            try single("o"), try single("k"),
        ]
        let chain = [try #require(prompt.last)] + spoken + [tok.eosID]
        try #require(Set(chain).count == chain.count, "the walk needs distinct tokens")

        let directory = try writeScriptedModel(chain: chain, tokenizer: fixture, sidecar: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let backend = try await CPUModelBackend(
            snapshotDirectory: directory, resident: false,
            thinkingMode: .off)

        // Counting resolves a tokenizer for the switch without generating.
        let count = try await backend.countPromptTokens(
            request(reasoning: RequestReasoning(thinkingMode: .on, effort: nil)))
        #expect(count > 0)

        // Generating does too, and the switch has to change the split: the
        // scripted model's "hm" is a thought only while thinking is on.
        let sink = Sink()
        let completion = try await backend.generate(
            request(reasoning: RequestReasoning(thinkingMode: .on, effort: nil))
        ) { sink.append($0) }
        #expect(completion.reasoning == "hm")
        #expect(completion.content == "ok")
        #expect(completion.finishReason == "stop")
    }
}
