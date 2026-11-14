import Foundation
import TinyTitan
import Tokenizers
import TinyTitanCore

// Install-dependent K2 checks used by the `k2gpu` command: the tokenizer's
// multi-turn and tool rendering, the structured decoder on real tokenizer
// output, the tool round trip and the greedy golden. They live here so the
// command itself stays under the file-length gate.
extension TinyTitanBench {
    /// The second half of a tool round trip: take the call the model made, feed
    /// a canned result back as history, and let the model answer with it. This
    /// exercises the template's `tool_calls`/`tool` branches against the real
    /// weights instead of only rendering them.
    static func k2ToolLoop(
        model: Model, context: MetalContext, tokenizer: GFTokenizer,
        prompt: String, promptIDs: [Int32], firstRun: [Int32]
    ) async throws {
        let allowed = Set(PromptTools.sample.map(\.name))
        let decoded = try k2Decode(
            tokenizer: tokenizer, promptIDs: promptIDs, tokens: firstRun,
            allowedTools: allowed)
        guard let call = decoded.calls.first else {
            print("tool loop: no call emitted; nothing to continue")
            return
        }
        let result = #"{"city": "Paris", "temp_c": 21, "condition": "sunny"}"#
        let history: [GFTokenizer.Message] = [
            GFTokenizer.Message(role: .user, content: prompt),
            GFTokenizer.Message(
                role: .assistant, content: decoded.content,
                toolCalls: [
                    GFTokenizer.HistoricalToolCall(
                        id: call.id, name: call.name, arguments: call.arguments)
                ],
                reasoning: decoded.reasoning),
            GFTokenizer.Message(
                role: .tool, content: result, toolCallID: call.id,
                name: call.name),
        ]
        let ids = try tokenizer.encodeToolChat(
            messages: history, tools: PromptTools.sample)
        let graph = try K2GraphProbe(
            model: model, context: context, maxSequence: ids.count + 65)
        let stops: Set<Int32> = [tokenizer.eosID, tokenizer.endOfTurnID]
        let run = try await k2Generate(
            graph: graph, ids: ids, limit: 64, stops: stops, repeatCount: 1)
        let reply = try k2Decode(
            tokenizer: tokenizer, promptIDs: ids, tokens: run.produced,
            allowedTools: nil)
        print(
            "tool loop: \(call.name)(\(call.arguments)) -> \(result); "
                + "reply=(\(reply.reasoning.debugDescription), "
                + "\(reply.content.debugDescription))")
    }

    /// Compare this run's token ids against a stored baseline, or capture one
    /// when the file does not exist (or `TINYTITAN_K2_GOLDEN_UPDATE=1`).
    ///
    /// A K2 baseline is one (machine, build, install, prompt) tuple: Metal
    /// reduction order is not guaranteed across GPU families, and a deliberate
    /// numerics change needs a re-capture. That is the same contract as
    /// `tools/golden-baseline.sh`, which cannot drive this path because it runs
    /// the CLI and `RealForwardRunner` refuses the family.
    static func k2GoldenCheck(
        path: String, install: URL, prompt: String, limit: Int,
        tokens: [Int32]
    ) throws {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let update =
            ProcessInfo.processInfo.environment["TINYTITAN_K2_GOLDEN_UPDATE"] == "1"
        let recorded = "tokens=" + tokens.map(String.init).joined(separator: ",")
        let header =
            "model=\(install.lastPathComponent) prompt=\(prompt.debugDescription) "
            + "max_new=\(limit)"
        if !FileManager.default.fileExists(atPath: url.path) || update {
            let body =
                "# K2-Horizon greedy baseline; machine-specific\n"
                + header + "\n" + recorded + "\n"
            try body.write(to: url, atomically: true, encoding: .utf8)
            print("golden: captured \(tokens.count) tokens to \(url.path)")
            return
        }
        // lint:allow-unbounded-read the baseline is a one-line file this
        // command itself wrote under .build; it is not model metadata.
        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
        guard let storedLine = lines.first(where: { $0.hasPrefix("tokens=") }) else {
            throw ModelError.internalInconsistency(
                detail: "golden \(url.path) has no tokens= line")
        }
        let stored = storedLine.dropFirst("tokens=".count)
            .split(separator: ",")
            .compactMap { Int32($0) }
        guard stored == tokens else {
            var firstDiff = 0
            while firstDiff < min(stored.count, tokens.count),
                stored[firstDiff] == tokens[firstDiff]
            {
                firstDiff += 1
            }
            FileHandle.standardError.write(
                Data(
                    ("golden: MISMATCH at token \(firstDiff) — stored \(stored.count) "
                        + "tokens, produced \(tokens.count)\n"
                        + "  stored:  \(stored.prefix(16).map(String.init).joined(separator: ","))\n"
                        + "  produced: \(tokens.prefix(16).map(String.init).joined(separator: ","))\n")
                        .utf8))
            throw ModelError.internalInconsistency(
                detail: "golden baseline mismatch (\(url.path))")
        }
        print("golden: ok, \(tokens.count) tokens match \(url.path)")
    }

    /// A sample tool set for the chat path, so a tool-templated prompt can be
    /// rendered and parsed without a server request.
    enum PromptTools {
        static let sample: [GFTokenizer.FunctionDefinition] = [
            GFTokenizer.FunctionDefinition(
                name: "get_weather",
                description: "Current weather for a city",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("city")]),
                ]))
        ]
    }

    /// Run a token stream through the same structured decoder the server uses
    /// and render its events, so the K2 think/tool surfaces are checked on the
    /// real tokenizer instead of only in unit tests with a fixture.
    struct K2Decoded {
        var reasoning = ""
        var content = ""
        var calls: [ParsedToolCall] = []
    }

    static func k2Decode(
        tokenizer: GFTokenizer, promptIDs: [Int32], tokens: [Int32],
        allowedTools: Set<String>?
    ) throws -> K2Decoded {
        let decoder = StructuredAssistantDecoder.forGeneration(
            tokenizer: tokenizer, promptIDs: promptIDs,
            allowedTools: allowedTools)
        var detokenizer = GFDetokenizer(tokenizer: tokenizer)
        var events: [StructuredAssistantEvent] = []
        for id in tokens {
            events += try decoder.consume(
                tokenID: id, delta: detokenizer.push(id))
        }
        events += try decoder.consumeTail(detokenizer.flush())
        try decoder.finish()
        var decoded = K2Decoded()
        for event in events {
            switch event {
            case .reasoning(let text): decoded.reasoning += text
            case .content(let text): decoded.content += text
            case .toolCall(let call): decoded.calls.append(call)
            }
        }
        return decoded
    }

    static func k2DecoderReport(
        tokenizer: GFTokenizer,
        promptIDs: [Int32],
        tokens: [Int32],
        allowedTools: Set<String>?
    ) throws -> String {
        let decoded = try k2Decode(
            tokenizer: tokenizer, promptIDs: promptIDs, tokens: tokens,
            allowedTools: allowedTools)
        var line =
            "reasoning=\(decoded.reasoning.debugDescription) "
            + "content=\(decoded.content.debugDescription)"
        if !decoded.calls.isEmpty {
            line +=
                " calls="
                + decoded.calls.map { "\($0.name)(\($0.arguments))" }.joined(separator: ",")
        }
        return line
    }

    /// Feed two canned streams through the decoder with the real tokenizer:
    /// one closes a thought and answers, one emits an xml tool call. This is
    /// install-dependent, so it lives behind TINYTITAN_K2_DECODER_SELFTEST.
    static func k2DecoderSelfTest(tokenizer: GFTokenizer) throws {
        let effort = ModelReasoningEffort.resolved() ?? .high
        var tag = "think"
        switch effort {
        case .medium: tag = "think_fast"
        case .low: tag = "think_faster"
        case .high, .xhigh: tag = "think"
        }
        let promptIDs = tokenizer.encode(
            "<|ifm|im_start|>assistant\n<ifm|\(tag)>\n", addBOS: false)

        let answer = "Simple.\n</ifm|\(tag)>\nThe capital is Paris."
        let tool =
            "<ifm|tool_call>get_weather\n"
            + "<ifm|arg_key>city</ifm|arg_key>\n"
            + "<ifm|arg_value>Paris</ifm|arg_value>\n"
            + "</ifm|tool_call>"
        // Multi-turn history: the K2 template raises on an assistant message
        // that carries no thinking field, so a replayed turn has to say which
        // block it had (or that it had an empty one).
        let history = [
            GFTokenizer.Message(role: .user, content: "What is a mutex?"),
            GFTokenizer.Message(
                role: .assistant, content: "A lock.",
                reasoning: "It guards shared state."),
            GFTokenizer.Message(role: .user, content: "And a semaphore?"),
        ]
        do {
            let rendered = try tokenizer.applyChatTemplate(history)
            let historyIDs = tokenizer.encode(rendered, addBOS: false)
            let keepsReasoning = rendered.contains("It guards shared state.")
            let closes = rendered.contains("</ifm|\(tag)>")
            print(
                "selftest history: rendered \(historyIDs.count) prompt tokens, "
                    + "reasoning kept=\(keepsReasoning), thought closed=\(closes)")
        } catch {
            print("selftest history: FAILED (\(error))")
        }

        // The template's own shape: the inner call inside the
        // `<ifm|tool_calls>` wrapper, with newlines the decoder must drop.
        let wrapped = "<ifm|tool_calls>\n" + tool + "\n</ifm|tool_calls>"
        // A tool round trip: the assistant's call and the tool's response are
        // both history by the time the next user turn is rendered.
        let toolCall = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "get_weather",
            arguments: .object(["city": .string("Paris")]))
        let roundTrip = [
            GFTokenizer.Message(role: .user, content: "What is the weather in Paris?"),
            GFTokenizer.Message(
                role: .assistant, content: "", toolCalls: [toolCall],
                reasoning: "Use the weather tool."),
            GFTokenizer.Message(
                role: .tool, content: #"{"temp_c": 21}"#,
                toolCallID: "call_1", name: "get_weather"),
            GFTokenizer.Message(role: .user, content: "And tomorrow?"),
        ]
        do {
            let rendered = try tokenizer.applyChatTemplate(roundTrip)
            let ids = tokenizer.encode(rendered, addBOS: false)
            let hasCall = rendered.contains("get_weather")
            let hasResult = rendered.contains("temp_c")
            print(
                "selftest roundtrip: rendered \(ids.count) prompt tokens, "
                    + "call kept=\(hasCall), result kept=\(hasResult)")
        } catch {
            print("selftest roundtrip: FAILED (\(error))")
        }

        // The closing tag spelled as ordinary tokens, split across two of
        // them: the ID-level decoder cannot see it, so the text-level fallback
        // has to split the thought here.
        var spelled: [Int32] = []
        spelled += tokenizer.encode("Need the city.\n", addBOS: false)
        spelled += tokenizer.encode("</ifm|", addBOS: false)
        spelled += tokenizer.encode("\(tag)>", addBOS: false)
        spelled += tokenizer.encode("\nParis", addBOS: false)
        let spelledReport = try k2DecoderReport(
            tokenizer: tokenizer, promptIDs: promptIDs, tokens: spelled,
            allowedTools: nil)
        print("selftest spelled: \(spelledReport)")

        for (label, text) in [
            ("answer", answer), ("tool", tool), ("wrapped", wrapped),
        ] {
            let tokens = tokenizer.encode(text, addBOS: false)
            let report = try k2DecoderReport(
                tokenizer: tokenizer, promptIDs: promptIDs, tokens: tokens,
                allowedTools: label == "answer" ? nil : ["get_weather"])
            print("selftest \(label): \(report)")
        }
    }
}
