import Foundation
import NIOCore
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

// A thinking model's generation, as it reaches the HTTP layer once the
// decoder has split it: thoughts, then the answer. Each surface must put the
// thoughts where its own clients look for reasoning -- and, with thinking
// off, must produce exactly what it did before reasoning existed.

/// unchecked-invariant: every access is under `lock`.
private final class SeenRequests: @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [ValidatedChatRequest] = []
    var all: [ValidatedChatRequest] { lock.withLock { _requests } }
    func record(_ request: ValidatedChatRequest) { lock.withLock { _requests.append(request) } }
}

private actor ThinkingBackend: ServerInferenceBackend {
    let seen = SeenRequests()
    let thinks: Bool

    init(thinks: Bool = true) { self.thinks = thinks }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        seen.record(request)
        if thinks {
            onEvent(.reasoning("Weigh "))
            onEvent(.reasoning("it."))
        }
        onEvent(.content("An"))
        onEvent(.content("swer."))
        return ServerCompletion(
            content: "Answer.", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 5, completionTokens: 9, totalTokens: 14,
                reasoningTokens: 4),
            reasoning: thinks ? "Weigh it." : "")
    }
}

private struct SSEEvent {
    let name: String?
    let object: [String: Any]
}

private func sseEvents(_ data: Data) throws -> [SSEEvent] {
    try data.lossyUTF8String.components(separatedBy: "\n\n").compactMap { block in
        var name: String?
        var payload: String?
        for line in block.split(separator: "\n") {
            if line.hasPrefix("event: ") { name = String(line.dropFirst(7)) }
            if line.hasPrefix("data: ") { payload = String(line.dropFirst(6)) }
        }
        guard let payload, payload != "[DONE]" else { return nil }
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
        return SSEEvent(name: name, object: object)
    }
}

private func post(_ port: Int, _ path: String, _ json: String) async throws -> (Data, Int) {
    var request = URLRequest(url: try localURL(port: port, path))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
    request.httpBody = Data(json.utf8)
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse).statusCode)
}

private func object(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

private func withServer<T>(
    _ backend: any ServerInferenceBackend,
    profile: ServerReasoningProfile = .default,
    _ body: (Int) async throws -> T
) async throws -> T {
    let server = TinyTitanHTTPServer(
        modelID: "test-model", queueLimit: 2,
        backend: backend, reasoningProfile: profile)
    let channel = try await server.start(port: 0)
    let port = try #require(channel.localAddress?.port)
    do {
        let result = try await body(port)
        try await server.shutdown()
        return result
    } catch {
        try await server.shutdown()
        throw error
    }
}

@Suite("Reasoning on the chat completions surface", .serialized)
struct ChatReasoningTests {
    private let body = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]"#

    @Test func reasoningContentRidesBesideTheAnswer() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, status) = try await post(port, "/v1/chat/completions", body + "}")
            #expect(status == 200)
            let choices = try #require(try object(data)["choices"] as? [[String: Any]])
            let message = try #require(choices[0]["message"] as? [String: Any])
            #expect(message["content"] as? String == "Answer.")
            #expect(message["reasoning_content"] as? String == "Weigh it.")
            let usage = try #require(try object(data)["usage"] as? [String: Any])
            #expect(usage["completion_tokens"] as? Int == 9)
            // How many of those tokens were spent thinking, reported apart from
            // the total: the clients that route reasoning read it to bill and to
            // budget, and it is a subset of completion_tokens, not an addition.
            let details = try #require(usage["completion_tokens_details"] as? [String: Any])
            #expect(details["reasoning_tokens"] as? Int == 4)
        }
    }

    @Test func streamedReasoningArrivesInItsOwnDeltas() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, _) = try await post(port, "/v1/chat/completions", body + #","stream":true}"#)
            let deltas = try sseEvents(data).compactMap {
                (($0.object["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any]
            }
            let keys = deltas.map { $0.keys.sorted() }
            #expect(
                keys == [
                    ["role"], ["reasoning_content"], ["reasoning_content"],
                    ["content"], ["content"], [],
                ])
            #expect(
                deltas.compactMap { $0["reasoning_content"] as? String }.joined() == "Weigh it.")
            #expect(deltas.compactMap { $0["content"] as? String }.joined() == "Answer.")
        }
    }

    @Test func withThinkingOffNoReasoningFieldAppears() async throws {
        try await withServer(ThinkingBackend(thinks: false)) { port in
            let (data, _) = try await post(port, "/v1/chat/completions", body + "}")
            let choices = try #require(try object(data)["choices"] as? [[String: Any]])
            let message = try #require(choices[0]["message"] as? [String: Any])
            #expect(message.keys.sorted() == ["content", "role"])
            let (stream, _) = try await post(
                port, "/v1/chat/completions", body + #","stream":true}"#)
            #expect(!stream.lossyUTF8String.contains("reasoning"))
        }
    }

    /// Clients that keep the vLLM field send it back on the assistant turn.
    /// It is accepted and never rendered: thoughts are not part of a prompt.
    @Test func replayedReasoningContentIsAcceptedAndIgnored() async throws {
        let backend = ThinkingBackend()
        try await withServer(backend) { port in
            let (_, status) = try await post(
                port, "/v1/chat/completions",
                """
                {"model":"test-model","messages":[
                  {"role":"user","content":"a"},
                  {"role":"assistant","content":"b","reasoning_content":"old thought"},
                  {"role":"user","content":"c"}]}
                """)
            #expect(status == 200)
            let seen = try #require(backend.seen.all.first).messages
            #expect(seen.map(\.role) == [.user, .assistant, .user])
            #expect(seen[1].content == "b")
        }
    }
}

@Suite("Reasoning on the Anthropic Messages surface", .serialized)
struct AnthropicReasoningTests {
    private let body =
        #"{"model":"test-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]"#

    @Test func aThinkingBlockPrecedesTheText() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, status) = try await post(port, "/v1/messages", body + "}")
            #expect(status == 200)
            let content = try #require(try object(data)["content"] as? [[String: Any]])
            #expect(content.map { $0["type"] as? String } == ["thinking", "text"])
            #expect(content[0]["thinking"] as? String == "Weigh it.")
            #expect(content[0]["signature"] as? String == "")
            #expect(content[1]["text"] as? String == "Answer.")
        }
    }

    @Test func thinkingStreamsAsItsOwnSignedBlock() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, _) = try await post(port, "/v1/messages", body + #","stream":true}"#)
            let events = try sseEvents(data)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "message_start",
                    "content_block_start", "content_block_delta", "content_block_delta",
                    "content_block_delta", "content_block_stop",
                    "content_block_start", "content_block_delta", "content_block_delta",
                    "content_block_stop", "message_delta", "message_stop",
                ])
            let start = try #require(events[1].object["content_block"] as? [String: Any])
            #expect(start["type"] as? String == "thinking")
            #expect(events[1].object["index"] as? Int == 0)
            let deltas = events[2...4].compactMap { $0.object["delta"] as? [String: Any] }
            #expect(
                deltas.map { $0["type"] as? String }
                    == ["thinking_delta", "thinking_delta", "signature_delta"])
            #expect(deltas[0]["thinking"] as? String == "Weigh ")
            #expect(deltas[2]["signature"] as? String == "")
            #expect(events[5].object["index"] as? Int == 0)
            let text = try #require(events[6].object["content_block"] as? [String: Any])
            #expect(text["type"] as? String == "text")
            #expect(events[6].object["index"] as? Int == 1)
        }
    }

    @Test func withThinkingOffTheStreamIsUnchanged() async throws {
        try await withServer(ThinkingBackend(thinks: false)) { port in
            let (data, _) = try await post(port, "/v1/messages", body + #","stream":true}"#)
            let types = try sseEvents(data).compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "message_start", "content_block_start", "content_block_delta",
                    "content_block_delta", "content_block_stop", "message_delta",
                    "message_stop",
                ])
            let (plain, _) = try await post(port, "/v1/messages", body + "}")
            let content = try #require(try object(plain)["content"] as? [[String: Any]])
            #expect(content.map { $0["type"] as? String } == ["text"])
        }
    }

    /// The request's `thinking` block decides the level, not the mode the
    /// server was loaded with: an `enabled` budget raises it to `xhigh` and a
    /// `disabled` block turns thinking off on a server started thinking on.
    /// The model is Qwen3.8-Flash-Next because it is the family whose template
    /// really renders effort levels; the validator maps them, and what it
    /// applied is what generation receives.
    @Test func requestThinkingOverridesTheLoadedProfile() async throws {
        let backend = ThinkingBackend()
        let loaded = ServerReasoningProfile(
            family: .qwen38flash, thinkingMode: .on,
            reasoningEffort: .low)
        try await withServer(backend, profile: loaded) { port in
            let asked =
                #"{"model":"test-model","max_tokens":65536,"messages":[{"role":"user","content":"hi"}]"#
            let (_, plainStatus) = try await post(port, "/v1/messages", asked + "}")
            #expect(plainStatus == 200)
            #expect(
                backend.seen.all.last?.reasoning
                    == RequestReasoning(thinkingMode: .on, effort: .low),
                "a request that names no level keeps the loaded one")

            let (_, onStatus) = try await post(
                port, "/v1/messages",
                asked
                    + #","thinking":{"type":"enabled","budget_tokens":32768}}"#)
            #expect(onStatus == 200)
            #expect(
                backend.seen.all.last?.reasoning
                    == RequestReasoning(thinkingMode: .on, effort: .xhigh),
                "an enabled budget raises the level above the loaded low")

            let (_, offStatus) = try await post(
                port, "/v1/messages",
                asked
                    + #","thinking":{"type":"disabled"}}"#)
            #expect(offStatus == 200)
            #expect(
                backend.seen.all.last?.reasoning
                    == RequestReasoning(thinkingMode: .off, effort: nil),
                "disabled is a real off on a server that was loaded thinking on")
        }
    }

    /// Claude Code replays every assistant turn with its thinking blocks,
    /// signed or redacted. They are accepted and dropped, never rendered.
    @Test func historyCarryingThinkingBlocksIsAccepted() async throws {
        let backend = ThinkingBackend()
        try await withServer(backend) { port in
            let (_, status) = try await post(
                port, "/v1/messages",
                """
                {"model":"test-model","max_tokens":64,"messages":[
                  {"role":"user","content":"a"},
                  {"role":"assistant","content":[
                    {"type":"thinking","thinking":"old thought","signature":""},
                    {"type":"redacted_thinking","data":"opaque"},
                    {"type":"text","text":"b"}]},
                  {"role":"user","content":"c"}]}
                """)
            #expect(status == 200)
            let seen = try #require(backend.seen.all.first).messages
            #expect(seen.map(\.role) == [.user, .assistant, .user])
            #expect(seen[1].content == "b")
        }
    }
}

@Suite("Reasoning on the Responses surface", .serialized)
struct ResponsesReasoningTests {
    @Test func aReasoningItemPrecedesTheMessage() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, status) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi"}"#)
            #expect(status == 200)
            let output = try #require(try object(data)["output"] as? [[String: Any]])
            #expect(output.map { $0["type"] as? String } == ["reasoning", "message"])
            #expect((output[0]["id"] as? String)?.hasPrefix("rs_") == true)
            let summary = try #require(output[0]["summary"] as? [[String: Any]])
            #expect(summary.count == 1)
            #expect(summary[0]["type"] as? String == "summary_text")
            #expect(summary[0]["text"] as? String == "Weigh it.")
        }
    }

    @Test func reasoningStreamsAndClosesBeforeTheMessageOpens() async throws {
        try await withServer(ThinkingBackend()) { port in
            let (data, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi","stream":true}"#)
            let events = try sseEvents(data)
            let types = events.compactMap { $0.object["type"] as? String }
            #expect(
                types == [
                    "response.created", "response.in_progress",
                    "response.output_item.added", "response.reasoning_summary_part.added",
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary_text.done", "response.reasoning_summary_part.done",
                    "response.output_item.done",
                    "response.output_item.added", "response.content_part.added",
                    "response.output_text.delta", "response.output_text.delta",
                    "response.output_text.done", "response.content_part.done",
                    "response.output_item.done", "response.completed",
                ])
            #expect(events[4].object["delta"] as? String == "Weigh ")
            #expect(events[4].object["output_index"] as? Int == 0)
            #expect(events[6].object["text"] as? String == "Weigh it.")
            #expect(events[11].object["output_index"] as? Int == 1)
            let reasoningID = (events[2].object["item"] as? [String: Any])?["id"] as? String
            let final = try #require(events.last?.object["response"] as? [String: Any])
            let output = try #require(final["output"] as? [[String: Any]])
            #expect(output.map { $0["type"] as? String } == ["reasoning", "message"])
            #expect(output[0]["id"] as? String == reasoningID)
        }
    }

    @Test func withThinkingOffThereIsNoReasoningItem() async throws {
        try await withServer(ThinkingBackend(thinks: false)) { port in
            let (data, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi"}"#)
            let output = try #require(try object(data)["output"] as? [[String: Any]])
            #expect(output.map { $0["type"] as? String } == ["message"])
            let (stream, _) = try await post(
                port, "/v1/responses",
                #"{"model":"test-model","input":"hi","stream":true}"#)
            #expect(!stream.lossyUTF8String.contains("reasoning_summary"))
        }
    }
}

/// The routing both engines share: what judges the answer never sees the
/// thought, and the thought still reaches the client.
@Suite struct AssistantOutputTests {
    /// unchecked-invariant: every access is under `lock`.
    private final class Sink: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [ServerInferenceEvent] = []
        func append(_ event: ServerInferenceEvent) { lock.withLock { events.append(event) } }
        var all: [ServerInferenceEvent] { lock.withLock { events } }
    }

    /// Reasoning tokens come from the channel each generated token landed in,
    /// not from the thought text: detokenizing and re-tokenizing is not an
    /// identity, so only the decoder's verdict is the count the usage object can
    /// advertise. The flush at the end of generation is released text rather
    /// than a token, so it is not counted either.
    @Test func reasoningTokensCountTokensNotCharacters() {
        let sink = Sink()
        var output = AssistantOutput(stops: [], onEvent: { sink.append($0) })
        output.publish([.reasoning("Weigh ")])
        output.publish([.reasoning("it.")])
        output.publish([.content("An")])
        // One token can close the thought block and open the answer.
        output.publish([.reasoning("hmm"), .content("!")])
        output.publish([.content(" tail")], isToken: false)
        #expect(output.reasoningTokens == 3)
        #expect(output.reasoning == "Weigh it.hmm")
        #expect(output.content == "An! tail")
    }

    /// Stop strings apply to the answer alone, and each watcher sees only
    /// its own text: the answer's detectors the answer, the reasoning loop
    /// detector the thought.
    @Test func stopStringsApplyToTheAnswerAndEachWatcherSeesItsOwnText() {
        let sink = Sink()
        var observed: [String] = []
        var thought: [String] = []
        var output = AssistantOutput(
            stops: ["STOP"], onEvent: { sink.append($0) },
            observeVisible: { observed.append($0) },
            observeReasoning: { thought.append($0) })
        output.publish([.reasoning("I could say STOP here.")])
        #expect(!output.isStopped, "a stop string in a thought does not end the answer")
        output.publish([.content("Fine. STOP and more")])
        output.finish()
        #expect(output.isStopped)
        #expect(output.reasoning == "I could say STOP here.")
        #expect(output.content == "Fine. ")
        #expect(observed == ["Fine. "])
        #expect(thought == ["I could say STOP here."])
        #expect(sink.all == [.reasoning("I could say STOP here."), .content("Fine. ")])
    }
}
