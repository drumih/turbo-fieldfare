//
//  CompactionTests.swift
//  TinyTitanServer
//
//  `/v1/responses/compact`: the spec's shape, the round trip that makes it
//  useful, and the budget path. No model is loaded.
//

import Foundation
import NIOCore
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
import TinyTitanCore

// A summariser stub: it records every request it is handed, answers with a
// scripted note, and reports scripted token counts so the over-budget path can
// be driven without a tokenizer.

private final class CompactionLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [ValidatedChatRequest] = []
    var requests: [ValidatedChatRequest] { lock.withLock { _requests } }
    func record(_ request: ValidatedChatRequest) { lock.withLock { _requests.append(request) } }
}

private actor CompactionBackend: ServerInferenceBackend, PromptTokenCounting {
    let log = CompactionLog()
    private let notes: [String]
    /// Counts returned in order; the last one repeats.
    private let counts: [Int]
    private var generated = 0
    private var counted = 0

    init(notes: [String], counts: [Int] = [10]) {
        self.notes = notes
        self.counts = counts
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        log.record(request)
        let note = notes[min(generated, notes.count - 1)]
        generated += 1
        onEvent(.content(note))
        return ServerCompletion(
            content: note, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(
                promptTokens: 20, completionTokens: 10,
                totalTokens: 30, cachedTokens: 0))
    }

    func countPromptTokens(_ request: ValidatedChatRequest) async throws -> Int {
        let value = counts[min(counted, counts.count - 1)]
        counted += 1
        return value
    }
}

private func post(
    _ port: Int, _ path: String,
    _ json: String
) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try localURL(port: port, path))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.httpBody = Data(json.utf8)
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func object(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

/// An input item, built the way a client sends one. A payload from
/// `ServerCompaction.encode` is base64, so it needs no JSON escaping.
private func decodeItem(_ json: String) throws -> ResponsesAPIRequest.Item {
    try JSONDecoder().decode(ResponsesAPIRequest.Item.self, from: Data(json.utf8))
}

private func compactionItem(payload: String) throws -> ResponsesAPIRequest.Item {
    try decodeItem(#"{"type":"compaction","id":"cmp_1","encrypted_content":"\#(payload)"}"#)
}

private func withServer<T>(
    _ backend: any ServerInferenceBackend,
    _ body: (Int) async throws -> T
) async throws -> T {
    let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
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

@Suite("Compaction")
struct CompactionTests {
    // MARK: - the payload and the policy

    @Test func theEnvelopeRoundTripsAndForeignPayloadsAreRefused() throws {
        let envelope = CompactionEnvelope(
            model: "qwen3.8-flash-next_4-Bit", createdAt: 1_789_000_000,
            mode: .model, summary: "Launch Tuesday; support first.")
        let payload = try ServerCompaction.encode(envelope)
        #expect(try ServerCompaction.decode(payload) == envelope)

        // Not base64 of our JSON: refused by name rather than dropped, because
        // dropping it would lose the history the caller just paid to compact.
        #expect(throws: ServerRequestError.self) { try ServerCompaction.decode("not-a-payload") }

        // A payload from another envelope version says so instead of being
        // misread by this build.
        let other = Data(#"{"v":99,"model":"m","createdAt":1,"mode":"model","summary":"s"}"#.utf8)
            .base64EncodedString()
        let error = #expect(throws: ServerRequestError.self) { try ServerCompaction.decode(other) }
        if case .invalid(_, _, let code)? = error { #expect(code == "compaction_payload_version") }
    }

    @Test func theBudgetFollowsTheContextAndNoLargerThanTheCallerAsked() {
        // One eighth of the window, capped, and never more than half of it.
        #expect(ServerCompaction.targetTokens(maxContext: 262_144, requested: nil) == 4096)
        #expect(ServerCompaction.targetTokens(maxContext: 8_192, requested: nil) == 1024)
        #expect(ServerCompaction.targetTokens(maxContext: 1_024, requested: nil) == 256)
        #expect(ServerCompaction.targetTokens(maxContext: 262_144, requested: 512) == 512)
        #expect(ServerCompaction.targetTokens(maxContext: 1_024, requested: 900) == 512)
    }

    @Test func theTranscriptNamesRolesAndKeepsToolCalls() {
        let messages = [
            OpenAIChatMessage(
                role: "system", content: .text("Be terse."),
                toolCalls: nil, toolCallID: nil, name: nil),
            OpenAIChatMessage(
                role: "user", content: .text("Read /tmp/a"),
                toolCalls: nil, toolCallID: nil, name: nil),
            OpenAIChatMessage(
                role: "assistant", content: nil,
                toolCalls: [
                    OpenAIToolCall(
                        id: "call_1", type: "function",
                        function: OpenAIFunctionCall(
                            name: "read",
                            arguments: #"{"path":"/tmp/a"}"#))
                ],
                toolCallID: nil, name: nil),
        ]
        let transcript = ServerCompaction.transcript(messages)
        #expect(transcript.contains("[system] Be terse."))
        #expect(transcript.contains("[user] Read /tmp/a"))
        // A tool call is kept: its arguments are usually the fact worth keeping.
        #expect(transcript.contains("[assistant] (tool call: read{\"path\":\"/tmp/a\"})"))
    }

    @Test func theExtractiveFallbackKeepsBothEndsAndSaysWhatItDropped() {
        let transcript = String(repeating: "a", count: 400) + String(repeating: "z", count: 400)
        let note = ServerCompaction.extractiveSummary(transcript: transcript, characterBudget: 200)
        #expect(note.contains("earlier turns dropped"))
        #expect(note.hasPrefix("aaa"))
        #expect(note.hasSuffix("zzz"))
        #expect(note.count <= 260)
    }

    /// A small model reads its own instruction back. The lines it copied are
    /// known exactly, so they are dropped rather than replayed as history — and
    /// a note left with nothing at all is a failed pass, not an empty session.
    @Test func anInstructionEchoIsStrippedLineByLine() {
        let instruction = ServerCompaction.instruction(limit: 4096)
        let note = """
            - Goal
            - Open questions
            Move the launch to Tuesday and tell support first.
            """
        let stripped = ServerCompaction.strippingInstructionEcho(note, instruction: instruction)
        #expect(stripped == "Move the launch to Tuesday and tell support first.")

        // Nothing but the instruction, renumbered: nothing survives.
        let echoOnly = "2. Next step\n6) Open questions"
        #expect(
            ServerCompaction.strippingInstructionEcho(echoOnly, instruction: instruction).isEmpty)
    }

    /// A repetition loop is not a compaction: measured on the 2B, the "note" was
    /// four transcript lines repeated a dozen times and was longer than the
    /// session it was supposed to replace.
    @Test func aRepetitionLoopIsRecognisedAsFailed() {
        let looped = Array(repeating: "[user] Where is the design recorded?", count: 10)
            .joined(separator: "\n")
        #expect(ServerCompaction.isDegenerate(looped))

        let real = """
            # Goal
            Wire /v1/responses/compact and re-run the suite.
            # Decisions
            Option B, because option A quadruples the wired expert cache.
            # Facts
            Eight of nine requests answered at width 4; peak concurrency four.
            # Next step
            Run the acceptance tests.
            """
        #expect(!ServerCompaction.isDegenerate(real))
        // Short notes are never judged degenerate: there is not enough there.
        #expect(!ServerCompaction.isDegenerate("# Goal\nLaunch Tuesday."))
    }

    // MARK: - the round trip

    @Test func aReplayedNoteBecomesStandingContext() throws {
        let envelope = CompactionEnvelope(
            model: "m", createdAt: 1, mode: .model,
            summary: "We agreed to launch on Tuesday.")
        let item = try compactionItem(payload: try ServerCompaction.encode(envelope))
        let messages = try ResponsesAPIMapper.chatMessages(
            items: [item], instructions: "Answer in one line.")
        let system = try #require(messages.first)
        #expect(system.role == "system")
        let text = try #require(
            {
                if case .text(let value)? = system.content { return value } else { return nil }
            }())
        // Instructions verbatim, then the note, in the one leading block.
        #expect(text.hasPrefix("Answer in one line."))
        #expect(text.contains("Compacted earlier session"))
        #expect(text.contains("We agreed to launch on Tuesday."))
    }

    @Test func aCompactionItemWithoutAReadablePayloadIsRefused() throws {
        let malformed = try compactionItem(payload: "garbage")
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatMessages(items: [malformed], instructions: nil)
        }
        let missing = try decodeItem(#"{"type":"compaction","id":"cmp_1"}"#)
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatMessages(items: [missing], instructions: nil)
        }
    }

    // MARK: - the endpoint

    /// A note that is only the instruction read back is a failed pass: it is
    /// retried without a menu to copy, rather than returned as history.
    @Test func anEchoedInstructionIsRetriedWithAPlainInstruction() async throws {
        let backend = CompactionBackend(notes: [
            "- Goal\n- Open questions",
            "Decision: launch Tuesday.",
        ])
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","input":[{"type":"message","role":"user","content":"launch Tuesday"}]}
                """)
            #expect(response.statusCode == 200)
            #expect(backend.log.requests.count == 2, "the echo is retried once")
            #expect(
                backend.log.requests[1].messages.contains {
                    ($0.content ?? "").contains("Summarise the session below")
                })
            let output = try #require(try object(data)["output"] as? [[String: Any]])
            let item = try #require(output.first { $0["type"] as? String == "compaction" })
            let note = try ServerCompaction.decode(
                try #require(item["encrypted_content"] as? String))
            #expect(note.summary == "Decision: launch Tuesday.")
            #expect(note.mode == .model)
        }
    }

    /// The acceptance suite's own body, and the validators it applies.
    @Test func theEndpointReturnsTheSpecShape() async throws {
        let backend = CompactionBackend(notes: ["Decision: launch Tuesday, notify support first."])
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","prompt_cache_key":"openresponses-compact-test",
                 "input":[
                   {"type":"message","role":"user","content":"We agreed to launch on Tuesday and notify support first."},
                   {"type":"message","role":"assistant","content":"Understood."}
                 ]}
                """)
            #expect(response.statusCode == 200)
            let body = try object(data)
            #expect(body["object"] as? String == "response.compaction")
            #expect(body["created_at"] is Int)
            #expect(body["usage"] is [String: Any])
            let output = try #require(body["output"] as? [[String: Any]])
            #expect(!output.isEmpty)
            let item = try #require(output.first { $0["type"] as? String == "compaction" })
            let payload = try #require(item["encrypted_content"] as? String)
            let note = try ServerCompaction.decode(payload)
            #expect(note.summary.contains("launch Tuesday"))
            #expect(note.mode == .model)
        }
    }

    @Test func aRequestWithoutAModelIsRefusedByParameter() async throws {
        let backend = CompactionBackend(notes: ["unused"])
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/responses/compact",
                #"{"input":[{"type":"message","role":"user","content":"x"}]}"#)
            #expect(response.statusCode == 400)
            let error = try #require(try object(data)["error"] as? [String: Any])
            #expect(error["param"] as? String == "model")
        }
    }

    /// A budget of zero is a cap the caller set, not an omitted one. It used to
    /// fall through to the server's own default, so a client that asked for no
    /// room got a note of up to 4096 tokens and no signal that its number was
    /// not the one used.
    @Test func aNonPositiveCompactionBudgetIsRefusedRatherThanReinterpreted() async throws {
        for requested in [0, -5] {
            let backend = CompactionBackend(notes: ["Decision: launch Tuesday."])
            try await withServer(backend) { port in
                let (data, response) = try await post(
                    port, "/v1/responses/compact",
                    """
                    {"model":"test-model","max_compaction_tokens":\(requested),
                     "input":[{"type":"message","role":"user","content":"launch Tuesday"}]}
                    """)
                #expect(
                    response.statusCode == 400,
                    "max_compaction_tokens \(requested) was accepted")
                let body = try object(data)
                let error = body["error"] as? [String: Any]
                #expect(error != nil, "no error envelope; keys \(body.keys.sorted())")
                #expect(error?["param"] as? String == "max_compaction_tokens")
                #expect(
                    backend.log.requests.isEmpty,
                    "a refused budget still ran the summariser")
            }
        }
    }

    /// The other side of the same boundary: a readable budget is used exactly as
    /// sent, so the refusal above is about the unreadable value and not about
    /// the field being unwelcome.
    @Test func aCompactionBudgetIsUsedExactlyAsTheCallerSentIt() async throws {
        let backend = CompactionBackend(notes: ["Decision: launch Tuesday."])
        try await withServer(backend) { port in
            let (_, response) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","max_compaction_tokens":256,
                 "input":[{"type":"message","role":"user","content":"launch Tuesday"}]}
                """)
            #expect(response.statusCode == 200)
            let request = try #require(backend.log.requests.first)
            #expect(request.maximumCompletionTokens == 256)
        }
    }

    /// The summariser must not think: a model that reasons inside its own output
    /// cap spends the cap on thoughts and returns an empty note.
    @Test func theSummariserRunsUnthinkingOnTheTranscript() async throws {
        let backend = CompactionBackend(notes: ["note"])
        try await withServer(backend) { port in
            let (_, response) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","input":[{"type":"message","role":"user","content":"hello there"}]}
                """)
            #expect(response.statusCode == 200)
            let request = try #require(backend.log.requests.first)
            #expect(request.reasoning?.thinkingMode == .off)
            #expect(request.messages.contains { ($0.content ?? "").contains("handover note") })
            #expect(request.messages.contains { ($0.content ?? "").contains("[user] hello there") })
        }
    }

    /// The point of the endpoint: what it returns is what the next request is
    /// made of, so the model must see the note as its standing context.
    @Test func aCompactedWindowReplaysIntoTheNextRequest() async throws {
        let backend = CompactionBackend(notes: ["Decision: launch Tuesday."])
        try await withServer(backend) { port in
            let (data, _) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","input":[{"type":"message","role":"user","content":"launch Tuesday"}]}
                """)
            let output = try #require(try object(data)["output"] as? [[String: Any]])
            let item = try #require(output.first { $0["type"] as? String == "compaction" })

            let itemJSON = try #require(
                String(
                    data: try JSONSerialization.data(withJSONObject: item),
                    encoding: .utf8))
            let replay = """
                {"model":"test-model","input":[
                  \(itemJSON),
                  {"type":"message","role":"user","content":"What did we decide?"}
                ]}
                """
            let (_, response) = try await post(port, "/v1/responses", replay)
            #expect(response.statusCode == 200)
            let last = try #require(backend.log.requests.last)
            // The message is bound first rather than asserted inline: Testing's
            // macro expansion of `#expect((a ?? "").contains(b))` emits the
            // `contains` call as a statement and warns that its result is
            // unused, which -warnings-as-errors would turn into a build
            // failure. The assertion itself is unchanged.
            let replayContent = last.messages.first?.content ?? ""
            #expect(replayContent.contains("Decision: launch Tuesday."))
            #expect(last.messages.contains { ($0.content ?? "").contains("What did we decide?") })
        }
    }

    /// An over-budget note is compressed, not truncated: truncation would drop
    /// the end of the session, which is what a continuation needs most.
    @Test func anOverBudgetNoteIsCompressedByASecondPass() async throws {
        let backend = CompactionBackend(
            notes: ["first note", "second shorter note"],
            counts: [99_999, 10])
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","input":[{"type":"message","role":"user","content":"a long session"}]}
                """)
            #expect(response.statusCode == 200)
            #expect(backend.log.requests.count == 2, "the second pass is the compression")
            let second = backend.log.requests[1]
            #expect(second.messages.contains { ($0.content ?? "").contains("still too long") })
            #expect(second.messages.contains { ($0.content ?? "").contains("first note") })

            let output = try #require(try object(data)["output"] as? [[String: Any]])
            let item = try #require(output.first { $0["type"] as? String == "compaction" })
            let note = try ServerCompaction.decode(
                try #require(item["encrypted_content"] as? String))
            #expect(note.mode == .compressed)
            #expect(note.summary == "second shorter note")
            // The caller is told the whole price of the note, both passes.
            let usage = try #require(try object(data)["usage"] as? [String: Any])
            #expect(usage["output_tokens"] as? Int == 20)
        }
    }

    /// Instructions are preserved verbatim in the window: a paraphrase of a
    /// constraint is not the constraint.
    @Test func theWindowKeepsInstructionsVerbatim() async throws {
        let backend = CompactionBackend(notes: ["note"])
        try await withServer(backend) { port in
            let (data, _) = try await post(
                port, "/v1/responses/compact",
                """
                {"model":"test-model","instructions":"Never touch /etc.",
                 "input":[{"type":"message","role":"user","content":"hello"}]}
                """)
            let output = try #require(try object(data)["output"] as? [[String: Any]])
            let message = try #require(output.first { $0["type"] as? String == "message" })
            #expect(message["role"] as? String == "developer")
            let content = try #require(message["content"] as? [[String: Any]])
            #expect(content.first?["text"] as? String == "Never touch /etc.")
        }
    }
}
