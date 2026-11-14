import Darwin
import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Synchronization
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

private actor ScriptedServerBackend: ServerInferenceBackend {
    let delay: Duration

    init(delay: Duration = .zero) {
        self.delay = delay
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if delay > .zero {
            try await Task.sleep(for: delay)
        }
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

/// Records the workspace each request arrived with, so a test can assert
/// what actually reached the generation path rather than what a parser
/// returned in isolation.
///
/// unchecked-invariant: `seen` is only touched under `lock`.
private final class WorkspaceRecordingBackend: ServerInferenceBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var seen: [String?] = []

    var workspaces: [String?] { lock.withLock { seen } }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        lock.withLock { seen.append(request.workspace) }
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private actor MultipleToolBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let first = ParsedToolCall(
            id: "call_000000000000000000000001",
            name: "read",
            arguments: .object(["path": .string("/tmp/a")]),
            argumentsJSON: #"{"path":"/tmp/a"}"#)
        let second = ParsedToolCall(
            id: "call_000000000000000000000002",
            name: "read",
            arguments: .object(["path": .string("/tmp/b")]),
            argumentsJSON: #"{"path":"/tmp/b"}"#)
        onEvent(.toolCall(first))
        onEvent(.toolCall(second))
        return ServerCompletion(
            content: "",
            toolCalls: [first, second],
            finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 8, totalTokens: 11))
    }
}

private actor ContentAndToolBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let content = "I will read it."
        let call = ParsedToolCall(
            id: "call_000000000000000000000003",
            name: "read",
            arguments: .object(["path": .string("/tmp/mixed")]),
            argumentsJSON: #"{"path":"/tmp/mixed"}"#)
        onEvent(.content(content))
        onEvent(.toolCall(call))
        return ServerCompletion(
            content: content,
            toolCalls: [call],
            finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 8, totalTokens: 11))
    }
}

private actor PipelinedRequestBackend: ServerInferenceBackend {
    private var continuation: CheckedContinuation<Void, Never>?
    private(set) var generationCount = 0

    var isWaiting: Bool { continuation != nil }

    func release() {
        continuation?.resume()
        continuation = nil
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        generationCount += 1
        await withCheckedContinuation { continuation = $0 }
        onEvent(.content("first"))
        return ServerCompletion(
            content: "first",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

private actor CancellableServerBackend: ServerInferenceBackend {
    private(set) var startedCount = 0
    private(set) var cancellationCount = 0

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        startedCount += 1
        do {
            try await Task.sleep(for: .seconds(30))
        } catch is CancellationError {
            cancellationCount += 1
            throw CancellationError()
        }
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private actor CapturingServerBackend: ServerInferenceBackend {
    private(set) var request: ValidatedChatRequest?

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        self.request = request
        onEvent(.content("captured"))
        return ServerCompletion(
            content: "captured",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

/// Emits one content event, then fails mid-stream (S5/S20: the stream must
/// still terminate with an error frame followed by [DONE]).
private actor ThrowingMidStreamBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("partial"))
        throw ServerRequestError.invalid(
            message: "synthetic mid-stream failure",
            param: "messages",
            code: "synthetic_stream_error")
    }
}

/// Counts how often a residency-managed backend was built, so the unload
/// endpoint's release/reload cycle is observable without a model on disk.
private final class LoadCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0

    var count: Int { lock.withLock { _count } }

    func increment() {
        lock.withLock { _count += 1 }
    }
}

/// A real pipeline context, taken the only sound way: the channel's own event
/// loop hands it over in `handlerAdded`. `ChannelHandlerContext` is not
/// `Sendable`, so a test reads it from here instead of capturing it across a
/// task boundary, and the embedded channel it belongs to stays alive until
/// that test finishes it.
///
/// unchecked-invariant: written once from `handlerAdded` on the channel's event
/// loop and read only by the test that owns that channel.
private final class ContextCatcher: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    nonisolated(unsafe) static var captured: ChannelHandlerContext?

    func handlerAdded(context: ChannelHandlerContext) {
        ContextCatcher.captured = context
    }
}

@Suite("OpenAI HTTP server", .serialized)
struct HTTPServerTests {
    @Test func healthModelsAndNonStreamingCompletion() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let health = try await URLSession.shared.data(
            from: try localURL(port: port, "/health")
        ).0
        #expect(health.lossyUTF8String.contains(#""status":"ok""#))

        let models = try await URLSession.shared.data(
            from: try localURL(port: port, "/v1/models")
        ).0
        #expect(models.lossyUTF8String.contains("test-model"))

        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "hello")
        let usage = try #require(object["usage"] as? [String: Any])
        let details = try #require(usage["prompt_tokens_details"] as? [String: Any])
        #expect(details["cached_tokens"] as? Int == 0)

        try await server.shutdown()
    }

    /// A streaming request the server refuses before admitting it still gets an
    /// HTTP response.
    ///
    /// The SSE head is written by `startStream`, which the coordinator calls on
    /// admission. A queue-full rejection happens before that, and the failure
    /// path used to queue error frames on an outbox whose drainer wrote them as
    /// a body with no status line -- the client saw `data: {...}` where a 429
    /// belonged, on every streaming surface. URLSession refuses a response with
    /// no head outright, so this fails loudly on the old behaviour.
    @Test func aStreamingRequestTheQueueRefusesStillGetsAStatus() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(delay: .seconds(2)))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        func streamingRequest() throws -> URLRequest {
            var request = URLRequest(
                url: try localURL(port: port, "/v1/chat/completions"))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(
                #"""
                {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
                """#.utf8)
            return request
        }

        // queueLimit 1 admits one active request plus one queued; the third is
        // refused. Launch the first two and give them time to take those slots.
        let active = Task { try await URLSession.shared.data(for: streamingRequest()) }
        try await Task.sleep(for: .milliseconds(300))
        let queued = Task { try await URLSession.shared.data(for: streamingRequest()) }
        try await Task.sleep(for: .milliseconds(300))

        let (data, response) = try await URLSession.shared.data(for: streamingRequest())
        let status = (response as? HTTPURLResponse)?.statusCode
        #expect(
            status == 429,
            "a pre-admission rejection must carry a status line; got \(status.map(String.init) ?? "no HTTP response")"
        )
        #expect(data.lossyUTF8String.contains("queue_full"))

        _ = try? await active.value
        _ = try? await queued.value
        try await server.shutdown()
    }

    /// An oversized header block is answered rather than absorbed.
    ///
    /// NIO caps one header field at 80 KiB and nothing else, and its decoder
    /// exposes no field-count or total-size limit, so the only bound available is
    /// ours: refuse the request before routing it, and say so. Without this the
    /// server would happily parse a request carrying thousands of headers and do
    /// the work of routing it.
    @Test func anOversizedHeaderBlockIsRefused() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var request = URLRequest(
            url: try localURL(port: port, "/v1/models"))
        // One more than the limit, so the count alone trips it (the byte total is
        // only ~1.4 KiB, well under the other bound).
        for index in 0...TinyTitanHTTPServer.maximumRequestHeaderFields {
            request.setValue("v", forHTTPHeaderField: "x-h-\(index)")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 431)
        #expect(
            data.lossyUTF8String
                .contains("request_headers_too_large"))

        try await server.shutdown()
    }

    /// An oversized body is refused, and the refusal is readable.
    ///
    /// The 413 used to be written at `.end`, so the server read and discarded the
    /// whole body first — a client could make it consume an arbitrary number of
    /// bytes before learning the request was refused. Answering at the moment the
    /// cap is crossed and then draining a bounded remainder keeps the error
    /// readable: closing immediately would turn it into a connection reset for a
    /// client that is still uploading.
    @Test func anOversizedBodyIsRefusedAndReadable() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            repeating: 0x20,
            count: TinyTitanHTTPServer.maximumBodyBytes + 4096)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 413)
        #expect(data.lossyUTF8String.contains("request_too_large"))

        try await server.shutdown()
    }

    @Test func streamingUsesStableShapeAndDoneMarker() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}],
             "stream":true,"stream_options":{"include_usage":true}}
            """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = data.lossyUTF8String
        #expect(text.contains(#""role":"assistant""#))
        #expect(text.contains(#""content":"hello""#))
        #expect(text.contains(#""finish_reason":"stop""#))
        #expect(text.contains(#""prompt_tokens":3"#))
        #expect(text.contains(#""cached_tokens":0"#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func wrongModelUsesOpenAIErrorEnvelope() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"wrong","messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(data.lossyUTF8String.contains("model_not_found"))

        try await server.shutdown()
    }

    /// A body whose JSON is fine but whose shape is not is an error in one named
    /// parameter, and Foundation hands the server exactly that name. Answering
    /// "malformed JSON request" instead sends the client to its serializer for a
    /// payload whose syntax was never the problem.
    @Test func aWrongTypedFieldInValidJSONNamesTheParameter() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":"abc"}
            """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let detail = try #require(object["error"] as? [String: Any])
        let message = try #require(detail["message"] as? String)
        let param = detail["param"] as? String
        #expect(param == "max_tokens", "param was \(String(describing: param)): \(message)")
        #expect(!message.contains("malformed JSON"), "message was \(message)")

        try await server.shutdown()
    }

    /// The half of the fix that must not move: bytes that are not JSON at all
    /// still answer `invalid_json`, because that is the one thing the client
    /// genuinely needs to hear.
    @Test func bytesThatAreNotJSONStillAnswerInvalidJSON() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("{not json".utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let body = data.lossyUTF8String
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(body.contains("invalid_json"), "body was \(body)")
        #expect(body.contains("malformed JSON request"), "body was \(body)")

        try await server.shutdown()
    }

    /// Foundation refuses a document nested past its parser's ceiling as a
    /// syntax failure, and `JSONSchemaNode`'s depth note records that the server
    /// answers it `invalid_json`. The new branch decides shape-vs-syntax by
    /// re-parsing with that same parser, so this is the case where the two
    /// answers must not come apart.
    @Test func aBodyPastTheParsersNestingLimitStillAnswersInvalidJSON() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        let depth = 600
        let body =
            String(repeating: "[", count: depth) + String(repeating: "]", count: depth)
        request.httpBody = Data(body.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let text = data.lossyUTF8String
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(text.contains("invalid_json"), "body was \(text)")

        try await server.shutdown()
    }

    /// The Anthropic surface has no `param` field, so the name has to survive in
    /// the message; it is the second route a client uses and the same refusal
    /// rule applies there.
    @Test func theAnthropicRouteNamesTheFieldItRefused() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: try localURL(port: port, "/v1/messages"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","max_tokens":"abc","messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let body = data.lossyUTF8String
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(body.contains("max_tokens"), "body was \(body)")
        #expect(!body.contains("malformed JSON request"), "body was \(body)")

        try await server.shutdown()
    }

    @Test func streamingHeartbeatKeepsSlowFirstEventAlive() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(delay: .milliseconds(50)),
            heartbeatInterval: .milliseconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
            """#.utf8)
        let data = try await URLSession.shared.data(for: request).0
        #expect(data.lossyUTF8String.contains(": ping\n\n"))

        try await server.shutdown()
    }

    @Test func streamingMultipleToolsUseDistinctIndexes() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: MultipleToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"read both"}],
             "stream":true}
            """#.utf8)
        let text = try await URLSession.shared.data(for: request).0.lossyUTF8String
        #expect(text.contains(#""index":0"#))
        #expect(text.contains(#""index":1"#))
        #expect(text.contains(#""finish_reason":"tool_calls""#))

        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"read both"}]}
            """#.utf8)
        let data = try await URLSession.shared.data(for: request).0
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] is NSNull)
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 2)

        try await server.shutdown()
    }

    @Test func nonStreamingToolCallRetainsVisibleContent() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ContentAndToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"read"}]}
            """#.utf8)

        let data = try await URLSession.shared.data(for: request).0
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try #require(object["choices"] as? [[String: Any]])
        let message = try #require(choices[0]["message"] as? [String: Any])
        #expect(message["content"] as? String == "I will read it.")
        #expect((message["tool_calls"] as? [[String: Any]])?.count == 1)
        #expect(choices[0]["finish_reason"] as? String == "tool_calls")

        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"read"}],"stream":true}
            """#.utf8)
        let stream = try await URLSession.shared.data(for: request).0.lossyUTF8String
        #expect(stream.contains(#""content":"I will read it.""#))
        #expect(stream.contains(#""tool_calls""#))
        #expect(stream.contains(#""finish_reason":"tool_calls""#))

        try await server.shutdown()
    }

    /// A drainer cancelled *between* iterations used to park forever. The task
    /// cancellation handler runs before the continuation is installed when the
    /// task is already cancelled, so it found nothing to resume and the `next()`
    /// that followed stored a continuation no frame would ever reach: the
    /// request task and the in-flight count it decrements stayed outstanding,
    /// which is what made `shutdown()` wait for a drain that had no drainer.
    @Test func aCancelledNextResolvesInsteadOfParking() async {
        let outbox = SSEOutbox(capacity: 4)
        // Polled rather than awaited: with the bug the drainer stays parked
        // forever, and a test that awaits it would hang the whole run instead of
        // failing. The parked task is abandoned when the test ends.
        let settled = Mutex(false)
        let drainer = Task {
            // Cancel before the first suspension, so `next()` registers its
            // handler on an already-cancelled task: the ordering that broke.
            withUnsafeCurrentTask { $0?.cancel() }
            _ = await outbox.next()
            settled.withLock { $0 = true }
        }
        let deadline = ContinuousClock.now + .seconds(2)
        while ContinuousClock.now < deadline, !settled.withLock({ $0 }) {
            try? await Task.sleep(for: .milliseconds(10))
        }
        #expect(
            settled.withLock { $0 },
            "next() parked after cancellation instead of resolving")
        drainer.cancel()
    }

    /// AUD2-034: a head counted as in-flight plus a body that never completes is
    /// a stalled upload, not a generation. The idle deadline reset on every read,
    /// then fired once and declined to close because `inFlightRequests` was 1,
    /// and was never re-armed -- so the body buffer and connection were held
    /// indefinitely. The never-close-mid-generation rule still applies to the
    /// control below: the head has been consumed there, so the timer only
    /// re-arms.
    @Test func aStalledRequestBodyIsClosedByTheIdleDeadline() async throws {
        let channel = EmbeddedChannel()
        let handler = ServerHTTPHandler(
            modelID: "test-model",
            backend: ScriptedServerBackend(),
            coordinator: ServerCoordinator(queueLimit: 1),
            heartbeatInterval: .seconds(5),
            reasoningProfile: .default,
            router: nil,
            childChannels: ChildChannelRegistry(maximumChannels: 2),
            responseStore: ResponseStore(capacity: 2),
            idleReadTimeout: .milliseconds(40))
        try await channel.pipeline.addHandler(handler).get()
        let closed = Mutex(false)
        channel.closeFuture.whenComplete { _ in closed.withLock { $0 = true } }

        try channel.writeInbound(
            HTTPServerRequestPart.head(
                HTTPRequestHead(
                    version: .http1_1, method: .POST, uri: "/v1/chat/completions")))
        try channel.writeInbound(
            HTTPServerRequestPart.body(ByteBuffer(bytes: Array(#"{"a":"#.utf8))))
        #expect(handler.head != nil)
        #expect(handler.inFlightRequests == 1)

        channel.embeddedEventLoop.advanceTime(by: .milliseconds(200))
        channel.embeddedEventLoop.run()
        #expect(
            closed.withLock { $0 },
            "a stalled body with no .end must be closed, not held in flight forever")
        _ = try? channel.finish()
    }

    /// The control for the test above: once the head is consumed and only the
    /// generation is outstanding, the deadline must not close the connection.
    /// Re-arming is what keeps the timer alive without violating that rule.
    @Test func anInFlightGenerationIsNotClosedByTheIdleDeadline() async throws {
        let channel = EmbeddedChannel()
        let handler = ServerHTTPHandler(
            modelID: "test-model",
            backend: ScriptedServerBackend(),
            coordinator: ServerCoordinator(queueLimit: 1),
            heartbeatInterval: .seconds(5),
            reasoningProfile: .default,
            router: nil,
            childChannels: ChildChannelRegistry(maximumChannels: 2),
            responseStore: ResponseStore(capacity: 2),
            idleReadTimeout: .milliseconds(40))
        try await channel.pipeline.addHandler(handler).get()
        let closed = Mutex(false)
        channel.closeFuture.whenComplete { _ in closed.withLock { $0 = true } }
        try channel.writeInbound(
            HTTPServerRequestPart.head(
                HTTPRequestHead(
                    version: .http1_1, method: .POST, uri: "/v1/chat/completions")))
        // As `.end` leaves it: the body was routed, the response is outstanding.
        handler.head = nil
        #expect(handler.inFlightRequests == 1)
        channel.embeddedEventLoop.advanceTime(by: .milliseconds(200))
        channel.embeddedEventLoop.run()
        #expect(
            !closed.withLock { $0 },
            "a generation must never be closed mid-flight")
        handler.idleCloseTask?.cancel()
        _ = try? channel.finish()
    }

    /// S4's cap is enforced mid-stream by failing the stream: an error frame,
    /// a closed outbox, a cancelled generation. The two finish frames went
    /// around that guard, so a client that was one frame too slow at the very
    /// end got `data: [DONE]` with no `finish_reason`, no usage, and no error
    /// — a stream that reads as complete and is not. The Messages and
    /// Responses surfaces route their terminal events through the same guard
    /// as everything else; this pins the chat surface to it.
    @Test func aRefusedFinishFrameFailsTheStreamInsteadOfEndingItCleanly() async throws {
        // A pipeline context, the way the handler itself gets one: taking one
        // out of a finished pipeline is the deprecated, unsound route.
        let channel = EmbeddedChannel()
        try await channel.pipeline.addHandler(ContextCatcher()).get()
        let handler = ServerHTTPHandler(
            modelID: "test-model",
            backend: ScriptedServerBackend(),
            coordinator: ServerCoordinator(queueLimit: 1),
            heartbeatInterval: .seconds(5),
            reasoningProfile: .default,
            router: nil,
            childChannels: ChildChannelRegistry(maximumChannels: 2),
            responseStore: ResponseStore(capacity: 2))
        // Filled to its cap by a reader that stopped pulling frames, which is
        // the state a slow client leaves when generation reaches its end.
        let outbox = SSEOutbox(capacity: 2)
        #expect(outbox.enqueue(Data("one".utf8)))
        #expect(outbox.enqueue(Data("two".utf8)))
        let completion = ServerCompletion(
            content: "hi",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 2, totalTokens: 5))
        // `failStream` cancels the task it runs in, as it must in production,
        // so the finish runs in a child task the drain below can survive in.
        await Task {
            guard let context = ContextCatcher.captured else { return }
            handler.finishStream(
                context, id: "chatcmpl-1", created: 1,
                completion: completion, includeUsage: true, outbox: outbox)
        }.value
        var received: [Data] = []
        while let frame = await outbox.next() {
            received.append(frame)
        }
        let text = received.map { $0.lossyUTF8String }.joined()
        #expect(
            text.contains("stream_overflow"),
            "the client was told the stream ended cleanly: \(text)")
        // The OpenAI shape ends a *failed* stream with `[DONE]` too, so the
        // terminator itself is not the tell: the error has to come first, and
        // the finish's own terminator has to be the no-op that a closed outbox
        // makes it. Two `[DONE]`s would mean the stream both failed and
        // completed.
        let terminators = text.components(separatedBy: "[DONE]").count - 1
        #expect(
            terminators == 1,
            "expected exactly one terminator, found \(terminators): \(text)")
        let errorAt = text.firstRange(of: "stream_overflow")?.lowerBound
        let doneAt = text.firstRange(of: "[DONE]")?.lowerBound
        if let errorAt, let doneAt {
            #expect(
                errorAt < doneAt,
                "the terminator arrived before the error that explains it: \(text)")
        }
        _ = try channel.finish()
    }

    /// The other half of the same guard: routing the finish frames through the
    /// mid-stream checkpoint must not turn a healthy stream into a failed one.
    /// A reader that kept up still gets `finish_reason`, then usage, then
    /// exactly one terminator, and no error frame at all.
    @Test func aFinishWithRoomToSpareStillDeliversEveryTerminalFrame() async throws {
        let channel = EmbeddedChannel()
        try await channel.pipeline.addHandler(ContextCatcher()).get()
        let handler = ServerHTTPHandler(
            modelID: "test-model",
            backend: ScriptedServerBackend(),
            coordinator: ServerCoordinator(queueLimit: 1),
            heartbeatInterval: .seconds(5),
            reasoningProfile: .default,
            router: nil,
            childChannels: ChildChannelRegistry(maximumChannels: 2),
            responseStore: ResponseStore(capacity: 2))
        let outbox = SSEOutbox(capacity: 8)
        #expect(outbox.enqueue(Data("data: a chunk\r\n\r\n".utf8)))
        let completion = ServerCompletion(
            content: "hi",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 2, totalTokens: 5))
        await Task {
            guard let context = ContextCatcher.captured else { return }
            handler.finishStream(
                context, id: "chatcmpl-1", created: 1,
                completion: completion, includeUsage: true, outbox: outbox)
        }.value
        var received: [Data] = []
        while let frame = await outbox.next() {
            received.append(frame)
        }
        let text = received.map { $0.lossyUTF8String }.joined()
        #expect(!text.contains("stream_overflow"), "a healthy stream failed: \(text)")
        #expect(text.contains(#""finish_reason":"stop""#), "no finish_reason: \(text)")
        #expect(text.contains(#""completion_tokens":2"#), "no usage: \(text)")
        let terminators = text.components(separatedBy: "[DONE]").count - 1
        #expect(
            terminators == 1,
            "expected exactly one terminator, found \(terminators): \(text)")
        _ = try channel.finish()
    }

    /// A HEAD response must have a head and nothing else. The two read routes
    /// answered head-only; every other path fell through to the method-error or
    /// not-found writer and came back with a JSON body, which a keep-alive client
    /// parses as the start of its next response.
    @Test func headRequestsNeverCarryABody() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        for path in ["/health", "/v1/models", "/v1/chat/completions", "/nope"] {
            let socket = try connectedSocket(port: port)
            defer { Darwin.close(socket) }
            try writeAll(
                socket: socket,
                text: "HEAD \(path) HTTP/1.1\r\nHost: localhost\r\n\r\n")
            let reply = try readAvailable(socket: socket, timeoutMilliseconds: 2_000)
            #expect(reply.hasPrefix("HTTP/1.1 "), "\(path): no status line in \(reply)")
            let separator = try #require(
                reply.range(of: "\r\n\r\n"),
                "\(path): no head terminator")
            let body = reply[separator.upperBound...]
            #expect(body.isEmpty, "\(path): HEAD carried a body: \(body.prefix(80))")
            #expect(
                reply.lowercased().contains("content-length: 0"),
                "\(path): HEAD did not declare an empty body: \(reply)")
        }
        try await server.shutdown()
    }

    @Test func pipelinedStreamingThenHealthResponsesRemainOrdered() async throws {
        let backend = PipelinedRequestBackend()
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            heartbeatInterval: .seconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)

        let body =
            #"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
        let firstRequest =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body
        let secondRequest =
            "GET /health HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Connection: close\r\n"
            + "\r\n"
        try writeAll(socket: socket, text: firstRequest + secondRequest)
        let waitDeadline = ContinuousClock.now + .seconds(2)
        while await !backend.isWaiting, ContinuousClock.now < waitDeadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(await backend.isWaiting)

        var response = try readAvailable(socket: socket, timeoutMilliseconds: 200)
        #expect(response.contains("text/event-stream"))
        #expect(response.components(separatedBy: "HTTP/1.1 200").count - 1 == 1)
        #expect(!response.contains(#""status":"ok""#))

        await backend.release()
        response += try readUntil(
            socket: socket,
            timeoutMilliseconds: 2_000,
            condition: { $0.contains(#""status":"ok""#) })
        #expect(response.components(separatedBy: "HTTP/1.1 200").count - 1 == 2)
        let done = try #require(response.range(of: "data: [DONE]"))
        let health = try #require(response.range(of: #""status":"ok""#))
        #expect(done.lowerBound < health.lowerBound)
        #expect(await backend.generationCount == 1)

        Darwin.close(socket)
        try await server.shutdown()
    }

    @Test func shutdownAfterListenerClosesIsIdempotent() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)

        try await channel.close().get()
        try await server.shutdown()
        try await server.shutdown()
    }

    @Test func shutdownCancelsActiveAndQueuedRequestsBeforeReturning() async throws {
        let backend = CancellableServerBackend()
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let firstSocket = try connectedSocket(port: port)
        let secondSocket = try connectedSocket(port: port)
        defer {
            Darwin.close(firstSocket)
            Darwin.close(secondSocket)
        }
        let body =
            #"{"model":"test-model","messages":[{"role":"user","content":"wait"}]}"#
        let request =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body

        try writeAll(socket: firstSocket, text: request)
        let activeDeadline = ContinuousClock.now + .seconds(2)
        while await backend.startedCount != 1, ContinuousClock.now < activeDeadline {
            await Task.yield()
        }
        #expect(await backend.startedCount == 1)

        try writeAll(socket: secondSocket, text: request)
        let queuedDeadline = ContinuousClock.now + .seconds(2)
        while await server.queuedRequestCount != 1, ContinuousClock.now < queuedDeadline {
            await Task.yield()
        }
        #expect(await server.queuedRequestCount == 1)
        #expect(await server.acceptedConnectionCount == 2)

        try await server.shutdown()

        #expect(await backend.cancellationCount == 1)
        #expect(await backend.startedCount == 1)
        #expect(await server.queuedRequestCount == 0)
        #expect(await !server.hasActiveRequest)
        #expect(await server.acceptedConnectionCount == 0)
        try await server.shutdown()
    }

    // MARK: - Model unload endpoint

    @Test func unloadEndpointReleasesTheModel() async throws {
        let counter = LoadCounter()
        let managed = ManagedModelBackend(
            plan: ModelSessionPlan(
                modelDirectory: URL(fileURLWithPath: "/nonexistent/model"),
                maxContext: 4_096,
                promptCacheMode: .multiPrefix,
                promptCacheMaximumEntries: 1,
                promptCacheMemoryLimitBytes: 1_048_576,
                promptCacheDiskDirectory: nil,
                promptCacheDiskLimitBytes: 1_048_576,
                prefillChunkTokens: nil,
                expertCacheSlots: nil,
                mtpModelDirectory: nil,
                mtpMemoryMiB: 0),
            facts: ModelSessionFacts(
                modelID: "test-model",
                prefillChunkTokens: 4_096,
                promptCacheMode: .multiPrefix),
            idleTimeout: nil,
            loader: { _, _ in
                counter.increment()
                return ScriptedServerBackend()
            })
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: managed)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        // The first completion loads the model.
        var completion = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        completion.httpMethod = "POST"
        completion.setValue("application/json", forHTTPHeaderField: "content-type")
        completion.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
            """#.utf8)
        let (_, response) = try await URLSession.shared.data(for: completion)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        #expect(counter.count == 1)

        // The unload endpoint releases it.
        var unload = URLRequest(
            url: try localURL(port: port, "/v1/models/unload"))
        unload.httpMethod = "POST"
        let (data, unloadResponse) = try await URLSession.shared.data(for: unload)
        #expect((unloadResponse as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["unloaded"] as? Bool == true)

        // A second completion reloads on demand.
        let (_, secondResponse) = try await URLSession.shared.data(for: completion)
        #expect((secondResponse as? HTTPURLResponse)?.statusCode == 200)
        #expect(counter.count == 2)

        try await server.shutdown()
    }

    @Test func unloadEndpointIsANoOpWithoutResidency() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        var unload = URLRequest(
            url: try localURL(port: port, "/v1/models/unload"))
        unload.httpMethod = "POST"
        let (data, response) = try await URLSession.shared.data(for: unload)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["unloaded"] as? Bool == false)

        try await server.shutdown()
    }

    @Test func unloadEndpointRejectsNonPostMethods() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (_, response) = try await URLSession.shared.data(
            from: try localURL(port: port, "/v1/models/unload"))
        #expect((response as? HTTPURLResponse)?.statusCode == 405)

        try await server.shutdown()
    }

    /// AUD-147's content-type gate is what stands between a page the operator
    /// visits and a mutating route on this loopback port, and `/v1/models/unload`
    /// was the one POST route in the table that did not carry it. A form POST
    /// cannot name `application/json`, so it is refused here; the request that
    /// the CLI and the tests above make -- `curl -X POST`, no content-type --
    /// still works, because the absent header is that route's contract.
    @Test func unloadRefusesABrowserFormPost() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for encoding in ["application/x-www-form-urlencoded", "text/plain", "multipart/form-data"] {
            var request = URLRequest(
                url: try localURL(port: port, "/v1/models/unload"))
            request.httpMethod = "POST"
            request.setValue(encoding, forHTTPHeaderField: "content-type")
            request.httpBody = Data("x=1".utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect(
                (response as? HTTPURLResponse)?.statusCode == 415,
                "\(encoding) is a form, not a client")
            #expect(data.lossyUTF8String.contains("unsupported_media_type"))
            #expect(!data.lossyUTF8String.contains("\"unloaded\""))
        }

        try await server.shutdown()
    }

    /// The five routes that *do* carry the gate had no test for it: nothing in the
    /// suite ever sent a non-JSON content type, so the guard AUD-147 added was
    /// pinned only by the reviewer's memory. Half a fix in this repository's terms,
    /// and the LAN sibling has three such tests.
    ///
    /// The two surfaces are checked apart, because they answer apart: the OpenAI
    /// envelope carries the `unsupported_media_type` code, and the Anthropic one is
    /// `{"type":"error","error":{"type":"invalid_request_error","message":...}}`
    /// with no code field at all — which the first draft of this test mistook for a
    /// missing gate until it printed the body. The message is matched up to its
    /// slash, because the encoder escapes `/` as `\/` and the full sentence does not
    /// appear in either body as written.
    @Test func everyJsonRouteRefusesANonJsonContentType() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let openAIRoutes = ["/v1/chat/completions", "/v1/responses", "/v1/responses/compact"]
        let anthropicRoutes = ["/v1/messages", "/v1/messages/count_tokens"]
        for route in openAIRoutes + anthropicRoutes {
            var request = URLRequest(url: try localURL(port: port, route))
            request.httpMethod = "POST"
            request.setValue("text/plain", forHTTPHeaderField: "content-type")
            request.httpBody = Data(
                #"""
                {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
                """#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            let text = data.lossyUTF8String
            #expect(
                (response as? HTTPURLResponse)?.statusCode == 415,
                "\(route) accepts a form POST")
            #expect(
                text.contains("content-type must be application"),
                "\(route) answered 415 without naming the rule")
            if openAIRoutes.contains(route) {
                #expect(text.contains("unsupported_media_type"), "\(route) lost its code")
            } else {
                #expect(text.contains("invalid_request_error"), "\(route) lost its type")
            }
        }

        try await server.shutdown()
    }

    // MARK: - SSE edge cases (T30)

    /// A backend failure mid-stream must emit the error frame and still
    /// terminate with [DONE] — the terminal marker may never go missing on
    /// the error path (S5/S20).
    @Test func errorDuringStreamEmitsErrorFrameAndTerminalDone() async throws {
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ThrowingMidStreamBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: try localURL(port: port, "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(
            #"""
            {"model":"test-model","messages":[{"role":"user","content":"fail"}],
             "stream":true}
            """#.utf8)
        let text = try await URLSession.shared.data(for: request).0.lossyUTF8String

        // The partial content arrived, then the error envelope...
        #expect(text.contains(#""content":"partial""#))
        #expect(text.contains(#""code":"synthetic_stream_error""#))
        #expect(text.contains("synthetic mid-stream failure"))
        // ...and the stream still terminated with [DONE] (never missing on
        // the error path), with the error frame preceding the terminal.
        #expect(text.hasSuffix("data: [DONE]\n\n"))
        let errorRange = try #require(text.range(of: "synthetic_stream_error"))
        let doneRange = try #require(text.range(of: "data: [DONE]"))
        #expect(errorRange.lowerBound < doneRange.lowerBound)

        try await server.shutdown()
    }

    /// A client that disconnects mid-stream must cancel the in-flight
    /// generation (S25: channelInactive cancels the active task).
    @Test func midStreamClientDisconnectCancelsGeneration() async throws {
        let backend = CancellableServerBackend()
        // A short heartbeat makes the server notice the dead peer quickly via
        // a failed ping write.
        let server = TinyTitanHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            heartbeatInterval: .milliseconds(50))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        let body =
            #"{"model":"test-model","messages":[{"role":"user","content":"wait"}],"stream":true}"#
        let request =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: keep-alive\r\n"
            + "\r\n"
            + body
        try writeAll(socket: socket, text: request)

        // The stream must actually start before we disconnect, so the cancel
        // lands on a live generation rather than a not-yet-routed request.
        _ = try readUntil(socket: socket, timeoutMilliseconds: 2_000) {
            $0.contains("data:")
        }
        let startedDeadline = ContinuousClock.now + .seconds(2)
        while await backend.startedCount != 1, ContinuousClock.now < startedDeadline {
            await Task.yield()
        }
        #expect(await backend.startedCount == 1)

        Darwin.close(socket)
        let cancelledDeadline = ContinuousClock.now + .seconds(2)
        while await backend.cancellationCount != 1, ContinuousClock.now < cancelledDeadline {
            await Task.yield()
        }
        #expect(
            await backend.cancellationCount == 1,
            "mid-stream client disconnect did not cancel the generation")

        try await server.shutdown()
    }

    /// `X-TinyTitan-Workspace` has to reach the generation path, over a real
    /// socket, on both API surfaces.
    ///
    /// A unit test on the header parser is not enough and was not: the
    /// parser was correct and the feature still did nothing, because the
    /// handler clears its stored request head when the body ends and the
    /// lookup ran after that, reading nil every time. Only a request that
    /// travels the whole way catches that.
    @Test func theWorkspaceHeaderReachesTheBackend() async throws {
        let backend = WorkspaceRecordingBackend()
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        func send(_ path: String, body: String, workspace: String?) async throws {
            var request = URLRequest(url: try localURL(port: port, path))
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            if let workspace {
                request.setValue(workspace, forHTTPHeaderField: "X-TinyTitan-Workspace")
            }
            request.httpBody = Data(body.utf8)
            _ = try await URLSession.shared.data(for: request)
        }

        let chat = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
        try await send("/v1/chat/completions", body: chat, workspace: "proj-alpha")
        try await send("/v1/chat/completions", body: chat, workspace: nil)
        // `input` is a list of items on this surface, not a bare string.
        try await send(
            "/v1/responses",
            body: #"{"model":"test-model","input":[{"role":"user","content":"hi"}]}"#,
            workspace: "proj-beta")
        // And the Anthropic surface, which arrived on a branch written before
        // the header was wired at all.
        try await send(
            "/v1/messages",
            body:
                #"{"model":"test-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
            workspace: "proj-gamma")

        #expect(backend.workspaces == ["proj-alpha", nil, "proj-beta", "proj-gamma"])
        try await server.shutdown()
    }

}

private enum RawSocketError: Error {
    case systemCall(String, Int32)
    case timeout
}

private func connectedSocket(port: Int) throws -> Int32 {
    let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else {
        throw RawSocketError.systemCall("socket", errno)
    }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = in_port_t(port).bigEndian
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let result = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard result == 0 else {
        let code = errno
        Darwin.close(descriptor)
        throw RawSocketError.systemCall("connect", code)
    }
    return descriptor
}

private func writeAll(socket: Int32, text: String) throws {
    let bytes = Array(text.utf8)
    var written = 0
    while written < bytes.count {
        let count = bytes.withUnsafeBytes { buffer -> Int in
            // The loop condition guarantees a byte remains, so the base address
            // is present; -1 reports the impossible case through the error path
            // below rather than trapping inside the socket helper.
            guard let base = buffer.baseAddress else { return -1 }
            return Darwin.send(
                socket, base.advanced(by: written),
                bytes.count - written, 0)
        }
        guard count > 0 else {
            throw RawSocketError.systemCall("send", errno)
        }
        written += count
    }
}

private func readAvailable(socket: Int32, timeoutMilliseconds: Int32) throws -> String {
    var result: [UInt8] = []
    var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
    while Darwin.poll(&descriptor, 1, timeoutMilliseconds) > 0 {
        var buffer = [UInt8](repeating: 0, count: 4_096)
        let count = Darwin.recv(socket, &buffer, buffer.count, 0)
        guard count >= 0 else {
            throw RawSocketError.systemCall("recv", errno)
        }
        if count == 0 { break }
        result.append(contentsOf: buffer.prefix(count))
        descriptor.revents = 0
    }
    return result.lossyUTF8String
}

private func readUntil(
    socket: Int32,
    timeoutMilliseconds: Int32,
    condition: (String) -> Bool
) throws -> String {
    let deadline = Date().addingTimeInterval(Double(timeoutMilliseconds) / 1_000)
    var result = ""
    while Date() < deadline {
        result += try readAvailable(socket: socket, timeoutMilliseconds: 50)
        if condition(result) { return result }
    }
    throw RawSocketError.timeout
}
