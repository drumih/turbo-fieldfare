import Darwin
import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private actor ScriptedServerBackend: ServerInferenceBackend {
    let delayNanoseconds: UInt64

    init(delayNanoseconds: UInt64 = 0) {
        self.delayNanoseconds = delayNanoseconds
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        if delayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
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

private struct DecodeFailure: Error {}

/// Fails after `onEvent` has already put content on the wire, the way a decode
/// failure does mid-generation.
private actor FailingMidStreamBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("partial"))
        throw DecodeFailure()
    }
}

/// Rejects the request from inside `generate`, the way an overlong prompt does
/// once the streaming head has been committed.
private actor RejectingBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw ServerRequestError.invalid(
            message: "prompt exceeds the configured context",
            param: "messages",
            code: "context_length_exceeded")
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

@Suite("OpenAI HTTP server", .serialized)
struct HTTPServerTests {
    @Test func healthModelsAndNonStreamingCompletion() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let health = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        #expect(String(decoding: health, as: UTF8.self).contains(#""status":"ok""#))

        let models = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!).0
        #expect(String(decoding: models, as: UTF8.self).contains("test-model"))

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
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

    @Test func streamingUsesStableShapeAndDoneMarker() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true,"stream_options":{"include_usage":true}}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""role":"assistant""#))
        #expect(text.contains(#""content":"hello""#))
        #expect(text.contains(#""finish_reason":"stop""#))
        #expect(text.contains(#""prompt_tokens":3"#))
        #expect(text.contains(#""cached_tokens":0"#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func streamingFailureAfterHeadReportsErrorInBand() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: FailingMidStreamBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        // A dropped connection fails this call: the body must end normally.
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""content":"partial""#))
        #expect(text.contains(#""code":"internal_error""#))
        #expect(text.contains(#""type":"server_error""#))
        #expect(!text.contains(#""finish_reason":"stop""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func streamingRejectionAfterHeadNamesTheCause() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: RejectingBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        let (data, _) = try await URLSession.shared.data(for: request)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"context_length_exceeded""#))
        #expect(text.contains(#""param":"messages""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        // The same request without a stream still gets a real status code.
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (blocking, blockingResponse) = try await URLSession.shared.data(for: request)
        #expect((blockingResponse as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: blocking, as: UTF8.self)
            .contains(#""code":"context_length_exceeded""#))

        try await server.shutdown()
    }

    @Test func wrongModelUsesOpenAIErrorEnvelope() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"wrong","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(String(decoding: data, as: UTF8.self).contains("model_not_found"))

        try await server.shutdown()
    }

    @Test func streamingHeartbeatKeepsSlowFirstEventAlive() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(delayNanoseconds: 50_000_000),
            heartbeatInterval: .milliseconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}
        """#.utf8)
        let data = try await URLSession.shared.data(for: request).0
        #expect(String(decoding: data, as: UTF8.self).contains(": ping\n\n"))

        try await server.shutdown()
    }

    @Test func streamingMultipleToolsUseDistinctIndexes() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: MultipleToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read both"}],
         "stream":true}
        """#.utf8)
        let text = String(decoding: try await URLSession.shared.data(for: request).0,
                          as: UTF8.self)
        #expect(text.contains(#""index":0"#))
        #expect(text.contains(#""index":1"#))
        #expect(text.contains(#""finish_reason":"tool_calls""#))

        request.httpBody = Data(#"""
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
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ContentAndToolBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
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

        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"read"}],"stream":true}
        """#.utf8)
        let stream = String(
            decoding: try await URLSession.shared.data(for: request).0,
            as: UTF8.self)
        #expect(stream.contains(#""content":"I will read it.""#))
        #expect(stream.contains(#""tool_calls""#))
        #expect(stream.contains(#""finish_reason":"tool_calls""#))

        try await server.shutdown()
    }

    @Test func pipelinedStreamingThenHealthResponsesRemainOrdered() async throws {
        let backend = PipelinedRequestBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            heartbeatInterval: .seconds(10))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)

        let body = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
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
        let server = TurboFieldfareHTTPServer(
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
        let server = TurboFieldfareHTTPServer(
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
        let count = bytes.withUnsafeBytes {
            Darwin.send(socket, $0.baseAddress!.advanced(by: written),
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
    return String(decoding: result, as: UTF8.self)
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
