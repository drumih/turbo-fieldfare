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

private actor QueueHeartbeatBackend: ServerInferenceBackend {
    private var firstContinuation: CheckedContinuation<Void, Never>?
    private(set) var startedCount = 0

    var firstIsWaiting: Bool { firstContinuation != nil }

    func releaseFirst() {
        firstContinuation?.resume()
        firstContinuation = nil
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        startedCount += 1
        if startedCount == 1 {
            await withCheckedContinuation { firstContinuation = $0 }
        }
        onEvent(.content("hello"))
        return ServerCompletion(
            content: "hello",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 3, completionTokens: 1, totalTokens: 4))
    }
}

private actor CancellationIgnoringPreparationBackend: ServerInferenceBackend {
    let delay: Duration
    private(set) var preparationStarted = false
    private(set) var cancellationCount = 0
    private(set) var generationCount = 0

    init(delay: Duration = .seconds(30)) {
        self.delay = delay
    }

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        preparationStarted = true
        do {
            try await Task.sleep(for: delay)
        } catch is CancellationError {
            cancellationCount += 1
        }
        return ServerPreparedRequest(request: request)
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        generationCount += 1
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private actor InvalidRequestServerBackend: ServerInferenceBackend {
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

private actor PreflightRejectingServerBackend: ServerInferenceBackend {
    private(set) var generationCount = 0

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        throw ServerRequestError.invalid(
            message: "prompt exceeds the configured context",
            param: "messages",
            code: "context_length_exceeded")
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        generationCount += 1
        return ServerCompletion(
            content: "unexpected",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private actor AdmissionBlockingPreparationBackend: ServerInferenceBackend {
    private var continuations: [CheckedContinuation<Void, Never>] = []
    private var released = false
    private(set) var preparationCount = 0

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        preparationCount += 1
        if !released {
            await withCheckedContinuation { continuations.append($0) }
        }
        try Task.checkCancellation()
        return ServerPreparedRequest(request: request)
    }

    func releaseAll() {
        released = true
        let waiting = continuations
        continuations.removeAll()
        for continuation in waiting {
            continuation.resume()
        }
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        ServerCompletion(
            content: "ready",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private enum TestGenerationError: Error, Sendable {
    case sensitiveFailure
}

private actor FailingServerBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw TestGenerationError.sensitiveFailure
    }
}

private actor BatchFailingServerBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw ServerRequestError.invalid(message: "synthetic failure",
                                         param: nil,
                                         code: "synthetic_failure")
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

    @Test func queryStringDoesNotChangeRoute() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let models = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models?xcode=1")!)
        #expect((models.1 as? HTTPURLResponse)?.statusCode == 200)
        #expect(String(decoding: models.0, as: UTF8.self).contains("test-model"))

        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions?client=xcode")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}]}
        """#.utf8)
        let completion = try await URLSession.shared.data(for: request)
        #expect((completion.1 as? HTTPURLResponse)?.statusCode == 200)
        #expect(String(decoding: completion.0, as: UTF8.self).contains("hello"))

        try await server.shutdown()
    }

    @Test func streamingPreflightErrorUsesHTTPStatusBeforeSSEStarts() async throws {
        let backend = PreflightRejectingServerBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true}
        """#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect((response as? HTTPURLResponse)?.value(
            forHTTPHeaderField: "content-type") == "application/json")
        #expect(String(decoding: data, as: UTF8.self).contains(
            #""code":"context_length_exceeded""#))
        #expect(await backend.generationCount == 0)

        try await server.shutdown()
    }

    @Test func unsupportedToolSchemaReturnsHTTP400BeforeSSEStarts() async throws {
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
        {
          "model":"test-model",
          "messages":[{"role":"user","content":"hi"}],
          "stream":true,
          "tools":[{"type":"function","function":{
            "name":"unsafe",
            "parameters":{"type":"object","properties":{
              "value":{"anyOf":[{"type":"string"},{"type":"object"}]}
            }}
          }}]
        }
        """#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"invalid_tool_schema""#))
        #expect(text.contains(#""param":"tools""#))
        #expect(!text.contains("data:"))

        try await server.shutdown()
    }

    @Test func streamingRequestErrorUsesEnvelopeAndCompletesTransport() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: InvalidRequestServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true}
        """#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""code":"context_length_exceeded""#))
        #expect(text.contains(#""param":"messages""#))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func streamingInternalErrorIsMaskedAndCompletesTransport() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: FailingServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"model":"test-model","messages":[{"role":"user","content":"hi"}],
         "stream":true}
        """#.utf8)

        let data = try await URLSession.shared.data(for: request).0
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains(#""type":"server_error""#))
        #expect(text.contains(#""code":"internal_error""#))
        #expect(!text.contains("sensitiveFailure"))
        #expect(text.hasSuffix("data: [DONE]\n\n"))

        try await server.shutdown()
    }

    @Test func batchRunsNonStreamingRequestsInOrder() async throws {
        let outputDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TurboFieldfareBatchTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: outputDirectory) }
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend(),
            batchOutputDirectory: outputDirectory)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"requests":[
          {"custom_id":"first","model":"test-model","messages":[{"role":"user","content":"first"}]},
          {"model":"test-model","messages":[{"role":"user","content":"second"}]}
        ]}
        """#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["object"] as? String == "batch")
        let batchID = try #require(object["id"] as? String)
        #expect(object["output_file_id"] is NSNull || object["output_file_id"] == nil)
        #expect((object["request_counts"] as? [String: Any])?["total"] as? Int == 2)

        let listData = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/batches?limit=1")!).0
        let list = try #require(JSONSerialization.jsonObject(with: listData) as? [String: Any])
        #expect(list["object"] as? String == "list")
        #expect((list["data"] as? [[String: Any]])?.first?["id"] as? String == batchID)
        #expect(list["has_more"] as? Bool == false)

        var status = ""
        var outputFileID: String?
        var usage: [String: Any]?
        for _ in 0..<100 {
            let statusData = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(batchID)")!).0
            let snapshot = try #require(JSONSerialization.jsonObject(with: statusData) as? [String: Any])
            status = try #require(snapshot["status"] as? String)
            if status == "completed" {
                outputFileID = try #require(snapshot["output_file_id"] as? String)
                usage = try #require(snapshot["usage"] as? [String: Any])
                break
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(status == "completed")
        #expect(outputFileID?.hasPrefix("file-") == true)
        #expect((usage?["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 0)
        let outputData = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/files/\(try #require(outputFileID))/content")!).0
        let output = String(decoding: outputData, as: UTF8.self)
        let filesData = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/files?purpose=batch_output&limit=1&order=desc")!).0
        let filesObject = try #require(JSONSerialization.jsonObject(with: filesData) as? [String: Any])
        let files = try #require(filesObject["data"] as? [[String: Any]])
        let listedOutput = try #require(files.first { $0["id"] as? String == outputFileID })
        #expect(listedOutput["purpose"] as? String == "batch_output")
        #expect(filesObject["has_more"] as? Bool == false)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 2)
        #expect(lines[0].contains("\"custom_id\":\"first\""))
        #expect(lines.allSatisfy { $0.contains("\"status_code\":200") })

        var reuseOutput = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        reuseOutput.httpMethod = "POST"
        reuseOutput.setValue("application/json", forHTTPHeaderField: "content-type")
        reuseOutput.httpBody = Data(#"{"input_file_id":"\#(try #require(outputFileID))","endpoint":"/v1/chat/completions","completion_window":"24h"}"#.utf8)
        let (reuseData, reuseResponse) = try await URLSession.shared.data(for: reuseOutput)
        #expect((reuseResponse as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: reuseData, as: UTF8.self).contains("input_file_id"))

        try await server.shutdown()
    }

    @Test func batchRejectsStreamingItems() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"requests":[{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}]}
        """#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("unsupported_value"))

        try await server.shutdown()
    }

    @Test func batchRejectsOversizedMetadataBeforeReadingInputFile() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let metadata = Dictionary(uniqueKeysWithValues: (0..<17).map { ("key\($0)", "value") })
        let body: [String: Any] = ["input_file_id": "file-missing", "endpoint": "/v1/chat/completions",
                                   "completion_window": "24h", "metadata": metadata]
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("metadata"))
        try await server.shutdown()
    }

    @Test func malformedBatchJSONUsesInvalidJSONError() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data("{".utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("invalid_json"))
        try await server.shutdown()
    }

    @Test func batchAcceptsOpenAIFileAndJSONLContract() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TurboFieldfareBatchFileTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend(), batchOutputDirectory: directory)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let boundary = "batch-test-boundary"
        let jsonl = #"{"custom_id":"row-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"hi"}]}}"# + "\n" +
            #"{"custom_id":"row-2","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"again"}]}}"# + "\n"
        let multipart = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n\(jsonl)\r\n--\(boundary)--\r\n"
        var upload = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/files")!)
        upload.httpMethod = "POST"
        upload.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
        upload.httpBody = Data(multipart.utf8)
        let (fileData, fileResponse) = try await URLSession.shared.data(for: upload)
        #expect((fileResponse as? HTTPURLResponse)?.statusCode == 200)
        let file = try #require(JSONSerialization.jsonObject(with: fileData) as? [String: Any])
        let fileID = try #require(file["id"] as? String)
        let files = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/files")!).0
        #expect((try #require(JSONSerialization.jsonObject(with: files) as? [String: Any]))["data"] as? [[String: Any]] != nil)
        var create = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        create.httpMethod = "POST"
        create.setValue("application/json", forHTTPHeaderField: "content-type")
        create.httpBody = Data(#"{"input_file_id":"\#(fileID)","endpoint":"/v1/chat/completions","completion_window":"24h","output_expires_after":{"anchor":"created_at","seconds":3600}}"#.utf8)
        let (batchData, batchResponse) = try await URLSession.shared.data(for: create)
        #expect((batchResponse as? HTTPURLResponse)?.statusCode == 200)
        let batch = try #require(JSONSerialization.jsonObject(with: batchData) as? [String: Any])
        #expect(batch["object"] as? String == "batch")
        #expect(batch["model"] as? String == "test-model")
        #expect(batch["input_file_id"] as? String == fileID)
        #expect(batch["completion_window"] as? String == "24h")
        #expect((batch["request_counts"] as? [String: Any])?["total"] as? Int == 2)
        #expect(batch["errors"] is NSNull || batch["errors"] == nil)
        #expect((batch["expires_at"] as? Int ?? 0) > (batch["created_at"] as? Int ?? 0))
        let batchID = try #require(batch["id"] as? String)
        var outputFileID: String?
        for _ in 0..<20 {
            let statusData = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(batchID)")!).0
            let status = try #require(JSONSerialization.jsonObject(with: statusData) as? [String: Any])
            if status["status"] as? String == "completed" {
                outputFileID = try #require(status["output_file_id"] as? String)
                break
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        let output = try #require(outputFileID)
        let outputFile = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/files/\(output)")!).0
        let outputObject = try #require(JSONSerialization.jsonObject(with: outputFile) as? [String: Any])
        #expect((outputObject["expires_at"] as? Int ?? 0) > (outputObject["created_at"] as? Int ?? 0))
        let outputLines = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/files/\(output)/content")!).0
        #expect(String(decoding: outputLines, as: UTF8.self).split(separator: "\n").count == 2)
        try await server.shutdown()
    }

    @Test func batchRejectsInvalidOutputExpiryPolicy() async throws {
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let body: [String: Any] = ["input_file_id": "file-missing", "endpoint": "/v1/chat/completions",
                                   "completion_window": "24h",
                                   "output_expires_after": ["anchor": "created_at", "seconds": 1]]
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("output_expires_after"))
        try await server.shutdown()
    }

    @Test func invalidBatchJSONLReturnsFailedBatchWithErrors() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TurboFieldfareInvalidBatchTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend(), batchOutputDirectory: directory)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let boundary = "invalid-batch-boundary"
        let multipart = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\nnot-json\n\r\n--\(boundary)--\r\n"
        var upload = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/files")!)
        upload.httpMethod = "POST"
        upload.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
        upload.httpBody = Data(multipart.utf8)
        let (fileData, _) = try await URLSession.shared.data(for: upload)
        let file = try #require(JSONSerialization.jsonObject(with: fileData) as? [String: Any])
        let fileID = try #require(file["id"] as? String)
        var create = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        create.httpMethod = "POST"
        create.setValue("application/json", forHTTPHeaderField: "content-type")
        create.httpBody = Data(#"{"input_file_id":"\#(fileID)","endpoint":"/v1/chat/completions","completion_window":"24h"}"#.utf8)
        let (batchData, response) = try await URLSession.shared.data(for: create)
        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        let batch = try #require(JSONSerialization.jsonObject(with: batchData) as? [String: Any])
        #expect(batch["status"] as? String == "failed")
        let errors = try #require(batch["errors"] as? [String: Any])
        #expect(errors["object"] as? String == "list")
        let data = try #require(errors["data"] as? [[String: Any]])
        #expect(!(data).isEmpty)
        #expect(data[0]["line"] as? Int == 1)
        try await server.shutdown()
    }

    @Test func blankBatchJSONLLineReturnsItsLineNumber() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TurboFieldfareBlankBatchTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend(), batchOutputDirectory: directory)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let boundary = "blank-batch-boundary"
        let requestLine = #"{"custom_id":"row-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"hi"}]}}"#
        let secondRequestLine = requestLine.replacingOccurrences(of: "row-1", with: "row-2")
        let multipart = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\nnot-json\n\n\(requestLine)\n\(secondRequestLine)\n\r\n--\(boundary)--\r\n"
        var upload = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/files")!)
        upload.httpMethod = "POST"
        upload.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
        upload.httpBody = Data(multipart.utf8)
        let (fileData, _) = try await URLSession.shared.data(for: upload)
        let file = try #require(JSONSerialization.jsonObject(with: fileData) as? [String: Any])
        let fileID = try #require(file["id"] as? String)
        var create = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        create.httpMethod = "POST"
        create.setValue("application/json", forHTTPHeaderField: "content-type")
        create.httpBody = Data(#"{"input_file_id":"\#(fileID)","endpoint":"/v1/chat/completions","completion_window":"24h"}"#.utf8)
        let (batchData, _) = try await URLSession.shared.data(for: create)
        let batch = try #require(JSONSerialization.jsonObject(with: batchData) as? [String: Any])
        let errors = try #require(batch["errors"] as? [String: Any])
        let data = try #require(errors["data"] as? [[String: Any]])
        #expect(data.map { $0["line"] as? Int } == [1, 2])
        try await server.shutdown()
    }

    @Test func batchFilesAreDiscardedOnServerRestart() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TurboFieldfareRestartBatchTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                             backend: ScriptedServerBackend(), batchOutputDirectory: directory)
        let firstChannel = try await first.start(port: 0)
        let firstPort = try #require(firstChannel.localAddress?.port)
        let boundary = "restart-batch-boundary"
        let jsonl = #"{"custom_id":"row-1","method":"POST","url":"/v1/chat/completions","body":{"model":"test-model","messages":[{"role":"user","content":"hi"}]}}"# + "\n"
        let multipart = "--\(boundary)\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n\(jsonl)\r\n--\(boundary)--\r\n"
        var upload = URLRequest(url: URL(string: "http://127.0.0.1:\(firstPort)/v1/files")!)
        upload.httpMethod = "POST"
        upload.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "content-type")
        upload.httpBody = Data(multipart.utf8)
        let (fileData, _) = try await URLSession.shared.data(for: upload)
        let file = try #require(JSONSerialization.jsonObject(with: fileData) as? [String: Any])
        let fileID = try #require(file["id"] as? String)
        try await first.shutdown()

        let second = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1,
                                              backend: ScriptedServerBackend(), batchOutputDirectory: directory)
        let secondChannel = try await second.start(port: 0)
        let secondPort = try #require(secondChannel.localAddress?.port)
        let (_, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(secondPort)/v1/files/\(fileID)")!)
        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        try await second.shutdown()
    }

    @Test func batchListPaginatesWithAfterCursor() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 3,
            backend: ScriptedServerBackend(delayNanoseconds: 20_000_000))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var createdIDs: [String] = []
        for index in 0..<3 {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data(#"""
            {"requests":[{"model":"test-model","messages":[{"role":"user","content":"\#(index)"}]}]}
            """#.utf8)
            let data = try await URLSession.shared.data(for: request).0
            let batch = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            createdIDs.append(try #require(batch["id"] as? String))
        }

        let firstPage = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/batches?limit=2")!).0
        let firstObject = try #require(JSONSerialization.jsonObject(with: firstPage) as? [String: Any])
        let firstData = try #require(firstObject["data"] as? [[String: Any]])
        #expect(firstData.count == 2)
        #expect(firstObject["has_more"] as? Bool == true)
        let after = try #require(firstObject["last_id"] as? String)

        let secondPage = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/batches?limit=2&after=\(after)")!).0
        let secondObject = try #require(JSONSerialization.jsonObject(with: secondPage) as? [String: Any])
        #expect((secondObject["data"] as? [[String: Any]])?.count == 1)
        #expect(secondObject["has_more"] as? Bool == false)
        #expect(Set(createdIDs).count == 3)

        try await server.shutdown()
    }

    @Test func batchCancellationReachesCancelledWithoutFailure() async throws {
        let backend = CancellableServerBackend()
        let server = TurboFieldfareHTTPServer(modelID: "test-model", queueLimit: 1, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"requests":[
          {"model":"test-model","messages":[{"role":"user","content":"one"}]},
          {"model":"test-model","messages":[{"role":"user","content":"two"}]}
        ]}
        """#.utf8)
        let batchData = try await URLSession.shared.data(for: request).0
        let batch = try #require(JSONSerialization.jsonObject(with: batchData) as? [String: Any])
        let id = try #require(batch["id"] as? String)
        for _ in 0..<100 where await backend.startedCount == 0 {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(await backend.startedCount == 1)

        var cancellation = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(id)/cancel")!)
        cancellation.httpMethod = "POST"
        let cancellationData = try await URLSession.shared.data(for: cancellation).0
        #expect(String(decoding: cancellationData, as: UTF8.self).contains("cancelling"))

        var status = ""
        var counts: [String: Any] = [:]
        for _ in 0..<100 {
            let statusData = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(id)")!).0
            let snapshot = try #require(JSONSerialization.jsonObject(with: statusData) as? [String: Any])
            status = try #require(snapshot["status"] as? String)
            counts = try #require(snapshot["request_counts"] as? [String: Any])
            if status == "cancelled" { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(status == "cancelled")
        #expect(counts["failed"] as? Int == 0)
        #expect(await backend.cancellationCount == 1)

        try await server.shutdown()
    }

    @Test func batchFailureWritesJSONLErrorRecord() async throws {
        let outputDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TurboFieldfareBatchFailureTest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: outputDirectory) }
        let server = TurboFieldfareHTTPServer(modelID: "test-model",
                                              queueLimit: 1,
                                              backend: BatchFailingServerBackend(),
                                              batchOutputDirectory: outputDirectory)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/batches")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"""
        {"requests":[{"custom_id":"will-fail","model":"test-model","messages":[{"role":"user","content":"fail"}]}]}
        """#.utf8)
        let batchData = try await URLSession.shared.data(for: request).0
        let batch = try #require(JSONSerialization.jsonObject(with: batchData) as? [String: Any])
        let id = try #require(batch["id"] as? String)
        var status = ""
        for _ in 0..<100 {
            let data = try await URLSession.shared.data(
                from: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(id)")!).0
            status = try #require((JSONSerialization.jsonObject(with: data) as? [String: Any])?["status"] as? String)
            if status == "completed" { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(status == "completed")
        let final = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/batches/\(id)")!).0
        let finalObject = try #require(JSONSerialization.jsonObject(with: final) as? [String: Any])
        let errorFileID = try #require(finalObject["error_file_id"] as? String)
        #expect(finalObject["output_file_id"] is NSNull || finalObject["output_file_id"] == nil)
        let output = try String(contentsOf: outputDirectory.appendingPathComponent(errorFileID)
            .appendingPathExtension("jsonl"), encoding: .utf8)
        #expect(output.contains("\"custom_id\":\"will-fail\""))
        #expect(output.contains("\"response\":null"))
        #expect(output.contains("\"synthetic_failure\""))
        #expect(output.contains("\"type\":\"invalid_request_error\""))

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

    @Test func queuedStreamingRequestReceivesHeartbeatBeforeItsTurn() async throws {
        let backend = QueueHeartbeatBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend,
            heartbeatInterval: .milliseconds(100))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let firstSocket = try connectedSocket(port: port)
        let secondSocket = try connectedSocket(port: port)
        defer {
            Darwin.close(firstSocket)
            Darwin.close(secondSocket)
        }

        let blockingBody =
            #"{"model":"test-model","messages":[{"role":"user","content":"first"}]}"#
        let streamingBody =
            #"{"model":"test-model","messages":[{"role":"user","content":"second"}],"stream":true}"#
        try writeAll(socket: firstSocket, text: httpRequest(
            port: port, body: blockingBody, connection: "keep-alive"))
        let activeDeadline = ContinuousClock.now + .seconds(2)
        while await !backend.firstIsWaiting, ContinuousClock.now < activeDeadline {
            await Task.yield()
        }
        #expect(await backend.firstIsWaiting)

        try writeAll(socket: secondSocket, text: httpRequest(
            port: port, body: streamingBody, connection: "close"))
        let queuedDeadline = ContinuousClock.now + .seconds(2)
        while await server.queuedRequestCount != 1, ContinuousClock.now < queuedDeadline {
            await Task.yield()
        }
        #expect(await server.queuedRequestCount == 1)

        let queuedResponse: String
        do {
            queuedResponse = try readUntil(
                socket: secondSocket,
                timeoutMilliseconds: 1_000,
                condition: { $0.contains(": ping\n\n") })
        } catch {
            await backend.releaseFirst()
            try? await server.shutdown()
            throw error
        }
        #expect(queuedResponse.contains("HTTP/1.1 200 OK"))
        #expect(queuedResponse.contains(#""role":"assistant""#))
        #expect(queuedResponse.contains(": ping\n\n"))
        #expect(await backend.startedCount == 1)

        await backend.releaseFirst()
        let completed = queuedResponse + (try readUntil(
            socket: secondSocket,
            timeoutMilliseconds: 2_000,
            condition: { $0.contains("data: [DONE]\n\n") }))
        #expect(completed.contains("data: [DONE]\n\n"))
        #expect(await backend.startedCount == 2)

        try await server.shutdown()
    }

    @Test func queueLimitBoundsRequestsBeforePreparation() async throws {
        let backend = AdmissionBlockingPreparationBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let sockets = try (0..<3).map { _ in try connectedSocket(port: port) }
        defer { sockets.forEach { _ = Darwin.close($0) } }
        let body = #"{"model":"test-model","messages":[{"role":"user","content":"wait"}]}"#

        try writeAll(socket: sockets[0], text: httpRequest(
            port: port, body: body, connection: "close"))
        try writeAll(socket: sockets[1], text: httpRequest(
            port: port, body: body, connection: "close"))
        let preparationDeadline = ContinuousClock.now + .seconds(2)
        while await backend.preparationCount != 2,
              ContinuousClock.now < preparationDeadline {
            await Task.yield()
        }
        #expect(await backend.preparationCount == 2)

        try writeAll(socket: sockets[2], text: httpRequest(
            port: port, body: body, connection: "close"))
        let rejected = try readUntil(
            socket: sockets[2],
            timeoutMilliseconds: 1_000,
            condition: { $0.contains(#""code":"queue_full""#) })
        #expect(rejected.contains("HTTP/1.1 429 Too Many Requests"))
        #expect(await backend.preparationCount == 2)

        await backend.releaseAll()
        for socket in sockets.prefix(2) {
            _ = try readUntil(
                socket: socket,
                timeoutMilliseconds: 2_000,
                condition: { $0.contains(#""content":"ready""#) })
        }
        try await server.shutdown()
    }

    @Test func disconnectDuringPreparationNeverStartsGeneration() async throws {
        let backend = CancellationIgnoringPreparationBackend(delay: .milliseconds(100))
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        let body = #"{"model":"test-model","messages":[{"role":"user","content":"wait"}],"stream":true}"#
        try writeAll(socket: socket, text: httpRequest(
            port: port, body: body, connection: "close"))

        let preparationDeadline = ContinuousClock.now + .seconds(2)
        while await !backend.preparationStarted,
              ContinuousClock.now < preparationDeadline {
            await Task.yield()
        }
        #expect(await backend.preparationStarted)
        abortSocket(socket)

        let completionDeadline = ContinuousClock.now + .seconds(2)
        while await server.acceptedConnectionCount != 0,
              ContinuousClock.now < completionDeadline {
            await Task.yield()
        }
        #expect(await backend.generationCount == 0)
        #expect(await server.acceptedConnectionCount == 0)

        try await server.shutdown()
        #expect(await !server.hasActiveRequest)
    }

    @Test func shutdownDuringPreparationNeverStartsGeneration() async throws {
        let backend = CancellationIgnoringPreparationBackend()
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model",
            queueLimit: 1,
            backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { Darwin.close(socket) }
        let body = #"{"model":"test-model","messages":[{"role":"user","content":"wait"}],"stream":true}"#
        try writeAll(socket: socket, text: httpRequest(
            port: port, body: body, connection: "keep-alive"))

        let preparationDeadline = ContinuousClock.now + .seconds(2)
        while await !backend.preparationStarted,
              ContinuousClock.now < preparationDeadline {
            await Task.yield()
        }
        #expect(await backend.preparationStarted)

        try await server.shutdown()
        #expect(await backend.cancellationCount == 1)
        #expect(await backend.generationCount == 0)
        #expect(await !server.hasActiveRequest)
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

private func httpRequest(port: Int, body: String, connection: String) -> String {
    "POST /v1/chat/completions HTTP/1.1\r\n"
        + "Host: 127.0.0.1:\(port)\r\n"
        + "Content-Type: application/json\r\n"
        + "Content-Length: \(body.utf8.count)\r\n"
        + "Connection: \(connection)\r\n"
        + "\r\n"
        + body
}

private func abortSocket(_ socket: Int32) {
    var option = linger(l_onoff: 1, l_linger: 0)
    withUnsafePointer(to: &option) {
        _ = Darwin.setsockopt(socket, SOL_SOCKET, SO_LINGER, $0,
                              socklen_t(MemoryLayout<linger>.size))
    }
    Darwin.close(socket)
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
