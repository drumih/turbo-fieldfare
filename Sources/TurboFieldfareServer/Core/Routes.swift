import Foundation
import Hummingbird
import NIOCore

/// Builds the HTTP API surface: the OpenAI-compatible endpoints plus the
/// Anthropic Messages endpoint. All runtime access goes through the
/// `GenerationSession` actor, so this layer stays Metal-free.
public func buildServerRouter(session: GenerationSession) -> Router<BasicRequestContext> {
    let router = Router(context: BasicRequestContext.self)

    router.get("health") { _, _ -> Response in
        jsonResponse(.ok, HealthResponse())
    }

    router.get("v1/models") { _, _ -> Response in
        let model = ModelObject(id: await session.modelID, created: await session.created)
        return jsonResponse(.ok, ModelList(models: [model]))
    }

    router.post("v1/chat/completions") { request, context -> Response in
        await handleChatCompletions(request: request, context: context, session: session)
    }

    router.post("v1/completions") { request, context -> Response in
        await handleCompletions(request: request, context: context, session: session)
    }

    router.post("v1/messages") { request, context -> Response in
        await handleMessages(request: request, context: context, session: session)
    }

    return router
}

private func handleChatCompletions(request: Request,
                                   context: some RequestContext,
                                   session: GenerationSession) async -> Response {
    do {
        let body = try await request.decode(as: ChatCompletionRequest.self, context: context)
        if body.hasTools {
            throw GenerationRequestError.unsupportedParameter("tools is not supported")
        }
        if body.hasToolChoice {
            throw GenerationRequestError.unsupportedParameter("tool_choice is not supported")
        }
        if body.hasResponseFormat {
            throw GenerationRequestError.unsupportedParameter("response_format is not supported")
        }
        if let n = body.n, n != 1 {
            throw GenerationRequestError.unsupportedParameter("n must be 1")
        }
        let sampling = SamplingParams(maxTokens: body.maxTokens,
                                      temperature: body.temperature,
                                      topP: body.topP,
                                      seed: body.seed,
                                      stop: body.stop)
        let prepared = try await session.prepareChat(messages: body.messages, sampling: sampling)
        let includeUsage = body.streamOptions?.includeUsage ?? false
        let modelID = await session.modelID
        if body.stream == true {
            return streamingResponse(kind: .chat,
                                     session: session,
                                     prepared: prepared,
                                     modelID: modelID,
                                     includeUsage: includeUsage)
        }
        let outcome = try await session.generate(prepared)
        let response = ChatCompletionResponse(
            id: openAIID(prefix: "chatcmpl"),
            created: unixNow(),
            model: modelID,
            choices: [ChatChoice(index: 0,
                                 message: ChatChoiceMessage(content: outcome.text),
                                 finishReason: outcome.finishReason.rawValue)],
            usage: outcome.usage)
        return jsonResponse(.ok, response)
    } catch let error as HTTPError {
        return errorResponse(status: error.status,
                             message: error.body ?? "invalid request body")
    } catch let error as GenerationRequestError {
        return errorResponse(status: .badRequest, message: error.description)
    } catch {
        // Deliberately generic: runtime errors carry internal paths/Metal
        // details that must not leak into HTTP responses.
        return errorResponse(status: .internalServerError,
                             message: "generation failed",
                             type: "server_error")
    }
}

private func handleCompletions(request: Request,
                               context: some RequestContext,
                               session: GenerationSession) async -> Response {
    do {
        let body = try await request.decode(as: CompletionRequest.self, context: context)
        if let n = body.n, n != 1 {
            throw GenerationRequestError.unsupportedParameter("n must be 1")
        }
        let sampling = SamplingParams(maxTokens: body.maxTokens,
                                      temperature: body.temperature,
                                      topP: body.topP,
                                      seed: body.seed,
                                      stop: body.stop)
        let prepared = try await session.prepareCompletion(prompt: body.prompt, sampling: sampling)
        let includeUsage = body.streamOptions?.includeUsage ?? false
        let modelID = await session.modelID
        if body.stream == true {
            return streamingResponse(kind: .raw,
                                     session: session,
                                     prepared: prepared,
                                     modelID: modelID,
                                     includeUsage: includeUsage)
        }
        let outcome = try await session.generate(prepared)
        let response = CompletionResponse(
            id: openAIID(prefix: "cmpl"),
            created: unixNow(),
            model: modelID,
            choices: [CompletionChoice(index: 0,
                                       text: outcome.text,
                                       finishReason: outcome.finishReason.rawValue)],
            usage: outcome.usage)
        return jsonResponse(.ok, response)
    } catch let error as HTTPError {
        return errorResponse(status: error.status,
                             message: error.body ?? "invalid request body")
    } catch let error as GenerationRequestError {
        return errorResponse(status: .badRequest, message: error.description)
    } catch {
        // Deliberately generic: runtime errors carry internal paths/Metal
        // details that must not leak into HTTP responses.
        return errorResponse(status: .internalServerError,
                             message: "generation failed",
                             type: "server_error")
    }
}

private func handleMessages(request: Request,
                            context: some RequestContext,
                            session: GenerationSession) async -> Response {
    do {
        let body = try await request.decode(as: AnthropicMessagesRequest.self, context: context)
        guard let maxTokens = body.maxTokens else {
            throw GenerationRequestError.unsupportedParameter("max_tokens is required")
        }
        if body.hasTools {
            throw GenerationRequestError.unsupportedParameter("tools is not supported")
        }
        if body.hasToolChoice {
            throw GenerationRequestError.unsupportedParameter("tool_choice is not supported")
        }
        if body.hasThinking {
            throw GenerationRequestError.unsupportedParameter("thinking is not supported")
        }
        if body.hasMetadata {
            throw GenerationRequestError.unsupportedParameter("metadata is not supported")
        }
        if let n = body.n, n != 1 {
            throw GenerationRequestError.unsupportedParameter("n must be 1")
        }
        var messages: [ChatMessage] = []
        if let system = body.system {
            messages.append(ChatMessage(role: "system", content: system.text))
        }
        for message in body.messages {
            guard message.role == "user" || message.role == "assistant" else {
                throw GenerationRequestError.unsupportedRole(message.role)
            }
            messages.append(ChatMessage(role: message.role, content: message.content.text))
        }
        let sampling = SamplingParams(maxTokens: maxTokens,
                                      temperature: body.temperature,
                                      topP: body.topP,
                                      seed: nil,
                                      stop: body.stopSequences.map { .multiple($0) })
        let prepared = try await session.prepareChat(messages: messages, sampling: sampling)
        // `modelID` is a nonisolated `let` on the actor; no await needed.
        let modelID = session.modelID
        if body.stream == true {
            return anthropicStreamingResponse(session: session, prepared: prepared, modelID: modelID)
        }
        let outcome = try await session.generate(prepared)
        let response = AnthropicMessageResponse(
            id: anthropicMessageID(),
            model: modelID,
            content: [AnthropicTextBlock(text: outcome.text)],
            stopReason: outcome.finishReason.anthropicStopReason,
            usage: AnthropicUsage(inputTokens: outcome.promptTokens,
                                  outputTokens: outcome.completionTokens))
        return jsonResponse(.ok, response)
    } catch let error as HTTPError {
        return anthropicErrorResponse(status: error.status,
                                      message: error.body ?? "invalid request body")
    } catch let error as GenerationRequestError {
        return anthropicErrorResponse(status: .badRequest, message: error.description)
    } catch {
        // Deliberately generic: runtime errors carry internal paths/Metal
        // details that must not leak into HTTP responses.
        return anthropicErrorResponse(status: .internalServerError,
                                      message: "generation failed",
                                      type: "api_error")
    }
}

private enum StreamKind {
    case chat
    case raw
}
/// Events bridged from the generation task to the SSE body writer.
private enum StreamEvent {
    case delta(String)
    case finished(GenerationOutcome)
}

/// SSE response. The generation runs in a child task bridged through an
/// AsyncThrowingStream; each chunk is written (and flushed) as it is
/// produced. If the client disconnects, the write throws, the loop exits and
/// `defer` cancels the generation task — `runRawCompletion` observes that via
/// `Task.checkCancellation()`.
// s5d:debt(ceiling="disconnect during prefill is detected only at the first decode write — prefill yields no deltas, so a dead client still pays the full prefill", trigger="prefill progress is surfaced through TokenGenerating or the runtime checks cancellation mid-prefill")
private func streamingResponse(kind: StreamKind,
                               session: GenerationSession,
                               prepared: PreparedGeneration,
                               modelID: String,
                               includeUsage: Bool) -> Response {
    let id = openAIID(prefix: kind == .chat ? "chatcmpl" : "cmpl")
    let created = unixNow()
    let body = ResponseBody { writer in
        let (stream, continuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        let generation = Task {
            do {
                let outcome = try await session.generate(prepared) { delta in
                    continuation.yield(.delta(delta))
                }
                continuation.yield(.finished(outcome))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        defer { generation.cancel() }
        if kind == .chat {
            let roleChunk = ChatCompletionChunk(
                id: id, created: created, model: modelID,
                choices: [ChatChunkChoice(index: 0,
                                          delta: ChatChunkDelta(role: "assistant"),
                                          finishReason: nil)])
            try await writer.write(ByteBuffer(data: SSEFramer.frame(roleChunk)))
        }
        for try await event in stream {
            switch event {
            case .delta(let text):
                try await writer.write(ByteBuffer(data: SSEFramer.frame(deltaChunk(
                    kind: kind, id: id, created: created, modelID: modelID, text: text))))
            case .finished(let outcome):
                try await writer.write(ByteBuffer(data: SSEFramer.frame(finalChunk(
                    kind: kind, id: id, created: created, modelID: modelID, outcome: outcome))))
                if includeUsage {
                    try await writer.write(ByteBuffer(data: SSEFramer.frame(usageChunk(
                        kind: kind, id: id, created: created, modelID: modelID, outcome: outcome))))
                }
            }
        }
        try await writer.write(ByteBuffer(data: SSEFramer.doneFrame))
        try await writer.finish(nil)
    }
    return Response(status: .ok,
                    headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
                    body: body)
}

/// Anthropic Messages SSE stream. Same bridge/disconnect semantics as
/// `streamingResponse`, but every frame carries an `event:` line, the event
/// order is message_start → content_block_start → deltas →
/// content_block_stop → message_delta → message_stop, and there is no
/// `[DONE]` sentinel.
// s5d:debt(ceiling="disconnect during prefill is detected only at the first decode write — prefill yields no deltas, so a dead client still pays the full prefill", trigger="prefill progress is surfaced through TokenGenerating or the runtime checks cancellation mid-prefill")
private func anthropicStreamingResponse(session: GenerationSession,
                                        prepared: PreparedGeneration,
                                        modelID: String) -> Response {
    let id = anthropicMessageID()
    let body = ResponseBody { writer in
        let (stream, continuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        let generation = Task {
            do {
                let outcome = try await session.generate(prepared) { delta in
                    continuation.yield(.delta(delta))
                }
                continuation.yield(.finished(outcome))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        defer { generation.cancel() }
        try await writer.write(ByteBuffer(data: SSEFramer.frame(
            event: "message_start",
            AnthropicMessageStart(message: AnthropicMessageResponse(
                id: id,
                model: modelID,
                content: [],
                stopReason: nil,
                usage: AnthropicUsage(inputTokens: prepared.promptTokens, outputTokens: 1))))))
        try await writer.write(ByteBuffer(data: SSEFramer.frame(
            event: "content_block_start",
            AnthropicContentBlockStart(index: 0, contentBlock: AnthropicTextBlock(text: "")))))
        for try await event in stream {
            switch event {
            case .delta(let text):
                try await writer.write(ByteBuffer(data: SSEFramer.frame(
                    event: "content_block_delta",
                    AnthropicContentBlockDelta(index: 0, delta: AnthropicTextDelta(text: text)))))
            case .finished(let outcome):
                try await writer.write(ByteBuffer(data: SSEFramer.frame(
                    event: "content_block_stop",
                    AnthropicContentBlockStop(index: 0))))
                try await writer.write(ByteBuffer(data: SSEFramer.frame(
                    event: "message_delta",
                    AnthropicMessageDelta(
                        delta: AnthropicMessageDeltaBody(
                            stopReason: outcome.finishReason.anthropicStopReason),
                        usage: AnthropicOutputUsage(outputTokens: outcome.completionTokens)))))
            }
        }
        try await writer.write(ByteBuffer(data: SSEFramer.frame(
            event: "message_stop",
            AnthropicMessageStop())))
        try await writer.finish(nil)
    }
    return Response(status: .ok,
                    headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
                    body: body)
}

private func deltaChunk(kind: StreamKind,
                        id: String,
                        created: Int,
                        modelID: String,
                        text: String) -> any Encodable {
    switch kind {
    case .chat:
        return ChatCompletionChunk(
            id: id, created: created, model: modelID,
            choices: [ChatChunkChoice(index: 0,
                                      delta: ChatChunkDelta(content: text),
                                      finishReason: nil)])
    case .raw:
        return CompletionChunk(
            id: id, created: created, model: modelID,
            choices: [CompletionChoice(index: 0, text: text, finishReason: nil)])
    }
}

private func finalChunk(kind: StreamKind,
                        id: String,
                        created: Int,
                        modelID: String,
                        outcome: GenerationOutcome) -> any Encodable {
    switch kind {
    case .chat:
        return ChatCompletionChunk(
            id: id, created: created, model: modelID,
            choices: [ChatChunkChoice(index: 0,
                                      delta: ChatChunkDelta(),
                                      finishReason: outcome.finishReason.rawValue)])
    case .raw:
        return CompletionChunk(
            id: id, created: created, model: modelID,
            choices: [CompletionChoice(index: 0,
                                       text: "",
                                       finishReason: outcome.finishReason.rawValue)])
    }
}

private func usageChunk(kind: StreamKind,
                        id: String,
                        created: Int,
                        modelID: String,
                        outcome: GenerationOutcome) -> any Encodable {
    switch kind {
    case .chat:
        return ChatCompletionChunk(id: id, created: created, model: modelID,
                                   choices: [], usage: outcome.usage)
    case .raw:
        return CompletionChunk(id: id, created: created, model: modelID,
                               choices: [], usage: outcome.usage)
    }
}

func jsonResponse<T: Encodable>(_ status: HTTPResponse.Status, _ value: T) -> Response {
    do {
        let data = try JSONEncoder().encode(value)
        return Response(status: status,
                        headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: ByteBuffer(data: data)))
    } catch {
        return Response(status: .internalServerError,
                        headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: ByteBuffer(
                            string: #"{"error":{"message":"response encoding failed","type":"server_error","code":null}}"#)))
    }
}

func errorResponse(status: HTTPResponse.Status,
                   message: String,
                   type: String = "invalid_request_error") -> Response {
    jsonResponse(status, OpenAIErrorResponse(message: message, type: type))
}

func unixNow() -> Int {
    Int(Date().timeIntervalSince1970)
}

func openAIID(prefix: String) -> String {
    let suffix = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
    return "\(prefix)-\(suffix)"
}

func anthropicErrorResponse(status: HTTPResponse.Status,
                            message: String,
                            type: String = "invalid_request_error") -> Response {
    jsonResponse(status, AnthropicErrorResponse(type: type, message: message))
}

/// Anthropic ids use an underscore separator (`msg_...`), unlike the OpenAI
/// `prefix-...` form above.
func anthropicMessageID() -> String {
    let suffix = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
    return "msg_\(suffix)"
}
