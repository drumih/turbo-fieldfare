//
//  HTTPServerHandler+Streaming.swift
//  TinyTitanServer
//
//  The streaming half of the response plumbing: the SSE head, chat-event
//  enqueueing, stream lifecycle and the outbound frame writers. Split out of
//  `HTTPServerHandler+Plumbing.swift` (2026-09-28) under the 500-line-per-file
//  rule (Task 8 of the cleanup runbook) as pure code motion.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitan
import TinyTitanLib
import TinyTitanCore

extension ServerHTTPHandler {

    // MARK: Shared response plumbing

    /// Start an SSE response: the head, then whatever frames the surface
    /// opens with.
    func writeStreamHead(
        _ context: ChannelHandlerContext,
        initialFrames: Data,
        extraHeaders: [(String, String)]
    ) -> EventLoopFuture<Void> {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
        for (name, value) in extraHeaders {
            headers.add(name: name, value: value)
        }
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)), promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: initialFrames.count)
            buffer.writeBytes(initialFrames)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    /// A request error in the surface's envelope, with the status the error
    /// maps to.
    func writeRequestError(
        _ context: ChannelHandlerContext,
        _ error: ServerRequestError,
        status: HTTPResponseStatus,
        surface: APISurface,
        requestID: String? = nil
    ) {
        switch surface {
        case .chat, .responses:
            writeCodable(context, status: status, error.envelope)
        case .anthropic:
            let id = requestID ?? AnthropicBuilder.requestID()
            writeCodable(
                context, status: status,
                AnthropicErrorEnvelope.from(error, requestID: id),
                extraHeaders: [("request-id", id)])
        }
    }

    func writeJSON(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        object: Any,
        surface: APISurface,
        requestID: String? = nil
    ) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        let headers: [(String, String)] =
            surface == .anthropic
            ? [("request-id", requestID ?? AnthropicBuilder.requestID())] : []
        writeData(context, status: status, data: data, extraHeaders: headers)
    }

    /// A generation that failed after the request was accepted. Streams get
    /// the surface's failure frames and are closed; a cancelled stream (the
    /// client left, or shutdown) just ends. Non-streaming requests get the
    /// error envelope with its status.
    func handleAsyncFailure(
        _ error: Error,
        context: ChannelHandlerContext,
        id: String,
        phase: String,
        stream: Bool,
        outbox: SSEOutbox?,
        streamState: StreamState,
        surface: APISurface,
        requestID: String? = nil,
        failureFrames: (@Sendable (OpenAIErrorEnvelope) -> [Data])? = nil
    ) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = HTTPResponseStatus(statusCode: requestError.httpStatus)
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TinyTitanServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if !(error is CancellationError) {
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        // The stream branch is only correct once the SSE head exists -- it is
        // written by `startStream`, which the coordinator calls when it admits
        // the request. A rejection that happens *before* admission (queue full,
        // shutting down) never gets there, and the frames queued here were then
        // written as a body with no status line: the client saw `data: {...}`
        // bytes where a 429 should have been, on every streaming surface.
        // Falling through writes a real response instead.
        if stream, let outbox, !streamState.isStarted {
            // Refused before admission, so no SSE head was ever written and the
            // paths below write this response in full. Retire the outbox so its
            // drainer neither waits forever nor appends an `end` to a response it
            // did not produce.
            outbox.abandon()
        }
        if stream, let outbox, streamState.isStarted {
            // S5/S20: never leave a streaming client without a terminal frame.
            if error is CancellationError {
                outbox.enqueueTerminal(
                    surface == .chat ? [Self.doneFrame()] : [],
                    closeWhenDrained: true)
            } else {
                let frames =
                    failureFrames?(envelope)
                    ?? Self.failureFrames(envelope, surface: surface, requestID: requestID)
                outbox.enqueueTerminal(frames, closeWhenDrained: true)
            }
            return
        }
        if error is CancellationError {
            // S20: the client disconnected or the server is shutting down;
            // there is no one to write to. Do not emit a misleading 500.
            return
        }
        if let requestError = error as? ServerRequestError {
            writeRequestError(
                context, requestError, status: status, surface: surface,
                requestID: requestID)
        } else if surface == .anthropic {
            let id = requestID ?? AnthropicBuilder.requestID()
            writeCodable(
                context, status: status,
                AnthropicErrorEnvelope(
                    type: "api_error", message: envelope.error.message,
                    requestID: id),
                extraHeaders: [("request-id", id)])
        } else {
            writeError(context, status: status, envelope)
        }
    }

    /// One streamed chat event as its chunk. Reasoning rides in
    /// `delta.reasoning_content`, the vLLM and DeepSeek convention that
    /// Qwen Code, OpenCode and their kind read; a client that knows no
    /// such field ignores it and sees the answer alone.
    func enqueueChatEvent(
        _ event: ServerInferenceEvent,
        id: String,
        created: Int,
        streamState: StreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        switch event {
        case .promptProcessed:
            // The wire's clients learn the prompt's size from the usage block at
            // the end, and OpenAI's streaming shape has no chunk for it, so
            // there is nothing to send here.
            break
        case .promptProgress(let done, let total):
            // Only a request that set `return_progress` reaches this arm. The
            // chunk keeps the chat.completion.chunk envelope with empty
            // `choices`, the shape the usage-only chunk already uses, and adds
            // the progress object beside it; a client that ignores unknown
            // fields sees nothing it did not expect.
            enqueueStreamChunk(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": responseModelID,
                    "choices": [Any](),
                    "prompt_progress": ["tokens": done, "total": total],
                ],
                outbox: outbox, context: context)
        case .content(let text):
            enqueueStreamChunk(
                chunk(id: id, created: created, delta: ["content": text], finishReason: nil),
                outbox: outbox, context: context)
        case .reasoning(let text):
            enqueueStreamChunk(
                chunk(
                    id: id, created: created, delta: ["reasoning_content": text],
                    finishReason: nil),
                outbox: outbox, context: context)
        case .toolCall(let call):
            enqueueToolCallChunks(
                id: id, created: created,
                toolIndex: streamState.nextToolIndex(), call: call,
                outbox: outbox, context: context)
        }
    }

    func writeCompletion(
        _ context: ChannelHandlerContext,
        id: String,
        created: Int,
        completion: ServerCompletion
    ) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
            ? NSNull()
            : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        // Absent rather than empty with thinking off, so that response is
        // byte for byte what it was.
        if !completion.reasoning.isEmpty {
            message["reasoning_content"] = completion.reasoning
        }
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": responseModelID,
            "choices": [
                [
                    "index": 0,
                    "message": message,
                    "finish_reason": completion.finishReason,
                ]
            ],
            "usage": usageObject(completion.usage),
        ]
        writeJSON(context, status: .ok, object: object)
    }

    func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(
                ServerRequestError.invalid(
                    message: "stream response could not be encoded",
                    param: nil,
                    code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(
                self.wrapOutboundOut(.head(head)),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    func enqueueToolCallChunks(
        id: String,
        created: Int,
        toolIndex: Int,
        call: ParsedToolCall,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for (index, fragment) in fragments.enumerated() {
            var function: [String: Any] = ["arguments": fragment]
            var tool: [String: Any] = ["index": toolIndex, "function": function]
            if index == 0 {
                function["name"] = call.name
                tool["id"] = call.id
                tool["type"] = "function"
                tool["function"] = function
            }
            enqueueStreamChunk(
                chunk(
                    id: id, created: created,
                    delta: ["tool_calls": [tool]],
                    finishReason: nil),
                outbox: outbox,
                context: context)
        }
    }

    func finishStream(
        _ context: ChannelHandlerContext,
        id: String,
        created: Int,
        completion: ServerCompletion,
        includeUsage: Bool,
        outbox: SSEOutbox
    ) {
        // Both finish frames go through the same guard as every other frame:
        // a refused one fails the stream with an error, and the closed outbox
        // makes the `[DONE]` below a no-op. Enqueue it directly and a client
        // that was one frame too slow at the end reads a complete stream whose
        // finish_reason never arrived.
        enqueueStreamChunk(
            chunk(
                id: id, created: created,
                delta: [:],
                finishReason: completion.finishReason),
            outbox: outbox,
            context: context)
        if includeUsage {
            enqueueStreamChunk(
                [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": responseModelID,
                    "choices": [],
                    "usage": usageObject(completion.usage),
                ],
                outbox: outbox,
                context: context)
        }
        outbox.enqueueTerminal([Self.doneFrame()], closeWhenDrained: false)
    }

    func chunk(
        id: String,
        created: Int,
        delta: [String: Any],
        finishReason: String?
    ) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": responseModelID,
            "choices": [
                [
                    "index": 0,
                    "delta": delta,
                    "finish_reason": encodedReason,
                ]
            ],
        ]
    }

    /// Enqueue one SSE frame. Encoding or backpressure failure fails the
    /// stream with a terminal frame instead of silently dropping the chunk
    /// (S4/S5).
    func enqueueStreamChunk(
        _ object: [String: Any],
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        guard let frame = streamFrame(object) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream response could not be encoded",
                code: "internal_error")
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream backpressure limit exceeded; client is too slow",
                code: "stream_overflow")
            return
        }
    }

    func streamFrame(_ object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("data: " + data.lossyUTF8String)
    }

    func failStream(
        outbox: SSEOutbox,
        context: ChannelHandlerContext,
        message: String,
        code: String,
        surface: APISurface = .chat
    ) {
        let envelope = OpenAIErrorEnvelope(
            message: message,
            code: code,
            type: "server_error")
        outbox.enqueueTerminal(
            Self.failureFrames(envelope, surface: surface),
            closeWhenDrained: true)
        // Cancel the generation this runs in, by identity: `failStream` is called
        // from the generation's own task (the event callback), so this is exact,
        // and it is not `activeTask` -- a pipelined follow-up request may already
        // have replaced that. The frames above are delivered by the drainer,
        // which runs in a separate task and is unaffected.
        withUnsafeCurrentTask { $0?.cancel() }
    }
}
