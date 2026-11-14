//
//  HTTPServerHandler+Anthropic.swift
//  TinyTitanServer
//
//  The Anthropic Messages API surface.
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
    // MARK: Anthropic Messages API

    /// Content-block bookkeeping for one /v1/messages stream.
    /// unchecked-invariant: guarded by `lock`, for the same reason as the
    /// Responses state above.
    final class AnthropicStreamState: @unchecked Sendable {
        /// The two kinds of block that stream as deltas. A tool_use block
        /// arrives whole and is never left open.
        enum Kind {
            case text
            case thinking
        }

        struct Block {
            let index: Int
            let kind: Kind
        }

        private let lock = NSLock()
        private var nextIndex = 0
        private var open: Block?
        private var _announced = false

        /// True once a text or tool_use block has been opened. Thinking does
        /// not count: a message whose only block is a thought still gets its
        /// (empty) text block, as the non-streamed content does.
        var announced: Bool { lock.withLock { _announced } }

        /// The open block of `kind`, opening one when none is. Returns the
        /// block of the other kind it had to close first, for the caller to
        /// stop, since a thought and the answer never share a block.
        func block(_ kind: Kind) -> (index: Int, first: Bool, closed: Block?) {
            lock.withLock {
                if kind == .text { _announced = true }
                if let block = open, block.kind == kind { return (block.index, false, nil) }
                let closed = open
                let index = nextIndex
                nextIndex += 1
                open = Block(index: index, kind: kind)
                return (index, true, closed)
            }
        }

        /// Close the open block, returning it if there was one.
        func close() -> Block? {
            lock.withLock {
                defer { open = nil }
                return open
            }
        }

        func allocate() -> Int {
            lock.withLock {
                _announced = true
                defer { nextIndex += 1 }
                return nextIndex
            }
        }
    }

    static func anthropicFrame(_ object: [String: Any]) -> Data? {
        guard let type = object["type"] as? String else { return nil }
        return eventFrame(name: type, object: object)
    }

    /// Anthropic Messages API endpoint (`POST /v1/messages`). Mapped onto the
    /// same validated chat request as the OpenAI paths; answered in the
    /// Messages API's own object and event shapes.
    func handleMessages(
        body: ByteBuffer,
        context: ChannelHandlerContext,
        workspace: String? = nil
    ) {
        let requestID = AnthropicBuilder.requestID()
        do {
            let decoded = try JSONDecoder().decode(
                AnthropicMessagesRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let chatRequest = try AnthropicMapper.chatRequest(
                decoded, maxContext: target.maximumContext)
            let request = try validate(chatRequest, for: target)
                .withWorkspace(workspace)
            let messageID = AnthropicBuilder.messageID()
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let blockState = AnthropicStreamState()
            let phaseState = requestPhaseState
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                    streamState.start(
                        eventLoop: contextBox.value.eventLoop,
                        interval: self.heartbeatInterval,
                        ping: {
                            self.writeAnthropicPing(contextBox.value)
                        })
                else { return }
                let future = self.beginAnthropicStream(
                    contextBox.value, id: messageID, requestID: requestID)
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: messageID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: messageID, streaming: request.stream)
                let outbox: SSEOutbox? =
                    request.stream
                    ? SSEOutbox(capacity: Self.maximumPendingStreamChunks)
                    : nil
                let drainer = outbox.map { outbox in
                    Task { [self] in
                        await self.drainOutbox(
                            contextBox.value, outbox: outbox,
                            streamState: streamState)
                    }
                }
                do {
                    let completion = try await self.coordinator.run(onQueued: onQueued) {
                        try Task.checkCancellation()
                        startStream()
                        try await streamState.waitUntilStarted()
                        try Task.checkCancellation()
                        phaseState.set("generating")
                        ServerLog.generating(id: messageID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            switch event {
                            case .promptProcessed:
                                // Anthropic's stream announces the input token
                                // count in `message_start`, which this handler
                                // has already sent by the time generation runs.
                                break
                            case .promptProgress:
                                // Reading progress is a chat-completions extension; this surface
                                // has no event for it and the flag is not parsed here.
                                break
                            case .content(let text):
                                self.enqueueAnthropicDelta(
                                    .text, text, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            case .reasoning(let text):
                                self.enqueueAnthropicDelta(
                                    .thinking, text, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            case .toolCall(let call):
                                self.enqueueAnthropicToolUse(
                                    call, blockState: blockState,
                                    outbox: outbox, context: contextBox.value)
                            }
                        }
                    }
                    ServerLog.completed(
                        id: messageID,
                        duration: started.duration(to: .now),
                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        self.finishAnthropicStream(
                            contextBox.value, completion: completion,
                            blockState: blockState, outbox: outbox)
                    } else {
                        let stop = AnthropicBuilder.stopReason(for: completion)
                        self.writeJSON(
                            contextBox.value, status: .ok,
                            object: AnthropicBuilder.messageObject(
                                id: messageID, model: self.responseModelID,
                                content: AnthropicBuilder.contentBlocks(completion),
                                stopReason: stop.reason, stopSequence: stop.sequence,
                                usage: AnthropicBuilder.usageObject(completion.usage)),
                            surface: .anthropic, requestID: requestID)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(
                        error, context: contextBox.value, id: messageID,
                        phase: phaseState.value, stream: request.stream, outbox: outbox,
                        streamState: streamState,
                        surface: .anthropic, requestID: requestID)
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(
                context, error,
                status: HTTPResponseStatus(statusCode: error.httpStatus),
                surface: .anthropic, requestID: requestID)
        } catch {
            writeRequestError(
                context, RequestDecodingFailure.serverError(error, body: body),
                status: .badRequest, surface: .anthropic, requestID: requestID)
        }
    }

    /// `POST /v1/messages/count_tokens`: the prompt tokens the request would
    /// occupy, from the backend's own tokenizer. 501 when the backend has
    /// none to count with.
    func handleCountTokens(
        body: ByteBuffer,
        context: ChannelHandlerContext
    ) {
        let requestID = AnthropicBuilder.requestID()
        do {
            let decoded = try JSONDecoder().decode(
                AnthropicCountTokensRequest.self, from: Data(body.readableBytesView))
            let target = try servedModel(named: decoded.model)
            let chatRequest = try AnthropicMapper.chatRequest(counting: decoded)
            let request = try validate(chatRequest, for: target)
            guard let counting = backend as? any PromptTokenCounting else {
                throw ServerRequestError.unsupportedOperation("count_tokens")
            }
            let contextBox = SendableContext(context)
            activeTask = childChannels.startTask {
                do {
                    let count = try await counting.countPromptTokens(request)
                    self.writeJSON(
                        contextBox.value, status: .ok, object: ["input_tokens": count],
                        surface: .anthropic, requestID: requestID)
                } catch let error as ServerRequestError {
                    self.writeRequestError(
                        contextBox.value, error,
                        status: HTTPResponseStatus(statusCode: error.httpStatus),
                        surface: .anthropic, requestID: requestID)
                } catch {
                    ServerLog.failed(id: requestID, phase: "counting", status: 500, error: error)
                    self.writeCodable(
                        contextBox.value, status: .internalServerError,
                        AnthropicErrorEnvelope(
                            type: "api_error",
                            message: "token counting failed",
                            requestID: requestID),
                        extraHeaders: [("request-id", requestID)])
                }
            }
        } catch let error as ServerRequestError {
            writeRequestError(
                context, error,
                status: HTTPResponseStatus(statusCode: error.httpStatus),
                surface: .anthropic, requestID: requestID)
        } catch {
            writeRequestError(
                context, RequestDecodingFailure.serverError(error, body: body),
                status: .badRequest, surface: .anthropic, requestID: requestID)
        }
    }

    func beginAnthropicStream(
        _ context: ChannelHandlerContext,
        id: String,
        requestID: String
    ) -> EventLoopFuture<Void> {
        let message = AnthropicBuilder.messageObject(
            id: id, model: responseModelID, content: [], stopReason: nil, stopSequence: nil,
            usage: [
                "input_tokens": 0, "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            ])
        let frame = Self.anthropicFrame(["type": "message_start", "message": message]) ?? Data()
        return writeStreamHead(
            context, initialFrames: frame, extraHeaders: [("request-id", requestID)])
    }

    func anthropicEvent(
        _ object: [String: Any],
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        guard let frame = Self.anthropicFrame(object) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream response could not be encoded",
                code: "internal_error", surface: .anthropic)
            return
        }
        guard outbox.enqueue(frame) else {
            failStream(
                outbox: outbox, context: context,
                message: "stream backpressure limit exceeded; client is too slow",
                code: "stream_overflow", surface: .anthropic)
            return
        }
    }

    /// Text or thinking, into the open block of that kind. A thought streams
    /// as its own `thinking` block ahead of the text, with `thinking_delta`s,
    /// as the Messages API streams extended thinking.
    func enqueueAnthropicDelta(
        _ kind: AnthropicStreamState.Kind,
        _ text: String,
        blockState: AnthropicStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        let (index, first, closed) = blockState.block(kind)
        if let closed {
            stopAnthropicBlock(closed, outbox: outbox, context: context)
        }
        if first {
            let start: [String: Any] =
                kind == .text
                ? ["type": "text", "text": ""]
                : [
                    "type": "thinking", "thinking": "",
                    "signature": AnthropicBuilder.thinkingSignature,
                ]
            anthropicEvent(
                [
                    "type": "content_block_start", "index": index,
                    "content_block": start,
                ],
                outbox: outbox, context: context)
        }
        let delta: [String: Any] =
            kind == .text
            ? ["type": "text_delta", "text": text]
            : ["type": "thinking_delta", "thinking": text]
        anthropicEvent(
            ["type": "content_block_delta", "index": index, "delta": delta],
            outbox: outbox, context: context)
    }

    /// End a streamed block. A thinking block is signed first, as the API
    /// always does just before its stop, so a client that assembles the
    /// block from its deltas ends up with the same object as the
    /// non-streamed content (see `AnthropicBuilder.thinkingSignature`).
    func stopAnthropicBlock(
        _ block: AnthropicStreamState.Block,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        if block.kind == .thinking {
            anthropicEvent(
                [
                    "type": "content_block_delta", "index": block.index,
                    "delta": [
                        "type": "signature_delta",
                        "signature": AnthropicBuilder.thinkingSignature,
                    ],
                ],
                outbox: outbox, context: context)
        }
        anthropicEvent(
            ["type": "content_block_stop", "index": block.index],
            outbox: outbox, context: context)
    }

    func enqueueAnthropicToolUse(
        _ call: ParsedToolCall,
        blockState: AnthropicStreamState,
        outbox: SSEOutbox,
        context: ChannelHandlerContext
    ) {
        if let open = blockState.close() {
            stopAnthropicBlock(open, outbox: outbox, context: context)
        }
        let index = blockState.allocate()
        anthropicEvent(
            [
                "type": "content_block_start", "index": index,
                "content_block": [
                    "type": "tool_use", "id": call.id,
                    "name": call.name, "input": [:],
                ],
            ],
            outbox: outbox, context: context)
        for fragment in utf8Fragments(call.argumentsJSON, maximumBytes: 1024) {
            anthropicEvent(
                [
                    "type": "content_block_delta", "index": index,
                    "delta": ["type": "input_json_delta", "partial_json": fragment],
                ],
                outbox: outbox, context: context)
        }
        anthropicEvent(
            ["type": "content_block_stop", "index": index],
            outbox: outbox, context: context)
    }

    func finishAnthropicStream(
        _ context: ChannelHandlerContext,
        completion: ServerCompletion,
        blockState: AnthropicStreamState,
        outbox: SSEOutbox
    ) {
        // A message always carries at least one text or tool_use block.
        if !blockState.announced {
            enqueueAnthropicDelta(
                .text, "", blockState: blockState, outbox: outbox,
                context: context)
        }
        if let open = blockState.close() {
            stopAnthropicBlock(open, outbox: outbox, context: context)
        }
        let stop = AnthropicBuilder.stopReason(for: completion)
        anthropicEvent(
            [
                "type": "message_delta",
                "delta": [
                    "stop_reason": stop.reason,
                    "stop_sequence": stop.sequence.map { $0 as Any } ?? NSNull(),
                ],
                "usage": AnthropicBuilder.usageObject(completion.usage),
            ],
            outbox: outbox, context: context)
        anthropicEvent(["type": "message_stop"], outbox: outbox, context: context)
        outbox.enqueueTerminal([], closeWhenDrained: false)
    }

    func writeAnthropicPing(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(
            string: "event: ping\ndata: {\"type\": \"ping\"}\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

}
