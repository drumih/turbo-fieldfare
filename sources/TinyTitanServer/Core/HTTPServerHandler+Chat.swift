//
//  HTTPServerHandler+Chat.swift
//  TinyTitanServer
//
//  The OpenAI-compatible chat completions surface.
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
    func handleCompletion(
        body: ByteBuffer,
        context: ChannelHandlerContext,
        workspace: String? = nil
    ) {
        do {
            // One copy out of the ByteBuffer, not two: a [UInt8] hop would
            // duplicate a body of up to `maximumBodyBytes` before decoding.
            let decoded = try JSONDecoder().decode(
                OpenAIChatRequest.self, from: Data(body.readableBytesView))
            let request = try validate(decoded, for: try servedModel(named: decoded.model))
                .withWorkspace(workspace)
            let responseID =
                "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = requestPhaseState
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                    streamState.start(
                        eventLoop: contextBox.value.eventLoop,
                        interval: self.heartbeatInterval,
                        ping: {
                            self.writeHeartbeat(contextBox.value)
                        })
                else { return }
                let future = self.beginStream(
                    contextBox.value,
                    self.chunk(
                        id: responseID, created: created,
                        delta: ["role": "assistant"],
                        finishReason: nil))
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
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
                        ServerLog.generating(id: responseID)
                        return try await self.backend.generate(request) { event in
                            guard request.stream, let outbox else { return }
                            self.enqueueChatEvent(
                                event, id: responseID, created: created,
                                streamState: streamState,
                                outbox: outbox, context: contextBox.value)
                        }
                    }
                    ServerLog.completed(
                        id: responseID,
                        duration: started.duration(to: .now),
                        completion: completion)
                    if request.stream, let outbox {
                        streamState.stop()
                        self.finishStream(
                            contextBox.value,
                            id: responseID,
                            created: created,
                            completion: completion,
                            includeUsage: request.includeUsage,
                            outbox: outbox)
                    } else {
                        self.writeCompletion(
                            contextBox.value,
                            id: responseID,
                            created: created,
                            completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncFailure(
                        error,
                        context: contextBox.value,
                        id: responseID,
                        phase: phaseState.value,
                        stream: request.stream,
                        outbox: outbox,
                        streamState: streamState,
                        surface: .chat)
                }
                if let drainer {
                    await Self.awaitDrainer(drainer)
                }
            }
        } catch let error as ServerRequestError {
            writeError(
                context,
                status: error == .unknownModel ? .notFound : .badRequest,
                error.envelope)
        } catch {
            writeError(
                context, status: .badRequest,
                RequestDecodingFailure.serverError(error, body: body).envelope)
        }
    }

    /// Per-request bookkeeping for a /v1/responses stream: the event sequence
    /// number and the output indices handed to items in the order the model
    /// produced them, so the streamed items and the final object agree.
    /// unchecked-invariant: every field is guarded by `lock`; the sequence is
    /// read from the queued callback (event loop) and the generation task.
    final class ResponsesStreamState: @unchecked Sendable {
        private let lock = NSLock()
        private var sequence = 0
        private var nextOutput = 0
        private var _messageIndex: Int?
        private var _callIndices: [Int] = []

        func nextSequence() -> Int {
            lock.withLock {
                defer { sequence += 1 }
                return sequence
            }
        }

        var messageIndex: Int? { lock.withLock { _messageIndex } }
        var callIndices: [Int] { lock.withLock { _callIndices } }

        /// The message item's output index, allocated on first use.
        func announceMessage() -> (index: Int, first: Bool) {
            lock.withLock {
                if let index = _messageIndex { return (index, false) }
                let index = nextOutput
                nextOutput += 1
                _messageIndex = index
                return (index, true)
            }
        }

        /// A function-call item's output index and its ordinal among calls.
        func allocateCall() -> (index: Int, ordinal: Int) {
            lock.withLock {
                let index = nextOutput
                nextOutput += 1
                _callIndices.append(index)
                return (index, _callIndices.count - 1)
            }
        }

        /// Every reasoning item so far, by output index, with its whole text:
        /// the done events and the final object both need the full thought.
        private var _reasoning: [(index: Int, text: String)] = []
        private var openReasoning: Int?

        var reasoningItems: [(index: Int, text: String)] { lock.withLock { _reasoning } }

        /// Appends to the open reasoning item, opening one when none is.
        /// The ordinal numbers the item among reasoning items, for its id.
        func appendReasoning(_ text: String) -> (index: Int, ordinal: Int, first: Bool) {
            lock.withLock {
                if let ordinal = openReasoning {
                    _reasoning[ordinal].text += text
                    return (_reasoning[ordinal].index, ordinal, false)
                }
                let index = nextOutput
                nextOutput += 1
                _reasoning.append((index, text))
                openReasoning = _reasoning.count - 1
                return (index, _reasoning.count - 1, true)
            }
        }

        /// Close the open reasoning item, returning it if there was one.
        func closeReasoning() -> (index: Int, ordinal: Int, text: String)? {
            lock.withLock {
                guard let ordinal = openReasoning else { return nil }
                openReasoning = nil
                return (_reasoning[ordinal].index, ordinal, _reasoning[ordinal].text)
            }
        }
    }

    static func eventFrame(name: String, object: [String: Any]) -> Data? {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            return nil
        }
        return Self.sseFrame("event: " + name + "\ndata: " + data.lossyUTF8String)
    }

    /// The frames that end a stream after a failure, in the surface's shape:
    /// chat sends an error object then [DONE]; the Responses API and the
    /// Messages API send a typed `error` event and no terminator.
    static func failureFrames(
        _ envelope: OpenAIErrorEnvelope,
        surface: APISurface,
        requestID: String? = nil
    ) -> [Data] {
        switch surface {
        case .chat:
            return errorFrame(envelope).map { [$0, doneFrame()] } ?? [doneFrame()]
        case .responses:
            let object: [String: Any] = [
                "type": "error", "code": envelope.error.code,
                "message": envelope.error.message,
                "param": envelope.error.param.map { $0 as Any } ?? NSNull(),
            ]
            return eventFrame(name: "error", object: object).map { [$0] } ?? []
        case .anthropic:
            let detail = AnthropicErrorEnvelope(
                type: envelope.error.type == "server_error" ? "api_error" : envelope.error.type,
                message: envelope.error.message, requestID: requestID)
            guard let data = try? JSONEncoder().encode(detail) else { return [] }
            return [sseFrame("event: error\ndata: " + data.lossyUTF8String)]
        }
    }

    /// An `item_reference` names an output item of a stored response; the
    /// item itself takes its place in the input. Unknown ids stay as they
    /// are and the mapper refuses them by id.
}
