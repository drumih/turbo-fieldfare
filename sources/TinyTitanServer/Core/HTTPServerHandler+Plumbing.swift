//
//  HTTPServerHandler+Plumbing.swift
//  TinyTitanServer
//
//  Response writing, SSE framing, streaming and the idle/deadline plumbing shared by
//  every surface.
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

    /// Drain the outbox with backpressure: each chunk's write future is
    /// awaited, so a slow reader stalls here (NIO holds the bytes in its
    /// outbound buffer) instead of letting pending memory grow without bound
    /// (S4). Write failures cancel the generation and close the connection.
    func drainOutbox(
        _ context: ChannelHandlerContext,
        outbox: SSEOutbox,
        streamState: StreamState
    ) async {
        while let frame = await outbox.next() {
            do {
                try await writeSSEChunk(context, frame)
            } catch {
                // The write failed (client gone or socket error): nothing more
                // can be delivered. Cancel the generation and close -- but only
                // when the slot still holds the generation this drainer belongs
                // to. This drainer runs in its own task, so `activeTask` is the
                // only handle it has, and under pipelining that is the *next*
                // request's task; cancelling it would stop the wrong generation.
                if inFlightRequests <= 1 { activeTask?.cancel() }
                context.close(promise: nil)
                return
            }
        }
        // Outbox closed and drained. End the HTTP response body — unless this
        // outbox was abandoned, which means the request was refused before a
        // stream head existed and the error response has already been written in
        // full: an `end` here would be a second one on the same request.
        let endWriteFailed: Bool
        if outbox.isAbandoned {
            endWriteFailed = false
        } else {
            // Stop the heartbeat *before* the terminal `end`.
            //
            // `writeHeartbeat` writes a body part gated only on
            // `started && !stopped`, and `stop()` otherwise runs in the request
            // task's `defer` — which is *after* this write. A ping landing in
            // between is a body with no head outstanding: NIO's
            // `HTTPServerProtocolErrorHandler` traps on that, and in release it
            // is a malformed response. The test suite schedules heartbeats every
            // 10 ms, so the window was being hit intermittently; production's 5 s
            // interval only makes it rarer.
            streamState.stop()
            do {
                try await writeSSEEnd(context)
                endWriteFailed = false
            } catch {
                endWriteFailed = true
            }
        }
        let close = outbox.closeWhenDrained
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
            if close || endWriteFailed {
                contextBox.value.close(promise: nil)
            }
        }
    }

    func writeSSEChunk(
        _ context: ChannelHandlerContext,
        _ frame: Data
    ) async throws {
        let promise = context.eventLoop.makePromise(of: Void.self)
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: frame.count)
            buffer.writeBytes(frame)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: promise)
        }
        try await promise.futureResult.get()
    }

    func writeSSEEnd(_ context: ChannelHandlerContext) async throws {
        let promise = context.eventLoop.makePromise(of: Void.self)
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: promise)
        }
        try await promise.futureResult.get()
    }

    func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    func writeHeadOnly(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus
    ) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "0")
            headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
            contextBox.value.write(
                self.wrapOutboundOut(
                    .head(
                        HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil))).whenFailure { _ in
                contextBox.value.close(promise: nil)
            }
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
        }
    }

    func writeCodable<T: Encodable>(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        _ value: T,
        extraHeaders: [(String, String)] = []
    ) {
        guard let data = try? JSONEncoder().encode(value) else {
            // S5: encoding failure must not silently drop the response; send a
            // minimal error envelope instead.
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data, extraHeaders: extraHeaders)
    }

    func writeError(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        _ error: OpenAIErrorEnvelope
    ) {
        writeCodable(context, status: status, error)
    }

    func writeJSON(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        object: Any
    ) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else {
            writeData(context, status: .internalServerError, data: Self.minimalErrorData)
            return
        }
        writeData(context, status: status, data: data)
    }

    func writeData(
        _ context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        data: Data,
        extraHeaders: [(String, String)] = []
    ) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            headers.add(name: Self.openAIVersionHeader.0, value: Self.openAIVersionHeader.1)
            for (name, value) in extraHeaders {
                headers.add(name: name, value: value)
            }
            contextBox.value.write(
                self.wrapOutboundOut(
                    .head(
                        HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            // S5: a failed response write leaves no terminal frame possible;
            // close the connection so the client never hangs.
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil))).whenFailure { _ in
                contextBox.value.close(promise: nil)
            }
            if self.inFlightRequests > 0 { self.inFlightRequests -= 1 }
            self.resetIdleDeadline(contextBox.value)
        }
    }

    /// S1: (re)arm the per-connection idle close deadline. Fires after
    /// `idleReadTimeout` without read activity. A request whose head has arrived
    /// but whose body has not completed is a stalled upload -- it is counted
    /// in flight, yet nothing will ever complete it -- so it is closed like an
    /// idle connection. A generation whose head was consumed is never closed
    /// mid-flight; its deadline is re-armed instead, because the old code
    /// stopped here and disarmed the timer for good (AUD2-034).
    func resetIdleDeadline(_ context: ChannelHandlerContext) {
        idleCloseTask?.cancel()
        let contextBox = SendableContext(context)
        idleCloseTask = context.eventLoop.scheduleTask(
            in: idleReadTimeout
        ) {
            if self.inFlightRequests == 0 || self.head != nil {
                contextBox.value.close(promise: nil)
            } else {
                self.resetIdleDeadline(contextBox.value)
            }
        }
    }

    static func awaitDrainer(_ drainer: Task<Void, Never>) async {
        await withTaskCancellationHandler {
            await drainer.value
        } onCancel: {
            drainer.cancel()
        }
    }

    static func sseFrame(_ text: String) -> Data {
        var bytes = Data(text.utf8)
        bytes.append(Data("\n\n".utf8))
        return bytes
    }

    static func doneFrame() -> Data {
        sseFrame("data: [DONE]")
    }

    static func errorFrame(_ envelope: OpenAIErrorEnvelope) -> Data? {
        guard let data = try? JSONEncoder().encode(envelope) else { return nil }
        return sseFrame("data: " + data.lossyUTF8String)
    }

    func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens
            ],
            "completion_tokens_details": [
                "reasoning_tokens": usage.completionTokensDetails.reasoningTokens
            ],
        ]
    }

    func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
        guard !text.isEmpty else { return [""] }
        var result: [String] = []
        var current = ""
        var bytes = 0
        for character in text {
            let size = String(character).utf8.count
            if bytes + size > maximumBytes, !current.isEmpty {
                result.append(current)
                current = ""
                bytes = 0
            }
            current.append(character)
            bytes += size
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}
