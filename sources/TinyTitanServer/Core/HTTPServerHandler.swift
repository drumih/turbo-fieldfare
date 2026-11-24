//
//  HTTPServerHandler.swift
//  TinyTitanServer
//
//  The per-connection NIO handler: its state, the channel lifecycle, and the
//  write path's entry points. The API surfaces it routes to live in the
//  `HTTPServerHandler+*.swift` files beside this one.
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore
import TinyTitanLib

/// unchecked-invariant: NIO calls every ChannelInboundHandler method on the
/// channel's own event loop, so the handler's per-request state is already
/// serialised. The exceptions are `activeTask`, which the SSE drainer and the
/// backpressure path touch from the cooperative pool -- that field is guarded by
/// `taskLock` -- and `responseModelID`, which the response builders read from
/// the generation task and which is guarded by `responseModelLock`.
final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    /// The OpenAI REST protocol/version identifier reported in the
    /// `openai-version` response header on every API response. The REST API
    /// major version is v1 (endpoints under /v1/...).
    static let openAIVersionHeader = ("openai-version", "2020-10-01")

    /// S4: cap on SSE frames waiting to be written; a slow reader that exceeds
    /// it fails the stream instead of growing the pending queue without bound.
    static let maximumPendingStreamChunks = 512

    static let minimalErrorData = Data(
        #"""
        {"error":{"message":"internal server error","type":"server_error","code":"internal_error"}}
        """#.utf8)

    let modelID: String
    let backend: any ServerInferenceBackend
    let coordinator: ServerCoordinator
    let heartbeatInterval: TimeAmount
    let reasoningProfile: ServerReasoningProfile
    let router: (any ModelRouting)?
    let childChannels: ChildChannelRegistry
    let responseStore: ResponseStore
    var head: HTTPRequestHead?

    /// The model the current request was validated for, echoed in every
    /// response object it produces. One request is in flight per connection
    /// (pipelining assistance holds the next head until this response ends),
    /// but the echo sites run on the cooperative pool, hence the lock.
    let responseModelLock = NSLock()
    var _responseModelID: String
    var responseModelID: String {
        get { responseModelLock.withLock { _responseModelID } }
        set { responseModelLock.withLock { _responseModelID = newValue } }
    }
    var body = ByteBuffer()
    var oversized = false
    /// This request has already been answered with an error (oversized headers
    /// or body), so its remaining body is discarded and `.end` closes instead of
    /// routing.
    var rejected = false
    var drainedSinceReject = 0

    // Access to activeTask is lock-guarded because the SSE drainer and the
    // backpressure fail path read it from the cooperative pool while the event
    // loop writes it for each new request.
    let taskLock = NSLock()
    var _activeTask: Task<Void, Never>?
    var activeTask: Task<Void, Never>? {
        get { taskLock.withLock { _activeTask } }
        set { taskLock.withLock { _activeTask = newValue } }
    }

    var requestPhaseState = RequestPhaseState()
    /// Requests currently being processed on this connection; idle closing and
    /// phase bookkeeping key off it (S10, S25, S1).
    var inFlightRequests = 0
    var idleCloseTask: Scheduled<Void>?
    /// The idle/mid-body deadline this connection arms. Injected so the
    /// stalled-upload path is testable without waiting two minutes.
    let idleReadTimeout: TimeAmount

    init(
        modelID: String,
        backend: any ServerInferenceBackend,
        coordinator: ServerCoordinator,
        heartbeatInterval: TimeAmount,
        reasoningProfile: ServerReasoningProfile,
        router: (any ModelRouting)?,
        childChannels: ChildChannelRegistry,
        responseStore: ResponseStore,
        idleReadTimeout: TimeAmount = TinyTitanHTTPServer.idleReadTimeout
    ) {
        self.modelID = modelID
        self._responseModelID = modelID
        self.router = router
        self.reasoningProfile = reasoningProfile
        self.backend = backend
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.childChannels = childChannels
        self.responseStore = responseStore
        self.idleReadTimeout = idleReadTimeout
    }

    /// Which API's wire shapes a request speaks. Error envelopes, stream
    /// terminators and heartbeat frames differ per surface; the generation
    /// underneath does not.
    enum APISurface: Sendable {
        case chat, responses, anthropic
    }

    func channelActive(context: ChannelHandlerContext) {
        // S1: start the per-connection idle deadline so a connection that
        // never sends anything is closed.
        resetIdleDeadline(context)
        context.fireChannelActive()
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        // S1: any read activity pushes the idle deadline out (slowloris
        // connections stall once the trickle stops).
        resetIdleDeadline(context)
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
            oversized = false
            rejected = false
            drainedSinceReject = 0
            // Checked before this request is routed or counted as in-flight: an
            // oversized header block is answered here and its body discarded
            // below. The head is still stored so `.end` can close the connection
            // in order rather than mid-upload.
            if Self.headerBlockExceedsLimits(head.headers) {
                rejected = true
                writeError(
                    context, status: .requestHeaderFieldsTooLarge,
                    OpenAIErrorEnvelope(
                        message: "request headers are too large",
                        code: "request_headers_too_large"))
                return
            }
            // S10/S25: reset per-request phase state and drop the reference to
            // any previous request's (finished) task when a new request head
            // arrives. Pipelined requests are serialized by NIO's pipeline
            // assistance, so an in-flight request is never mid-generation
            // when a later head is delivered.
            requestPhaseState = RequestPhaseState()
            activeTask = nil
            inFlightRequests += 1
        case .body(var part):
            if rejected {
                _ = drainAfterReject(context, bytes: part.readableBytes)
                return
            }
            if body.readableBytes + part.readableBytes > TinyTitanHTTPServer.maximumBodyBytes {
                rejected = true
                oversized = true
                body.clear()
                // Answer now rather than at `.end`. Waiting meant reading and
                // discarding the whole body first, so a client could make the
                // server consume an arbitrary number of bytes and as much time
                // as it liked before learning the request was refused.
                writeError(
                    context, status: .payloadTooLarge,
                    OpenAIErrorEnvelope(
                        message: "request body is too large",
                        code: "request_too_large"))
            } else {
                body.writeBuffer(&part)
            }
        case .end:
            // A refused request was answered when the limit was crossed, and the
            // connection closes once its body has been consumed: reusing a
            // keep-alive connection whose request was refused buys nothing and
            // the client is about to read an error, not a response.
            if rejected {
                self.head = nil
                body.clear()
                closeAfterPendingWrites(context)
                return
            }
            guard let head else { return }
            self.head = nil
            // S35: do not retain the (up to 1 MiB) request body buffer across
            // keep-alive requests.
            defer { body = ByteBuffer() }
            if oversized {
                // Already answered when the cap was crossed; this is the tail of
                // a body we stopped reading.
                return
            }
            route(head: head, body: body, context: context)
        }
    }

    /// Closes the connection *after* any response already queued has been
    /// written.
    ///
    /// `writeData` and friends queue their head/body/end with
    /// `eventLoop.execute`, and `channelRead` already runs on the event loop — so
    /// closing inline here closes before that block runs and the client sees a
    /// dropped connection instead of the error it was sent. Scheduling the close
    /// onto the loop puts it behind the queued write, and the loop's task queue is
    /// FIFO.
    func closeAfterPendingWrites(_ context: ChannelHandlerContext) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute { contextBox.value.close(promise: nil) }
    }

    /// Discards what is left of a refused request's body, up to
    /// `maximumDrainedBytesAfterReject`, then closes the connection. Returns
    /// false once the connection has been closed.
    func drainAfterReject(
        _ context: ChannelHandlerContext,
        bytes: Int
    ) -> Bool {
        drainedSinceReject += bytes
        guard drainedSinceReject <= TinyTitanHTTPServer.maximumDrainedBytesAfterReject else {
            closeAfterPendingWrites(context)
            return false
        }
        return true
    }

    /// Whether a request head exceeds the aggregate limits above.
    static func headerBlockExceedsLimits(_ headers: HTTPHeaders) -> Bool {
        guard headers.count <= TinyTitanHTTPServer.maximumRequestHeaderFields else { return true }
        var total = 0
        for header in headers {
            // Name + value + the ": " and CRLF the wire form costs.
            total += header.name.utf8.count + header.value.utf8.count + 4
            if total > TinyTitanHTTPServer.maximumRequestHeaderBytes { return true }
        }
        return false
    }

    func channelInactive(context: ChannelHandlerContext) {
        // S25: cancel by identity — capture this connection's current task,
        // clear the property, then cancel so a stale reference can never
        // cancel a task that belongs to a later request.
        let task = activeTask
        activeTask = nil
        task?.cancel()
        idleCloseTask?.cancel()
        idleCloseTask = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        // An I/O error (e.g. a write to a disconnected client) means the
        // stream can no longer be delivered; cancel the generation so it
        // stops promptly, then let the pipeline handle the error.
        //
        // Only when one request is in flight on this connection. `activeTask` is
        // a single slot, so with a pipelined follow-up already started it holds
        // *that* request's task and cancelling it would stop a generation whose
        // bytes are not the ones that failed -- the identity rule S25 applies in
        // `channelInactive`, which this call did not. With more than one in
        // flight the failure belongs to one response and is handled where that
        // response is written; the generation there ends on its own at worst.
        if inFlightRequests <= 1 { activeTask?.cancel() }
        context.fireErrorCaught(error)
    }

}
