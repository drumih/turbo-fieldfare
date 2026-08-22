import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TurboFieldfare

public actor TurboFieldfareHTTPServer {
    public static let maximumBodyBytes = StreamingChatRequestBody.maximumWireBytes
    /// Accepted sockets are unbounded otherwise, and each one costs a file
    /// descriptor plus whatever its half-sent request has staged.
    public static let maximumConnections = 128
    /// How long a connection may sit without being read from before it is
    /// closed. Generous, because it must never interrupt a slow client that is
    /// still uploading images.
    public static let idleTimeout = TimeAmount.seconds(120)

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let visionCapability: String
    private let attachmentRoot: URL
    private let idleTimeout: TimeAmount
    private let childChannels = ChildChannelRegistry()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: any ServerInferenceBackend,
                heartbeatInterval: TimeAmount = .seconds(5),
                visionCapability: String = "missing",
                attachmentRoot: URL = ServerAttachmentDirectory.root,
                idleTimeout: TimeAmount = TurboFieldfareHTTPServer.idleTimeout,
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
        self.visionCapability = visionCapability
        self.attachmentRoot = attachmentRoot
        self.idleTimeout = idleTimeout
        ServerAttachmentDirectory.sweepAbandoned(in: attachmentRoot)
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let childChannels = self.childChannels
        let visionCapability = self.visionCapability
        let attachmentRoot = self.attachmentRoot
        let idleTimeout = self.idleTimeout
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.addHandler(
                    IdleStateHandler(readTimeout: idleTimeout)
                ).flatMap {
                    channel.pipeline.configureHTTPServerPipeline(
                        withPipeliningAssistance: true,
                        withErrorHandling: true)
                }.flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        modelID: modelID,
                        backend: backend,
                        coordinator: coordinator,
                        heartbeatInterval: heartbeatInterval,
                        visionCapability: visionCapability,
                        attachmentRoot: attachmentRoot,
                        childChannels: childChannels))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let coordinator = self.coordinator
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await coordinator.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await coordinator.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await coordinator.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

private final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let childChannels: ChildChannelRegistry
    private let visionCapability: String
    private let attachmentRoot: URL
    private var bodyParser: StreamingChatRequestBody?
    private var bodyError: (any Error)?
    private var receivedBodyBytes = 0
    /// Past the wire cap the drain toward `.end` is unbounded - a chunked
    /// stream may never send one - so the request is answered immediately
    /// and everything after it dropped until the connection closes.
    private var discardingUntilClose = false
    /// Requests still producing an answer. A generation reads nothing for as
    /// long as it runs, so the idle timeout must not close under it.
    private var inFlightRequests = 0
    private var head: HTTPRequestHead?
    private var activeTask: Task<Void, Never>?

    init(modelID: String,
         backend: any ServerInferenceBackend,
         coordinator: ServerCoordinator,
         heartbeatInterval: TimeAmount,
         visionCapability: String,
         attachmentRoot: URL,
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.backend = backend
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.visionCapability = visionCapability
        self.attachmentRoot = attachmentRoot
        self.childChannels = childChannels
    }

    static let chatCompletionsPath = "/v1/chat/completions"

    static func requestPath(_ head: HTTPRequestHead) -> String {
        head.uri.split(separator: "?", maxSplits: 1,
                       omittingEmptySubsequences: false)
            .first.map(String.init) ?? head.uri
    }

    static func carriesChatBody(_ head: HTTPRequestHead) -> Bool {
        head.method == .POST
            && requestPath(head) == chatCompletionsPath
            && head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            guard !discardingUntilClose else { return }
            self.head = head
            // The parser stages inline images to disk, so it is created only
            // once the request is known to carry a chat body. A body sent
            // elsewhere is counted and dropped.
            bodyParser = Self.carriesChatBody(head)
                ? StreamingChatRequestBody(attachmentRoot: attachmentRoot,
                                           visionCapability: visionCapability)
                : nil
            bodyError = nil
            receivedBodyBytes = 0
        case .body(var part):
            guard !discardingUntilClose else { return }
            receivedBodyBytes += part.readableBytes
            if receivedBodyBytes > StreamingChatRequestBody.maximumWireBytes {
                // A body already rejected for its own reason keeps it: telling a
                // client its image was too large is more useful than telling it
                // the request was, and the drain past the cap is what forces the
                // answer now.
                let capError = (bodyError as? ServerRequestError)
                    ?? ServerRequestError.invalid(
                        message: "request body is too large",
                        param: nil, code: "request_too_large")
                bodyError = nil
                bodyParser = nil
                head = nil
                discardingUntilClose = true
                writeError(context, status: capError.httpStatus, capError.envelope,
                           closeAfter: true)
                return
            }
            guard bodyError == nil else { return }
            guard let parser = bodyParser else { return }
            do { try parser.feed(&part) }
            catch {
                bodyError = error
                // Rejection is already certain, so release the staged bytes now.
                bodyParser = nil
            }
        case .end:
            guard !discardingUntilClose else { return }
            guard let head else { return }
            self.head = nil
            let parser = bodyParser
            // Cleared on every path below: a retained parser pins its staged
            // images until the connection closes.
            bodyParser = nil
            if let bodyError {
                self.bodyError = nil
                if let requestError = bodyError as? ServerRequestError {
                    writeError(context, status: requestError.httpStatus,
                               requestError.envelope)
                } else {
                    writeError(context, status: .badRequest,
                               OpenAIErrorEnvelope(message: "malformed JSON request",
                                                   code: "invalid_json"))
                }
                return
            }
            do {
                let parsed = try parser?.finish()
                    ?? ParsedChatRequestBody(json: Data(), stagedImages: [:], lease: nil)
                route(head: head, body: parsed, context: context)
            } catch let error as ServerRequestError {
                writeError(context, status: error.httpStatus, error.envelope)
            } catch {
                writeError(context, status: .badRequest,
                           OpenAIErrorEnvelope(message: "malformed JSON request",
                                               code: "invalid_json"))
            }
        }
    }


    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if event is IdleStateHandler.IdleStateEvent, inFlightRequests == 0 {
            context.close(promise: nil)
            return
        }
        context.fireUserInboundEventTriggered(event)
    }

    func channelInactive(context: ChannelHandlerContext) {
        activeTask?.cancel()
        activeTask = nil
        // Frees the staged images of a request abandoned mid-body.
        bodyParser = nil
        bodyError = nil
        head = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    private func route(head: HTTPRequestHead,
                       body: ParsedChatRequestBody,
                       context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: [
                "status": "ok",
                "vision": visionCapability,
            ])
        case (.GET, "/v1/models"):
            let response = OpenAIModelList(
                object: "list",
                data: [.init(id: modelID,
                             object: "model",
                             created: 0,
                             ownedBy: "turbofieldfare",
                             capabilities: visionCapability == "ready"
                                ? ["text", "image"] : ["text"])])
            writeCodable(context, status: .ok, response)
        case (.POST, "/v1/chat/completions"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(body: body, context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    private func handleCompletion(body: ParsedChatRequestBody,
                                  context: ChannelHandlerContext) {
        do {
            let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: body.json)
            let request = try OpenAIRequestValidator.validate(
                decoded,
                modelID: modelID,
                preStagedImages: body.stagedImages,
                attachmentLease: body.lease)
            let responseID = "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = RequestPhaseState()
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginStream(
                    contextBox.value,
                    self.chunk(id: responseID, created: created,
                               delta: ["role": "assistant"],
                               finishReason: nil))
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            inFlightRequests += 1
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                // Back on the event loop, where `inFlightRequests` lives. The hop
                // also orders this after the assignment above, which is still
                // executing in this same event-loop tick.
                defer {
                    contextBox.value.eventLoop.execute {
                        self.inFlightRequests -= 1
                    }
                }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
                do {
                    let completion = try await self.coordinator.runPreparing(
                        onQueued: onQueued,
                        prepare: {
                            let prepared = try await self.backend.prepare(request)
                            phaseState.set("prepared")
                            ServerLog.prepared(id: responseID,
                                               promptTokens: prepared.promptTokenCount)
                            return prepared
                        },
                        operation: { prepared in
                            try Task.checkCancellation()
                            startStream()
                            try await streamState.waitUntilStarted()
                            try Task.checkCancellation()
                            phaseState.set("generating")
                            ServerLog.generating(id: responseID)
                            return try await self.backend.generate(prepared) { event in
                                guard request.stream else { return }
                                switch event {
                                case .content(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   delta: ["content": text],
                                                   finishReason: nil))
                                case .toolCall(let call):
                                    self.writeToolCall(contextBox.value,
                                                       id: responseID,
                                                       created: created,
                                                       toolIndex: streamState.nextToolIndex(),
                                                       call: call)
                                }
                            }
                    })
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream {
                        streamState.stop()
                        self.finishStream(contextBox.value,
                                          id: responseID,
                                          created: created,
                                          completion: completion,
                                          includeUsage: request.includeUsage)
                    } else {
                        self.writeCompletion(contextBox.value,
                                             id: responseID,
                                             created: created,
                                             completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: streamState.isStarted,
                                          started: started)
                }
            }
        } catch let error as ServerRequestError {
            writeError(context,
                       status: error.httpStatus,
                       error.envelope)
        } catch {
            writeError(context, status: .badRequest,
                       OpenAIErrorEnvelope(message: "malformed JSON request",
                                           code: "invalid_json"))
        }
    }

    private func writeCompletion(_ context: ChannelHandlerContext,
                                 id: String,
                                 created: Int,
                                 completion: ServerCompletion) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
                ? NSNull()
                : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": completion.finishReason,
            ]],
            "usage": usageObject(completion.usage),
        ]
        writeJSON(context, status: .ok, object: object)
    }

    private func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(ServerRequestError.invalid(
                message: "stream response could not be encoded",
                param: nil,
                code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)),
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

    private func writeToolCall(_ context: ChannelHandlerContext,
                               id: String,
                               created: Int,
                               toolIndex: Int,
                               call: ParsedToolCall) {
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
            writeStreamChunk(
                context,
                chunk(id: id, created: created,
                      delta: ["tool_calls": [tool]],
                      finishReason: nil))
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              id: String,
                              created: Int,
                              completion: ServerCompletion,
                              includeUsage: Bool) {
        writeStreamChunk(
            context,
            chunk(id: id, created: created,
                  delta: [:],
                  finishReason: completion.finishReason))
        if includeUsage {
            writeStreamChunk(context, [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": modelID,
                "choices": [],
                "usage": usageObject(completion.usage),
            ])
        }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            let buffer = contextBox.value.channel.allocator.buffer(string: "data: [DONE]\n\n")
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func chunk(id: String,
                       created: Int,
                       delta: [String: Any],
                       finishReason: String?) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
        ]
    }

    private func writeStreamChunk(_ context: ChannelHandlerContext,
                                  _ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
    }

    private func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    private func handleAsyncError(_ error: Error,
                                  context: ChannelHandlerContext,
                                  id: String,
                                  phase: String,
                                  stream: Bool,
                                  started: ContinuousClock.Instant) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = requestError.httpStatus
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TurboFieldfareServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if error is CancellationError {
            // Not a failure, and not silence either: without a terminal line the
            // request's last record stayed `generating` forever, so a reader could
            // not tell running from abandoned from crashed.
            ServerLog.cancelled(id: id, phase: phase,
                                duration: started.duration(to: .now))
        } else {
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        if stream {
            finishStreamWithError(context, envelope: envelope)
            return
        }
        writeError(context, status: status, envelope)
    }

    private func finishStreamWithError(_ context: ChannelHandlerContext,
                                       envelope: OpenAIErrorEnvelope) {
        guard let data = try? JSONEncoder().encode(envelope) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            guard contextBox.value.channel.isActive else { return }
            var buffer = contextBox.value.channel.allocator.buffer(
                capacity: data.count + 32)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\ndata: [DONE]\n\n")
            contextBox.value.write(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
                                            status: HTTPResponseStatus,
                                            _ value: T,
                                            closeAfter: Bool = false) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        writeData(context, status: status, data: data, closeAfter: closeAfter)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope,
                            closeAfter: Bool = false) {
        writeCodable(context, status: status, error, closeAfter: closeAfter)
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data,
                           closeAfter: Bool = false) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            if closeAfter {
                headers.add(name: "connection", value: "close")
            }
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            if closeAfter {
                let promise = contextBox.value.eventLoop.makePromise(of: Void.self)
                let channel = contextBox.value.channel
                promise.futureResult.whenComplete { _ in
                    // Lingering close: an immediate close RSTs away the very
                    // response that explains the rejection while the client's
                    // remaining bytes are still in flight. The grace is a hard
                    // bound - a client that streams past it is cut off with its
                    // response undelivered, which is on the client.
                    _ = channel.eventLoop.scheduleTask(in: .seconds(2)) {
                        channel.close(promise: nil)
                    }
                }
                contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: promise)
            } else {
                contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
            }
        }
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
    }

    private func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    private func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
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

/// The one mapping from a request rejection to its wire status. Every write
/// site derives the status from here, so a new resource-limit code cannot
/// return 400 at one site and 413 at another.
private extension ServerRequestError {
    var httpStatus: HTTPResponseStatus {
        switch self {
        case .unknownModel:
            .notFound
        case .queueFull:
            .tooManyRequests
        case .invalid(_, _, let code):
            switch code {
            case "request_too_large", "image_too_large", "too_many_images":
                .payloadTooLarge
            default:
                .badRequest
            }
        }
    }
}

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())

    /// Registers a connection, or closes it because the server is shutting
    /// down or already at its connection cap.
    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown,
                  $0.channels.count < TurboFieldfareHTTPServer.maximumConnections
            else { return true }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

private final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

private final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(eventLoop: EventLoop,
               interval: TimeAmount,
               ping: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval) { [weak self] _ in
                    guard self?.shouldPing == true else { return }
                    ping()
                }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}
