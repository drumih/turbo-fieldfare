import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TurboFieldfare

public actor TurboFieldfareHTTPServer {
    public static let maximumBodyBytes = 1_048_576
    public static let maximumBatchFileBytes = 200 * 1_024 * 1_024

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let batches: BatchRegistry
    private let files: BatchFileStore
    private let heartbeatInterval: TimeAmount
    private let childChannels = ChildChannelRegistry()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: any ServerInferenceBackend,
                heartbeatInterval: TimeAmount = .seconds(5),
                batchOutputDirectory: URL? = nil,
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        let batchDirectory = batchOutputDirectory ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("TurboFieldfare/batches", isDirectory: true)
        let files = BatchFileStore(directory: batchDirectory)
        self.files = files
        self.batches = BatchRegistry(outputDirectory: batchDirectory) { id, expiresAt in
            _ = try? await files.registerBatchOutput(id, expiresAt: expiresAt)
        }
        self.heartbeatInterval = heartbeatInterval
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let batches = self.batches
        let files = self.files
        let heartbeatInterval = self.heartbeatInterval
        let childChannels = self.childChannels
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        modelID: modelID,
                        backend: backend,
                        coordinator: coordinator,
                        batches: batches,
                        files: files,
                        heartbeatInterval: heartbeatInterval,
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
    private let batches: BatchRegistry
    private let files: BatchFileStore
    private let heartbeatInterval: TimeAmount
    private let childChannels: ChildChannelRegistry
    private var head: HTTPRequestHead?
    private var body = ByteBuffer()
    private var oversized = false
    private var activeTask: Task<Void, Never>?

    init(modelID: String,
         backend: any ServerInferenceBackend,
         coordinator: ServerCoordinator,
         batches: BatchRegistry,
         files: BatchFileStore,
         heartbeatInterval: TimeAmount,
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.backend = backend
        self.coordinator = coordinator
        self.batches = batches
        self.files = files
        self.heartbeatInterval = heartbeatInterval
        self.childChannels = childChannels
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
            oversized = false
        case .body(var part):
            let limit = head?.uri.split(separator: "?", maxSplits: 1).first == "/v1/files"
                ? TurboFieldfareHTTPServer.maximumBatchFileBytes
                : TurboFieldfareHTTPServer.maximumBodyBytes
            if body.readableBytes + part.readableBytes > limit {
                oversized = true
            } else {
                body.writeBuffer(&part)
            }
        case .end:
            guard let head else { return }
            self.head = nil
            if oversized {
                writeError(context, status: .payloadTooLarge,
                           OpenAIErrorEnvelope(message: "request body is too large",
                                               code: "request_too_large"))
                return
            }
            route(head: head, body: body, context: context)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        activeTask?.cancel()
        activeTask = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    private func route(head: HTTPRequestHead,
                       body: ByteBuffer,
                       context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            let response = OpenAIModelList(
                object: "list",
                data: [.init(id: modelID,
                             object: "model",
                             created: 0,
                             ownedBy: "turbofieldfare")])
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
        case (.POST, "/v1/batches"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleBatchJob(body: body, context: context)
        case (.POST, "/v1/files"):
            handleFileUpload(head: head, body: body, context: context)
        case (.GET, "/v1/files"):
            handleFileList(uri: head.uri, context: context)
        case (.GET, let uri) where uri.hasPrefix("/v1/files/") && uri.hasSuffix("/content"):
            handleFileContent(id: String(uri.dropFirst("/v1/files/".count).dropLast("/content".count)), context: context)
        case (.GET, let uri) where uri.hasPrefix("/v1/files/"):
            handleFileStatus(id: String(uri.dropFirst("/v1/files/".count)), context: context)
        case (.DELETE, let uri) where uri.hasPrefix("/v1/files/"):
            handleFileDelete(id: String(uri.dropFirst("/v1/files/".count)), context: context)
        case (.GET, "/v1/batches"):
            handleBatchList(uri: head.uri, context: context)
        case (.GET, let uri) where uri.hasPrefix("/v1/batches/"):
            handleBatchStatus(uri: uri, context: context)
        case (.POST, let uri) where uri.hasPrefix("/v1/batches/") && uri.hasSuffix("/cancel"):
            handleBatchCancel(uri: uri, context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/batches"), (_, "/v1/files"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    private func handleBatchJob(body: ByteBuffer, context: ChannelHandlerContext) {
        let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
        let box = SendableContext(context)
        activeTask = childChannels.startTask {
            do {
                    let decoded = try? JSONDecoder().decode(OpenAIBatchCreateRequest.self, from: Data(bytes))
                    let requests: [BatchRequest]
                    var inputFileID: String?
                    var completionWindow: String?
                    var metadata: [String: String]?
                    var outputExpiresAfterSeconds: Int?
                    if let create = decoded {
                        guard (create.metadata?.count ?? 0) <= 16,
                              create.metadata?.allSatisfy({ $0.key.utf8.count <= 64 && $0.value.utf8.count <= 512 }) ?? true else {
                            throw ServerRequestError.invalid(message: "metadata supports at most 16 entries with keys up to 64 and values up to 512 UTF-8 bytes", param: "metadata", code: "invalid_value")
                        }
                        guard create.endpoint == "/v1/chat/completions" else {
                            throw ServerRequestError.invalid(message: "only /v1/chat/completions batches are supported", param: "endpoint", code: "unsupported_value")
                        }
                        guard create.completionWindow == "24h" else {
                            throw ServerRequestError.invalid(message: "only completion_window=24h is supported", param: "completion_window", code: "unsupported_value")
                        }
                        if let outputExpiresAfter = create.outputExpiresAfter {
                            guard outputExpiresAfter.anchor == "created_at",
                                  (3_600...2_592_000).contains(outputExpiresAfter.seconds) else {
                                throw ServerRequestError.invalid(message: "output_expires_after requires anchor=created_at and seconds between 3600 and 2592000", param: "output_expires_after", code: "invalid_value")
                            }
                            outputExpiresAfterSeconds = outputExpiresAfter.seconds
                        }
                        guard let input = try await self.files.contents(create.inputFileID) else {
                            throw ServerRequestError.invalid(message: "input file not found", param: "input_file_id", code: "invalid_value")
                        }
                        do {
                            requests = try self.decodeBatchInput(input)
                        } catch let error as ServerRequestError {
                            let snapshot = await self.batches.createFailed(
                                modelID: self.modelID,
                                inputFileID: create.inputFileID,
                                completionWindow: create.completionWindow,
                                metadata: create.metadata,
                                error: .init(code: error.envelope.error.code,
                                             message: error.envelope.error.message,
                                             param: error.envelope.error.param,
                                             line: nil))
                            self.writeCodable(box.value, status: .ok, snapshot)
                            return
                        }
                        inputFileID = create.inputFileID
                        completionWindow = create.completionWindow
                        metadata = create.metadata
                    } else {
                        let legacy = try JSONDecoder().decode(LegacyOpenAIChatBatchRequest.self, from: Data(bytes))
                        guard !legacy.requests.isEmpty else { throw ServerRequestError.invalid(message: "requests must not be empty", param: "requests", code: "invalid_value") }
                        requests = try legacy.requests.enumerated().map { index, item in
                            guard item.stream != true else { throw ServerRequestError.invalid(message: "streaming is not supported for batch requests", param: "requests.stream", code: "unsupported_value") }
                            return BatchRequest(customID: item.customID ?? "request-\(index)", request: try OpenAIRequestValidator.validate(item, modelID: self.modelID))
                        }
                    }
                    let snapshot = try await self.batches.create(requests: requests,
                                                                 backend: self.backend,
                                                                 coordinator: self.coordinator,
                                                                 modelID: self.modelID,
                                                                 inputFileID: inputFileID,
                                                                 completionWindow: completionWindow,
                                                                 metadata: metadata,
                                                                 outputExpiresAfterSeconds: outputExpiresAfterSeconds)
                    self.writeCodable(box.value, status: .ok, snapshot)
            } catch let error as ServerRequestError {
                self.writeError(box.value, status: error == .unknownModel ? .notFound : .badRequest, error.envelope)
            } catch {
                self.writeError(box.value, status: .badRequest,
                                OpenAIErrorEnvelope(message: "malformed JSON request", code: "invalid_json"))
            }
        }
    }

    private func decodeBatchInput(_ data: Data) throws -> [BatchRequest] {
        struct Line: Decodable { let customID: String; let method: String; let url: String; let body: OpenAIChatRequest
            enum CodingKeys: String, CodingKey { case customID = "custom_id", method, url, body } }
        let text = String(decoding: data, as: UTF8.self)
        let lines = text.split(whereSeparator: \.isNewline)
        guard !lines.isEmpty else { throw ServerRequestError.invalid(message: "input file must contain JSONL requests", param: "input_file_id", code: "invalid_value") }
        guard lines.count <= 50_000 else { throw ServerRequestError.invalid(message: "batch input may contain at most 50,000 requests", param: "input_file_id", code: "invalid_value") }
        var ids = Set<String>()
        return try lines.enumerated().map { index, line in
            guard let lineData = String(line).data(using: .utf8) else { throw ServerRequestError.invalid(message: "invalid UTF-8 JSONL line", param: "input_file_id", code: "invalid_value") }
            let item: Line
            do { item = try JSONDecoder().decode(Line.self, from: lineData) }
            catch { throw ServerRequestError.invalid(message: "invalid JSONL request at line \(index + 1)", param: "input_file_id", code: "invalid_value") }
            guard !item.customID.isEmpty, item.customID.utf8.count <= 512 else {
                throw ServerRequestError.invalid(message: "custom_id must contain 1 through 512 UTF-8 bytes", param: "input_file_id", code: "invalid_value")
            }
            guard item.method == "POST", item.url == "/v1/chat/completions", ids.insert(item.customID).inserted else {
                throw ServerRequestError.invalid(message: "invalid Batch request at line \(index + 1)", param: "input_file_id", code: "invalid_value")
            }
            guard item.body.stream != true else { throw ServerRequestError.invalid(message: "streaming is not supported for batch requests", param: "input_file_id", code: "unsupported_value") }
            return BatchRequest(customID: item.customID, request: try OpenAIRequestValidator.validate(item.body, modelID: modelID))
        }
    }

    private func handleFileUpload(head: HTTPRequestHead, body: ByteBuffer, context: ChannelHandlerContext) {
        do {
            guard let contentType = head.headers.first(name: "content-type"),
                  let boundary = contentType.split(separator: ";").map({ $0.trimmingCharacters(in: .whitespaces) }).first(where: { $0.hasPrefix("boundary=") })?.dropFirst("boundary=".count) else {
                throw ServerRequestError.invalid(message: "content-type must be multipart/form-data", param: nil, code: "unsupported_media_type")
            }
            let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
            let parsed = try parseMultipart(Data(bytes), boundary: String(boundary).trimmingCharacters(in: CharacterSet(charactersIn: "\"")))
            let box = SendableContext(context)
            activeTask = childChannels.startTask { do {
                let file = try await self.files.create(filename: parsed.filename, purpose: parsed.purpose, contents: parsed.contents)
                self.writeCodable(box.value, status: .ok, file)
            } catch let error as ServerRequestError { self.writeError(box.value, status: .badRequest, error.envelope) }
              catch { self.writeError(box.value, status: .internalServerError, OpenAIErrorEnvelope(message: "could not store file", code: "internal_error")) } }
        } catch let error as ServerRequestError { writeError(context, status: .badRequest, error.envelope) }
          catch { writeError(context, status: .badRequest, OpenAIErrorEnvelope(message: "malformed multipart request", code: "invalid_request_error")) }
    }

    private func parseMultipart(_ data: Data, boundary: String) throws -> (filename: String, purpose: String, contents: Data) {
        let text = String(decoding: data, as: UTF8.self)
        let parts = text.components(separatedBy: "--\(boundary)")
        var purpose: String?
        var filename: String?
        var contents: Data?
        for rawPart in parts {
            let part = rawPart.trimmingCharacters(in: CharacterSet(charactersIn: "\r\n-"))
            guard let range = part.range(of: "\r\n\r\n") else { continue }
            let headers = String(part[..<range.lowerBound])
            var value = String(part[range.upperBound...])
            if value.hasSuffix("\r\n") { value.removeLast(2) }
            if headers.contains("name=\"purpose\"") { purpose = value }
            if headers.contains("name=\"file\"") {
                filename = headers.components(separatedBy: "filename=\"").dropFirst().first?.components(separatedBy: "\"").first
                contents = Data(value.utf8)
            }
        }
        guard let purpose, let filename, let contents else {
            throw ServerRequestError.invalid(message: "multipart request requires file and purpose", param: nil, code: "invalid_value")
        }
        return (filename, purpose, contents)
    }

    private func handleFileStatus(id: String, context: ChannelHandlerContext) {
        let box = SendableContext(context)
        activeTask = childChannels.startTask { guard let file = await self.files.get(id) else { self.writeError(box.value, status: .notFound, OpenAIErrorEnvelope(message: "file not found", code: "not_found")); return }; self.writeCodable(box.value, status: .ok, file) }
    }

    private func handleFileList(uri: String, context: ChannelHandlerContext) {
        struct List: Encodable {
            let object = "list"
            let data: [BatchFileStore.File]
            let firstID: String?
            let lastID: String?
            let hasMore: Bool
            enum CodingKeys: String, CodingKey {
                case object, data
                case firstID = "first_id"
                case lastID = "last_id"
                case hasMore = "has_more"
            }
        }
        guard let components = URLComponents(string: "http://localhost\(uri)") else {
            writeError(context, status: .badRequest, OpenAIErrorEnvelope(message: "invalid query", code: "invalid_value"))
            return
        }
        let query = components.queryItems ?? []
        let limit = query.first(where: { $0.name == "limit" })?.value.flatMap(Int.init) ?? 10_000
        let order = query.first(where: { $0.name == "order" })?.value ?? "desc"
        guard (1...10_000).contains(limit), order == "asc" || order == "desc" else {
            writeError(context, status: .badRequest, OpenAIErrorEnvelope(message: "invalid Files list query", code: "invalid_value"))
            return
        }
        let after = query.first(where: { $0.name == "after" })?.value
        let purpose = query.first(where: { $0.name == "purpose" })?.value
        let box = SendableContext(context)
        activeTask = childChannels.startTask {
            let all = await self.files.list(order: order, purpose: purpose)
            let start = after.flatMap { id in all.firstIndex(where: { $0.id == id }).map { $0 + 1 } } ?? 0
            let page = Array(all.dropFirst(start).prefix(limit))
            self.writeCodable(box.value, status: .ok, List(data: page, firstID: page.first?.id,
                                                            lastID: page.last?.id,
                                                            hasMore: start + page.count < all.count))
        }
    }

    private func handleFileDelete(id: String, context: ChannelHandlerContext) {
        let box = SendableContext(context)
        activeTask = childChannels.startTask { do {
            guard try await self.files.delete(id) else { self.writeError(box.value, status: .notFound, OpenAIErrorEnvelope(message: "file not found", code: "not_found")); return }
            self.writeJSON(box.value, status: .ok, object: ["id": id, "object": "file", "deleted": true])
        } catch { self.writeError(box.value, status: .internalServerError, OpenAIErrorEnvelope(message: "could not delete file", code: "internal_error")) } }
    }

    private func handleFileContent(id: String, context: ChannelHandlerContext) {
        let box = SendableContext(context)
        activeTask = childChannels.startTask { do { guard let data = try await self.files.contents(id) else { self.writeError(box.value, status: .notFound, OpenAIErrorEnvelope(message: "file not found", code: "not_found")); return }; self.writeRaw(box.value, status: .ok, contentType: "application/jsonl", data: data) } catch { self.writeError(box.value, status: .internalServerError, OpenAIErrorEnvelope(message: "could not read file", code: "internal_error")) } }
    }

    private func handleBatchList(uri: String, context: ChannelHandlerContext) {
        guard let components = URLComponents(string: "http://localhost\(uri)") else {
            writeError(context, status: .badRequest, OpenAIErrorEnvelope(message: "invalid query", code: "invalid_value"))
            return
        }
        let limit = components.queryItems?.first(where: { $0.name == "limit" })?.value
            .flatMap { Int($0) } ?? 20
        guard (1...100).contains(limit) else {
            writeError(context, status: .badRequest, OpenAIErrorEnvelope(message: "limit must be between 1 and 100", param: "limit", code: "invalid_value"))
            return
        }
        let box = SendableContext(context)
        activeTask = childChannels.startTask {
            self.writeCodable(box.value, status: .ok,
                              await self.batches.list(
                                limit: limit,
                                after: components.queryItems?.first(where: { $0.name == "after" })?.value))
        }
    }

    private func handleBatchStatus(uri: String, context: ChannelHandlerContext) {
        let id = String(uri.dropFirst("/v1/batches/".count))
        let box = SendableContext(context)
        activeTask = childChannels.startTask { guard let batch = await self.batches.get(id) else { self.writeError(box.value, status: .notFound, OpenAIErrorEnvelope(message: "batch not found", code: "not_found")); return }; self.writeCodable(box.value, status: .ok, batch) }
    }

    private func handleBatchCancel(uri: String, context: ChannelHandlerContext) {
        let id = String(uri.dropFirst("/v1/batches/".count).dropLast("/cancel".count))
        let box = SendableContext(context)
        activeTask = childChannels.startTask { guard let batch = await self.batches.cancel(id) else { self.writeError(box.value, status: .notFound, OpenAIErrorEnvelope(message: "batch not found", code: "not_found")); return }; self.writeCodable(box.value, status: .ok, batch) }
    }

    private func handleCompletion(body: ByteBuffer,
                                  context: ChannelHandlerContext) {
        do {
            let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
            let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(bytes))
            let request = try OpenAIRequestValidator.validate(decoded, modelID: modelID)
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
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
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
                                          stream: streamState.isStarted)
                }
            }
        } catch let error as ServerRequestError {
            writeError(context,
                       status: error == .unknownModel ? .notFound : .badRequest,
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
        writeJSON(context, status: .ok,
                  object: completionObject(id: id, created: created, completion: completion))
    }

    private func completionObject(id: String,
                                  created: Int,
                                  completion: ServerCompletion) -> [String: Any] {
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
        return object
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
                                  stream: Bool) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = requestError == .queueFull ? .tooManyRequests : .badRequest
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TurboFieldfareServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if !(error is CancellationError) {
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
                                            _ value: T) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope) {
        writeCodable(context, status: status, error)
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func writeRaw(_ context: ChannelHandlerContext, status: HTTPResponseStatus,
                          contentType: String, data: Data) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: contentType)
            headers.add(name: "content-length", value: "\(data.count)")
            contextBox.value.write(self.wrapOutboundOut(.head(HTTPResponseHead(version: .http1_1, status: status, headers: headers))), promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
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

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())

    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown else { return true }
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
