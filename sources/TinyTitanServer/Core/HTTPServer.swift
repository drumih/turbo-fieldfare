import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TinyTitanCore
import TinyTitanLib

public actor TinyTitanHTTPServer {
    public static let maximumBodyBytes = 1_048_576

    /// S1: close a connection that has been idle (no request in flight) for
    /// this long. Also used as the pipeline read timeout, bounding slowloris
    /// style partial-request stalls.
    public static let idleReadTimeout: TimeAmount = .seconds(120)

    /// S1: reject connections beyond this cap to bound FD/memory usage.
    public static let maximumConcurrentConnections = 64

    /// Ceiling on the *aggregate* request header block, checked when the head
    /// arrives.
    ///
    /// NIO caps a single header field at 80 KiB and nothing else: it has no
    /// field-count or total-size limit, and `HTTPDecoder` exposes none to
    /// configure (only `leftOverBytesStrategy` and
    /// `informationalResponseStrategy`). So a client can send an unbounded
    /// number of small headers and the decoder will accumulate all of them
    /// before this handler is handed the head. What this bound can do is refuse
    /// the request before any routing or generation work and say why; it cannot
    /// stop NIO's own buffering, which is why the value is generous rather than
    /// tight — 16 KiB is far more than any real client sends and small enough
    /// that a request that trips it is not one worth serving.
    public static let maximumRequestHeaderBytes = 16 * 1024

    /// Ceiling on the number of header fields, for the same reason. A field
    /// count is what a hostile client can inflate without inflating bytes.
    public static let maximumRequestHeaderFields = 128

    /// How much of a refused request's body is read and discarded before the
    /// connection is closed.
    ///
    /// Answering and closing immediately is the tempting version and it is
    /// wrong: a client mid-upload often sees a connection reset instead of the
    /// response it was just sent, which turns "your body is too large" into "the
    /// connection broke". Draining a bounded remainder lets the client finish
    /// writing, read the refusal, and shut down in order — while still bounding
    /// what a multi-gigabyte body can make the server consume to this much.
    public static let maximumDrainedBytesAfterReject = 4 * 1024 * 1024

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let reasoningProfile: ServerReasoningProfile
    /// Set when the server routes between catalog models; nil keeps the
    /// single-model server exactly as it was.
    private let router: (any ModelRouting)?
    private let childChannels = ChildChannelRegistry(
        maximumChannels: maximumConcurrentConnections)
    /// Finished /v1/responses kept for previous_response_id and retrieval.
    private let responseStore = ResponseStore()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    package init(
        modelID: String,
        queueLimit: Int,
        maxConcurrentSequences: Int = 1,
        backend: any ServerInferenceBackend,
        heartbeatInterval: TimeAmount = .seconds(5),
        reasoningProfile: ServerReasoningProfile = .default,
        group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1),
        router: (any ModelRouting)? = nil,
        coordinator: ServerCoordinator? = nil
    ) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        // Injectable so a resident side-engine can read the same
        // `generating` signal the coordinator raises for every client
        // generation, which is how it chooses its width.
        self.coordinator =
            coordinator
            ?? ServerCoordinator(
                queueLimit: queueLimit,
                width: maxConcurrentSequences)
        self.heartbeatInterval = heartbeatInterval
        self.reasoningProfile = reasoningProfile
        self.router = router
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let reasoningProfile = self.reasoningProfile
        let router = self.router
        let childChannels = self.childChannels
        let responseStore = self.responseStore
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(
                        ServerHTTPHandler(
                            modelID: modelID,
                            backend: backend,
                            coordinator: coordinator,
                            heartbeatInterval: heartbeatInterval,
                            reasoningProfile: reasoningProfile,
                            router: router,
                            childChannels: childChannels,
                            responseStore: responseStore))
                }
            }
        // S29: so_reuseaddr belongs on the listening socket only, not on
        // accepted sockets.
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

/// The workspace a request names, from `X-TinyTitan-Workspace`.
///
/// This is how one server serves several checkouts, and it was documented in
/// three places and read in none: `ValidatedChatRequest.workspace` had no
/// caller at all, so two projects sharing a server silently shared a memory
/// store — the exact cross-project mixing the workspace design exists to
/// prevent.
///
/// Only the shape is checked here. Whether the name is *allowed* — the
/// reserved shared workspace, characters a scope forbids — belongs to the
/// memory layer, which already refuses those and disables memory for the
/// request rather than failing it. A bad workspace must never cost someone
/// their answer.
enum WorkspaceHeader {
    static let name = "x-tinytitan-workspace"

    static func value(in head: HTTPRequestHead?) -> String? {
        guard let raw = head?.headers.first(name: name) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        // A proxy that adds the header with nothing after it must not create
        // a workspace called "".
        return trimmed.isEmpty ? nil : trimmed
    }
}
