import Darwin
import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Ingress-side resource ownership: what a request is allowed to allocate
/// before it has been accepted, and when those allocations come back.
///
/// The unit under test is the connection, not the completion, so the backend
/// only has to answer.
private actor EchoBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

@Suite struct ServerIngressHardeningTests {
    /// A staging root of this test's own. The process-wide one is shared with
    /// every other server in the suite, so counting files under it measures
    /// whatever else happens to be running.
    private static func makeRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ingress-\(UUID().uuidString)", isDirectory: true)
    }

    /// Every file staged under this root, however deep. A leaked lease shows up
    /// here as a directory that outlives its request.
    private static func stagedFileCount(_ root: URL) -> Int {
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { return 0 }
        var count = 0
        for case let url as URL in walker {
            let regular = (try? url.resourceValues(forKeys: [.isRegularFileKey]))?
                .isRegularFile ?? false
            if regular { count += 1 }
        }
        return count
    }

    private static func imageBody(count: Int) -> String {
        // A one-pixel PNG, valid enough to be staged; the request is rejected
        // later, which is exactly the case that used to leak.
        let pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        let parts = (0..<count).map {
            _ in #"{"type":"image_url","image_url":{"url":"data:image/png;base64,\#(pixel)"}}"#
        }.joined(separator: ",")
        return #"{"model":"test-model","messages":[{"role":"user","content":[\#(parts)]}]}"#
    }

    /// A rejected request must give its staged images back immediately. The
    /// client is under no obligation to close a keep-alive connection, so
    /// holding them until `channelInactive` means holding them forever.
    @Test func rejectedRequestReleasesStagedImagesWithoutClosingTheConnection() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend(),
            visionCapability: "ready", attachmentRoot: root)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        // Over the per-request image cap, so staging happens and then fails.
        let body = Self.imageBody(count: ServerAttachmentStore.maximumRequestImages + 4)
        try writeAll(socket: socket, text: httpRequest(
            port: port, body: body, connection: "keep-alive"))
        let response = try readUntil(
            socket: socket, timeoutMilliseconds: 4_000,
            condition: { $0.contains("HTTP/1.1 4") })
        #expect(response.contains("too_many_images"))

        // Connection deliberately still open.
        #expect(Self.stagedFileCount(root) == 0,
                "staged images survived the rejection that discarded them")
        try await server.shutdown()
    }

    /// A server that cannot serve images must refuse them during the body
    /// parse — before a byte is decoded to disk — not after the request has
    /// queued behind a generation.
    @Test func imagesAreRefusedWithoutStagingWhenVisionIsUnavailable() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend(),
            visionCapability: "missing", attachmentRoot: root)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        try writeAll(socket: socket, text: httpRequest(
            port: port, body: Self.imageBody(count: 1), connection: "keep-alive"))
        let response = try readUntil(
            socket: socket, timeoutMilliseconds: 4_000,
            condition: { $0.contains("HTTP/1.1 4") })
        #expect(response.contains("vision_unavailable"))
        #expect(Self.stagedFileCount(root) == 0,
                "an image was staged on a server that cannot serve images")
        try await server.shutdown()
    }

    /// Nothing may be written to disk for a request the router will not accept.
    /// Staging used to run before the method, path and content-type were even
    /// looked at, so `PUT /health` could fill the attachment directory.
    @Test func bodiesOnUnroutedRequestsAreNeverStaged() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend(),
            attachmentRoot: root)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let body = Self.imageBody(count: 3)

        func send(_ head: String, expecting status: String) throws {
            let socket = try connectedSocket(port: port)
            defer { _ = Darwin.close(socket) }
            try writeAll(socket: socket, text: head
                + "Host: 127.0.0.1:\(port)\r\n"
                + "Content-Length: \(body.utf8.count)\r\n"
                + "Connection: close\r\n\r\n" + body)
            let response = try readUntil(
                socket: socket, timeoutMilliseconds: 4_000,
                condition: { $0.contains("HTTP/1.1 ") })
            #expect(response.contains(status), "got \(response.prefix(64))")
            #expect(Self.stagedFileCount(root) == 0,
                    "a body was staged before its request was routed")
        }

        try send("PUT /health HTTP/1.1\r\nContent-Type: application/json\r\n",
                 expecting: "405")
        try send("POST /v1/nope HTTP/1.1\r\nContent-Type: application/json\r\n",
                 expecting: "404")
        try send("POST /v1/chat/completions HTTP/1.1\r\nContent-Type: text/plain\r\n",
                 expecting: "415")
        try await server.shutdown()
    }

    /// A body on an unrouted request is discarded rather than parsed, so the
    /// parser's size cap no longer applies to it. It still has to be bounded.
    @Test func discardedBodiesAreStillBounded() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        let chunk = String(repeating: "x", count: 1 << 20)
        let total = StreamingChatRequestBody.maximumWireBytes + (4 << 20)
        try writeAll(socket: socket, text: "POST /v1/nope HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(total)\r\n"
            + "Connection: close\r\n\r\n")
        var sent = 0
        var response = ""
        while sent < total {
            do { try writeAll(socket: socket, text: chunk) } catch { break }
            sent += chunk.utf8.count
            // The server answers as soon as the cap is passed and then closes.
            if let peek = try? readAvailable(socket: socket, timeoutMilliseconds: 0),
               peek.contains("HTTP/1.1 ") {
                response = peek
                break
            }
        }
        if !response.contains("HTTP/1.1 ") {
            response = try readUntil(
                socket: socket, timeoutMilliseconds: 4_000,
                condition: { $0.contains("HTTP/1.1 ") })
        }
        #expect(response.contains("413") || response.contains("request_too_large"),
                "unbounded discarded body: \(response.prefix(64))")
        try await server.shutdown()
    }

    /// An idle keep-alive connection holds a slot against the connection cap
    /// forever unless the server closes it.
    @Test func idleConnectionsAreClosed() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend(),
            idleTimeout: .milliseconds(200))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        // One completed request, then silence.
        try writeAll(socket: socket, text: httpRequest(
            port: port, body: #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
            connection: "keep-alive"))
        _ = try readUntil(socket: socket, timeoutMilliseconds: 4_000,
                          condition: { $0.contains("\"content\":\"ok\"") })

        // recv returning 0 is the peer's FIN.
        var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
        var closed = false
        let deadline = Date().addingTimeInterval(4)
        while Date() < deadline {
            guard Darwin.poll(&descriptor, 1, 100) > 0 else { continue }
            var buffer = [UInt8](repeating: 0, count: 1_024)
            if Darwin.recv(socket, &buffer, buffer.count, 0) == 0 { closed = true; break }
        }
        #expect(closed, "idle connection was never closed")
        try await server.shutdown()
    }

    /// A request whose parse already failed must not become an unbounded
    /// drain: past the wire cap the server answers and closes, because a
    /// chunked stream may never send the .end that would otherwise produce
    /// the response, and every read resets the idle timer.
    @Test func bodiesPastTheWireCapAreAnsweredAndClosed() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        let chunk = String(repeating: "x", count: 1 << 20)
        let total = StreamingChatRequestBody.maximumWireBytes + (4 << 20)
        try writeAll(socket: socket, text: "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(total)\r\n\r\n")
        var sent = 0
        var response = ""
        while sent < total {
            do { try writeAll(socket: socket, text: chunk) } catch { break }
            sent += chunk.utf8.count
            if let peek = try? readAvailable(socket: socket, timeoutMilliseconds: 0),
               peek.contains("HTTP/1.1 ") {
                response = peek
                break
            }
        }
        if !response.contains("HTTP/1.1 ") {
            response = try readUntil(
                socket: socket, timeoutMilliseconds: 4_000,
                condition: { $0.contains("HTTP/1.1 ") })
        }
        #expect(response.contains("413") || response.contains("request_too_large"),
                "unbounded latched body: \(response.prefix(64))")

        var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
        var closed = false
        let deadline = Date().addingTimeInterval(4)
        while Date() < deadline {
            guard Darwin.poll(&descriptor, 1, 100) > 0 else { continue }
            var buffer = [UInt8](repeating: 0, count: 4_096)
            if Darwin.recv(socket, &buffer, buffer.count, 0) == 0 { closed = true; break }
        }
        #expect(closed, "a connection past the wire cap stayed open")
        try await server.shutdown()
    }

    /// A client that sends a request head and goes silent is a stalled upload,
    /// not a request in flight: it must not hold a slot against the connection
    /// cap past the read timeout. An active uploader is safe — every body
    /// chunk it sends resets the timer.
    @Test func stalledUploadsAreReapedByTheIdleTimeout() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend(),
            idleTimeout: .milliseconds(200))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        // A head promising a body that never arrives.
        try writeAll(socket: socket, text: "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: 1024\r\n\r\n{\"model\":")

        var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
        var closed = false
        let deadline = Date().addingTimeInterval(4)
        while Date() < deadline {
            guard Darwin.poll(&descriptor, 1, 100) > 0 else { continue }
            var buffer = [UInt8](repeating: 0, count: 1_024)
            if Darwin.recv(socket, &buffer, buffer.count, 0) == 0 { closed = true; break }
        }
        #expect(closed, "a stalled upload was never reaped")
        try await server.shutdown()
    }

    /// Accepted sockets were unbounded: each costs a descriptor and can pin a
    /// staging directory, so the count has to be capped.
    @Test func connectionsBeyondTheCapAreClosed() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let cap = TurboFieldfareHTTPServer.maximumConnections
        var sockets: [Int32] = []
        defer { sockets.forEach { _ = Darwin.close($0) } }
        for _ in 0..<cap {
            sockets.append(try connectedSocket(port: port))
        }
        // The accept loop runs on the event loop; wait for it to catch up.
        let deadline = Date().addingTimeInterval(4)
        while await server.acceptedConnectionCount < cap, Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(await server.acceptedConnectionCount == cap)

        let overflow = try connectedSocket(port: port)
        sockets.append(overflow)
        // A refused connection is accepted by the kernel and then closed, so it
        // reads EOF rather than failing to connect.
        var descriptor = pollfd(fd: overflow, events: Int16(POLLIN), revents: 0)
        var closed = false
        let overflowDeadline = Date().addingTimeInterval(4)
        while Date() < overflowDeadline {
            guard Darwin.poll(&descriptor, 1, 100) > 0 else { continue }
            var buffer = [UInt8](repeating: 0, count: 64)
            if Darwin.recv(overflow, &buffer, buffer.count, 0) == 0 { closed = true; break }
        }
        #expect(closed, "connection past the cap stayed open")
        #expect(await server.acceptedConnectionCount == cap)
        try await server.shutdown()
    }

    /// Pins the listen backlog to the connection cap. With a backlog of 16
    /// under a cap of 128, a burst of connects overflowed the listen queue;
    /// macOS 27 answers that with RST, so `connectionsBeyondTheCapAreClosed`
    /// failed there with ECONNRESET on connect (issue #151), while macOS 26
    /// only stalls the burst and the race stays invisible.
    @Test func listenBacklogCoversTheConnectionCap() async throws {
        let server = TurboFieldfareHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: EchoBackend())
        let channel = try await server.start(port: 0)
        let backlog = try await channel.getOption(ChannelOptions.backlog)
        #expect(backlog >= Int32(TurboFieldfareHTTPServer.maximumConnections))
        try await server.shutdown()
    }
}

/// The sweep is what reclaims staging directories left by a process that died
/// before its leases could release them.
@Suite struct ServerAttachmentSweepTests {
    @Test func sweepReclaimsDeadProcessesAndSparesLiveOnes() throws {
        let manager = FileManager.default
        // An isolated root: a server starting in a parallel test sweeps the
        // real one, which would race this test's fixtures.
        let root = manager.temporaryDirectory
            .appendingPathComponent("sweep-\(UUID().uuidString)", isDirectory: true)
        defer { try? manager.removeItem(at: root) }
        try manager.createDirectory(at: root, withIntermediateDirectories: true)

        // A pid that cannot be running: pid_t values are positive and this one
        // is past the system maximum.
        let dead = root.appendingPathComponent("pid-4194300", isDirectory: true)
        let deadFile = dead.appendingPathComponent("staged.bin")
        try manager.createDirectory(at: dead, withIntermediateDirectories: true)
        try Data([0x1]).write(to: deadFile)

        // A directory from an older layout, owned by no process at all.
        let legacy = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: legacy, withIntermediateDirectories: true)

        // This process is alive, so its own directory must survive.
        let live = try ServerAttachmentDirectory.makeLease(in: root)
        let liveFile = live.directoryURL.appendingPathComponent("staged.bin")
        try Data([0x2]).write(to: liveFile)

        let reclaimed = ServerAttachmentDirectory.sweepAbandoned(in: root)
        #expect(reclaimed.contains("pid-4194300"))
        #expect(reclaimed.contains(legacy.lastPathComponent))
        #expect(!manager.fileExists(atPath: dead.path))
        #expect(!manager.fileExists(atPath: legacy.path))
        #expect(manager.fileExists(atPath: liveFile.path),
                "the sweep deleted a live process's staged images")

        withExtendedLifetime(live) {}
    }

    @Test func leasesAreScopedUnderThisProcess() throws {
        let lease = try ServerAttachmentDirectory.makeLease()
        defer { try? FileManager.default.removeItem(at: lease.directoryURL) }
        #expect(lease.directoryURL.deletingLastPathComponent().lastPathComponent
                    == "pid-\(getpid())")
        #expect(lease.directoryURL.deletingLastPathComponent()
                    .deletingLastPathComponent().lastPathComponent
                    == ServerAttachmentDirectory.rootName)
    }
}
