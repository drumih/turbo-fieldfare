import Darwin
import Foundation
import Synchronization
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

@Suite(.serialized) struct DecodeServiceWriteLifetimeTests {
    @Test func concurrentWriteAndShutdownDoNotCrash() throws {
        for _ in 0..<256 {
            let socket = try DecodeServiceTestSocket()
            let client = DecodeServiceInferenceClient(testConnection: { (socket.input, socket.output) })
            let handles = try client.ensureProcess()
            let start = DispatchSemaphore(value: 0)
            let finished = DispatchGroup()
            finished.enter()
            Thread {
                defer { finished.leave() }
                start.wait()
                do {
                    try client.write(.cancel, to: handles.input, expecting: handles.responses)
                } catch AppInferenceError.connectionLost {
                } catch {
                    Issue.record("Unexpected write failure: \(error)")
                }
            }.start()
            finished.enter()
            Thread {
                defer { finished.leave() }
                start.wait()
                client.shutdownForTermination()
            }.start()
            start.signal()
            start.signal()
            try #require(finished.wait(timeout: .now() + 5) == .success)
            #expect(client.installedRouter == nil)
        }
    }

    @Test func shutdownInterruptsAWriteToAServiceThatStoppedReading() throws {
        let socket = try DecodeServiceTestSocket()
        let client = DecodeServiceInferenceClient(testConnection: { (socket.input, socket.output) })
        let handles = try client.ensureProcess()
        let command = DecodeServiceCommand.load(DecodeLoadRequest(
            modelPath: String(repeating: "x", count: 1_048_576),
            maxContextTokens: 512, runtimeOptions: DecodeRuntimeOptions(),
            forceLogitsHead: false))
        let written = DispatchSemaphore(value: 0)
        let stopped = DispatchSemaphore(value: 0)
        let failure = Mutex<String?>(nil)
        Thread {
            defer { written.signal() }
            do {
                try client.write(command, to: handles.input, expecting: handles.responses)
            } catch AppInferenceError.connectionLost {
                failure.withLock { $0 = "connectionLost" }
            } catch {
                failure.withLock { $0 = String(describing: error) }
            }
        }.start()
        defer {
            shutdown(socket.peer.fileDescriptor, SHUT_RDWR)
            client.shutdownForTermination()
        }
        // Wait for bytes, not a sleep: the frame exceeds the socket capacity,
        // so its writer cannot finish while the peer refuses to drain it.
        var ready = pollfd(fd: socket.peer.fileDescriptor, events: Int16(POLLIN), revents: 0)
        try #require(poll(&ready, 1, 5_000) == 1)
        #expect(written.wait(timeout: .now()) == .timedOut)
        Thread {
            client.shutdownForTermination()
            stopped.signal()
        }.start()
        #expect(stopped.wait(timeout: .now() + 5) == .success,
                "Teardown waited behind a blocked write")
        #expect(written.wait(timeout: .now() + 5) == .success,
                "The blocked write outlived teardown")
        #expect(failure.withLock { $0 } == "connectionLost")
    }

    @Test(arguments: [false, true])
    func retiredHandleFailsWithoutClosingReplacement(reconnect: Bool) throws {
        let first = try DecodeServiceTestSocket()
        let second = try DecodeServiceTestSocket()
        let attempts = Mutex(0)
        let client = DecodeServiceInferenceClient(testConnection: {
            let index = attempts.withLock { value in
                defer { value += 1 }
                return value
            }
            let socket = index == 0 ? first : second
            return (socket.input, socket.output)
        })
        defer { client.shutdownForTermination() }
        let retired = try client.ensureProcess()
        client.shutdownForTermination()
        let replacement = reconnect ? try client.ensureProcess() : nil

        // Without the identity check, a retained snapshot writes through the
        // closed FileHandle and raises an Objective-C exception, not a Swift error.
        do {
            try client.write(.cancel, to: retired.input, expecting: retired.responses)
            Issue.record("A retired connection accepted a command")
        } catch AppInferenceError.connectionLost {
        } catch {
            Issue.record("Wrong error for a retired connection: \(error)")
        }
        if let replacement {
            #expect(client.installedRouter === replacement.responses)
            try client.write(.cancel, to: replacement.input, expecting: replacement.responses)
            guard case .cancel = try DecodeFrameCodec.read(
                DecodeServiceCommand.self, from: second.peer) else {
                Issue.record("Replacement did not receive the command")
                return
            }
        } else {
            #expect(client.installedRouter == nil)
        }
    }
}

final class DecodeServiceTestSocket: @unchecked Sendable {
    let input: FileHandle
    let output: FileHandle
    let peer: FileHandle

    func readCommand() async throws -> DecodeServiceCommand {
        try await withCheckedThrowingContinuation { continuation in
            Thread { [peer] in
                continuation.resume(with: Result {
                    var ready = pollfd(fd: peer.fileDescriptor, events: Int16(POLLIN), revents: 0)
                    let result = poll(&ready, 1, 5_000)
                    guard result > 0 else {
                        throw POSIXError(result == 0 ? .ETIMEDOUT : (.init(rawValue: errno) ?? .EIO))
                    }
                    return try DecodeFrameCodec.read(DecodeServiceCommand.self, from: peer)
                })
            }.start()
        }
    }

    init() throws {
        var pair: [Int32] = [-1, -1]
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        let reader = dup(pair[0])
        guard reader >= 0 else {
            let error = POSIXError(.init(rawValue: errno) ?? .EIO)
            Darwin.close(pair[0])
            Darwin.close(pair[1])
            throw error
        }
        DecodeUnixSocket.disableSIGPIPE(on: pair[0])
        input = FileHandle(fileDescriptor: pair[0], closeOnDealloc: true)
        output = FileHandle(fileDescriptor: reader, closeOnDealloc: true)
        peer = FileHandle(fileDescriptor: pair[1], closeOnDealloc: true)
    }
}
