import Darwin
import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct InstallLockTests {
    @Test func twoContendersForCanonicalTargetCannotBothAcquire() throws {
        let root = temporaryRoot("contenders")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        var first: InstallLock? = try InstallLock.acquire(outputDirectory: output)

        #expect(throws: RepackError.self) {
            _ = try InstallLock.acquire(outputDirectory: output)
        }

        first = nil
        _ = try InstallLock.acquire(outputDirectory: output)
        #expect(first == nil)
    }

    @Test func asyncInstallHoldsLockUntilOperationFinishes() async throws {
        let root = temporaryRoot("async-operation")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        HangingInstallURLProtocol.reset()
        let task = Task {
            try await RemoteStreamingRepacker(
                options: RemoteStreamingRepackOptions(
                    repoID: "owner/model",
                    revision: "main",
                    outputDir: output,
                    minFreeReserveBytes: 0,
                    overwrite: true,
                    downloadSession: RemoteDownloadSession(
                        protocolClasses: [HangingInstallURLProtocol.self]),
                    baseURL: URL(string: "https://lock.test")!,
                    rangeRetryAttempts: 0,
                    retryBaseDelayNs: 0)
            ).run()
        }

        for _ in 0..<200 where !HangingInstallURLProtocol.started {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(HangingInstallURLProtocol.started)
        do {
            _ = try InstallLock.acquire(outputDirectory: output)
            Issue.record("second contender acquired the active install lock")
        } catch let error as RepackError {
            guard case .installBusy = error else {
                Issue.record("expected installBusy, got \(error)")
                task.cancel()
                return
            }
        }

        task.cancel()
        await #expect(throws: (any Error).self) {
            _ = try await task.value
        }
        _ = try InstallLock.acquire(outputDirectory: output)
    }

    @Test func symlinkedParentAliasesContendOnOnePhysicalLock() throws {
        let root = temporaryRoot("alias")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let physical = (root as NSString).appendingPathComponent("physical")
        let alias = (root as NSString).appendingPathComponent("alias")
        try Posix.mkdirP(physical)
        try FileManager.default.createSymbolicLink(
            atPath: alias,
            withDestinationPath: physical)
        let first = try InstallLock.acquire(
            outputDirectory: (physical as NSString).appendingPathComponent("model.gturbo"))

        #expect(throws: RepackError.self) {
            _ = try InstallLock.acquire(
                outputDirectory: (alias as NSString).appendingPathComponent("model.gturbo"))
        }
        withExtendedLifetime(first) {}
    }

    @Test func symlinkedLockIsRejectedWithoutFollowingIt() throws {
        let root = temporaryRoot("lock-symlink")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        let victim = (root as NSString).appendingPathComponent("victim")
        FileManager.default.createFile(atPath: victim, contents: Data())
        try FileManager.default.createSymbolicLink(
            atPath: output + ".install.lock",
            withDestinationPath: victim)

        #expect(throws: RepackError.self) {
            _ = try InstallLock.acquire(outputDirectory: output)
        }
    }

    @Test func lockIsReleasedWhenOwningProcessIsKilled() throws {
        let root = temporaryRoot("process-death")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        let paths = try RemoteInstallPaths(outputDirectory: output)
        let holder = Process()
        holder.executableURL = URL(fileURLWithPath: "/usr/bin/lockf")
        holder.arguments = ["-k", paths.lockFile, "/bin/sleep", "30"]
        try holder.run()
        defer {
            if holder.isRunning {
                holder.terminate()
                holder.waitUntilExit()
            }
        }

        var observedContention = false
        for _ in 0..<100 {
            do {
                _ = try InstallLock.acquire(outputDirectory: output)
            } catch {
                observedContention = true
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        #expect(observedContention)
        #expect(throws: RepackError.self) {
            _ = try InstallLock.acquire(outputDirectory: output)
        }

        #expect(kill(holder.processIdentifier, SIGKILL) == 0)
        holder.waitUntilExit()
        _ = try InstallLock.acquire(outputDirectory: output)
    }

    @Test func openedLockDescriptorDetectsPathReplacement() throws {
        let root = temporaryRoot("inode-replacement")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let path = (root as NSString).appendingPathComponent("model.install.lock")
        let descriptor = try Posix.openLock(path)
        defer { close(descriptor) }

        #expect(try Posix.descriptorMatchesPath(descriptor, path: path))
        try FileManager.default.removeItem(atPath: path)
        FileManager.default.createFile(atPath: path, contents: Data())

        #expect(try !Posix.descriptorMatchesPath(descriptor, path: path))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-lock-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }
}

private final class HangingInstallURLProtocol: URLProtocol, @unchecked Sendable {
    private static let state = Mutex(false)

    static var started: Bool {
        state.withLock { $0 }
    }

    static func reset() {
        state.withLock { $0 = false }
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.state.withLock { $0 = true }
    }

    override func stopLoading() {}
}
