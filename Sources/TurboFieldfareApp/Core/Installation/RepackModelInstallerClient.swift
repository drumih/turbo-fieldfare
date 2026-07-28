import Foundation
import TurboFieldfareRepackCore
import Synchronization

public final class RepackModelInstallerClient: AppModelInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        URL,
        @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL
    typealias DiscardRunner = @Sendable (URL) async throws -> Void

    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let descriptor: AppModelInstallDescriptor
    private let runInstall: InstallRunner
    private let runDiscard: DiscardRunner
    private let taskState = InstallTaskState()

    public init(descriptor: AppModelInstallDescriptor = .default) {
        self.descriptor = descriptor
        self.runInstall = { outputDirectory, progress in
            let paths = try RemoteInstallPaths(outputDirectory: outputDirectory.path)
            let resume = FileManager.default.fileExists(atPath: paths.checkpointFile)
            let options = RemoteStreamingRepackOptions(
                repoID: descriptor.repoID,
                revision: descriptor.revision,
                outputDir: outputDirectory.path,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                requireKnownSource: true,
                minFreeReserveBytes: descriptor.reserveBytes,
                overwrite: true,
                resume: resume)
            let result = try await RemoteStreamingRepacker(options: options).run(progress: progress)
            return URL(fileURLWithPath: result.outputDir).standardizedFileURL
        }
        self.runDiscard = { outputDirectory in
            try RemoteStreamingRepacker.discardPartial(
                outputDirectory: outputDirectory.path)
        }
    }

    init(descriptor: AppModelInstallDescriptor = .default,
         runInstall: @escaping InstallRunner,
         runDiscard: @escaping DiscardRunner = { _ in }) {
        self.descriptor = descriptor
        self.runInstall = runInstall
        self.runDiscard = runDiscard
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        let saved = try RemoteStreamingRepacker.inspectPersistentInstall(
            outputDirectory: outputDirectory.path,
            repoID: descriptor.repoID,
            requestedRevision: descriptor.revision)
        let remainingBytes: UInt64
        if let saved {
            let checkpointPath = try RemoteInstallPaths(
                outputDirectory: outputDirectory.path).checkpointFile
            let reused = try saved.validatedDestinationBytes(
                maximum: descriptor.installedBytes,
                path: checkpointPath)
            remainingBytes = descriptor.installedBytes > reused
                ? descriptor.installedBytes - reused
                : 0
        } else {
            remainingBytes = descriptor.installedBytes
        }
        let requested = remainingBytes.addingReportingOverflow(
            descriptor.rangeStagingBytes)
        guard !requested.overflow else {
            throw RepackError.configurationInvalid(
                detail: "model install requirement overflows UInt64")
        }
        let requirement = try DiskSpaceChecker.assess(
            path: outputDirectory.path,
            bytes: requested.partialValue,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: requirement.path,
                                          requiredBytes: requirement.requiredBytes,
                                          availableBytes: requirement.availableBytes)
    }

    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    continuation.yield(.checking)
                    let completedDirectory = try await runInstall(outputDirectory) { progress in
                        continuation.yield(Self.event(for: progress))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(completedDirectory))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()

            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    public func discardPartialInstall(outputDirectory: URL) async throws {
        let directory = outputDirectory.standardizedFileURL
        try await Task.detached(priority: .utility) { [runDiscard] in
            try await runDiscard(directory)
        }.value
    }

    static func event(for progress: ModelInstallProgress) -> AppModelInstallEvent {
        switch progress {
        case .downloadingMetadata:
            return .downloadingMetadata
        case .planning:
            return .planning
        case .checkingDisk:
            return .checking
        case .reservingOutput:
            return .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            return .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            return .hashingOutput(file)
        case .finalizing:
            return .finalizing
        }
    }
}
