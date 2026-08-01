import Foundation
import TurboFieldfareRepackCore
import Synchronization

public final class RepackModelInstallerClient: AppModelInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        ModelCatalogEntry,
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
        self.runInstall = { entry, outputDirectory, progress in
            let paths = try RemoteInstallPaths(outputDirectory: outputDirectory.path)
            let resume = FileManager.default.fileExists(atPath: paths.checkpointFile)
            let options = Self.repackOptions(
                entry: entry,
                outputDirectory: outputDirectory,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
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

    /// Extracted so the trust decision is testable without a network install:
    /// `requireKnownSource` is the gate that keeps a curated model pinned to the
    /// project's fingerprint, and getting it backwards would silently accept an
    /// unvetted upload in the verified tier.
    static func repackOptions(entry: ModelCatalogEntry,
                              outputDirectory: URL,
                              token: String?,
                              resume: Bool) -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: entry.repoID,
            revision: entry.revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: ModelTrustPolicy.requiresKnownSource(for: entry),
            minFreeReserveBytes: entry.reserveBytes,
            overwrite: true,
            resume: resume)
    }

    public func checkInstallRequirement(entry: ModelCatalogEntry,
                                        outputDirectory: URL) throws -> AppModelInstallRequirement {
        try Self.checkInstallRequirement(
            descriptor: AppModelInstallDescriptor(entry: entry),
            outputDirectory: outputDirectory)
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        try Self.checkInstallRequirement(descriptor: descriptor,
                                         outputDirectory: outputDirectory)
    }

    private static func checkInstallRequirement(
        descriptor: AppModelInstallDescriptor,
        outputDirectory: URL) throws -> AppModelInstallRequirement {
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
        // Uses this client's own descriptor rather than the curated head so an
        // injected descriptor still drives the install, and maps to the curated
        // tier to preserve the pre-multi-model behaviour of always requiring a
        // known source on this path.
        install(entry: ModelCatalogEntry(descriptor: descriptor, trustTier: .curated),
                outputDirectory: outputDirectory)
    }

    public func install(entry: ModelCatalogEntry,
                        outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    continuation.yield(.checking)
                    let completedDirectory = try await runInstall(entry, outputDirectory) { progress in
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
