import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareRepackCore

public final class RepackVisionPackInstallerClient: AppVisionPackInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        URL,
        @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL
    typealias MutationRunner = @Sendable (URL) async throws -> Void

    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let descriptor: AppModelInstallDescriptor
    private let runInstall: InstallRunner
    private let runDiscard: MutationRunner
    private let runRemove: MutationRunner
    private let taskState = InstallTaskState()

    public init(descriptor: AppModelInstallDescriptor = .visionCompanion) {
        self.descriptor = descriptor
        self.runInstall = { textModelDirectory, progress in
            let output = try VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            let paths = try RemoteInstallPaths(outputDirectory: output.path)
            let resume = FileManager.default.fileExists(atPath: paths.checkpointFile)
            let options = RemoteVisionPackInstallOptions(
                repoID: descriptor.repoID,
                revision: descriptor.revision,
                textModelDirectory: textModelDirectory.path,
                outputDirectory: output.path,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                minFreeReserveBytes: descriptor.reserveBytes,
                overwrite: true,
                resume: resume)
            let result = try await RemoteVisionPackInstaller(options: options)
                .prepare(progress: progress)
            return URL(fileURLWithPath: result.outputDirectory).standardizedFileURL
        }
        self.runDiscard = { textModelDirectory in
            let output = try VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            try RemoteVisionPackInstaller.discardPartial(
                outputDirectory: output.path)
        }
        self.runRemove = { textModelDirectory in
            let output = try VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            try RemoteVisionPackInstaller.removeInstalled(
                outputDirectory: output.path)
        }
    }

    init(
        descriptor: AppModelInstallDescriptor = .visionCompanion,
        runInstall: @escaping InstallRunner,
        runDiscard: @escaping MutationRunner = { _ in },
        runRemove: @escaping MutationRunner = { _ in }
    ) {
        self.descriptor = descriptor
        self.runInstall = runInstall
        self.runDiscard = runDiscard
        self.runRemove = runRemove
    }

    public func checkInstallRequirement(
        textModelDirectory: URL
    ) throws -> AppModelInstallRequirement {
        let output = try VisionPackLocation.companionURL(
            forTextModel: textModelDirectory)
        let saved = try RemoteVisionPackInstaller.inspectPersistentInstall(
            outputDirectory: output.path,
            repoID: descriptor.repoID,
            requestedRevision: descriptor.revision)
        let reused = try saved?.validatedDestinationBytes(
            maximum: descriptor.installedBytes,
            path: RemoteInstallPaths(outputDirectory: output.path).checkpointFile) ?? 0
        let remaining = descriptor.installedBytes > reused
            ? descriptor.installedBytes - reused
            : 0
        let requested = remaining.addingReportingOverflow(descriptor.rangeStagingBytes)
        guard !requested.overflow else {
            throw RepackError.configurationInvalid(
                detail: "vision install requirement overflows UInt64")
        }
        let requirement = try DiskSpaceChecker.assess(
            path: output.path,
            bytes: requested.partialValue,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(
            probePath: requirement.path,
            requiredBytes: requirement.requiredBytes,
            availableBytes: requirement.availableBytes)
    }

    public func install(
        textModelDirectory: URL
    ) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    continuation.yield(.checking)
                    let directory = try await runInstall(textModelDirectory) { progress in
                        continuation.yield(RepackModelInstallerClient.event(for: progress))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.readyToActivate(directory))
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

    public func discardPartialInstall(textModelDirectory: URL) async throws {
        try await mutate(textModelDirectory, using: runDiscard)
    }

    public func preparedInstallIsValid(textModelDirectory: URL) -> Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: textModelDirectory) else { return false }
        return RemoteVisionPackInstaller.preparedInstallIsValid(
            outputDirectory: output.path,
            textModelDirectory: textModelDirectory.path)
    }

    public func activatePreparedInstall(
        textModelDirectory: URL,
        onVerifyProgress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let textModelDirectory = textModelDirectory.standardizedFileURL
        let output = try VisionPackLocation.companionURL(
            forTextModel: textModelDirectory)
        try await runCancellableDetached { [descriptor] in
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: output.path,
                textModelDirectory: textModelDirectory.path,
                repoID: descriptor.repoID,
                requestedRevision: descriptor.revision,
                onVerifyProgress: { hashed, total in
                    // Cancellation rides the progress callback: verification is
                    // the only phase that can be abandoned safely, and it is the
                    // only one that takes minutes.
                    try Task.checkCancellation()
                    guard let onVerifyProgress, total > 0 else { return }
                    onVerifyProgress(min(max(Double(hashed) / Double(total), 0), 1))
                })
        }
        return output.standardizedFileURL
    }

    public func removeInstalled(textModelDirectory: URL) async throws {
        try await mutate(textModelDirectory, using: runRemove)
    }

    /// Runs blocking work off the calling task, with cancellation forwarded.
    ///
    /// `Task.detached` inherits nothing, cancellation included, so a detached
    /// body that cooperatively checks `Task.isCancelled` is checking a flag
    /// nobody ever sets: cancelling the caller left the vision pack's
    /// multi-minute verification hash running to completion and activating the
    /// pack anyway, while the UI sat on "Cancelling" with every model action
    /// disabled. Extracted so the forwarding itself can be tested; inline, the
    /// only way to exercise it was a real 1.14 GB pack.
    func runCancellableDetached<T: Sendable>(
        priority: TaskPriority = .utility,
        _ body: @escaping @Sendable () throws -> T
    ) async throws -> T {
        let work = Task.detached(priority: priority, operation: body)
        return try await withTaskCancellationHandler {
            try await work.value
        } onCancel: {
            work.cancel()
        }
    }

    private func mutate(_ directory: URL, using runner: @escaping MutationRunner) async throws {
        try await Task.detached(priority: .utility) {
            try await runner(directory.standardizedFileURL)
        }.value
    }
}
