import Foundation
import Synchronization
import TurboFieldfareRepackCore
@testable import TurboFieldfareAppCore

final class MockVisionPackInstallerClient: AppVisionPackInstallerClient, Sendable {
    let descriptor: AppModelInstallDescriptor
    let requirement: AppModelInstallRequirement
    let events: [AppModelInstallEvent]
    let holdOpen: Bool
    let preparedValid: Bool
    let activationError: RepackError?
    /// Fractions the activation reports before it finishes, with a pause after
    /// each so a cancel can land mid-verification.
    let activationProgress: [Double]
    private let task = Mutex<Task<Void, Never>?>(nil)
    private let activations = Mutex(0)

    var activationCount: Int { activations.withLock { $0 } }

    init(
        descriptor: AppModelInstallDescriptor = .visionCompanion,
        requirement: AppModelInstallRequirement = AppModelInstallRequirement(
            probePath: "/",
            requiredBytes: 1,
            availableBytes: UInt64.max),
        events: [AppModelInstallEvent] = [],
        holdOpen: Bool = false,
        preparedValid: Bool = false,
        activationError: RepackError? = nil,
        activationProgress: [Double] = []
    ) {
        self.descriptor = descriptor
        self.requirement = requirement
        self.events = events
        self.holdOpen = holdOpen
        self.preparedValid = preparedValid
        self.activationError = activationError
        self.activationProgress = activationProgress
    }

    func checkInstallRequirement(
        textModelDirectory: URL
    ) throws -> AppModelInstallRequirement {
        requirement
    }

    func install(
        textModelDirectory: URL
    ) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [events, holdOpen] in
                do {
                    for event in events {
                        try Task.checkCancellation()
                        continuation.yield(event)
                        await Task.yield()
                    }
                    if holdOpen {
                        try await Task.sleep(for: .seconds(60))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            self.task.withLock { $0 = task }
            continuation.onTermination = { [weak self] _ in
                self?.task.withLock { active in
                    active?.cancel()
                    active = nil
                }
            }
        }
    }

    func discardPartialInstall(textModelDirectory: URL) async throws {}
    func removeInstalled(textModelDirectory: URL) async throws {}
    func preparedInstallIsValid(textModelDirectory: URL) -> Bool { preparedValid }
    func activatePreparedInstall(
        textModelDirectory: URL,
        onVerifyProgress: (@Sendable (Double) -> Void)?
    ) async throws -> URL {
        activations.withLock { $0 += 1 }
        for fraction in activationProgress {
            try Task.checkCancellation()
            onVerifyProgress?(fraction)
            try await Task.sleep(for: .milliseconds(30))
        }
        try Task.checkCancellation()
        if let activationError { throw activationError }
        return textModelDirectory
    }

    func cancel() {
        task.withLock { $0?.cancel() }
    }
}
