import Foundation
import Synchronization
@testable import TurboFieldfareAppCore

final class MockModelInstallerClient: AppModelInstallerClient, Sendable {
    let events: [AppModelInstallEvent]
    let failure: Error?
    let holdOpen: Bool
    let requirement: AppModelInstallRequirement
    let descriptor: AppModelInstallDescriptor
    private struct State {
        var task: Task<Void, Never>?
        var cancelCalled = false
    }
    private final class TaskState: Sendable {
        let value = Mutex(State())
    }
    private let taskState = TaskState()

    var cancelCalled: Bool { taskState.value.withLock { $0.cancelCalled } }

    init(events: [AppModelInstallEvent] = [],
         failure: Error? = nil,
         requirement: AppModelInstallRequirement = AppModelInstallRequirement(
            requiredBytes: 1,
            availableBytes: UInt64.max),
         descriptor: AppModelInstallDescriptor = .default,
         holdOpen: Bool = false) {
        self.events = events
        self.failure = failure
        self.requirement = requirement
        self.descriptor = descriptor
        self.holdOpen = holdOpen
    }

    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        requirement
    }

    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let events = self.events
            let failure = self.failure
            let holdOpen = self.holdOpen
            let task = Task {
                do {
                    for event in events {
                        try Task.checkCancellation()
                        continuation.yield(event)
                        await Task.yield()
                    }
                    if holdOpen {
                        try await Task.sleep(for: .seconds(60))
                    }
                    if let failure {
                        continuation.finish(throwing: failure)
                    } else {
                        continuation.finish()
                    }
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            taskState.value.withLock { $0.task = task }
            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { state -> Task<Void, Never>? in
                    defer { state.task = nil }
                    return state.task
                }
                task?.cancel()
            }
        }
    }

    func cancel() {
        let task = taskState.value.withLock { state -> Task<Void, Never>? in
            state.cancelCalled = true
            return state.task
        }
        task?.cancel()
    }
}
