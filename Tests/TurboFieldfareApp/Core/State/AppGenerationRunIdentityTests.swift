import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareAppCore

private actor RunIdentityGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
    }
}

private final class RunIdentityInferenceClient: AppModelLifecycleClient, Sendable {
    private let callCount = Mutex(0)
    let firstThrow = RunIdentityGate()
    let secondTerminal = RunIdentityGate()

    func ensureLoaded(modelDirectory: URL,
                      maxContextTokens: Int,
                      options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    func unload() async {}
    func resetConversation(epoch: UUID) async throws {}
    func cancel() {}

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        let call = callCount.withLock { count in
            defer { count += 1 }
            return count
        }
        return AsyncThrowingStream { continuation in
            let task = Task { [firstThrow, secondTerminal] in
                let message = call == 0 ? "first terminal" : "second terminal"
                if call == 0 {
                    continuation.yield(.failed(.unknown(message), partial: nil))
                    await firstThrow.wait()
                } else {
                    await secondTerminal.wait()
                    continuation.yield(.failed(.unknown(message), partial: nil))
                }
                continuation.finish(throwing: AppInferenceError.unknown("\(message) throw"))
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@Suite struct AppGenerationRunIdentityTests {
    @MainActor
    @Test func staleStreamFailureCannotTerminateTheNextRun() async throws {
        let directory = FileManager.default.temporaryDirectory
        let client = RunIdentityInferenceClient()
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)

        model.promptText = "first"
        model.run()
        while model.isRunning { await Task.yield() }

        model.promptText = "second"
        model.run()
        let secondIdentity = model.runIdentity
        #expect(model.isRunning)
        #expect(model.conversation.hasTurnInFlight)
        #expect(model.outputPromptText == "second")

        await client.firstThrow.open()
        try await Task.sleep(for: .milliseconds(20))

        #expect(model.runIdentity == secondIdentity)
        #expect(model.isRunning, "the first stream's catch terminated the second run")
        #expect(model.conversation.hasTurnInFlight)
        #expect(model.outputPromptText == "second")
        #expect(model.error == nil)

        await client.secondTerminal.open()
        while model.isRunning { await Task.yield() }
    }
}
