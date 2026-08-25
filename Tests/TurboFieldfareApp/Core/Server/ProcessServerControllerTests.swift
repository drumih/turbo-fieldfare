import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct ProcessServerControllerTests {
    @Test func anOutOfRangePortReportsAFailedStateQuickly() async throws {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareServer")
        let controller = ProcessServerController(executableURL: executable)
        let states = StateCollector()

        controller.start(arguments: [
            "--model", "/tmp/turbofieldfare-arguments-only-\(UUID().uuidString).gturbo",
            "--port", "0",
        ]) { state in
            states.append(state)
        }

        let deadline = Date().addingTimeInterval(10)
        while !states.containsFailure(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }

        #expect(states.containsFailure(), "expected a failed state before the deadline")
        #expect(states.failureMessages().contains {
            $0.contains("--port must be between 1 and 65535")
        })
    }

    @Test func aMissingExecutableReportsAFailedStateImmediately() {
        let executable = URL(fileURLWithPath: "/tmp/turbofieldfare-no-such-binary")
        let controller = ProcessServerController(executableURL: executable)
        let states = StateCollector()

        controller.start(arguments: ["--model", "/tmp/anything"]) { state in
            states.append(state)
        }

        #expect(states.containsFailure())
    }
}

private final class StateCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppServerState] = []

    func append(_ state: AppServerState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func containsFailure() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return states.contains { if case .failed = $0 { return true }; return false }
    }

    func failureMessages() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return states.compactMap { state in
            if case .failed(let message) = state { return message }
            return nil
        }
    }
}
