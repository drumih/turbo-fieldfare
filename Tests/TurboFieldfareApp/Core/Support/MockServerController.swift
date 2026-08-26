import Foundation
@testable import TurboFieldfareAppCore

final class MockServerController: AppServerController, @unchecked Sendable {
    private let lock = NSLock()
    private var _startCalls: [[String]] = []
    private var _stopCallCount = 0
    private var latestHandler: (@Sendable (AppServerState) -> Void)?

    var startCalls: [[String]] {
        lock.lock()
        defer { lock.unlock() }
        return _startCalls
    }

    var stopCallCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _stopCallCount
    }

    func start(arguments: [String],
              onStateChange: @escaping @Sendable (AppServerState) -> Void) {
        lock.lock()
        _startCalls.append(arguments)
        latestHandler = onStateChange
        lock.unlock()
    }

    func stop() {
        lock.lock()
        _stopCallCount += 1
        lock.unlock()
    }

    /// Simulates the controller reporting a new state, the way the real
    /// `ProcessServerController` does from its stdout/termination handlers.
    func emit(_ state: AppServerState) {
        let handler: (@Sendable (AppServerState) -> Void)?
        lock.lock()
        handler = latestHandler
        lock.unlock()
        handler?(state)
    }
}
