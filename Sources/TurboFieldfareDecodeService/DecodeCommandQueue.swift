import Foundation
import TurboFieldfareDecodeProtocol

final class DecodeCommandQueue: @unchecked Sendable {
    private let condition = NSCondition()
    private var commands: [DecodeServiceCommand] = []
    private var closed = false

    func append(_ command: DecodeServiceCommand) {
        condition.lock()
        commands.append(command)
        condition.signal()
        condition.unlock()
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
    }

    func next() -> DecodeServiceCommand? {
        condition.lock()
        defer { condition.unlock() }
        while commands.isEmpty && !closed { condition.wait() }
        guard !commands.isEmpty else { return nil }
        return commands.removeFirst()
    }
}
