import Foundation
import TurboFieldfareDecodeProtocol

final class DecodeServiceResponseRouter: @unchecked Sendable {
    private struct State {
        var pending: [UUID: [DecodeServiceEvent]] = [:]
        var terminalError: Error?
    }

    private let condition = NSCondition()
    private var state = State()

    init(output: FileHandle) {
        let reader = Thread { [weak self] in
            self?.readFrames(from: output)
        }
        reader.name = "TurboFieldfare.DecodeService.ResponseRouter"
        reader.qualityOfService = .userInitiated
        reader.start()
    }

    func next(matching requestID: UUID) async throws -> DecodeServiceEvent {
        try await Task.detached(priority: .userInitiated) { [self] in
            try waitForEvent(matching: requestID)
        }.value
    }

    private func waitForEvent(matching requestID: UUID) throws -> DecodeServiceEvent {
        condition.lock()
        defer { condition.unlock() }
        while state.pending[requestID]?.isEmpty != false,
              state.terminalError == nil {
            condition.wait()
        }
        if var events = state.pending[requestID], !events.isEmpty {
            let event = events.removeFirst()
            if events.isEmpty {
                state.pending.removeValue(forKey: requestID)
            } else {
                state.pending[requestID] = events
            }
            return event
        }
        throw state.terminalError ?? DecodeFrameError.unexpectedEOF
    }

    private func readFrames(from output: FileHandle) {
        do {
            while true {
                let event = try DecodeFrameCodec.read(
                    DecodeServiceEvent.self, from: output)
                condition.lock()
                state.pending[event.generationID, default: []].append(event)
                condition.broadcast()
                condition.unlock()
            }
        } catch {
            condition.lock()
            state.terminalError = error
            condition.broadcast()
            condition.unlock()
        }
    }
}
