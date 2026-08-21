import Foundation
import Synchronization

enum ServerLog {
    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func progress(id: String,
                         progress: ServerGenerationProgress,
                         duration: Duration) {
        switch progress {
        case .prefill(let done, let total):
            write("request \(id) generating phase=prefill "
                + "progress=\(done)/\(total) elapsed=\(format(duration))")
        case .decode(let completionTokens):
            write("request \(id) generating phase=decode "
                + "completion=\(completionTokens) elapsed=\(format(duration))")
        }
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        write("request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)")
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

final class ServerProgressLogLimiter: Sendable {
    static let prefillTokenInterval = 1_024
    static let decodeTokenInterval = 16

    private struct State {
        var lastPrefillDone = 0
        var lastDecodeTokens = 0
    }

    private let state = Mutex(State())

    func shouldLog(_ progress: ServerGenerationProgress) -> Bool {
        state.withLock { state in
            switch progress {
            case .prefill(let done, let total):
                guard done > state.lastPrefillDone else { return false }
                let crossedInterval = done / Self.prefillTokenInterval
                    > state.lastPrefillDone / Self.prefillTokenInterval
                let finished = done >= total && state.lastPrefillDone < total
                guard crossedInterval || finished else { return false }
                state.lastPrefillDone = done
                return true
            case .decode(let completionTokens):
                guard completionTokens > state.lastDecodeTokens else { return false }
                let firstToken = state.lastDecodeTokens == 0
                let crossedInterval = completionTokens / Self.decodeTokenInterval
                    > state.lastDecodeTokens / Self.decodeTokenInterval
                guard firstToken || crossedInterval else { return false }
                state.lastDecodeTokens = completionTokens
                return true
            }
        }
    }
}
