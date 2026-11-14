import Foundation
import TurboFieldfare

struct LoadedRequestTiming: Sendable, Equatable {
    let requestNanos: UInt64
}

/// Runs finish exactly once, on every operation outcome, before returning or
/// throwing. The loaded-request clock includes the runtime's common drain.
/// The outer finish is normally empty after the runtime has drained its readers.
func withFinishedRequest<Value>(
    operation: () async throws -> Value,
    finish: () throws -> Void
) async throws -> (value: Value, timing: LoadedRequestTiming) {
    let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let result: Result<Value, any Error>
    do { result = .success(try await operation()) }
    catch { result = .failure(error) }
    do {
        try finish()
    } catch {
        if case .failure(let primary) = result {
            throw RequestCompletionFailure(primary: primary, drain: error)
        }
        throw error
    }
    let ended = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    return (try result.get(), LoadedRequestTiming(
        requestNanos: ended - started))
}

func loadedRequestFooter(_ timing: LoadedRequestTiming) -> String {
    "[request=\(String(format: "%.6f", Double(timing.requestNanos) / 1e9))s]\n"
}
