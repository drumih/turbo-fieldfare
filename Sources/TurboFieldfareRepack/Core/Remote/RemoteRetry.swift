import Foundation

public struct RemoteRetryPolicy: Sendable {
    public var attempts: Int
    public var baseDelayNs: UInt64
    public var maxDelayNs: UInt64

    public init(attempts: Int = 4,
                baseDelayNs: UInt64 = 1_000_000_000,
                maxDelayNs: UInt64 = 16_000_000_000) {
        self.attempts = attempts
        self.baseDelayNs = baseDelayNs
        self.maxDelayNs = maxDelayNs
    }

    public static func isRetryable(_ error: Error) -> Bool {
        if let e = error as? URLError {
            switch e.code {
            case .timedOut, .networkConnectionLost, .cannotConnectToHost,
                 .dnsLookupFailed, .notConnectedToInternet, .resourceUnavailable:
                return true
            default:
                return false
            }
        }
        if case RepackError.remoteHTTPStatus(_, let status) = error {
            return status == 429 || (500...599).contains(status)
        }
        if case RepackError.remoteProtocolInvalid(let detail) = error,
           detail.contains("wrote") {
            return true
        }
        return false
    }
}

func withRemoteRetries<T>(_ policy: RemoteRetryPolicy,
                                 audit: RepackAudit?,
                                 op: () async throws -> T) async throws -> T {
    let attempts = max(policy.attempts, 1)
    var delay = policy.baseDelayNs
    var lastError: Error?
    for attempt in 1...attempts {
        try Task.checkCancellation()
        do {
            return try await op()
        } catch {
            lastError = error
            guard attempt < attempts,
                  RemoteRetryPolicy.isRetryable(error) else {
                throw error
            }
            audit?.recordRemoteRetry()
            if delay > 0 {
                let jitter = UInt64.random(in: 0..<250_000_000)
                try await Task.sleep(nanoseconds: delay &+ jitter)
            }
            delay = min(delay &* 2, policy.maxDelayNs)
        }
    }
    throw lastError ?? RepackError.remoteProtocolInvalid(detail: "retry loop exited without an error")
}
