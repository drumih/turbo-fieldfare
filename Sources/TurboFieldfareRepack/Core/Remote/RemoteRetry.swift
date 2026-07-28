import Foundation

public struct RemoteRetryPolicy: Sendable {
    public var attempts: Int
    public var baseDelayNs: UInt64
    public var maxDelayNs: UInt64
    public var maxServerDelayNs: UInt64

    public init(attempts: Int = 4,
                baseDelayNs: UInt64 = 1_000_000_000,
                maxDelayNs: UInt64 = 16_000_000_000,
                maxServerDelayNs: UInt64 = 60 * 60 * 1_000_000_000) {
        self.attempts = attempts
        self.baseDelayNs = baseDelayNs
        self.maxDelayNs = maxDelayNs
        self.maxServerDelayNs = maxServerDelayNs
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
            return retryableStatus(status)
        }
        if case RepackError.remoteHTTPResponse(_, let status, _) = error {
            return retryableStatus(status)
        }
        if case RepackError.remoteBodyTruncated = error {
            return true
        }
        return false
    }

    public func retryDelayNs(error: Error,
                             localDelayNs: UInt64,
                             now: Date = Date()) throws -> UInt64 {
        let raw: String?
        if case RepackError.remoteHTTPResponse(_, _, let retryAfter) = error {
            raw = retryAfter
        } else {
            raw = nil
        }
        guard let raw else { return min(localDelayNs, maxServerDelayNs) }
        let server = try parseRetryAfterNs(raw, now: now)
        guard server <= maxServerDelayNs else {
            throw RepackError.remoteRetryDelayExceeded(
                value: raw,
                capSeconds: maxServerDelayNs / 1_000_000_000)
        }
        return max(min(localDelayNs, maxServerDelayNs), server)
    }
}

public typealias RemoteRetrySleeper = @Sendable (UInt64) async throws -> Void
public typealias RemoteRetryJitter = @Sendable () -> UInt64

public func withRemoteRetries<T>(_ policy: RemoteRetryPolicy,
                                 label: String,
                                 audit: RepackAudit?,
                                 now: @Sendable () -> Date = Date.init,
                                 jitter: @escaping RemoteRetryJitter = {
                                     UInt64.random(in: 0..<250_000_000)
                                 },
                                 sleeper: @escaping RemoteRetrySleeper = {
                                     try await Task.sleep(nanoseconds: $0)
                                 },
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
            let localDelay = delay == 0 ? 0 : saturatingAdd(delay, jitter())
            let retryDelay = try policy.retryDelayNs(
                error: error,
                localDelayNs: localDelay,
                now: now())
            audit?.recordRemoteRetry(
                label: label,
                attempt: attempt,
                detail: String(describing: error))
            if retryDelay > 0 {
                try await sleeper(retryDelay)
            }
            let doubled = delay.multipliedReportingOverflow(by: 2)
            delay = min(doubled.overflow ? UInt64.max : doubled.partialValue,
                        policy.maxDelayNs)
        }
    }
    throw lastError ?? RepackError.remoteProtocolInvalid(detail: "retry loop exited without an error")
}

private func retryableStatus(_ status: Int) -> Bool {
    status == 408 || status == 429 || status == 500 || status == 502
        || status == 503 || status == 504
}

private func parseRetryAfterNs(_ raw: String, now: Date) throws -> UInt64 {
    let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if let seconds = UInt64(value) {
        let (nanoseconds, overflow) = seconds.multipliedReportingOverflow(by: 1_000_000_000)
        return overflow ? UInt64.max : nanoseconds
    }
    let formats = [
        "EEE',' dd MMM yyyy HH':'mm':'ss zzz",
        "EEEE',' dd-MMM-yy HH':'mm':'ss zzz",
        "EEE MMM d HH':'mm':'ss yyyy",
    ]
    let date = formats.lazy.compactMap { format -> Date? in
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = format
        return formatter.date(from: value)
    }.first
    guard let date else {
        throw RepackError.remoteProtocolInvalid(detail: "invalid Retry-After \(value)")
    }
    let seconds = max(date.timeIntervalSince(now), 0)
    guard seconds < Double(UInt64.max / 1_000_000_000) else {
        return UInt64.max
    }
    return UInt64(seconds * 1_000_000_000)
}

private func saturatingAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
    let result = lhs.addingReportingOverflow(rhs)
    return result.overflow ? UInt64.max : result.partialValue
}
