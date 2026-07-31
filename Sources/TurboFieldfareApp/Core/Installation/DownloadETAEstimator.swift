import Foundation

public enum DownloadETAPresentation: Equatable, Sendable {
    case hidden
    case estimating
    case remaining(seconds: Double)
}

public struct DownloadETAObservation: Equatable, Sendable {
    public let reusedBytes: UInt64
    public let downloadedThisRunBytes: UInt64
    public let totalBytes: UInt64

    public init(reusedBytes: UInt64,
                downloadedThisRunBytes: UInt64,
                totalBytes: UInt64) {
        self.reusedBytes = reusedBytes
        self.downloadedThisRunBytes = downloadedThisRunBytes
        self.totalBytes = totalBytes
    }
}

public struct DownloadETAEstimator: Sendable {
    private static let warmupSeconds = 10.0
    private static let warmupBytes: UInt64 = 32 * 1024 * 1024
    private static let updateSeconds = 30.0

    private var startTime: Double?
    private var startBytes: UInt64 = 0
    private var previousBytes: UInt64?
    private var lastUpdateTime: Double?
    private var lastETA: Double?

    public init() {}

    public mutating func reset() {
        self = Self()
    }

    public mutating func update(
        _ observation: DownloadETAObservation,
        timestamp: Double
    ) -> DownloadETAPresentation {
        guard timestamp.isFinite else { return .hidden }
        let downloaded = observation.downloadedThisRunBytes
        if let previousBytes, downloaded < previousBytes {
            self = Self()
        }
        previousBytes = downloaded

        if startTime == nil {
            startTime = timestamp
            startBytes = downloaded
            return .estimating
        }
        let useful = downloaded - min(downloaded, startBytes)
        let elapsed = timestamp - (startTime ?? timestamp)
        guard elapsed >= Self.warmupSeconds,
              useful >= Self.warmupBytes else {
            return .estimating
        }

        if let lastUpdateTime,
           timestamp - lastUpdateTime < Self.updateSeconds,
           let lastETA {
            return .remaining(seconds: lastETA)
        }
        let rate = Double(useful) / elapsed
        let completed = min(
            observation.totalBytes,
            observation.reusedBytes.addingReportingOverflow(downloaded).overflow
                ? UInt64.max
                : observation.reusedBytes + downloaded)
        guard completed < observation.totalBytes, rate > 0 else {
            return .hidden
        }
        let estimate = Double(observation.totalBytes - completed) / rate
        guard estimate.isFinite, estimate >= 0 else { return .estimating }

        lastETA = estimate
        lastUpdateTime = timestamp
        return .remaining(seconds: estimate)
    }
}

public enum DownloadETAFormatter {
    public static func string(for presentation: DownloadETAPresentation) -> String? {
        switch presentation {
        case .hidden:
            return nil
        case .estimating:
            return "Estimating time remaining…"
        case .remaining(let seconds):
            return remainingString(seconds)
        }
    }

    public static func remainingString(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else {
            return "Estimating time remaining…"
        }
        if seconds >= 24 * 60 * 60 {
            return "More than a day remaining"
        }
        if seconds < 60 {
            return "Less than a minute remaining"
        }
        if seconds < 60 * 60 {
            return "About \(Int((seconds / 60).rounded())) min remaining"
        }
        let hours = Int((seconds / 3600).rounded())
        return "About \(hours) hr remaining"
    }
}
