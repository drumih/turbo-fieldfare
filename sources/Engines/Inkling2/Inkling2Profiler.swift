import Foundation

/// Opt-in stage timings for the CPU/GPU hybrid path.
public enum Inkling2Profiler {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var totals: [String: Double] = [:]
    nonisolated(unsafe) private static var counts: [String: Int] = [:]
    public static let enabled =
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING_PROFILE"] != nil

    public static func record(_ stage: String, _ seconds: Double) {
        guard enabled else { return }
        lock.withLock {
            totals[stage, default: 0] += seconds
            counts[stage, default: 0] += 1
        }
    }

    public static func report() -> String {
        lock.withLock {
            totals.keys.sorted().map { key in
                let count = counts[key] ?? 0
                let seconds = totals[key] ?? 0
                return "\(key): \(String(format: "%.3f", seconds))s / \(count)"
            }.joined(separator: "\n")
        }
    }
}
