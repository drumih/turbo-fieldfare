import Foundation

/// Single opt-in gate for suites that allocate real decode shapes. Off by
/// default so an ordinary `swift test` stays inside the 8 GB budget; the suites
/// behind it are run one filter at a time with `--no-parallel`.
enum HeavyTestGate {
    static let enabled: Bool = ProcessInfo.processInfo.environment["GTURBO_HEAVY"] == "1"
}
