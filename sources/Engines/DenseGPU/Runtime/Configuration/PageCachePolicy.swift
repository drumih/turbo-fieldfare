import Foundation

/// Whether routed-expert reads go through the OS page cache.
///
/// The bounded reader (`F_NOCACHE`) is the product default on a machine that
/// cannot hold the install: the slot budget is then the process's real
/// footprint, which is what makes "a 35B model on 24 GB" true. On a machine
/// whose RAM comfortably exceeds the install, the page cache is the faster
/// path (measured +4.7% at 8-bit on the 24 GB host, and slower there at 4-bit),
/// because a repeat read becomes a memory read instead of an SSD read.
public enum PageCachePolicy: String, Sendable, Equatable {
    /// Bypass the page cache (`F_NOCACHE`). The declared slot budget is the
    /// only expert cache.
    case bounded
    /// Allow the OS to cache expert reads. Faster when the install fits, but
    /// it adds an undeclared cache the process does not account for.
    case pageCache = "page-cache"

    public var bypassesCache: Bool { self == .bounded }
}

/// The decision and the inputs it was made from, for the startup banner and
/// for tests that must distinguish "bounded because small" from "bounded
/// because the size could not be read".
public struct PageCacheDecision: Sendable, Equatable {
    public let policy: PageCachePolicy
    public let reason: String
    public let physicalMemoryBytes: UInt64
    public let installBytes: UInt64?

    public init(
        policy: PageCachePolicy,
        reason: String,
        physicalMemoryBytes: UInt64,
        installBytes: UInt64?
    ) {
        self.policy = policy
        self.reason = reason
        self.physicalMemoryBytes = physicalMemoryBytes
        self.installBytes = installBytes
    }
}

/// Machine-size-aware page-cache selection (T25).
///
/// Precedence is explicit override, then the automatic rule:
///
/// 1. `TINYTITAN_BOUNDED_IO=0` forces the page cache, `=1` forces the bounded
///    reader. Any other value is refused as bounded rather than read as
///    "enable" -- a typo must not create the undeclared cache.
/// 2. With no override, the page cache is enabled only when physical memory is
///    at least twice the install size. The factor of two is the task's
///    "comfortably exceeds" rule: the install is not the only resident claim
///    (weights, KV, the expert slot cache and the process all live beside it).
/// 3. If physical memory or the install size cannot be read, the answer is
///    bounded. The footprint guarantee fails closed; it is never guessed into
///    an unbounded cache.
///
/// On the 24 GB development host every installed model is larger than half of
/// RAM (19/34/96 GB against a 12 GB threshold), so the automatic rule is a
/// no-op there and the shipped behaviour is unchanged.
public enum PageCachePolicyResolver {
    public static let overrideVariable = "TINYTITAN_BOUNDED_IO"

    /// Total install bytes from the manifest's file sizes.
    ///
    /// An overflowing sum is reported as `UInt64.max`, which the resolver then
    /// refuses as a 2x overflow rather than wrapping into a small install that
    /// would wrongly enable the page cache.
    public static func installSize(fileSizes: some Sequence<UInt64>) -> UInt64 {
        fileSizes.reduce(UInt64(0)) { total, size in
            let (sum, overflow) = total.addingReportingOverflow(size)
            return overflow ? UInt64.max : sum
        }
    }

    public static func resolve(
        physicalMemory: UInt64,
        installBytes: UInt64?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> PageCacheDecision {
        func decision(_ policy: PageCachePolicy, _ reason: String)
            -> PageCacheDecision
        {
            PageCacheDecision(
                policy: policy,
                reason: reason,
                physicalMemoryBytes: physicalMemory,
                installBytes: installBytes)
        }

        if let raw = environment[overrideVariable] {
            switch raw {
            case "0":
                return decision(
                    .pageCache, "\(overrideVariable)=0 (explicit)")
            case "1":
                return decision(
                    .bounded, "\(overrideVariable)=1 (explicit)")
            default:
                return decision(
                    .bounded,
                    "\(overrideVariable)=\(raw) is not 0 or 1; "
                        + "refusing to enable an undeclared cache")
            }
        }
        guard physicalMemory > 0 else {
            return decision(.bounded, "physical memory is unavailable")
        }
        guard let installBytes, installBytes > 0 else {
            return decision(.bounded, "install size is unavailable")
        }
        let (twice, overflow) = installBytes.multipliedReportingOverflow(by: 2)
        guard !overflow else {
            return decision(.bounded, "install size overflows the 2x rule")
        }
        guard physicalMemory >= twice else {
            return decision(
                .bounded,
                "install is \(installBytes) bytes; \(physicalMemory) bytes "
                    + "of RAM is below the 2x threshold")
        }
        return decision(
            .pageCache,
            "install is \(installBytes) bytes and RAM is at least 2x it")
    }
}
