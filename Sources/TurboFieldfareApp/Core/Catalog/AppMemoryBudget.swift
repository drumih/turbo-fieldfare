import Foundation

/// How much RAM the app is willing to hold in resident weights plus KV cache.
///
/// The cap is not just about fitting: resident state competes with the page
/// cache that keeps streamed experts warm, so spending the machine's whole
/// budget on KV makes decode slower even when it fits.
public enum AppMemoryBudget {
    public static let absoluteCapBytes: UInt64 = 4 * 1_073_741_824
    public static let floorBytes: UInt64 = UInt64(1.5 * 1_073_741_824)

    /// Attention, embedding, and shared-expert weights stay resident for the
    /// whole session while routed experts stream, so this is the fixed part of
    /// the budget a context length has to share with.
    public static let residentWeightEstimateBytes: UInt64 = UInt64(1.6 * 1_073_741_824)

    public static func ceilingBytes(installedRAMBytes: UInt64) -> UInt64 {
        let proportional = installedRAMBytes / 4
        return min(absoluteCapBytes, max(floorBytes, proportional))
    }

    public static var installedRAMBytes: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }
}
