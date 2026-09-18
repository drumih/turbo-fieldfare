import Foundation

/// The one host-memory admission rule for a requested context length.
///
/// Every surface that can pick a context — the app menu, the CLI, the server,
/// the decode service — asks this type whether the host can back it, so a
/// refusal, a hidden menu row and a warning all quote the same arithmetic. The
/// numbers are FP16 KV geometry plus two measured constants; nothing here
/// allocates or touches the model.
public enum ContextAdmission {

    /// Everything resident that does not scale with context: the LM weights,
    /// the expert-cache slots, prefill scratch, the tokenizer and the Metal
    /// library. Measured at 1.75-1.94 GB across PR 156's five rungs and the
    /// M2-8 8K row; rounded up to 2 GiB so the rule never under-reserves.
    public static let nonKVRuntimeFootprintBytes: UInt64 = 2 << 30

    /// Headroom left to the OS and to a minimum routed-expert page cache. A
    /// host that satisfied only the projected footprint would run with every
    /// expert read faulting against a cold cache.
    public static let hostReserveBytes: UInt64 = 3 << 30

    public enum Availability: Equatable, Sendable {
        case available
        case needsMemory(minimumHostBytes: UInt64)
    }

    /// Rows the production runtime allocates per sliding-window layer.
    ///
    /// `RealForwardRunner` floors its `maxPrefillChunkTokens` at the pooled
    /// image-token count, and `KVCacheManager` sizes the ring at
    /// `slidingWindow + maxPrefillChunkTokens`, so the widest chunk the runtime
    /// may see — not the default text chunk — is what the ring costs. 1,304
    /// today.
    public static func productionSlidingRingRows(config: ArchConfig, maxContext: Int) -> Int {
        let chunkRows = max(PrefillRuntimeConfig.defaultChunked.chunkTokens,
                            VisionConfig().maximumPooledTokens)
        return min(maxContext, config.slidingWindow + chunkRows)
    }

    /// FP16 K and V bytes the runtime allocates for `maxContext`.
    ///
    /// Sliding-window layers are ring-capped; only the full-attention layers
    /// grow with the context, at 20,480 B per token across the five of them.
    public static func fp16KVBytes(config: ArchConfig, maxContext: Int) -> UInt64 {
        let fullLayers = config.fullAttentionLayerMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        let slidingLayers = config.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = productionSlidingRingRows(config: config, maxContext: maxContext)
        let slidingBytesPerRow = config.numKVHeads * config.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = config.numFullKVHeads * config.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * maxContext * fullBytesPerRow)
    }

    /// What the process is expected to occupy at this context.
    public static func projectedFootprintBytes(config: ArchConfig, maxContext: Int) -> UInt64 {
        fp16KVBytes(config: config, maxContext: maxContext) + nonKVRuntimeFootprintBytes
    }

    /// The smallest host memory that admits this context.
    public static func minimumHostMemoryBytes(config: ArchConfig, maxContext: Int) -> UInt64 {
        projectedFootprintBytes(config: config, maxContext: maxContext) + hostReserveBytes
    }

    public static func availability(config: ArchConfig,
                                    maxContext: Int,
                                    hostMemoryBytes: UInt64) -> Availability {
        let minimum = minimumHostMemoryBytes(config: config, maxContext: maxContext)
        return hostMemoryBytes >= minimum ? .available : .needsMemory(minimumHostBytes: minimum)
    }

    public static var hostMemoryBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// One sentence naming what the context needs and what this Mac has, for a
    /// refusal, a hidden menu row's caption or a CLI warning.
    public static func needDescription(config: ArchConfig,
                                       maxContext: Int,
                                       hostMemoryBytes: UInt64) -> String {
        let need = marketingGigabytesRoundedUp(
            minimumHostMemoryBytes(config: config, maxContext: maxContext))
        let have = marketingGigabytesNearest(hostMemoryBytes)
        return "A \(grouped(maxContext))-token context needs \(need) GB of memory; this Mac has \(have) GB."
    }

    /// Memory sizes Apple actually ships, so the sentence names a machine the
    /// reader can recognize instead of "10.25 GiB".
    private static let machineSizesGB = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512]

    /// The requirement is rounded in GiB because that is the unit
    /// `physicalMemory` reports: an "8 GB Mac" answers 8,589,934,592, so a need
    /// of 8,320,122,880 is met by it and must print as 8 GB, not 16.
    private static func marketingGigabytesRoundedUp(_ bytes: UInt64) -> Int {
        if let size = machineSizesGB.first(where: { UInt64($0) << 30 >= bytes }) { return size }
        return Int((bytes + (1 << 30) - 1) >> 30)
    }

    /// The host is reported in decimal GB, which is how the same machine is
    /// labeled in About This Mac.
    private static func marketingGigabytesNearest(_ bytes: UInt64) -> Int {
        let decimal = Double(bytes) / 1e9
        guard let nearest = machineSizesGB.min(by: {
            abs(Double($0) - decimal) < abs(Double($1) - decimal)
        }) else { return Int(decimal.rounded()) }
        return nearest
    }

    /// Grouped by threes without a locale, so the message reads the same in
    /// every region and in test assertions.
    private static func grouped(_ value: Int) -> String {
        let digits = Array(String(value))
        var out = ""
        for (index, digit) in digits.enumerated() {
            if index > 0, (digits.count - index) % 3 == 0 { out.append(",") }
            out.append(digit)
        }
        return out
    }
}
