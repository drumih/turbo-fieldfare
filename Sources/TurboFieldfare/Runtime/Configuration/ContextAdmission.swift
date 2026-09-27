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
    /// M2-8 8K row at 16 cache slots; extra slots are accounted for separately.
    public static let nonKVRuntimeFootprintBytes: UInt64 = 2 << 30

    // The pinned IT pack uses 3,358,720 bytes per routed expert. Each layer
    // owns its slots; the streamer rounds each allocation to the host page size.
    private static let expertStrideBytes = 3_358_720

    private static func additionalExpertCacheBytes(config: ArchConfig, slots: Int) -> UInt64 {
        let page = Int(getpagesize())
        let slotBytes = ((expertStrideBytes + page - 1) / page) * page
        // Do not discount the measured baseline for smaller caches: the rest
        // of that allowance has not been measured separately.
        return UInt64(max(0, slots - 16)) * UInt64(config.numLayers) * UInt64(slotBytes)
    }

    /// KV bytes the retained prompt-cache lineages add on top of the single
    /// lineage `fp16KVBytes` already accounts for.
    ///
    /// `--prompt-cache-slots N` configures `N + 1` lineages: N that retain a
    /// conversation's prefix, plus one that serves requests sending
    /// `prompt_cache_mode: "off"`, so an opted-out request cannot evict a
    /// conversation while explicitly asking not to participate. Slot 0 is the
    /// lineage `fp16KVBytes` already charges, so only slots past the first are
    /// added here — at the default of 1 this is zero and admission is exactly
    /// the arithmetic it was before retention existed.
    ///
    /// The opt-out lineage is deliberately NOT charged, and that is the one
    /// judgement in this file worth disagreeing with. It is allocated on first
    /// use like every lineage past slot 0, and only a client that sends the
    /// `prompt_cache_mode` field can open it, so charging it would make every
    /// operator pay for a lineage most deployments never allocate. The
    /// omission is real rather than negligible at the top of the range — one
    /// lineage at 262,144 is 5.25 GiB, well past `hostReserveBytes` — so it is
    /// named here instead of buried.
    private static func additionalPromptCacheBytes(config: ArchConfig,
                                                   maxContext: Int,
                                                   slots: Int) -> UInt64 {
        UInt64(max(0, slots - 1)) * fp16KVBytes(config: config, maxContext: maxContext)
    }

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
    public static func projectedFootprintBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16,
                                               promptCacheSlots: Int = 1) -> UInt64 {
        fp16KVBytes(config: config, maxContext: maxContext) + nonKVRuntimeFootprintBytes
            + additionalExpertCacheBytes(config: config, slots: expertCacheSlots)
            + additionalPromptCacheBytes(config: config, maxContext: maxContext, slots: promptCacheSlots)
    }

    /// The smallest host memory that admits this context.
    public static func minimumHostMemoryBytes(config: ArchConfig, maxContext: Int, expertCacheSlots: Int = 16,
                                              promptCacheSlots: Int = 1) -> UInt64 {
        projectedFootprintBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                promptCacheSlots: promptCacheSlots) + hostReserveBytes
    }

    public static func availability(config: ArchConfig,
                                    maxContext: Int,
                                    hostMemoryBytes: UInt64, expertCacheSlots: Int = 16,
                                    promptCacheSlots: Int = 1) -> Availability {
        let minimum = minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                             promptCacheSlots: promptCacheSlots)
        return hostMemoryBytes >= minimum ? .available : .needsMemory(minimumHostBytes: minimum)
    }

    public static var hostMemoryBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// One sentence naming what the context needs and what this Mac has, for a
    /// refusal, a hidden menu row's caption or a CLI warning.
    public static func needDescription(config: ArchConfig,
                                       maxContext: Int,
                                       hostMemoryBytes: UInt64, expertCacheSlots: Int = 16,
                                       promptCacheSlots: Int = 1) -> String {
        let need = marketingGigabytesRoundedUp(
            minimumHostMemoryBytes(config: config, maxContext: maxContext, expertCacheSlots: expertCacheSlots,
                                   promptCacheSlots: promptCacheSlots))
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
