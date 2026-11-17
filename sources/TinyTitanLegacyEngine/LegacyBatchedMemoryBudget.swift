import TinyTitan
import TinyTitanCore

package enum LegacyBatchedMemoryBudget {
    package static func perSlotBytes(
        config: ArchConfig,
        maxContext: Int,
        precision: KVCachePrecision,
        fp16RingEnabled: Bool,
        slidingWindow: Int,
        maxPrefillChunkTokens: Int,
        vocab: Int,
        hostKVWindowTokens: Int? = nil
    ) -> Int {
        let kv = KVCacheManager.worstCaseBytes(
            config: config, maxContext: maxContext, precision: precision,
            slots: 1, fp16RingEnabled: fp16RingEnabled,
            slidingWindow: slidingWindow,
            maxPrefillChunkTokens: maxPrefillChunkTokens,
            windowTokens: hostKVWindowTokens)
        let hostArena =
            hostKVWindowTokens.map { _ in
                KVCacheManager.worstCaseHostArenaBytes(
                    config: config, maxContext: maxContext,
                    precision: precision, slots: 1)
            } ?? 0
        let gdn = GDNStateManager.worstCaseBytes(config: config, slots: 1)
        let scratch = 2 * vocab * MemoryLayout<Float16>.size + MemoryLayout<UInt32>.size
        return kv + hostArena + gdn + scratch
    }
}
