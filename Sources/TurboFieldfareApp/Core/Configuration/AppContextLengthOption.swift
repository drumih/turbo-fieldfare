import TurboFieldfare

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    public var fp16KVBytes: UInt64 {
        let architecture = ArchConfig.gemma4_26B_A4B
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        let slidingLayers = architecture.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        // The ring the runtime actually allocates: the window plus the widest
        // prefill chunk it may see, which is the pooled image-token count, not
        // the default text chunk. Using the smaller number understates every
        // estimate by about 29.69 MiB.
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + max(
                PrefillRuntimeConfig.defaultChunked.chunkTokens,
                VisionConfig().maximumPooledTokens))
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var menuLabel: String {
        switch self {
        // Measured against the 8K default, not against 4K: moving the default
        // without moving the baseline would have left every delta describing a
        // size the user is no longer starting from.
        case .fourK: "4K, -85 MB"
        case .eightK: "8K, Default"
        case .sixteenK: "16K, +170 MB"
        case .thirtyTwoK: "32K, +500 MB"
        case .sixtyFourK: "64K, +1.17 GB"
        }
    }
}
