import Foundation
import TurboFieldfare

/// One selectable context length with its measured cost.
public struct AppContextOption: Equatable, Sendable, Identifiable {
    public var id: Int { tokens }

    public let tokens: Int
    public let kvBytes: UInt64
    public let isEnabled: Bool
    public let disabledReason: String?

    public var shortLabel: String { "\(tokens / 1_024)K" }

    public var label: String {
        let megabytes = Double(kvBytes) / (1_024 * 1_024)
        let cost = megabytes >= 1_024
            ? String(format: "%.2f GB", megabytes / 1_024)
            : String(format: "%.0f MB", megabytes)
        return "\(shortLabel), \(cost)"
    }
}

public enum AppContextLengthOption {
    /// Candidate lengths offered to the user. Entries beyond 64K exist for
    /// machines whose budget can hold them; they render disabled elsewhere.
    public static let candidateTokens = [4_096, 8_192, 16_384, 32_768,
                                         65_536, 131_072, 262_144]

    public static let defaultTokens = 4_096

    public static func availableOptions(architecture: ArchConfig,
                                        residentWeightBytes: UInt64,
                                        installedRAMBytes: UInt64) -> [AppContextOption] {
        let ceiling = AppMemoryBudget.ceilingBytes(installedRAMBytes: installedRAMBytes)
        return candidateTokens.map { tokens in
            let kvBytes = fp16KVBytes(architecture: architecture, tokens: tokens)
            let total = residentWeightBytes + kvBytes
            let fits = total <= ceiling
            return AppContextOption(
                tokens: tokens,
                kvBytes: kvBytes,
                isEnabled: fits,
                disabledReason: fits
                    ? nil
                    : "Needs \(formatted(total)) of \(formatted(ceiling)) available.")
        }
    }

    /// Sliding-window layers stop growing once the window plus one prefill
    /// chunk is covered; full-attention layers grow with the whole context.
    public static func fp16KVBytes(architecture: ArchConfig, tokens: Int) -> UInt64 {
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        let slidingLayers = architecture.numLayers - fullLayers
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    private static func formatted(_ bytes: UInt64) -> String {
        let gigabytes = Double(bytes) / (1_024 * 1_024 * 1_024)
        return String(format: "%.1f GB", gigabytes)
    }
}
