import Foundation

/// Model-free runtime rung constants shared by CLI/server argument parsers,
/// help text, and engine load planners (TT-134).
public enum RuntimeLimits {
    public static let supportedContextTokens = [
        4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
    ]
    public static let nativeMaximumContextTokens = 262_144
    public static let supportedYaRNContextTokens = (9...14).map { 1 << ($0 + 10) }
    public static let defaultYaRNContextTokens = 1_048_576
    public static let maximumContextTokens = 16_777_216
    public static let allowedExpertCacheSlots = [
        8, 16, 24, 32, 40, 48, 64, 96, 112, 128, 160, 192, 256,
    ]
    public static let minimumExpertCacheSlots = 8
    public static let minimumProcessTargetBytes = 4 << 30
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedProductionChunkTokens
}
