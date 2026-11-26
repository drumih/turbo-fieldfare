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
    public static let maximumConcurrentSequences = 256
    public static let defaultMTPMemoryMiB = 384
    public static let allowedMTPMemoryMiB = 256...512
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedProductionChunkTokens

    /// Native context limit declared by an install's manifest, or
    /// `nativeMaximumContextTokens` when the manifest omits a larger window.
    package static func nativeContextCeiling(forModelDirectory path: String?) -> Int {
        guard let path, !path.isEmpty else { return nativeMaximumContextTokens }
        guard
            let manifest = try? SSDAIManifestReader.read(
                directoryURL: URL(fileURLWithPath: path)),
            let declared = manifest.arch.variantGeometry?.scalars["contextLength"],
            declared.isFinite, declared > Double(nativeMaximumContextTokens)
        else {
            return nativeMaximumContextTokens
        }
        return Int(declared)
    }

    /// Context rungs supported without YaRN up to `ceiling`.
    package static func supportedNativeContextTokens(ceiling: Int) -> [Int] {
        supportedContextTokens + supportedYaRNContextTokens.filter { $0 <= ceiling }
    }

    public static func parseBudgetBytes(_ text: String) -> Int? {
        let raw = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard !raw.isEmpty else { return nil }
        let multipliers: [(String, Int)] = [
            ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
            ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
            ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10),
        ]
        for (suffix, scale) in multipliers where raw.hasSuffix(suffix) {
            let number = String(raw.dropLast(suffix.count))
                .trimmingCharacters(in: .whitespaces)
            guard let value = Double(number), value > 0 else { return nil }
            let bytes = value * Double(scale)
            guard bytes.isFinite, bytes >= 1, bytes < Double(Int.max) else { return nil }
            return Int(bytes)
        }
        guard let plain = Int(raw), plain > 0 else { return nil }
        return plain
    }
}
