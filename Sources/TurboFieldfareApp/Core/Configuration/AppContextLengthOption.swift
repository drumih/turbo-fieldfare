import TurboFieldfare

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536
    case oneTwentyEightK = 131_072
    case twoFiftySixK = 262_144

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    /// The architecture every figure on this type is derived from. The menu
    /// only ever offers contexts for the model the app ships with.
    private static let architecture = ArchConfig.gemma4_26B_A4B

    /// Delegated rather than recomputed: the menu label, the loader's refusal,
    /// the server's and the CLI's answers all have to be the same arithmetic,
    /// and they were not when each surface carried its own copy.
    public var fp16KVBytes: UInt64 {
        ContextAdmission.fp16KVBytes(config: Self.architecture, maxContext: tokens)
    }

    public var projectedFootprintBytes: UInt64 {
        ContextAdmission.projectedFootprintBytes(config: Self.architecture,
                                                 maxContext: tokens)
    }

    public var minimumHostMemoryBytes: UInt64 {
        ContextAdmission.minimumHostMemoryBytes(config: Self.architecture,
                                                maxContext: tokens)
    }

    public func availability(hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> ContextAdmission.Availability {
        ContextAdmission.availability(config: Self.architecture,
                                      maxContext: tokens,
                                      hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots)
    }

    public func needDescription(hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> String {
        ContextAdmission.needDescription(config: Self.architecture,
                                         maxContext: tokens,
                                         hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots)
    }

    /// The options this host can actually back, in ascending order.
    ///
    /// A 262,144-token KV is charged in full the moment a command buffer binds
    /// it — one full-attention layer over sixteen rows already costs the whole
    /// 1,085 MiB — so admission is decided by the cap, never by how much of it
    /// a conversation happens to use.
    public static func available(on hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> [AppContextLengthOption] {
        allCases.filter { $0.availability(hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots) == .available }
    }

    /// The largest context this host can back, for clamping a stored choice it
    /// cannot. A host too small even for the smallest option still gets one:
    /// the loader refuses an unbacked context before it allocates, so the
    /// refusal is where that machine is told, not a menu with no rows in it.
    public static func largestAvailable(on hostMemoryBytes: UInt64, expertCacheSlots: Int = 16) -> AppContextLengthOption {
        available(on: hostMemoryBytes, expertCacheSlots: expertCacheSlots).last ?? .fourK
    }

    public static let defaultOption: AppContextLengthOption = .eightK

    /// The default row is labeled as such; every other row shows its FP16 KV
    /// allocation relative to the default, computed from `fp16KVBytes` rather
    /// than hardcoded so a default change moves every delta with it.
    public var menuLabel: String {
        guard self != Self.defaultOption else { return "\(shortLabel), Default" }
        let delta = Int64(fp16KVBytes) - Int64(Self.defaultOption.fp16KVBytes)
        return "\(shortLabel), \(Self.formattedDelta(delta))"
    }

    private static func formattedDelta(_ bytes: Int64) -> String {
        let sign = bytes < 0 ? "-" : "+"
        let megabytes = Double(abs(bytes)) / 1_000_000
        guard megabytes < 1_000 else {
            return String(format: "%@%.2f GB", sign, megabytes / 1_000)
        }
        let rounded = Int((megabytes / 5).rounded()) * 5
        return "\(sign)\(rounded) MB"
    }
}
