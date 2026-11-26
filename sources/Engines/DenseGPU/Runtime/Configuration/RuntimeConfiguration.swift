import Foundation
import TinyTitanCore

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let supportedContextTokens = [
        4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
    ]
    /// The ceiling for a backend that has no model card to ask, so it is what
    /// a request's `max_context` defaults to. Equal to
    /// `supportedContextTokens.last` by declaration, pinned by
    /// `RuntimeConfigurationTests.publicContextChoicesReachQwenMaximum` --
    /// which is why the sites that want it name this rather than taking
    /// `.max()` of the list and inventing a fallback for a branch that cannot
    /// happen.
    public static let nativeMaximumContextTokens = 262_144
    /// The YaRN rungs. `2_097_152` is the LongRoPE-scale window: at the native
    /// 262 144 it is exactly the 8x that arXiv 2402.13753 reports without
    /// fine-tuning, and `YaRNRoPEParameters` derives factor 8 from the two
    /// numbers. It is offered because the engine serves hosts other than the one
    /// it was built on — the KV cost at 2M (12 KB/token on this family, 24.6 GB
    /// at 8-bit) is the operator's to weigh, and the memory guard refuses rather
    /// than silently clamping. `docs/plan-longrope-2m.md` carries the arithmetic.
    public static let supportedYaRNContextTokens = (9...14).map { 1 << ($0 + 10) }
    public static let defaultYaRNContextTokens = 1_048_576
    public static let maximumContextTokens = 16_777_216
    // 112 exists because the useful range ends between 96 and 128 on a 24 GiB
    // machine: 96 measured an 85.4% hit rate, 128 reaches 89.8% but needs
    // ~17 GB of cache and swaps, costing 68%. Without a value in between
    // there was no way to ask whether the extra hit rate is reachable.
    //
    // 160, 192 and 256 (2026-09-05): the 35B family at 128 slots per layer
    // still spent a fifth of its 4-bit token in exposed expert reads (87.7%
    // hit rate); its route traces put the ceiling at 192 (95.5%, the rest
    // compulsory). 160 at 4-bit is 10 GiB and measured +4% with swap flat;
    // 192 is 12 GiB, +8%, and pushed 1.5 GB to swap on a 24 GB machine.
    public static let allowedExpertCacheSlots = [
        8, 16, 24, 32, 40, 48, 64, 96, 112, 128, 160, 192, 256,
    ]

    /// The smallest rung `allowedExpertCacheSlots` offers.
    ///
    /// The list is a constant, so "no rungs at all" is not a case a caller has
    /// to answer: the sites that need the floor name this instead of writing
    /// `.first ?? 8` and putting the 8 in seven places. Pinned by
    /// `RuntimeConfigurationTests.expertCacheSlotFloorIsTheSmallestRung`, which
    /// is what catches the two drifting apart.
    public static let minimumExpertCacheSlots = 8

    /// Target bytes for the routed-expert slot cache when no count is given.
    ///
    /// 8 GiB, which is a third of a 24 GB machine and deliberate. The slot cache
    /// has to hold the routing working set, and a routing trace over 383 real
    /// tokens measured **131 distinct experts per layer** across a 128-token
    /// window. 128 slots is the first budget that holds it.
    ///
    /// Under the bounded policy (the automatic answer on this 24 GB host, and
    /// on any host where the install is above half of RAM), expert reads
    /// bypass the page cache (see `PageCachePolicy` and
    /// `ParallelExpertReader`). There is then no second cache to fall back on:
    /// whatever the slots do not hold is fetched from SSD every token. That makes the curve a cliff rather than a
    /// slope. Measured, 4-bit, short prompt, bounded:
    ///
    ///      16 slots  1.05 GB   8.73 tok/s   io 49.4 ms
    ///      32 slots  2.11 GB   8.94         io 41.3
    ///      64 slots  4.22 GB   9.91         io 28.3
    ///     128 slots  8.44 GB  18.91         io  7.2
    ///
    /// A smaller budget does not trade throughput gently for memory -- it falls off
    /// by 2.2x while saving RAM that the OS would otherwise have to hold anyway.
    ///
    /// A note here used to claim this inverts under the page-cache policy - the OS
    /// holding the working set and slot memory being redundant pressure, with 4-bit
    /// measuring "13.61 tok/s at 16 slots against 8.78 at 128". **That no longer
    /// reproduces and the note was removed rather than left to guide tuning**: swept
    /// against the page-cache reader on an 8 GB mini, 8 / 16 / 24 / 32 / 40 slots
    /// give 6.319 / 6.769 / 7.395 / 8.138 / 8.552 tok/s with the await falling
    /// 5564 -> 3358 ms. Monotonic, the opposite sign, and 16 slots costs 21%
    /// against 40 - so more slots is right under either reader, which is what the
    /// budget below selects. Re-tune at the shipped `--max-context`, never a
    /// reduced one.
    ///
    /// The reader finding is left as a finding, not acted on: the serial page-cache
    /// reader beats the bounded parallel default by 8.334 against 7.717 tok/s, but
    /// `docs/v4-core-design.md` records that trade as deliberate - about 20% for a
    /// footprint that is actually bounded - so reversing it is a product decision
    /// about memory predictability, not a defect to fix.
    public static let defaultExpertCacheBudgetBytes = 8 << 30

    /// One source of truth: `PrefillRuntimeConfig.allowedProductionChunkTokens` (Core).
    public static let allowedPrefillChunkTokens = PrefillRuntimeConfig.allowedProductionChunkTokens
    public static let qwenLongPrefillChunkTokens = 4_096

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let decodeExpertExecution: RuntimeDecodeExpertExecution
    public let expertIOSynchronization: RuntimeExpertIOSynchronization
    public let expertIOSubmission: RuntimeExpertIOSubmission
    public let expertIOTimeline: RuntimeExpertIOTimeline
    public let kvCachePrecision: KVCachePrecision
    /// Full-attention KV window in VRAM; nil keeps the contiguous cache.
    public let hostKVWindowTokens: Int?
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let yarnContextTokens: Int

    public init(
        expertCacheSlots: Int = 64,
        expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
        rdadvisePolicy: RDAdvicePolicyMode = .default,
        prefillEnabled: Bool = true,
        prefillChunkTokens: Int = 128,
        prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
        forceLogitsHead: Bool = false,
        decodeExpertExecution: RuntimeDecodeExpertExecution = .hitFixup,
        expertIOSynchronization: RuntimeExpertIOSynchronization = .host,
        expertIOSubmission: RuntimeExpertIOSubmission = .deferred,
        expertIOTimeline: RuntimeExpertIOTimeline = .shared,
        kvCachePrecision: KVCachePrecision = .int8,
        hostKVWindowTokens: Int? = nil,
        ropeScalingMode: RuntimeRoPEScalingMode = .none,
        yarnContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens
    ) throws {
        guard Self.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw RuntimeConfigurationError.invalidExpertCacheSlots(expertCacheSlots)
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw RuntimeConfigurationError.invalidPrefillChunkTokens(prefillChunkTokens)
        }
        guard Self.supportedYaRNContextTokens.contains(yarnContextTokens) else {
            throw RuntimeConfigurationError.invalidYaRNContextTokens(yarnContextTokens)
        }
        if let hostKVWindowTokens,
            !(1...Self.maximumContextTokens).contains(hostKVWindowTokens)
        {
            // A window larger than the largest context this engine serves
            // cannot hold a meaningful selection, and the arena it would size
            // is per layer and per slot: refuse at parse, not at allocation.
            throw RuntimeConfigurationError.invalidHostKVWindowTokens(
                hostKVWindowTokens)
        }
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.decodeExpertExecution = decodeExpertExecution
        self.expertIOSynchronization = expertIOSynchronization
        self.expertIOSubmission = expertIOSubmission
        self.expertIOTimeline = expertIOTimeline
        self.kvCachePrecision = kvCachePrecision
        self.hostKVWindowTokens = hostKVWindowTokens
        self.ropeScalingMode = ropeScalingMode
        self.yarnContextTokens = yarnContextTokens
    }

    public func validate(maxContext: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        switch ropeScalingMode {
        case .none:
            guard maxContext <= Self.nativeMaximumContextTokens else {
                throw RuntimeConfigurationError.contextRequiresYaRN(maxContext)
            }
        case .yarn:
            // The cap and the fixed extension profile are independent.
            guard maxContext <= Self.maximumContextTokens else {
                throw RuntimeConfigurationError.contextExceedsMaximum(maxContext)
            }
        }
    }

    public static var production: RuntimeConfiguration {
        // lint:allow-force every default is a compile-time constant on the
        // allowed lists, so the validating init cannot throw here;
        // RuntimeConfigurationTests pins that. The SwiftLint disable below
        // restates that audited reason for the second gate rather than waiving
        // it silently: there is no non-trapping fallback, because any
        // fallback value would ship a configuration no caller asked for.
        // swiftlint:disable:next force_try
        try! RuntimeConfiguration()
    }

    /// Production pins the sliding-window ring on. This is deliberately a
    /// constant and not a stored option: `KVCacheManager` takes the flag as a
    /// real parameter (tests construct it both ways to cover the non-ring
    /// path), but the shipping runtime has exactly one supported setting, and
    /// the value is part of `ServerPromptCacheDomain` — making it settable
    /// would let two processes disagree about the layout of a persisted KV
    /// snapshot. Read-only here is the guarantee, not an oversight.
    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    /// The configured policy as the streaming stack spells it. The
    /// `TINYTITAN_EXPERT_CACHE_POLICY` override that used to sit here also
    /// reached two experiment variants (aging-lfu and a decayed use count), and
    /// every arm measured a wash on this engine, so the switch and the variants
    /// are gone and the configured lfu/lru choice is the whole surface.
    public var modelExpertCachePolicy: ExpertCachePolicy {
        return expertCachePolicy == .lru ? .lru : .lfu
    }
}
