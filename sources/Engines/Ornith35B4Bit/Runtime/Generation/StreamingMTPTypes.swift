import Foundation
import Metal

// The native-MTP value types: the memory plan, the error type, the checkpoint
// and verification shapes, the statistics and the verify-schedule enum.
//
// Split out of `StreamingMTP.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
/// Memory contract for native Qwen3.6 multi-token prediction. The target
/// model remains SSD-streamed; this plan accounts only for incremental MTP
/// state that can become resident during a request.
public struct StreamingMTPMemoryPlan: Sendable, Equatable {
    public static let allowedBudgetMiB = 256...512
    public static let defaultBudgetMiB = 384
    public static let defaultDraftKVTokens = 65_536
    public static let defaultExpertSlots = 8
    public static let allowedExpertSlots = [8, 16, 24, 32, 64]

    /// Routed-expert cache slots for the MTP sidecar, which is a 1-layer MoE
    /// with 256 experts. Tunable because it shipped hard-coded and nothing in
    /// the build could measure the alternative -- the target model has had
    /// `--expert-cache-slots` all along, and the sidecar not having an
    /// equivalent is why this value went unexamined for so long.
    ///
    /// The default stays at 8: an interleaved A/B (4 samples each, warmup
    /// discarded) measured 8 slots at 6.202 tok/s (sd 0.26) against 32 slots
    /// at 6.008 (sd 0.35) -- no gain, and 32 costs another 42 MiB of the
    /// budget. Sequential sweeps appear to show 32 winning by ~11%, but that
    /// is page-cache warming from running the configs in order; measure this
    /// interleaved or not at all.
    ///
    /// Read once: the plan is constructed per session and the value must not
    /// change under a running decoder.
    public static let expertSlots: Int = {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_MTP_EXPERT_SLOTS"],
            let value = Int(raw), allowedExpertSlots.contains(value)
        else {
            return defaultExpertSlots
        }
        return value
    }()

    public let budgetBytes: Int
    public let residentTensorBytes: Int
    public let streamedExpertCacheBytes: Int
    public let draftKVBytes: Int
    public let targetRollbackBytes: Int
    public let scratchBytes: Int

    public var requiredBytes: Int {
        residentTensorBytes + streamedExpertCacheBytes + draftKVBytes
            + targetRollbackBytes + scratchBytes
    }

    public init(
        budgetMiB: Int = defaultBudgetMiB,
        residentTensorBytes: Int,
        expertStrideBytes: Int,
        draftKVTokens: Int = defaultDraftKVTokens,
        kvCachePrecision: KVCachePrecision = .fp16,
        targetRollbackBytes: Int,
        scratchBytes: Int
    ) throws {
        guard Self.allowedBudgetMiB.contains(budgetMiB) else {
            throw StreamingMTPError.invalidMemoryBudgetMiB(budgetMiB)
        }
        guard residentTensorBytes >= 0, expertStrideBytes >= 0,
            draftKVTokens > 0, targetRollbackBytes >= 0,
            scratchBytes >= 0
        else {
            throw StreamingMTPError.invalidMemoryComponent
        }
        let budgetBytes = budgetMiB * 1_048_576
        let streamedExpertCacheBytes = expertStrideBytes * Self.expertSlots
        // One MTP attention layer, two KV tensors, 2 heads x 256 dimensions.
        let elementsPerRow = 2 * 256
        let valueBytes = (elementsPerRow * kvCachePrecision.rawValue + 7) / 8
        let alignedValueBytes = (valueBytes + 1) & ~1
        let groupCount =
            (elementsPerRow + KVCacheManager.quantizationGroupSize - 1)
            / KVCacheManager.quantizationGroupSize
        let rowBytes =
            kvCachePrecision == .fp16
            ? elementsPerRow * MemoryLayout<Float16>.stride
            : alignedValueBytes + groupCount * 2 * MemoryLayout<Float16>.stride
        let draftKVBytes = draftKVTokens * 2 * rowBytes
        let required =
            residentTensorBytes + streamedExpertCacheBytes
            + draftKVBytes + targetRollbackBytes + scratchBytes
        guard required <= budgetBytes else {
            throw StreamingMTPError.memoryBudgetExceeded(
                requiredBytes: required,
                budgetBytes: budgetBytes)
        }
        self.budgetBytes = budgetBytes
        self.residentTensorBytes = residentTensorBytes
        self.streamedExpertCacheBytes = streamedExpertCacheBytes
        self.draftKVBytes = draftKVBytes
        self.targetRollbackBytes = targetRollbackBytes
        self.scratchBytes = scratchBytes
    }
}

public enum StreamingMTPError: Error, Equatable, CustomStringConvertible {
    case invalidMemoryBudgetMiB(Int)
    case invalidMemoryComponent
    case memoryBudgetExceeded(requiredBytes: Int, budgetBytes: Int)
    case targetMustBeQwen36
    case sidecarMustBeQwen36MTP
    case greedyOnly
    case draftNotReady
    case yaRNUnsupported
    case invalidVerifySchedule(String)
    /// A logic invariant the decoder believes is impossible was violated.
    /// Distinct from `.draftNotReady` so a genuine internal bug is debuggable
    /// instead of masquerading as "call advance before prepare" (R24).
    case internalInconsistency(String)

    public var description: String {
        switch self {
        case .invalidMemoryBudgetMiB(let value):
            "MTP memory budget must be 256...512 MiB, got \(value)"
        case .invalidMemoryComponent:
            "MTP memory-plan components must be non-negative"
        case .memoryBudgetExceeded(let required, let budget):
            "MTP requires \(required) bytes, exceeding its \(budget)-byte budget"
        case .targetMustBeQwen36:
            "MTP target must be a compatible Qwen3.5-MoE 35B-A3B model"
        case .sidecarMustBeQwen36MTP:
            "MTP sidecar has the wrong architecture"
        case .greedyOnly:
            "native MTP currently preserves exact output only for greedy decoding"
        case .draftNotReady:
            "MTP draft state has not been aligned with the target prompt"
        case .yaRNUnsupported:
            "MTP cannot use YaRN until its 65536-token draft cache supports extended logical positions"
        case .invalidVerifySchedule(let value):
            "unsupported MTP verify schedule '\(value)'; allowed: pair, tile"
        case .internalInconsistency(let detail):
            "MTP internal inconsistency: \(detail)"
        }
    }
}

/// Lightweight target checkpoint: target KV is append-only and only its
/// cursor is rewound; the fixed-size Gated-DeltaNet state is copied because it
/// is updated in place by the two-token verification batch.
struct SpeculativeInferenceCheckpoint: Sendable {
    let position: Int
}

struct TargetPairVerification: Sendable {
    let predictionAfterFirst: Int32
    let predictionAfterSecond: Int32
    /// Two contiguous FP16 pre-final-norm target hidden rows.
    let hiddenRows: Data
    /// Phase attribution for the B1 investigation (docs/v4.4 Track B):
    /// wall nanos in the 40-layer prefill-path traversal, the two-row final
    /// head command buffer, and the two CPU argmax scans respectively.
    let backboneNanos: UInt64
    let headNanos: UInt64
    let argmaxNanos: UInt64
}

struct MTPPrefillResult: Sendable {
    let target: PrefillResult
    let lastTargetHidden: Data
}

public struct MTPStatistics: Sendable, Equatable {
    public private(set) var draftedTokens = 0
    public private(set) var acceptedTokens = 0
    public private(set) var targetBackbonePasses = 0
    public private(set) var emittedTokens = 0

    /// Wall-clock phase attribution across all `advance` calls, in
    /// nanoseconds. Recording is unconditional — a handful of clock reads per
    /// ~100 ms pass — because the B1 question ("where does the 1.965x verify
    /// cost actually go?") must be answerable from any qualification run, not
    /// only specially instrumented ones.
    public private(set) var proposalNanos: UInt64 = 0
    public private(set) var checkpointNanos: UInt64 = 0
    public private(set) var verifyNanos: UInt64 = 0
    public private(set) var verifyBackboneNanos: UInt64 = 0
    public private(set) var verifyHeadNanos: UInt64 = 0
    public private(set) var verifyArgmaxNanos: UInt64 = 0
    public private(set) var commitNanos: UInt64 = 0
    public private(set) var rollbackNanos: UInt64 = 0

    mutating func recordPhases(
        proposal: UInt64,
        checkpoint: UInt64,
        verify: UInt64,
        verifyBackbone: UInt64,
        verifyHead: UInt64,
        verifyArgmax: UInt64,
        commit: UInt64,
        rollback: UInt64
    ) {
        proposalNanos &+= proposal
        checkpointNanos &+= checkpoint
        verifyNanos &+= verify
        verifyBackboneNanos &+= verifyBackbone
        verifyHeadNanos &+= verifyHead
        verifyArgmaxNanos &+= verifyArgmax
        commitNanos &+= commit
        rollbackNanos &+= rollback
    }

    public var acceptanceRate: Double {
        draftedTokens == 0 ? 0 : Double(acceptedTokens) / Double(draftedTokens)
    }
    public var emittedTokensPerTargetPass: Double {
        targetBackbonePasses == 0
            ? 0
            : Double(emittedTokens) / Double(targetBackbonePasses)
    }

    mutating func record(accepted: Bool, emitted: Int, targetPasses: Int) {
        draftedTokens += 1
        acceptedTokens += accepted ? 1 : 0
        targetBackbonePasses += targetPasses
        emittedTokens += emitted
    }
}

public struct MTPDecodeBatch: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let backedPrefixCount: Int
    public let acceptedDraft: Bool
    public let statistics: MTPStatistics
}

/// Which routed-MoE schedule serves the width-2 MTP verify pass.
///
/// `pair` is the B2 schedule: one union cache plan over both rows' experts,
/// one parallel miss fetch overlapped with the shared expert, and the decode
/// phase-1/phase-2 kernels per row. `tile` is the pre-B2 behavior — the
/// width-4096 prefill tile scheduler at width 2 — kept as the measured
/// control arm. B1 attributed the old verify's 2.30x-token backbone to that
/// path: per-tile fetches degraded the hit rate 81% -> 75.1% and read 497 MB
/// per pass against a 1.585x union model, the shared expert was waited on
/// synchronously so nothing overlapped the fetch, and the grouped tile
/// kernels cost ~71 ms/pass GPU against ~22 for the decode kernels.
public enum RuntimeMTPVerifySchedule: String, Codable, Sendable {
    case pair
    case tile

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeMTPVerifySchedule {
        guard let raw = environment["TINYTITAN_MTP_VERIFY"] else { return .pair }
        guard let value = RuntimeMTPVerifySchedule(rawValue: raw) else {
            throw StreamingMTPError.invalidVerifySchedule(raw)
        }
        return value
    }
}
