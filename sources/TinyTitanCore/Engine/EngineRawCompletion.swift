import Foundation
import Metal

public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct WideVerifyCostModel: Sendable, Equatable {
    public var scalarMilliseconds: Double
    public var passBaseMilliseconds: Double
    public var passRowMilliseconds: Double

    public static let planProjection = WideVerifyCostModel(
        scalarMilliseconds: 211, passBaseMilliseconds: 370, passRowMilliseconds: 55)
    public static let measuredChunkedPrefill = WideVerifyCostModel(
        scalarMilliseconds: 293.4, passBaseMilliseconds: 1352, passRowMilliseconds: 150.7)
    public static let measuredDedicatedVerify = WideVerifyCostModel(
        scalarMilliseconds: 299.2, passBaseMilliseconds: 259, passRowMilliseconds: 103.4)

    public init(
        scalarMilliseconds: Double, passBaseMilliseconds: Double, passRowMilliseconds: Double
    ) {
        precondition(
            scalarMilliseconds > 0 && passBaseMilliseconds >= 0 && passRowMilliseconds >= 0,
            "cost model needs a positive scalar token and non-negative pass terms")
        self.scalarMilliseconds = scalarMilliseconds
        self.passBaseMilliseconds = passBaseMilliseconds
        self.passRowMilliseconds = passRowMilliseconds
    }

    public func passMilliseconds(rows: Int) -> Double {
        passBaseMilliseconds + passRowMilliseconds * Double(rows)
    }
}

public struct WideVerifyDecodeOptions: Sendable, Equatable {
    public var maxRows: Int
    public var costModel: WideVerifyCostModel
    public var minN: Int
    public var maxN: Int

    public init(
        maxRows: Int = 9, costModel: WideVerifyCostModel = .measuredDedicatedVerify,
        minN: Int = 2, maxN: Int = 8
    ) {
        self.maxRows = maxRows
        self.costModel = costModel
        self.minN = minN
        self.maxN = maxN
    }

    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> WideVerifyDecodeOptions? {
        guard environment["TINYTITAN_WIDE_VERIFY"] == "1" else { return nil }
        var options = WideVerifyDecodeOptions()
        if let raw = environment["TINYTITAN_WIDE_VERIFY_ROWS"], let rows = Int(raw), rows >= 2 {
            options.maxRows = rows
        }
        return options
    }
}

public struct WideVerifyDecodeStats: Sendable, Equatable {
    public var maxRows = 0
    public var passes = 0
    public var scalarSteps = 0
    public var draftedTokens = 0
    public var acceptedDrafts = 0
    public var rejectingPasses = 0
    public var emittedTokens = 0
    public var passesCutAtStop = 0
    public var passNanos: UInt64 = 0

    public init() {}

    public var acceptanceRate: Double {
        draftedTokens > 0 ? Double(acceptedDrafts) / Double(draftedTokens) : 0
    }

    public var summary: String {
        String(
            format:
                "TinyTitan wide-verify rows=%d passes=%d drafted=%d accepted=%d "
                + "acceptance=%.1f%% rejecting_passes=%d emitted=%d emitted_per_pass=%.3f "
                + "scalar_steps=%d pass_ms=%.1f",
            maxRows, passes, draftedTokens, acceptedDrafts, acceptanceRate * 100,
            rejectingPasses, emittedTokens,
            passes > 0 ? Double(emittedTokens) / Double(passes) : 0,
            scalarSteps, passes > 0 ? Double(passNanos) / Double(passes) / 1_000_000 : 0)
    }
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public var wideVerify: WideVerifyDecodeStats?

    public init(
        prefillTokens: Int, cachedPromptTokens: Int, computedPrefillTokens: Int,
        prefillSeconds: Double, newTokens: Int, decodeSeconds: Double, reason: StopReason,
        kvPosition: Int, kvBackedTokenIDs: [Int32], uncommittedBoundaryTokenIDs: [Int32],
        wideVerify: WideVerifyDecodeStats? = nil
    ) {
        self.prefillTokens = prefillTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.computedPrefillTokens = computedPrefillTokens
        self.prefillSeconds = prefillSeconds
        self.newTokens = newTokens
        self.decodeSeconds = decodeSeconds
        self.reason = reason
        self.kvPosition = kvPosition
        self.kvBackedTokenIDs = kvBackedTokenIDs
        self.uncommittedBoundaryTokenIDs = uncommittedBoundaryTokenIDs
        self.wideVerify = wideVerify
    }
}

/// unchecked-invariant: the selected factory owns the payload and its exclusive-use contract.
public struct RawCompletionScratch: @unchecked Sendable {
    public let payload: (any Sendable)?
    public init(payload: (any Sendable)? = nil) { self.payload = payload }
}

extension GenerationConfig {
    public var isPureGreedy: Bool {
        temperature == 0 && presencePenalty == 0 && repetitionPenalty == 1
    }
}

public struct RawCompletionRequest: Sendable {
    public let producer: any LogitProducer
    public let tokenizer: GFTokenizer
    public let promptIDs: [Int32]
    public let configuration: GenerationConfig
    public let context: MetalContext
    public let scratch: RawCompletionScratch
    public let prefillConfiguration: PrefillRuntimeConfig
    public let start: RawCompletionStart
    public let slot: Int
    public let checkpointPositions: [Int]
    public let onPrefillCheckpoint: (@Sendable (Int) -> Void)?
    public let wideVerify: (any Sendable)?
    public let shouldStop: @Sendable () -> Bool
    public let onProgress: @Sendable (RawDecodeProgress) -> Void

    public init(
        producer: any LogitProducer, tokenizer: GFTokenizer, promptIDs: [Int32],
        configuration: GenerationConfig, context: MetalContext, scratch: RawCompletionScratch,
        prefillConfiguration: PrefillRuntimeConfig, start: RawCompletionStart, slot: Int,
        checkpointPositions: [Int], onPrefillCheckpoint: (@Sendable (Int) -> Void)?,
        wideVerify: (any Sendable)?, shouldStop: @escaping @Sendable () -> Bool,
        onProgress: @escaping @Sendable (RawDecodeProgress) -> Void
    ) {
        self.producer = producer
        self.tokenizer = tokenizer
        self.promptIDs = promptIDs
        self.configuration = configuration
        self.context = context
        self.scratch = scratch
        self.prefillConfiguration = prefillConfiguration
        self.start = start
        self.slot = slot
        self.checkpointPositions = checkpointPositions
        self.onPrefillCheckpoint = onPrefillCheckpoint
        self.wideVerify = wideVerify
        self.shouldStop = shouldStop
        self.onProgress = onProgress
    }
}
