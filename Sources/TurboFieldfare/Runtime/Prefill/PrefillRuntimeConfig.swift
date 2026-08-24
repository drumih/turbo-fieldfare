import Foundation

public enum PrefillError: Error, CustomStringConvertible, Equatable {
    public static let chunkedRequiresChunkedRunnerReason =
        "chunked prefill requires a ChunkedPrefillRunner-backed runtime"

    case chunkedUnsupported(String)
    case chunkedRunnerDirty(String)
    case prefillCursorMismatch(String)
    case unsupportedPrefillSeed(String)

    public var description: String {
        switch self {
        case .chunkedUnsupported(let reason),
             .chunkedRunnerDirty(let reason),
             .prefillCursorMismatch(let reason),
             .unsupportedPrefillSeed(let reason):
            return reason
        }
    }
}

struct PrefillChunkCommitState: Sendable, Equatable {
    private(set) var isDirty = false
    private(set) var inFlightStartPosition: Int?
    private(set) var inFlightTokenCount: Int?

    var inFlightEndPosition: Int? {
        guard let start = inFlightStartPosition,
              let count = inFlightTokenCount else { return nil }
        return start + count
    }

    init() {}

    mutating func markDirty(startPosition: Int, tokenCount: Int) {
        precondition(startPosition >= 0, "prefill dirty startPosition must be non-negative")
        precondition(tokenCount > 0, "prefill dirty tokenCount must be positive")
        isDirty = true
        inFlightStartPosition = startPosition
        inFlightTokenCount = tokenCount
    }

    mutating func markCommitted() {
        isDirty = false
        inFlightStartPosition = nil
        inFlightTokenCount = nil
    }

    mutating func reset() {
        markCommitted()
    }

    func requireClean(operation: String) throws {
        guard !isDirty else {
            let range: String
            if let start = inFlightStartPosition, let end = inFlightEndPosition {
                range = " for in-flight chunk [\(start), \(end))"
            } else {
                range = ""
            }
            throw PrefillError.chunkedRunnerDirty(
                "\(operation) rejected because a previous chunked prefill wrote KV rows\(range) but did not commit; call reset() before reusing the runner")
        }
    }
}

struct PrefillChunkSpan: Sendable, Equatable {
    let tokenOffset: Int
    let tokenCount: Int
    let startPosition: Int
    let completedCount: Int

    init(tokenOffset: Int,
                tokenCount: Int,
                startPosition: Int,
                completedCount: Int) {
        self.tokenOffset = tokenOffset
        self.tokenCount = tokenCount
        self.startPosition = startPosition
        self.completedCount = completedCount
    }

}

/// One item of a multimodal prefill turn: either a run of text small enough to
/// fit the chunk scratch, or exactly one image span, which executes whole.
struct PrefillWorkItem: Sendable, Equatable {
    let range: Range<Int>
    let imageIndex: Int?

    init(range: Range<Int>, imageIndex: Int?) {
        self.range = range
        self.imageIndex = imageIndex
    }

    var isImage: Bool { imageIndex != nil }
}

enum PrefillChunkPlanner {
    /// Splits a multimodal prompt into the items a turn executes, in order.
    ///
    /// Text is cut at the same clamp the scratch layout applies, so the split
    /// and the buffer it runs against can never disagree; image spans pass
    /// through whole, because an image's features are one indivisible block.
    /// `imageRanges` must be sorted, non-overlapping, and inside `tokenCount`.
    static func multimodalWork(tokenCount: Int,
                               imageRanges: [Range<Int>],
                               chunkTokens: Int) -> [PrefillWorkItem] {
        precondition(tokenCount >= 0, "prefill tokenCount must be non-negative")
        let chunk = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        var work: [PrefillWorkItem] = []

        func appendText(_ range: Range<Int>) {
            guard !range.isEmpty else { return }
            var offset = range.lowerBound
            while offset < range.upperBound {
                let end = min(range.upperBound, offset + chunk)
                work.append(PrefillWorkItem(range: offset..<end, imageIndex: nil))
                offset = end
            }
        }

        var cursor = 0
        for (index, imageRange) in imageRanges.enumerated() {
            appendText(cursor..<imageRange.lowerBound)
            work.append(PrefillWorkItem(range: imageRange, imageIndex: index))
            cursor = imageRange.upperBound
        }
        appendText(cursor..<tokenCount)
        return work
    }

    static func spans(tokenCount: Int,
                             startPosition: Int,
                             config: PrefillRuntimeConfig) -> [PrefillChunkSpan] {
        spans(tokenCount: tokenCount,
              startPosition: startPosition,
              chunkTokens: config.chunkTokens)
    }

    static func spans(tokenCount: Int,
                             startPosition: Int,
                             chunkTokens: Int) -> [PrefillChunkSpan] {
        precondition(tokenCount >= 0, "prefill tokenCount must be non-negative")
        precondition(startPosition >= 0, "prefill startPosition must be non-negative")
        let chunk = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        guard tokenCount > 0 else { return [] }

        var spans: [PrefillChunkSpan] = []
        spans.reserveCapacity((tokenCount + chunk - 1) / chunk)
        var offset = 0
        while offset < tokenCount {
            let count = min(chunk, tokenCount - offset)
            let completed = offset + count
            spans.append(PrefillChunkSpan(tokenOffset: offset,
                                          tokenCount: count,
                                          startPosition: startPosition + offset,
                                          completedCount: completed))
            offset = completed
        }
        return spans
    }
}

public enum PrefillKVStorageMode: String, Sendable, Equatable {
    case fp16
}

public enum PrefillExecutedMode: String, Sendable, Equatable {
    case off
    case scalarFallback
    case chunked
    case mixed
    case unsupported
}

public enum PrefillChunkCompleteness: String, Sendable, Equatable {
    case complete
    case unsupported
}

public enum PrefillExecutionPath: String, Sendable, Equatable {
    case scalarFallback
    case chunked
    case mixed
}

public struct PrefillWorkDiagnostics: Sendable, Equatable {
    public let executionPath: PrefillExecutionPath
    public let scalarForwardCount: Int
    public let chunkPassCount: Int
    public let commandBufferCount: Int
    public let embeddingNanos: UInt64
    public let mixerNanos: UInt64
    public let deltaNetMixerNanos: UInt64
    public let fullAttentionMixerNanos: UInt64
    public let moePrepareNanos: UInt64
    public let expertFetchNanos: UInt64
    public let routedMoENanos: UInt64
    public let moeReduceNanos: UInt64
    public let finalHeadNanos: UInt64
    public let routedExpertCacheHitCount: Int
    public let routedExpertCacheMissCount: Int
    public let routedExpertEstimatedBytes: UInt64
    public let expertReadCount: Int
    public let expertReadNanos: UInt64
    public let expertReadMaxNanos: UInt64

    public init(executionPath: PrefillExecutionPath,
                scalarForwardCount: Int,
                chunkPassCount: Int,
                commandBufferCount: Int,
                embeddingNanos: UInt64 = 0,
                mixerNanos: UInt64 = 0,
                deltaNetMixerNanos: UInt64 = 0,
                fullAttentionMixerNanos: UInt64 = 0,
                moePrepareNanos: UInt64 = 0,
                expertFetchNanos: UInt64 = 0,
                routedMoENanos: UInt64 = 0,
                moeReduceNanos: UInt64 = 0,
                finalHeadNanos: UInt64 = 0,
                routedExpertCacheHitCount: Int = 0,
                routedExpertCacheMissCount: Int = 0,
                routedExpertEstimatedBytes: UInt64 = 0,
                expertReadCount: Int = 0,
                expertReadNanos: UInt64 = 0,
                expertReadMaxNanos: UInt64 = 0) {
        self.executionPath = executionPath
        self.scalarForwardCount = scalarForwardCount
        self.chunkPassCount = chunkPassCount
        self.commandBufferCount = commandBufferCount
        self.embeddingNanos = embeddingNanos
        self.mixerNanos = mixerNanos
        self.deltaNetMixerNanos = deltaNetMixerNanos
        self.fullAttentionMixerNanos = fullAttentionMixerNanos
        self.moePrepareNanos = moePrepareNanos
        self.expertFetchNanos = expertFetchNanos
        self.routedMoENanos = routedMoENanos
        self.moeReduceNanos = moeReduceNanos
        self.finalHeadNanos = finalHeadNanos
        self.routedExpertCacheHitCount = routedExpertCacheHitCount
        self.routedExpertCacheMissCount = routedExpertCacheMissCount
        self.routedExpertEstimatedBytes = routedExpertEstimatedBytes
        self.expertReadCount = expertReadCount
        self.expertReadNanos = expertReadNanos
        self.expertReadMaxNanos = expertReadMaxNanos
    }
}

struct PrefillWorkCounter {
    private(set) var scalarForwardCount = 0
    private(set) var chunkPassCount = 0
    private(set) var commandBufferCount = 0
    private(set) var embeddingNanos: UInt64 = 0
    private(set) var mixerNanos: UInt64 = 0
    private(set) var deltaNetMixerNanos: UInt64 = 0
    private(set) var fullAttentionMixerNanos: UInt64 = 0
    private(set) var moePrepareNanos: UInt64 = 0
    private(set) var expertFetchNanos: UInt64 = 0
    private(set) var routedMoENanos: UInt64 = 0
    private(set) var moeReduceNanos: UInt64 = 0
    private(set) var finalHeadNanos: UInt64 = 0
    private(set) var routedExpertCacheHitCount = 0
    private(set) var routedExpertCacheMissCount = 0
    private(set) var routedExpertEstimatedBytes: UInt64 = 0
    private(set) var expertReadCount = 0
    private(set) var expertReadNanos: UInt64 = 0
    private(set) var expertReadMaxNanos: UInt64 = 0

    mutating func recordScalarForward() {
        scalarForwardCount += 1
    }

    mutating func recordChunkPass() {
        chunkPassCount += 1
    }

    mutating func recordCommandBuffers(_ count: Int) {
        precondition(count >= 0, "prefill command-buffer count must be non-negative")
        commandBufferCount += count
    }

    mutating func recordStageTimings(
        embedding: UInt64 = 0,
        mixer: UInt64 = 0,
        deltaNetMixer: UInt64 = 0,
        fullAttentionMixer: UInt64 = 0,
        moePrepare: UInt64 = 0,
        expertFetch: UInt64 = 0,
        routedMoE: UInt64 = 0,
        moeReduce: UInt64 = 0,
        finalHead: UInt64 = 0
    ) {
        embeddingNanos += embedding
        mixerNanos += mixer
        deltaNetMixerNanos += deltaNetMixer
        fullAttentionMixerNanos += fullAttentionMixer
        moePrepareNanos += moePrepare
        expertFetchNanos += expertFetch
        routedMoENanos += routedMoE
        moeReduceNanos += moeReduce
        finalHeadNanos += finalHead
    }

    mutating func recordExpertReads(
        cacheHits: Int,
        cacheMisses: Int,
        estimatedBytes: UInt64,
        readCount: Int,
        readNanos: UInt64,
        readMaxNanos: UInt64
    ) {
        routedExpertCacheHitCount += cacheHits
        routedExpertCacheMissCount += cacheMisses
        routedExpertEstimatedBytes += estimatedBytes
        expertReadCount += readCount
        expertReadNanos += readNanos
        expertReadMaxNanos = max(expertReadMaxNanos, readMaxNanos)
    }

    mutating func merge(_ diagnostics: PrefillWorkDiagnostics) {
        scalarForwardCount += diagnostics.scalarForwardCount
        chunkPassCount += diagnostics.chunkPassCount
        commandBufferCount += diagnostics.commandBufferCount
        recordStageTimings(
            embedding: diagnostics.embeddingNanos,
            mixer: diagnostics.mixerNanos,
            deltaNetMixer: diagnostics.deltaNetMixerNanos,
            fullAttentionMixer: diagnostics.fullAttentionMixerNanos,
            moePrepare: diagnostics.moePrepareNanos,
            expertFetch: diagnostics.expertFetchNanos,
            routedMoE: diagnostics.routedMoENanos,
            moeReduce: diagnostics.moeReduceNanos,
            finalHead: diagnostics.finalHeadNanos)
        recordExpertReads(
            cacheHits: diagnostics.routedExpertCacheHitCount,
            cacheMisses: diagnostics.routedExpertCacheMissCount,
            estimatedBytes: diagnostics.routedExpertEstimatedBytes,
            readCount: diagnostics.expertReadCount,
            readNanos: diagnostics.expertReadNanos,
            readMaxNanos: diagnostics.expertReadMaxNanos)
    }

    var diagnostics: PrefillWorkDiagnostics? {
        guard scalarForwardCount > 0 || chunkPassCount > 0 else { return nil }
        let path: PrefillExecutionPath
        if scalarForwardCount > 0, chunkPassCount > 0 {
            path = .mixed
        } else if chunkPassCount > 0 {
            path = .chunked
        } else {
            path = .scalarFallback
        }
        return PrefillWorkDiagnostics(executionPath: path,
                                      scalarForwardCount: scalarForwardCount,
                                      chunkPassCount: chunkPassCount,
                                      commandBufferCount: commandBufferCount,
                                      embeddingNanos: embeddingNanos,
                                      mixerNanos: mixerNanos,
                                      deltaNetMixerNanos: deltaNetMixerNanos,
                                      fullAttentionMixerNanos: fullAttentionMixerNanos,
                                      moePrepareNanos: moePrepareNanos,
                                      expertFetchNanos: expertFetchNanos,
                                      routedMoENanos: routedMoENanos,
                                      moeReduceNanos: moeReduceNanos,
                                      finalHeadNanos: finalHeadNanos,
                                      routedExpertCacheHitCount: routedExpertCacheHitCount,
                                      routedExpertCacheMissCount: routedExpertCacheMissCount,
                                      routedExpertEstimatedBytes: routedExpertEstimatedBytes,
                                      expertReadCount: expertReadCount,
                                      expertReadNanos: expertReadNanos,
                                      expertReadMaxNanos: expertReadMaxNanos)
    }
}

public struct PrefillExecutionDiagnostics: Sendable, Equatable {
    public let requestedMode: PrefillRuntimeConfig.Mode
    public let executedMode: PrefillExecutedMode
    public let kvStorageMode: PrefillKVStorageMode?
    public let chunkCompleteness: PrefillChunkCompleteness
    public let unsupportedReason: String?
    public let scalarForwardCount: Int?
    public let chunkPassCount: Int?
    public let commandBufferCount: Int?

    public init(config: PrefillRuntimeConfig,
                executedMode: PrefillExecutedMode,
                kvStorageMode: PrefillKVStorageMode? = nil,
                chunkCompleteness: PrefillChunkCompleteness? = nil,
                unsupportedReason: String? = nil,
                work: PrefillWorkDiagnostics? = nil) {
        self.requestedMode = config.mode
        switch work?.executionPath {
        case .scalarFallback:
            self.executedMode = .scalarFallback
        case .chunked:
            self.executedMode = .chunked
        case .mixed:
            self.executedMode = .mixed
        case nil:
            self.executedMode = executedMode
        }
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
            ?? (executedMode == .unsupported ? .unsupported : .complete)
        self.unsupportedReason = unsupportedReason
        self.scalarForwardCount = work?.scalarForwardCount
        self.chunkPassCount = work?.chunkPassCount
        self.commandBufferCount = work?.commandBufferCount
    }

    public static func unsupported(config: PrefillRuntimeConfig,
                                   kvStorageMode: PrefillKVStorageMode? = nil,
                                   reason: String) -> PrefillExecutionDiagnostics {
        PrefillExecutionDiagnostics(config: config,
                                    executedMode: .unsupported,
                                    kvStorageMode: kvStorageMode,
                                    chunkCompleteness: .unsupported,
                                    unsupportedReason: reason)
    }
}

public struct PrefillRuntimeConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable {
        case off
        case chunked
    }

    /// Chunked prefill loops chunks outside layers, so each chunk streams that
    /// layer's experts again and read volume tracks chunk count and nothing
    /// else. On an 11,612-token prompt that meant 578 GB read from a 12 GB pool
    /// with no cache hits at all.
    ///
    /// 256 is free: the runner floors the chunk at
    /// `VisionConfig().maximumPooledTokens` (280) for the KV ring and for the
    /// multimodal scratch layout, so every size up to 280 produces byte-identical
    /// geometry. 512 is the first size that costs anything, and raising this past
    /// 280 should come with a test that asserts ring bytes the way
    /// `PrefillChunkScratchTests` asserts scratch.
    public static let maxChunkTokens = 256

    /// Chunk sizes a caller may select. Capped at `maxChunkTokens` - see there
    /// for why the larger sizes the scratch layout can handle are not offered.
    public static let allowedChunkTokens = [32, 64, 128, 256]

    /// The largest selectable chunk no greater than `requested`.
    ///
    /// Capping at `maxChunkTokens` is not enough on its own: a bare cap lets a
    /// caller ask for a size `--prefill-chunk-tokens` rejects. Snapping to the
    /// same list keeps the two doors agreeing on what is legal.
    public static func supportedChunkTokens(_ requested: Int) -> Int {
        let capped = max(1, min(requested, maxChunkTokens))
        return allowedChunkTokens.last { $0 <= capped } ?? capped
    }

    /// The smallest allowed chunk that covers `promptTokens`, capped.
    ///
    /// One chunk means each layer's experts are read once for the whole
    /// prefill, which is the floor, so this asks for the smallest size that
    /// reaches it and never more than the prompt needs.
    public static func autoChunkTokens(
        promptTokens: Int,
        cap: Int = maxChunkTokens
    ) -> Int {
        let ceiling = min(cap, maxChunkTokens)
        for candidate in allowedChunkTokens where candidate >= promptTokens {
            return min(candidate, ceiling)
        }
        return ceiling
    }

    public let mode: Mode
    public let chunkTokens: Int

    private init(mode: Mode, chunkTokens: Int) {
        self.mode = mode
        self.chunkTokens = chunkTokens
    }

    public var enabled: Bool { mode == .chunked }

    public static var off: PrefillRuntimeConfig {
        PrefillRuntimeConfig(mode: .off, chunkTokens: 128)
    }

    public static var defaultChunked: PrefillRuntimeConfig {
        production(chunkTokens: 128)
    }

    /// This config at a different chunk size, all other settings intact. The
    /// multimodal path runs each work item at the size it was planned at.
    public func replacingChunkTokens(_ tokens: Int) -> PrefillRuntimeConfig {
        PrefillRuntimeConfig(mode: mode, chunkTokens: max(1, tokens))
    }

    /// Whether the chunked path an image prompt needs is switched on. This tree
    /// has no per-stage toggles, so the mode is the whole requirement.
    ///
    /// This is deliberately not "the image turn will succeed": a chunked config
    /// is still held to its prompt-token budget afterwards.
    public var servesImagePrompt: Bool { mode == .chunked }

    /// The config an image prompt can run under, or nil when it already runs.
    ///
    /// Image spans are only served by the chunked multimodal path, so anything
    /// short of it threw after the model was loaded and every image had been
    /// encoded on the GPU. Prefill mode is a performance choice; whether images
    /// work at all is not, so an image turn is coerced rather than refused.
    /// Callers report the coercion.
    public func coercedForImagePrompt() -> PrefillRuntimeConfig? {
        servesImagePrompt ? nil : .defaultChunked
    }

    public static func production(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunkTokens),
                     "unsupported prefill chunk size")
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }
}
