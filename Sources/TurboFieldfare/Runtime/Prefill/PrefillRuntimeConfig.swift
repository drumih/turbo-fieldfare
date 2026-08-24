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
    case chunked
    case unsupported
}

public enum PrefillChunkCompleteness: String, Sendable, Equatable {
    case complete
    case unsupported
}

public struct PrefillExecutionDiagnostics: Sendable, Equatable {
    public let requestedMode: PrefillRuntimeConfig.Mode
    public let executedMode: PrefillExecutedMode
    public let kvStorageMode: PrefillKVStorageMode?
    public let chunkCompleteness: PrefillChunkCompleteness
    public let unsupportedReason: String?

    public init(config: PrefillRuntimeConfig,
                executedMode: PrefillExecutedMode,
                kvStorageMode: PrefillKVStorageMode? = nil,
                chunkCompleteness: PrefillChunkCompleteness? = nil,
                unsupportedReason: String? = nil) {
        self.requestedMode = config.mode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
            ?? (executedMode == .unsupported ? .unsupported : .complete)
        self.unsupportedReason = unsupportedReason
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
