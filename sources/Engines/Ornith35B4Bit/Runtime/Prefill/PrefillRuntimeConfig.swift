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
            let count = inFlightTokenCount
        else { return nil }
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
                "\(operation) rejected because a previous chunked prefill wrote KV rows\(range) but did not commit; call reset() before reusing the runner"
            )
        }
    }
}

struct PrefillChunkSpan: Sendable, Equatable {
    let tokenOffset: Int
    let tokenCount: Int
    let startPosition: Int
    let completedCount: Int
}

enum PrefillChunkPlanner {
    static func spans(
        tokenCount: Int,
        startPosition: Int,
        config: PrefillRuntimeConfig
    ) -> [PrefillChunkSpan] {
        spans(
            tokenCount: tokenCount,
            startPosition: startPosition,
            chunkTokens: config.chunkTokens)
    }

    static func spans(
        tokenCount: Int,
        startPosition: Int,
        chunkTokens: Int
    ) -> [PrefillChunkSpan] {
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
            spans.append(
                PrefillChunkSpan(
                    tokenOffset: offset,
                    tokenCount: count,
                    startPosition: startPosition + offset,
                    completedCount: completed))
            offset = completed
        }
        return spans
    }
}

public struct PrefillRuntimeConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable {
        case off
        case chunked
    }

    /// Larger chunks reduce the number of routed-expert file sweeps during
    /// long prompt prefill. Scratch and KV-ring allocations are sized from the
    /// configured chunk, so the larger ceiling does not change memory use for
    /// callers that retain a smaller setting.
    public static let maxChunkTokens = 4_096

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

    /// A chunk size chosen by the KV tier rather than the CLI's allowed list.
    /// Host-KV streaming caps the chunk at `windowTokens / selectionWidth`, a
    /// value the user never types and the production factory would reject.
    static func hostKVChunked(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(chunkTokens > 0 && chunkTokens <= maxChunkTokens)
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }

    public static func production(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(
            RuntimeConfiguration.allowedPrefillChunkTokens.contains(chunkTokens),
            "unsupported prefill chunk size")
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }
}
