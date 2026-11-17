import Foundation

/// The chunked-prefill value vocabulary, promoted from the 6.x runtime
/// (whose shipped variant carried a byte-identical copy). The internal
/// planning helpers became `package` so the engine modules keep using them;
/// nothing else about the types changed.

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

package struct PrefillChunkCommitState: Sendable, Equatable {
    private(set) var isDirty = false
    private(set) var inFlightStartPosition: Int?
    private(set) var inFlightTokenCount: Int?

    var inFlightEndPosition: Int? {
        guard let start = inFlightStartPosition,
            let count = inFlightTokenCount
        else { return nil }
        return start + count
    }

    package init() {}

    package mutating func markDirty(startPosition: Int, tokenCount: Int) {
        precondition(startPosition >= 0, "prefill dirty startPosition must be non-negative")
        precondition(tokenCount > 0, "prefill dirty tokenCount must be positive")
        isDirty = true
        inFlightStartPosition = startPosition
        inFlightTokenCount = tokenCount
    }

    package mutating func markCommitted() {
        isDirty = false
        inFlightStartPosition = nil
        inFlightTokenCount = nil
    }

    package mutating func reset() {
        markCommitted()
    }

    package func requireClean(operation: String) throws {
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

package struct PrefillChunkSpan: Sendable, Equatable {
    package let tokenOffset: Int
    package let tokenCount: Int
    package let startPosition: Int
    package let completedCount: Int
}

package enum PrefillChunkPlanner {
    package static func spans(
        tokenCount: Int,
        startPosition: Int,
        config: PrefillRuntimeConfig
    ) -> [PrefillChunkSpan] {
        spans(
            tokenCount: tokenCount,
            startPosition: startPosition,
            chunkTokens: config.chunkTokens)
    }

    package static func spans(
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

    /// The chunk sizes a production configuration may take. The canonical
    /// copy lives here so the runtime's `RuntimeConfiguration` row and every
    /// variant validate against one list instead of three that can drift.
    public static let allowedProductionChunkTokens: [Int] = [
        32, 64, 128, 256, 512, 1_024, 2_048, 4_096,
    ]

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
    package static func hostKVChunked(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(chunkTokens > 0 && chunkTokens <= maxChunkTokens)
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }

    public static func production(chunkTokens: Int) -> PrefillRuntimeConfig {
        precondition(
            allowedProductionChunkTokens.contains(chunkTokens),
            "unsupported prefill chunk size")
        return PrefillRuntimeConfig(mode: .chunked, chunkTokens: chunkTokens)
    }
}
