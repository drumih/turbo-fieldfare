import Metal

/// Projected image rows that replace ordinary token embeddings during prompt
/// prefill. The buffer stores tightly packed FP16 rows in language hidden size.
public struct PrefillEmbeddingOverlay: @unchecked Sendable {
    public let tokenRange: Range<Int>
    public let buffer: MTLBuffer
    public let bufferOffsetBytes: Int
    public let hiddenSize: Int

    public init(tokenRange: Range<Int>,
                buffer: MTLBuffer,
                bufferOffsetBytes: Int = 0,
                hiddenSize: Int) {
        precondition(!tokenRange.isEmpty, "embedding overlay must not be empty")
        precondition(bufferOffsetBytes >= 0, "embedding overlay offset must be non-negative")
        precondition(hiddenSize > 0, "embedding overlay hidden size must be positive")
        let bytes = tokenRange.count * hiddenSize * MemoryLayout<Float16>.stride
        precondition(bufferOffsetBytes + bytes <= buffer.length,
                     "embedding overlay exceeds its Metal buffer")
        self.tokenRange = tokenRange
        self.buffer = buffer
        self.bufferOffsetBytes = bufferOffsetBytes
        self.hiddenSize = hiddenSize
    }
}

public struct MultimodalPrefillInput: @unchecked Sendable {
    public let embeddingOverlays: [PrefillEmbeddingOverlay]

    public init(embeddingOverlays: [PrefillEmbeddingOverlay]) {
        self.embeddingOverlays = embeddingOverlays
    }

    public var visionTokenRanges: [Range<Int>] {
        embeddingOverlays.map(\.tokenRange)
    }
}

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

protocol MultimodalChunkedPrefillRunner: ChunkedPrefillRunner {
    func prefillMultimodal(tokens: ArraySlice<Int32>,
                           startPosition: Int,
                           input: MultimodalPrefillInput,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult
}
