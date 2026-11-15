import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
    /// Slot-aware decode: run one token into sequence `slot`'s KV and GDN
    /// regions. Required (not merely an extension) so a slot-aware producer is
    /// reached through the existential `any LogitProducer`; the default serves
    /// the single-sequence producers that have no slots.
    func produce(
        token: Int32, position: Int, slot: Int,
        into logits: MTLBuffer) async throws

    /// Clear one sequence's state before it starts. A single-sequence producer
    /// resets everything; a batched one must reset only `slot`, or starting a
    /// request would wipe the sequences already running beside it. Gated, so it
    /// cannot clear shared scratch mid-step.
    func resetSequence(slot: Int) async
}

extension LogitProducer {
    public func resetSequence(slot: Int) async {
        reset()
    }
}

extension LogitProducer {
    public func produce(
        token: Int32, position: Int, slot: Int,
        into logits: MTLBuffer
    ) async throws {
        try await produce(token: token, position: position, into: logits)
    }
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
    func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult

    /// Slot-aware chunked prefill: the chunk is written into `slot`'s KV and
    /// GDN regions. A requirement (not an extension method) so the existential
    /// dispatch reaches the runner; the default serves producers with one slot.
    func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult
}

extension ChunkedPrefillRunner {
    func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await prefillChunked(
            tokens: tokens, startPosition: startPosition,
            outputMode: outputMode, config: config,
            into: logits, onProgress: onProgress)
    }
}
