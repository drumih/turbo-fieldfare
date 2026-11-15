import Foundation
import Metal

/// MTLBuffer is not Sendable; the gate only carries the handle to its owner,
/// which is the runner the gate was created for.
/// unchecked-invariant: the buffer is one request's logits scratch, alive for
/// the suspended decode step, and the gate hands it back only to the runner
/// that owns it; no second writer reads or writes it.
private struct LogitsBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
}

/// Groups concurrent slots' decode steps into one `produceBatch` call.
///
/// The runner's layer loop still runs per row; the fusion today is the head
/// (two rows per two-row kernel, one weight read). The gate is what lets the
/// server's concurrent sessions reach that call together: each session's
/// `produce` parks on a continuation, the first row schedules a short flush,
/// and `width` rows (or the timeout, whichever comes first) run as one batch.
/// A lone session waits at most `flushDelay` a token, well under a
/// millisecond.
///
/// The batch is emitted under the runner's own `ForwardStepGate`, so a prefill
/// or a reset cannot interleave with it.
/// Park concurrent slots' decode steps and run them as one batch.
/// unchecked-invariant: the wrapped runner is itself @unchecked Sendable and
/// every call into it — produceBatch, prefill, reset — is serialised by the
/// batch gate and the runner's own ForwardStepGate, so no two steps touch the
/// runner's scratch at once.
public final class BatchedLogitProducer: LogitProducer, ContinuableLogitProducer,
    @unchecked Sendable
{
    private let runner: RealForwardRunner
    private let gate: BatchGate

    public init(runner: RealForwardRunner, width: Int) {
        self.runner = runner
        self.gate = BatchGate(width: width)
    }

    public func reset() {
        runner.reset()
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produce(token: token, position: position, slot: 0, into: logits)
    }

    public func produce(
        token: Int32, position: Int, slot: Int, into logits: MTLBuffer
    ) async throws {
        try await gate.submit(
            runner: runner, token: token, position: position, slot: slot,
            logits: LogitsBuffer(buffer: logits))
    }

    public func resetSequence(slot: Int) async {
        await runner.resetSequence(slot: slot)
    }

    public var maxContext: Int { runner.maxContext }
    public var continuationPosition: Int { runner.continuationPosition }

    public func prepareForContinuation(expectedPosition: Int) throws {
        try runner.prepareForContinuation(expectedPosition: expectedPosition)
    }

    public func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await runner.prefillChunked(
            tokens: tokens, startPosition: startPosition,
            outputMode: outputMode, config: config,
            into: logits, onProgress: onProgress)
    }

    func prefillChunked(
        tokens: ArraySlice<Int32>,
        startPosition: Int,
        slot: Int,
        outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig,
        into logits: MTLBuffer,
        onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await runner.prefillChunked(
            tokens: tokens, startPosition: startPosition, slot: slot,
            outputMode: outputMode, config: config,
            into: logits, onProgress: onProgress)
    }
}

/// The parking lot in front of `produceBatch`.
private actor BatchGate {
    private struct Pending {
        let token: Int32
        let position: Int
        let slot: Int
        let logits: LogitsBuffer
        let continuation: CheckedContinuation<Void, Error>
    }

    private let width: Int
    private var pending: [Pending] = []
    private var flushScheduled = false
    private var busy = false
    /// Long enough for a sibling session to arrive, short enough that a lone
    /// request does not pay a measurable per-token delay.
    private static let flushDelay: Duration = .microseconds(250)

    init(width: Int) {
        self.width = max(2, width)
    }

    func submit(
        runner: RealForwardRunner, token: Int32, position: Int, slot: Int,
        logits: LogitsBuffer
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            pending.append(
                Pending(
                    token: token, position: position, slot: slot,
                    logits: logits, continuation: continuation))
            if pending.count >= width {
                Task { await self.flush(runner: runner) }
            } else if !flushScheduled {
                flushScheduled = true
                Task {
                    try? await Task.sleep(for: Self.flushDelay)
                    await self.flush(runner: runner)
                }
            }
        }
    }

    private func flush(runner: RealForwardRunner) async {
        guard !busy, !pending.isEmpty else { return }
        busy = true
        flushScheduled = false
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        do {
            nonisolated(unsafe) let logits = batch.map(\.logits.buffer)
            try await runner.produceBatch(
                batch.map { (token: $0.token, position: $0.position, slot: $0.slot) },
                logits: logits)
            for row in batch { row.continuation.resume() }
        } catch {
            for row in batch { row.continuation.resume(throwing: error) }
        }
        busy = false
        if !pending.isEmpty {
            if pending.count >= width {
                await flush(runner: runner)
            } else if !flushScheduled {
                flushScheduled = true
                Task {
                    try? await Task.sleep(for: Self.flushDelay)
                    await self.flush(runner: runner)
                }
            }
        }
    }
}

/// The slot-aware and prefilling capabilities the runner has, forwarded
/// unchanged: the batching is only in `produce`.
extension BatchedLogitProducer: ChunkedPrefillRunner, ContextWindowReporting {}
