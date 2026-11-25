import Foundation
import Metal
import TinyTitanCore

/// The Inkling 2 Q4 decoder behind the shared session's runner contract
/// (TT-122). Every call is the decoder's own prefill or step, so the session
/// path computes the same bits the Inkling route did: greedy tokens are the
/// GPU argmax and logits are the decoder's softcapped FP32 values.
///
/// Multi-slot (TT-124): each sequence slot owns its own KV caches, convolution
/// histories and position counter. `lock` serialises GPU execution over the
/// shared chunk scratch so concurrent slots never interleave mid-step.
///
/// unchecked-invariant: `positions`, `decoder`, and `decoder.state` are only
/// mutated under `lock`, so concurrent tasks driving distinct slots cannot race
/// on shared scratch or position state.
package final class InklingQ4Runner: EngineRunner, @unchecked Sendable {
    package let model: InklingQ4EngineModel
    package let decoder: InklingQ4Decoder
    private let lock = NSLock()
    /// Per-slot token counts whose K/V and convolution state the decoder holds.
    private var positions: [Int]

    package init(model: InklingQ4EngineModel, decoder: InklingQ4Decoder) {
        self.model = model
        self.decoder = decoder
        self.positions = Array(repeating: 0, count: decoder.state.slots)
    }

    package var maxContext: Int { decoder.state.maxContext }
    package var continuationPosition: Int { lock.withLock { positions[0] } }

    package func position(forSlot slot: Int) -> Int {
        lock.withLock {
            guard slot >= 0, slot < positions.count else { return 0 }
            return positions[slot]
        }
    }

    package func prepareForContinuation(expectedPosition: Int) throws {
        try prepareForContinuation(expectedPosition: expectedPosition, slot: 0)
    }

    package func prepareForContinuation(expectedPosition: Int, slot: Int) throws {
        try lock.withLock {
            guard slot >= 0, slot < positions.count, expectedPosition == positions[slot] else {
                throw InferenceStateSnapshotError.invalidPosition(expectedPosition)
            }
        }
    }

    package func reset() {
        lock.withLock {
            decoder.state.reset()
            decoder.mtpDraft?.reset()
            for index in positions.indices { positions[index] = 0 }
        }
    }

    package func reset(slot: Int) {
        lock.withLock {
            guard slot >= 0, slot < positions.count else { return }
            decoder.state.reset(slot: slot)
            if slot == 0 { decoder.mtpDraft?.reset() }
            positions[slot] = 0
        }
    }

    package var hasMTP: Bool { decoder.mtpDraft != nil }

    package func prefillMTP(_ tokens: [Int], at start: Int) throws -> (next: Int, lastTapRow: Int) {
        try lock.withLock {
            guard start == positions[0] else {
                throw InferenceStateSnapshotError.invalidPosition(start)
            }
            let result = try decoder.prefillWithMTP(tokens, position: start)
            positions[0] = start + tokens.count
            return result
        }
    }

    package func draftMTP(anchorToken: Int, position: Int, lastTapRow: Int) throws -> [Int] {
        try lock.withLock {
            guard let mtp = decoder.mtpDraft else {
                throw MetalError.invalidState("MTP draft model is not enabled")
            }
            return try mtp.draftBlock(
                anchorToken: anchorToken, position: position, lastTapRow: lastTapRow)
        }
    }

    package func verifyMTP(_ tokens: [Int], at start: Int) throws -> [Int] {
        try lock.withLock {
            guard start == positions[0] else {
                throw InferenceStateSnapshotError.invalidPosition(start)
            }
            return try decoder.verifyBlock(tokens, position: start)
        }
    }

    package func commitMTP(at start: Int, acceptedCount: Int) throws {
        try lock.withLock {
            guard start == positions[0] else {
                throw InferenceStateSnapshotError.invalidPosition(start)
            }
            try decoder.commitVerified(position: start, acceptedCount: acceptedCount)
            positions[0] = start + acceptedCount
        }
    }

    package func resetSequence(slot: Int) async {
        reset(slot: slot)
    }

    /// Feeds `tokens` at `start` into `slot` and returns the greedy next token,
    /// with the softcapped logits of the vocabulary when asked.
    package func feed(_ tokens: [Int], at start: Int, slot: Int = 0, logits: Bool) throws
        -> (next: Int, logits: [Float]?)
    {
        try lock.withLock {
            guard slot >= 0, slot < positions.count else {
                throw MetalError.invalidState(
                    "Inkling 2 slot \(slot) out of range 0..<\(positions.count)")
            }
            guard start == positions[slot] else {
                throw InferenceStateSnapshotError.invalidPosition(start)
            }
            let result = try decoder.prefill(
                tokens, position: start, slot: slot, returnLogits: logits)
            positions[slot] = start + tokens.count
            return result
        }
    }

    package func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produce(token: token, position: position, slot: 0, into: logits)
    }

    package func produce(
        token: Int32, position: Int, slot: Int, into logits: MTLBuffer
    ) async throws {
        let result = try feed([Int(token)], at: position, slot: slot, logits: true)
        Self.write(result.logits ?? [], into: logits)
    }

    package func produceBatch(
        _ rows: [(token: Int32, position: Int, slot: Int)], logits: [MTLBuffer]
    ) async throws {
        for (row, buffer) in zip(rows, logits) {
            try await produce(
                token: row.token, position: row.position, slot: row.slot, into: buffer)
        }
    }

    package func prefillChunked(
        tokens: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig, into logits: MTLBuffer, onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try await prefillChunked(
            tokens: tokens, startPosition: startPosition, slot: 0, outputMode: outputMode,
            config: config, into: logits, onProgress: onProgress)
    }

    package func prefillChunked(
        tokens: ArraySlice<Int32>, startPosition: Int, slot: Int, outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig, into logits: MTLBuffer, onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        let wantsLogits = outputMode == .logits
        let result = try feed(
            tokens.map(Int.init), at: startPosition, slot: slot, logits: wantsLogits)
        let newPosition = startPosition + tokens.count
        onProgress(tokens.count)
        if wantsLogits {
            Self.write(result.logits ?? [], into: logits)
            return PrefillResult(newPosition: newPosition, seed: .logitsWritten)
        }
        return PrefillResult(newPosition: newPosition, seed: .greedyToken(UInt32(result.next)))
    }

    /// FP32 logits as the FP16 rows the shared sampler reads.
    static func write(_ values: [Float], into buffer: MTLBuffer) {
        let count = min(values.count, buffer.length / MemoryLayout<Float16>.size)
        let out = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for index in 0..<count { out[index] = Float16(values[index]) }
    }

    // MARK: - Inference state (prompt cache)

    package var supportsInferenceStateRestore: Bool { true }

    /// The state that defines a prefix of `position` tokens, as (buffer, live
    /// bytes) in a fixed order: per visit its K and V rows, then every visit's
    /// convolution history. Position `p` lives in row `p % capacity`, so a
    /// prefix shorter than the ring holds rows `0 ..< position` and a longer
    /// one (a sliding visit past its window) holds the whole ring.
    private func segments(at position: Int) -> (kv: [(MTLBuffer, Int)], conv: [(MTLBuffer, Int)]) {
        let state = decoder.state
        let rowBytes = state.kvWidth * MemoryLayout<Float16>.size
        var kv: [(MTLBuffer, Int)] = []
        for (visit, capacity) in state.capacities.enumerated() {
            let bytes = min(capacity, position) * rowBytes
            kv.append((state.kCaches[visit], bytes))
            kv.append((state.vCaches[visit], bytes))
        }
        return (kv, state.convStates.map { ($0, $0.length) })
    }

    package func captureInferenceState(maximumBytes: Int?) throws -> InferenceStateSnapshot {
        try lock.withLock {
            let current = positions[0]
            let layout = segments(at: current)
            let parts = layout.kv + layout.conv
            let total = parts.reduce(0) { $0 + $1.1 }
            if let maximumBytes, total > maximumBytes {
                throw InferenceStateSnapshotError.exceedsLimit(bytes: total, limit: maximumBytes)
            }
            var payload = Data(capacity: total)
            for (buffer, length) in parts {
                payload.append(Data(bytes: buffer.contents(), count: length))
            }
            return InferenceStateSnapshot(
                descriptor: InferenceStateSnapshotDescriptor(
                    position: current, kvSegmentLengths: layout.kv.map(\.1),
                    gdnSegmentLengths: layout.conv.map(\.1), payloadBytes: total),
                payload: payload)
        }
    }

    package func restoreInferenceState(
        _ snapshot: InferenceStateSnapshot, auxiliary: InferenceAuxiliaryState?
    ) throws {
        try lock.withLock {
            let descriptor = snapshot.descriptor
            guard descriptor.version == InferenceStateSnapshotDescriptor.currentVersion else {
                throw InferenceStateSnapshotError.unsupportedVersion(descriptor.version)
            }
            guard descriptor.position > 0, descriptor.position <= maxContext else {
                throw InferenceStateSnapshotError.invalidPosition(descriptor.position)
            }
            let layout = segments(at: descriptor.position)
            guard descriptor.kvSegmentLengths == layout.kv.map(\.1),
                descriptor.gdnSegmentLengths == layout.conv.map(\.1)
            else { throw InferenceStateSnapshotError.invalidLayout }
            let parts = layout.kv + layout.conv
            let total = parts.reduce(0) { $0 + $1.1 }
            guard snapshot.payload.count == total, descriptor.payloadBytes == total else {
                throw InferenceStateSnapshotError.invalidPayloadSize(
                    expected: total, actual: snapshot.payload.count)
            }
            decoder.state.reset(slot: 0)
            snapshot.payload.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                var offset = 0
                for (buffer, length) in parts {
                    memcpy(buffer.contents(), base + offset, length)
                    offset += length
                }
            }
            positions[0] = descriptor.position
        }
    }

    package func captureAuxiliaryState() throws -> InferenceAuxiliaryState? { nil }

    // MARK: - Diagnostics

    package var statistics: EngineStatistics { .zero }
    package func resetStatistics() {}
    package func decodeExpertIO() -> (hits: UInt64, misses: UInt64, bytes: UInt64)? { nil }

    package func expertStreamingStatistics() -> ExpertStreamingStatistics {
        guard let cache = decoder.expertCache else { return .zero }
        let stats = cache.statistics
        return ExpertStreamingStatistics(
            plans: stats.rounds, requestedExperts: stats.hits + stats.misses,
            hits: stats.hits, misses: stats.misses, bytesRead: stats.bytesRead,
            readOperations: stats.misses + stats.prefetchLoads, evictions: 0, reloads: 0,
            loadBatches: stats.loadCalls,
            totalLoadNanos: UInt64(max(0, stats.readSeconds) * 1e9), maximumLoadNanos: 0,
            latencyHistogram: [], residentSlots: cache.slotCount, loadingSlots: 0,
            pinnedSlots: 0, peakLoadingSlots: 0)
    }

    package var prefetchRingSummary: String? { nil }
    package var prefillTensorOpsLogLine: String { "" }
    package func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] { [] }
    package func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double) { (0, 0) }
    package func kernelGPUGaps() -> [(transition: String, millis: Double, count: Int)] { [] }
    package func resetKernelGPUTimings() {}
}
