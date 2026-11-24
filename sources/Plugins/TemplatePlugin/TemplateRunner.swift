import Foundation
import Metal
import TinyTitanCore

/// The runner side of the engine (`EngineRunner`): the shared session prefills,
/// decodes, resets, snapshots and restores through it. The template's "model"
/// is a counter -- the most likely next token is the last one plus one -- so
/// every contract below is real and checkable without weights. A real plugin
/// keeps this shape and runs its graph where the counter is.
///
/// unchecked-invariant: the session drives one generation at a time on this
/// runner (one sequence), so `tokens` is only touched by that generation.
public final class TemplateRunner: EngineRunner, @unchecked Sendable {
    public let model: TemplateModel
    public let maxContext: Int
    /// The state: every token fed so far. A real engine keeps K/V and
    /// recurrent state here; its length is the position.
    private(set) var tokens: [Int32] = []

    public init(model: TemplateModel, maxContext: Int) {
        self.model = model
        self.maxContext = maxContext
    }

    public var continuationPosition: Int { tokens.count }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition == tokens.count else {
            throw TemplateError.position(expected: tokens.count, got: expectedPosition)
        }
    }

    public func reset() { tokens.removeAll() }
    public func reset(slot: Int) { reset() }

    /// The next-token logits after `tokens`: the counter's choice gets 1,
    /// everything else 0.
    public func logits() -> [Float] {
        var row = [Float](repeating: 0, count: TemplateModel.vocabulary)
        row[nextGreedy()] = 1
        return row
    }

    public func nextGreedy() -> Int {
        ((tokens.last.map(Int.init) ?? -1) + 1) % TemplateModel.vocabulary
    }

    /// Feeds `chunk` at `start`, the native entry the completion loop uses.
    public func feed(_ chunk: [Int32], at start: Int) throws {
        guard start == tokens.count, start + chunk.count <= maxContext else {
            throw TemplateError.position(expected: tokens.count, got: start)
        }
        tokens += chunk
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try feed([token], at: position)
        Self.write(self.logits(), into: logits)
    }

    public func produceBatch(
        _ rows: [(token: Int32, position: Int, slot: Int)], logits: [MTLBuffer]
    ) async throws {
        for (row, buffer) in zip(rows, logits) {
            try await produce(token: row.token, position: row.position, into: buffer)
        }
    }

    public func prefillChunked(
        tokens chunk: ArraySlice<Int32>, startPosition: Int, outputMode: PrefillOutputMode,
        config: PrefillRuntimeConfig, into logits: MTLBuffer, onProgress: (Int) -> Void
    ) async throws -> PrefillResult {
        try feed(Array(chunk), at: startPosition)
        onProgress(chunk.count)
        if outputMode == .logits {
            Self.write(self.logits(), into: logits)
            return PrefillResult(newPosition: tokens.count, seed: .logitsWritten)
        }
        return PrefillResult(newPosition: tokens.count, seed: .greedyToken(UInt32(nextGreedy())))
    }

    static func write(_ values: [Float], into buffer: MTLBuffer) {
        let count = min(values.count, buffer.length / MemoryLayout<Float16>.size)
        let out = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        for index in 0..<count { out[index] = Float16(values[index]) }
    }

    // MARK: State snapshots (the prompt cache)

    public var supportsInferenceStateRestore: Bool { true }

    public func captureInferenceState(maximumBytes: Int?) throws -> InferenceStateSnapshot {
        let payload = tokens.withUnsafeBufferPointer { Data(buffer: $0) }
        if let maximumBytes, payload.count > maximumBytes {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: payload.count, limit: maximumBytes)
        }
        return InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: tokens.count, kvSegmentLengths: [payload.count],
                gdnSegmentLengths: [], payloadBytes: payload.count),
            payload: payload)
    }

    public func restoreInferenceState(
        _ snapshot: InferenceStateSnapshot, auxiliary: InferenceAuxiliaryState?
    ) throws {
        let descriptor = snapshot.descriptor
        let expected = descriptor.position * MemoryLayout<Int32>.size
        guard descriptor.position > 0, descriptor.position <= maxContext else {
            throw InferenceStateSnapshotError.invalidPosition(descriptor.position)
        }
        guard snapshot.payload.count == expected, descriptor.kvSegmentLengths == [expected] else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: expected, actual: snapshot.payload.count)
        }
        tokens = snapshot.payload.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    public func captureAuxiliaryState() throws -> InferenceAuxiliaryState? { nil }

    // MARK: Diagnostics: a plugin forwards its own counters here.

    public var statistics: EngineStatistics { .zero }
    public func resetStatistics() {}
    public func decodeExpertIO() -> (hits: UInt64, misses: UInt64, bytes: UInt64)? { nil }
    public func expertStreamingStatistics() -> ExpertStreamingStatistics { .zero }
    public var prefetchRingSummary: String? { nil }
    public var prefillTensorOpsLogLine: String { "\(TemplatePlugin.displayName): CPU counter" }
    public func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] { [] }
    public func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double) { (0, 0) }
    public func kernelGPUGaps() -> [(transition: String, millis: Double, count: Int)] { [] }
    public func resetKernelGPUTimings() {}
}
