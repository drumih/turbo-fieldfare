import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
import TinyTitanCore

/// T21: an interior checkpoint captured from a live runner must restore into a
/// fresh runner and continue the prompt with the same logits as the run that
/// never paused.
@Suite struct InteriorCheckpointRestoreTests {
    private let maxContext = 64
    private let vocab = 1024

    private func makeRunner(
        _ ctx: MetalContext
    ) throws -> (URL, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            slots: 1, runtimeConfiguration: runtime)
        return (dir, runner)
    }

    private func makeLogits(_ ctx: MetalContext) throws -> MTLBuffer {
        try #require(
            ctx.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
    }

    private func makeStore() throws -> PromptCheckpointStore {
        try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 1 << 20, diskDirectory: nil,
                diskLimitBytes: 0))
    }

    private func values(_ buffer: MTLBuffer) -> [Float16] {
        let pointer = buffer.contents().bindMemory(
            to: Float16.self, capacity: vocab)
        return (0..<vocab).map { pointer[$0] }
    }

    private func domain(_ template: String = "template")
        -> ServerPromptCacheDomain
    {
        ServerPromptCacheDomain(
            modelID: "qwen-toy",
            sourceSnapshotHash: "source",
            runtimeProfileHash: "profile",
            maximumContext: maxContext,
            kvStorage: "int8",
            fp16RingEnabled: false,
            templateSHA256: template)
    }

    @Test func restoredCheckpointContinuesIdentically() async throws {
        let ctx = try MetalContext()
        let tokens: [Int32] = [11, 7, 5, 3, 9, 13, 17]
        let boundary = 3
        let tokensPerChunk = PrefillRuntimeConfig.production(chunkTokens: 32)
        let logits = try makeLogits(ctx)

        // Reference: one uninterrupted prefill over the whole prompt.
        let (referenceDir, reference) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: referenceDir) }
        _ = try await reference.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: tokensPerChunk, into: logits, onProgress: { _ in })
        let expected = values(logits)

        // Capture a checkpoint at the boundary from a second run that prefills
        // only the prefix, then continue it to produce the same logits.
        let (captureDir, capture) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: captureDir) }
        _ = try await capture.prefillChunked(
            tokens: tokens[0..<boundary], startPosition: 0, outputMode: .logits,
            config: tokensPerChunk, into: logits, onProgress: { _ in })
        let snapshot = try capture.captureInferenceState()
        #expect(snapshot.descriptor.position == boundary)

        let store = try makeStore()
        let checkpoint = PromptCheckpointMetadata(
            key: PromptCheckpointKey.make(
                domain: domain(), prefixTokenIDs: Array(tokens[0..<boundary])),
            domain: domain(),
            kind: .lastTurnStart,
            conversationKey: "conversation",
            prefixTokenIDs: Array(tokens[0..<boundary]),
            descriptor: snapshot.descriptor)
        try await store.save(checkpoint, snapshot: snapshot)

        // The prefix's key never matches a different prefix.
        let otherKey = PromptCheckpointKey.make(
            domain: domain(), prefixTokenIDs: [11, 7, 6])
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(otherKey, domain: self.domain())
        }
        // A different runtime domain never matches either.
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(checkpoint.key, domain: self.domain("other"))
        }

        let loaded = try await store.load(checkpoint.key, domain: domain())
        let (restoredDir, restored) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: restoredDir) }
        try restored.restoreInferenceState(loaded.snapshot)
        #expect(restored.continuationPosition == boundary)
        _ = try await restored.prefillChunked(
            tokens: tokens[boundary...], startPosition: boundary,
            outputMode: .logits, config: tokensPerChunk, into: logits,
            onProgress: { _ in })
        #expect(values(logits) == expected)

        // The uninterrupted capture, continued without a restore, is the same
        // reference: the pause itself changes nothing.
        _ = try await capture.prefillChunked(
            tokens: tokens[boundary...], startPosition: boundary,
            outputMode: .logits, config: tokensPerChunk, into: logits,
            onProgress: { _ in })
        #expect(values(logits) == expected)
    }

    @Test func checkpointPositionMustMatchItsPrefix() async throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx)
        let tokens: [Int32] = [11, 7, 5, 3]
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        let snapshot = try runner.captureInferenceState()
        let metadata = PromptCheckpointMetadata(
            key: PromptCheckpointKey.make(
                domain: domain(), prefixTokenIDs: Array(tokens[0..<3])),
            domain: domain(),
            kind: .systemEnd,
            conversationKey: "conversation",
            prefixTokenIDs: Array(tokens[0..<3]),
            descriptor: snapshot.descriptor)
        let store = try makeStore()
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.save(metadata, snapshot: snapshot)
        }
    }

    /// Qwen3.8 needs the QSA indexer state restored beside the KV snapshot; a
    /// snapshot without it must still refuse rather than return a model that
    /// ranks blocks from the wrong prefix.
    @Test func qsaAuxiliaryStateRestoresTheSameContinuation() async throws {
        let ctx = try MetalContext()
        let tokens: [Int32] = [11, 7, 5, 3, 9, 13, 17]
        let boundary = 3
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)
        let logits = try makeLogits(ctx)
        let dir = try QwenToySynthetic.write(weightBits: 4, qsa: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToyQSA())
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)

        func runner() throws -> RealForwardRunner {
            try RealForwardRunner(
                model: model, context: ctx, maxContext: maxContext,
                slots: 1, runtimeConfiguration: runtime)
        }

        let reference = try runner()
        _ = try await reference.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: config, into: logits, onProgress: { _ in })
        let expected = values(logits)

        let capture = try runner()
        _ = try await capture.prefillChunked(
            tokens: tokens[0..<boundary], startPosition: 0,
            outputMode: .logits, config: config, into: logits,
            onProgress: { _ in })
        let snapshot = try capture.captureInferenceState()
        let auxiliary = try #require(try capture.captureAuxiliaryState())
        #expect(auxiliary.position == boundary)

        // A bare snapshot still refuses, and resets the runner rather than
        // leaving it half-restored.
        let refused = try runner()
        #expect(throws: InferenceStateSnapshotError.self) {
            try refused.restoreInferenceState(snapshot)
        }

        let restored = try runner()
        try restored.restoreInferenceState(snapshot, auxiliary: auxiliary)
        _ = try await restored.prefillChunked(
            tokens: tokens[boundary...], startPosition: boundary,
            outputMode: .logits, config: config, into: logits,
            onProgress: { _ in })
        #expect(values(logits) == expected)
    }
}
