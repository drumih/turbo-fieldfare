import Foundation
import Metal
import Testing
@testable import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib

/// T22/T23: an edited last turn resumes from the longest matching interior
/// checkpoint instead of regenerating the conversation before it. The restored
/// run must produce the same logits as an uninterrupted prefill of the edited
/// prompt, and a changed prefix must fall back to the next checkpoint (or a
/// full prefill), never restore the wrong one.
@Suite struct EditedTurnCheckpointTests {
    private let maxContext = 64
    private let vocab = 1024
    private let boundary = 9
    private let systemEnd = 3

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

    private func values(_ buffer: MTLBuffer) -> [Float16] {
        let pointer = buffer.contents().bindMemory(
            to: Float16.self, capacity: vocab)
        return (0..<vocab).map { pointer[$0] }
    }

    private func domain() -> ServerPromptCacheDomain {
        ServerPromptCacheDomain(
            modelID: "qwen-toy",
            sourceSnapshotHash: "source",
            runtimeProfileHash: "profile",
            maximumContext: maxContext,
            kvStorage: "int8",
            fp16RingEnabled: false,
            templateSHA256: "template")
    }

    private func makeStore() throws -> PromptCheckpointStore {
        try PromptCheckpointStore(
            configuration: ServerPromptCacheStorageConfiguration(
                memoryLimitBytes: 1 << 20, diskDirectory: nil,
                diskLimitBytes: 0))
    }

    private func prefill(
        _ runner: RealForwardRunner,
        _ tokens: [Int32],
        from start: Int,
        logits: MTLBuffer
    ) async throws {
        _ = try await runner.prefillChunked(
            tokens: tokens[start...],
            startPosition: start,
            outputMode: .logits,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
    }

    private func capture(
        _ runner: RealForwardRunner,
        prompt: [Int32],
        position: Int,
        kind: PromptCheckpointKind,
        store: PromptCheckpointStore,
        domain: ServerPromptCacheDomain
    ) async throws {
        let snapshot = try runner.captureInferenceState()
        #expect(snapshot.descriptor.position == position)
        let prefix = Array(prompt.prefix(position))
        let metadata = PromptCheckpointMetadata(
            key: PromptCheckpointKey.make(
                domain: domain, prefixTokenIDs: prefix),
            domain: domain,
            kind: kind,
            conversationKey: "conversation",
            prefixTokenIDs: prefix,
            descriptor: snapshot.descriptor)
        try await store.save(metadata, snapshot: snapshot)
    }

    @Test func editedTurnResumesFromTheLastTurnStart() async throws {
        let ctx = try MetalContext()
        let store = try makeStore()
        let logits = try makeLogits(ctx)
        let prefix: [Int32] = [101, 102, 103, 104, 105, 106, 107, 108]
        let oldTurn: [Int32] = [109, 110, 111, 112]
        let generation: [Int32] = [116]
        let oldPrompt = prefix + oldTurn + generation
        let editedPrompt =
            prefix + [201, 202, 203, 204] + generation

        // Reference: the edited prompt prefilled in one uninterrupted run.
        let (referenceDir, reference) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: referenceDir) }
        try await prefill(
            reference, editedPrompt, from: 0, logits: logits)
        let expected = values(logits)

        // Capture the old prompt's last-turn-start checkpoint from a runner
        // paused at exactly the boundary the server hook pauses at.
        let (prefixDir, prefixRunner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: prefixDir) }
        _ = try await prefixRunner.prefillChunked(
            tokens: oldPrompt[0..<boundary], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32),
            into: logits, onProgress: { _ in })
        try await capture(
            prefixRunner, prompt: oldPrompt, position: boundary,
            kind: .lastTurnStart, store: store, domain: domain())

        // A changed last turn makes the old key a miss; the checkpoint is
        // keyed by the exact prefix tokens, not by "last turn".
        let editedKey = PromptCheckpointKey.make(
            domain: domain(), prefixTokenIDs: Array(editedPrompt.prefix(boundary)))
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(editedKey, domain: self.domain())
        }

        // The old key still restores, and the edited suffix runs from it.
        let (restoredDir, restored) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: restoredDir) }
        let loaded = try await store.load(
            PromptCheckpointKey.make(
                domain: domain(), prefixTokenIDs: Array(oldPrompt.prefix(boundary))),
            domain: domain())
        try restored.restoreInferenceState(loaded.snapshot)
        #expect(restored.continuationPosition == boundary)
        try await prefill(restored, editedPrompt, from: boundary, logits: logits)
        #expect(values(logits) == expected)
    }

    @Test func changedPriorTurnFallsBackToTheSystemEndCheckpoint() async throws {
        let ctx = try MetalContext()
        let store = try makeStore()
        let logits = try makeLogits(ctx)
        let oldPrompt: [Int32] = [
            101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 116,
        ]
        // The second user turn changed too, so the last-turn-start checkpoint
        // cannot apply; only the system-end prefix is still common.
        let editedPrompt: [Int32] = [
            101, 102, 103, 104, 105, 106, 107, 108,
            211, 212, 213, 214, 116,
        ]
        let (referenceDir, reference) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: referenceDir) }
        try await prefill(reference, editedPrompt, from: 0, logits: logits)
        let expected = values(logits)

        let (captureDir, captureRunner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: captureDir) }
        // System end is the end of the system block; the prompt's first
        // boundary. Capture it with a prefix prefill.
        _ = try await captureRunner.prefillChunked(
            tokens: oldPrompt[0..<systemEnd], startPosition: 0,
            outputMode: .logits, config: .production(chunkTokens: 32),
            into: logits, onProgress: { _ in })
        try await capture(
            captureRunner, prompt: oldPrompt, position: systemEnd,
            kind: .systemEnd, store: store, domain: domain())

        let (restoredDir, restored) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: restoredDir) }
        let loaded = try await store.load(
            PromptCheckpointKey.make(
                domain: domain(),
                prefixTokenIDs: Array(oldPrompt.prefix(systemEnd))),
            domain: domain())
        try restored.restoreInferenceState(loaded.snapshot)
        #expect(restored.continuationPosition == systemEnd)
        try await prefill(
            restored, editedPrompt, from: systemEnd, logits: logits)
        #expect(values(logits) == expected)
    }

    @Test func noMatchingPrefixMeansFullPrefill() async throws {
        let store = try makeStore()
        let prompt: [Int32] = [101, 102, 103, 104, 105, 106, 107, 108]
        let key = PromptCheckpointKey.make(
            domain: domain(), prefixTokenIDs: Array(prompt.prefix(3)))
        await #expect(throws: PromptCheckpointStoreError.self) {
            try await store.load(key, domain: self.domain())
        }
    }
}
