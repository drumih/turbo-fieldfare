import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The rollback window end to end on a real runner: the qwen toy's hybrid
/// stack (linear layers 0 and 2, gated full attention on 1 and 3) runs a
/// decode-order verify pass through `runVerifyGreedyRows`, which opens the
/// window itself and retains each row from `encodeLinearAttentionDecodeOrderRows`.
/// Rolled back to k rows, the runner's whole KV and GDN snapshot must equal
/// the one k scalar decode steps leave from the same starting state.
@Suite struct WideVerifyRollbackRunnerTests {
    static let prompt: [Int32] = [1, 2, 3, 4, 5, 6]
    static let draft: [Int32] = [9, 14, 3, 27, 8]

    private func makeRunner() throws -> (URL, MetalContext, RealForwardRunner) {
        // Varied GDN weights: with the toy's uniform ones every qkv channel and
        // both of `a` and `b` are equal, and a mixed-up retention goes unseen.
        let dir = try QwenToySynthetic.write(weightBits: 4, variedLinearAttention: true)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 64, slots: 1,
            runtimeConfiguration: runtime)
        return (dir, ctx, runner)
    }

    @Test(arguments: [1, 3])
    func decodeOrderPassRolledBackEqualsDecodeSteps(accepted: Int) async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try #require(
            ctx.device.makeBuffer(
                length: runner.cfg.vocabSize * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
        _ = try await runner.prefillChunked(
            tokens: Self.prompt[...], startPosition: 0,
            outputMode: .greedyIfAvailable, config: .production(chunkTokens: 32),
            into: logits, onProgress: { _ in })
        let start = Self.prompt.count
        let primed = try runner.captureInferenceState()

        // Reference: k scalar decode steps.
        for (offset, token) in Self.draft.prefix(accepted).enumerated() {
            try await runner.produce(token: token, position: start + offset, into: logits)
        }
        let want = try runner.captureInferenceState()

        // The pass opens the window and leaves it open with every linear
        // layer's rows retained, one row at a time by the decode-order path.
        try runner.restoreInferenceState(primed)
        try runner.prepareForContinuation(expectedPosition: start)
        _ = try await runner.runVerifyGreedyRows(Self.draft, startPosition: start)
        let window = try #require(runner.fusedDecodeState.wideVerifyRollback)
        #expect(window.isOpen)
        let linear = (0..<runner.cfg.numLayers).filter {
            runner.gdnState?.isLinear(layer: $0) ?? false
        }
        #expect(!linear.isEmpty)
        for layer in linear {
            #expect(window.retainedRows(layer: layer) == Self.draft.count)
        }
        let afterPass = try runner.captureInferenceState()
        #expect(afterPass.descriptor.position == start + Self.draft.count)

        try runner.rollbackWideVerify(
            acceptedRows: accepted, tokens: Self.draft, startPosition: start)
        #expect(!window.isOpen)
        let got = try runner.captureInferenceState()
        #expect(got.descriptor == want.descriptor)
        let mismatch = got.payload.indices.first { got.payload[$0] != want.payload[$0] }
        #expect(
            got.payload == want.payload,
            "first differing payload byte \(mismatch.map(String.init) ?? "-") of \(got.payload.count)"
        )
    }

    /// A caller that never rolls back leaves no live window behind once it
    /// restores its snapshot, as the probe and pass-cost benches do.
    @Test func restoreClosesAWindowNobodyRolledBack() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try #require(
            ctx.device.makeBuffer(
                length: runner.cfg.vocabSize * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
        _ = try await runner.prefillChunked(
            tokens: Self.prompt[...], startPosition: 0,
            outputMode: .greedyIfAvailable, config: .production(chunkTokens: 32),
            into: logits, onProgress: { _ in })
        let primed = try runner.captureInferenceState()
        try runner.prepareForContinuation(expectedPosition: Self.prompt.count)
        _ = try await runner.runVerifyGreedyRows(Self.draft, startPosition: Self.prompt.count)
        #expect(runner.fusedDecodeState.wideVerifyRollback?.isOpen == true)
        try runner.restoreInferenceState(primed)
        #expect(runner.fusedDecodeState.wideVerifyRollback?.isOpen == false)
    }
}
