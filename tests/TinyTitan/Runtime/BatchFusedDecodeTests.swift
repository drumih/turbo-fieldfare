import Foundation
import Metal
import Testing

@testable import TinyTitan
import TinyTitanCore

/// T34: the fused width-2 loop must match the per-row decode step for the same
/// tokens, positions and slots, and the parity must come from the fused path
/// actually running.
@Suite struct BatchFusedDecodeTests {
    private let maxContext = 32

    private func makeRunner(
        _ ctx: MetalContext, slots: Int = 2
    ) throws -> (URL, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            slots: slots, runtimeConfiguration: runtime)
        return (dir, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        try #require(
            ctx.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
    }

    private func values(_ buffer: MTLBuffer, vocab: Int) -> [Float16] {
        let p = buffer.contents().bindMemory(to: Float16.self, capacity: vocab)
        return (0..<vocab).map { p[$0] }
    }

    @Test func fusedWidthTwoMatchesPerRow() async throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab = runner.cfg.vocabSize
        let steps: [[(Int32, Int, Int)]] = [
            [(11, 0, 0), (7, 0, 1)],
            [(5, 1, 0), (3, 1, 1)],
        ]

        // Reference: the per-row decode, one row at a time per step.
        var reference: [[[Float16]]] = []
        for step in steps {
            var rowLogits: [[Float16]] = []
            for (token, position, slot) in step {
                let logits = try makeLogits(ctx, vocab: vocab)
                try await runner.produce(
                    token: token, position: position, slot: slot,
                    into: logits)
                rowLogits.append(values(logits, vocab: vocab))
            }
            reference.append(rowLogits)
        }

        // Fused: the same tokens through the locked batch.
        runner.reset()
        runner.fusedDecodeOverride = true
        var fused: [[[Float16]]] = []
        for step in steps {
            var buffers: [MTLBuffer] = []
            for _ in step { buffers.append(try makeLogits(ctx, vocab: vocab)) }
            try await runner.produceBatch(
                step.map { (token: $0.0, position: $0.1, slot: $0.2) },
                logits: buffers)
            fused.append(buffers.map { values($0, vocab: vocab) })
        }

        #expect(runner.fusedBatchDecodeLaunches == steps.count)
        for (stepIndex, step) in steps.enumerated() {
            for row in step.indices {
                for i in 0..<vocab {
                    let got = fused[stepIndex][row][i]
                    let want = reference[stepIndex][row][i]
                    #expect(
                        abs(Float(got) - Float(want)) < 1e-3,
                        "step \(stepIndex) row \(row) logit \(i): \(got) vs \(want)")
                }
            }
        }
    }

    @Test func fusedWidthFourMatchesPerRow() async throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx, slots: 4)
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab = runner.cfg.vocabSize
        let steps: [[(Int32, Int, Int)]] = [
            [(11, 0, 0), (7, 0, 1), (9, 0, 2), (4, 0, 3)],
            [(5, 1, 0), (3, 1, 1), (8, 1, 2), (6, 1, 3)],
        ]

        var reference: [[[Float16]]] = []
        for step in steps {
            var rowLogits: [[Float16]] = []
            for (token, position, slot) in step {
                let logits = try makeLogits(ctx, vocab: vocab)
                try await runner.produce(
                    token: token, position: position, slot: slot,
                    into: logits)
                rowLogits.append(values(logits, vocab: vocab))
            }
            reference.append(rowLogits)
        }

        runner.reset()
        runner.fusedDecodeOverride = true
        var fused: [[[Float16]]] = []
        for step in steps {
            var buffers: [MTLBuffer] = []
            for _ in step { buffers.append(try makeLogits(ctx, vocab: vocab)) }
            try await runner.produceBatch(
                step.map { (token: $0.0, position: $0.1, slot: $0.2) },
                logits: buffers)
            fused.append(buffers.map { values($0, vocab: vocab) })
        }

        #expect(runner.fusedBatchDecodeLaunches == steps.count)
        for (stepIndex, step) in steps.enumerated() {
            for row in step.indices {
                for i in 0..<vocab {
                    let got = fused[stepIndex][row][i]
                    let want = reference[stepIndex][row][i]
                    #expect(
                        abs(Float(got) - Float(want)) < 1e-3,
                        "step \(stepIndex) row \(row) logit \(i): \(got) vs \(want)")
                }
            }
        }
    }
}
