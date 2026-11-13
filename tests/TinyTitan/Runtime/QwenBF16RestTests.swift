import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// A bf16-rest install (Qwen3.8 BF16Rest's shape: routed experts affine, every
/// dense matrix -- embedding and lm_head included -- at the checkpoint's bf16)
/// on the qwen toy runner: the widths the model reports, the bf16 embedding
/// lookups and head against a CPU reference, and decode, prefill and the
/// batched head end to end.
@Suite struct QwenBF16RestTests {
    private func makeRunner(
        bf16Rest: Bool = true, weightBits: Int = 4, slots: Int = 1
    ) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: weightBits, bf16Rest: bf16Rest)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 64, slots: slots,
            runtimeConfiguration: try RuntimeConfiguration(forceLogitsHead: true))
        return (dir, ctx, runner)
    }

    private func buffer(_ ctx: MetalContext, halves count: Int) throws -> MTLBuffer {
        try #require(Fp16Buffer.make(ctx.device, count: count))
    }

    /// The bf16 matrix `view` as floats, read straight from the resident bytes.
    private func table(_ view: TensorView) -> [Float] {
        let count = Int(view.length) / 2
        let base = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Float(bitPattern: UInt32(base[$0]) << 16) }
    }

    /// The width is the tensor's, not the slot's: a bf16 table reads 16 inside
    /// a 4-bit slot, and a quantized install reads exactly its slot as before.
    @Test(arguments: [(true, 4), (false, 4), (false, 8)])
    func modelReportsTheTensorsOwnWidths(bf16Rest: Bool, bits: Int) throws {
        let dir = try QwenToySynthetic.write(weightBits: bits, bf16Rest: bf16Rest)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        #expect(model.embeddingWeightBits == (bf16Rest ? 16 : bits))
        #expect(model.lmHeadWeightBits == (bf16Rest ? 16 : bits))
        #expect(try model.embedding().dtype == (bf16Rest ? 1 : 0))
        #expect(try model.lmHead().dtype == (bf16Rest ? 1 : 0))
    }

    /// `embed_lookup_bf16` and its prompt form write `half(bf16 * scale)` per
    /// element, exactly, and zeros for an out-of-range token.
    @Test(arguments: [Float(1), Float(8)])
    func embeddingLookupsMatchCPU(scale: Float) throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.bf16Rest.embedding != nil)
        let d = runner.cfg.hiddenSize
        let vocab = runner.cfg.vocabSize
        let rows = table(try runner.model.embedding())
        let tokens: [UInt32] = [0, 1, 517, UInt32(vocab - 1), UInt32(vocab)]
        func expected(_ token: UInt32) -> [Float] {
            guard token < UInt32(vocab) else { return [Float](repeating: 0, count: d) }
            let start = Int(token) * d
            return rows[start..<(start + d)].map { Float(Float16($0 * scale)) }
        }

        let row = try buffer(ctx, halves: d)
        for token in tokens {
            _ = try runner.runSync { cb in
                try runner.encodeEmbeddingLookup(
                    commandBuffer: cb, tokenId: token, out: row, outScale: scale)
            }
            #expect(Fp16Buffer.read(row, count: d) == expected(token), "token \(token)")
        }

        let tokenBuffer = try #require(
            ctx.device.makeBuffer(
                bytes: tokens, length: tokens.count * 4, options: .storageModeShared))
        let block = try buffer(ctx, halves: tokens.count * d)
        _ = try runner.runSync { cb in
            try runner.encodeEmbeddingBlock(
                commandBuffer: cb, tokens: tokenBuffer, out: block,
                t: tokens.count, outScale: scale)
        }
        #expect(Fp16Buffer.read(block, count: tokens.count * d) == tokens.flatMap(expected))
    }

    /// The bf16 head against a float dot product per vocabulary row, and its
    /// two-row and W-row forms against the single row byte for byte.
    @Test func bf16HeadMatchesCPUAndItsMultiRowForms() throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.headIsBF16)
        #expect(runner.affineHead == nil)
        #expect(runner.headGEMVFormat == .bf16)
        let d = runner.cfg.hiddenSize
        let vocab = runner.cfg.vocabSize
        let lm = try runner.model.lmHead()
        let weights = table(lm)
        let inputs: [[Float]] = (0..<3).map { row in
            (0..<d).map { Float(Float16(Float(($0 &* 7 &+ row &* 3) % 13 - 6) * 0.15)) }
        }
        let x = try #require(Fp16Buffer.make(ctx.device, values: inputs.flatMap { $0 }))
        let single = try buffer(ctx, halves: 3 * vocab)
        let paired = try buffer(ctx, halves: 2 * vocab)
        let multi = try buffer(ctx, halves: 3 * vocab)
        _ = try runner.runSync { cb in
            for row in 0..<3 {
                try runner.encodeHeadGEMV(
                    commandBuffer: cb,
                    weights: lm.buffer, weightsOffset: Int(lm.offset),
                    scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                    x: x, xOffset: row * d * 2, y: single, yOffset: row * vocab * 2,
                    m: UInt32(vocab), n: UInt32(d))
            }
            try runner.encodeHeadTwoRows(
                commandBuffer: cb,
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                x: x, xOffset: 0, y: paired, yOffset: 0,
                m: UInt32(vocab), n: UInt32(d))
            try runner.multiRowGEMV.encode(
                commandBuffer: cb,
                segments: [
                    .init(
                        view: lm, format: runner.headGEMVFormat, rows: vocab,
                        y: multi, yRowStride: vocab)
                ],
                x: x, xRowStride: d, activationRows: 3, columns: d)
        }
        let got = Fp16Buffer.read(single, count: 3 * vocab)
        var worst: Float = 0
        for row in 0..<3 {
            for r in 0..<vocab {
                var reference: Float = 0
                for k in 0..<d { reference += weights[r * d + k] * inputs[row][k] }
                worst = max(worst, abs(got[row * vocab + r] - reference))
            }
        }
        #expect(worst < 2e-3, "worst |bf16 head - CPU| = \(worst)")
        #expect(Set(got).count > 1, "the head must vary by row")
        #expect(Fp16Buffer.read(paired, count: 2 * vocab) == Array(got[0..<(2 * vocab)]))
        #expect(Fp16Buffer.read(multi, count: 3 * vocab) == got)
    }

    /// A chunked prefill of a prompt gives the logits decoding it token by
    /// token gives, on the bf16 embedding, projections and head.
    @Test func prefillMatchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab = runner.cfg.vocabSize
        let logits = try buffer(ctx, halves: vocab)
        let tokens: [Int32] = [11, 7, 5, 3, 9]
        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let reference = Fp16Buffer.read(logits, count: vocab)
        #expect(reference.allSatisfy { $0.isFinite })
        #expect(Set(reference).count > 1, "the logits must vary by row")

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
        let actual = Fp16Buffer.read(logits, count: vocab)
        for (index, (lhs, rhs)) in zip(actual, reference).enumerated() {
            #expect(abs(lhs - rhs) < 0.05, "logit \(index): \(lhs) vs \(rhs)")
        }
    }

    /// `produceBatch` runs two rows through the bf16 head at once; each row's
    /// logits must be what that row decodes to alone.
    @Test func produceBatchMatchesPerRowLogits() async throws {
        let rows: [(token: Int32, slot: Int)] = [(11, 0), (3, 1)]
        var alone: [[Float]] = []
        for row in rows {
            let (dir, ctx, runner) = try makeRunner()
            defer { try? FileManager.default.removeItem(at: dir) }
            let logits = try buffer(ctx, halves: runner.cfg.vocabSize)
            try await runner.produce(token: row.token, position: 0, into: logits)
            alone.append(Fp16Buffer.read(logits, count: runner.cfg.vocabSize))
        }
        let (dir, ctx, runner) = try makeRunner(slots: 2)
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab = runner.cfg.vocabSize
        let logits = [try buffer(ctx, halves: vocab), try buffer(ctx, halves: vocab)]
        try await runner.produceBatch(
            rows.map { (token: $0.token, position: 0, slot: $0.slot) }, logits: logits)
        for (index, buffer) in logits.enumerated() {
            #expect(Fp16Buffer.read(buffer, count: vocab) == alone[index], "row \(index)")
        }
    }
}
