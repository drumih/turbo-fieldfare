import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// End to end on the hybrid Qwen toy: a W-row decode-order verify pass must
/// leave every row with the residual scalar decode reaches for the same token
/// at the same position, byte for byte -- the last state before the head, so
/// the head then sees identical input. This is the property the wide verify
/// rests on; `MultiRowGEMVTests` cover the projections in isolation, this
/// covers their wiring into the GDN and gated full-attention layers and
/// everything around them. The toy is written with a varied lm_head and final
/// norm (`variedHead`): the default constant pattern makes every logit equal.
@Suite struct WideVerifyIdentityTests {
    static let vocab = 1024

    private func makeRunner(weightBits: Int) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(
            weightBits: weightBits, qsa: false, variedHead: true)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 64, slots: 1,
            runtimeConfiguration: runtime)
        return (dir, ctx, runner)
    }

    private func argmax(_ values: [UInt16]) -> Int32 {
        let floats = values.map { Float(Float16(bitPattern: $0)) }
        var best = 0
        for index in 1..<floats.count where floats[index] > floats[best] { best = index }
        return Int32(best)
    }

    /// `halves` fp16 values of `source` from `offset`, through a shared copy
    /// (the prefill scratch is device-private).
    private func read(
        _ ctx: MetalContext, _ source: MTLBuffer, offsetHalves: Int, halves: Int
    ) throws -> [UInt16] {
        let bytes = halves * MemoryLayout<Float16>.stride
        let staging = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        let blit = try #require(cb.makeBlitCommandEncoder())
        blit.copy(
            from: source, sourceOffset: offsetHalves * MemoryLayout<Float16>.stride,
            to: staging, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let values = staging.contents().assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: values, count: halves))
    }

    @Test(arguments: [(4, 2), (4, 5), (4, 9)] as [(Int, Int)])
    func verifyRowsMatchScalarDecodeResidual(weightBits: Int, rows: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: weightBits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let prompt: [Int32] = [11, 7, 3, 42]
        let drafts: [Int32] = Array([5, 9, 2, 14, 6, 300, 77, 8, 1000].prefix(rows))
        let logits = try #require(
            ctx.device.makeBuffer(
                length: Self.vocab * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let start = prompt.count
        let width = runner.residualWidth
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)

        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: config, into: logits, onProgress: { _ in })
        var reference: [[UInt16]] = []
        var referenceLogits: [[UInt16]] = []
        for (offset, token) in drafts.enumerated() {
            try await runner.produce(token: token, position: start + offset, into: logits)
            reference.append(try read(ctx, runner.hidden, offsetHalves: 0, halves: width))
            referenceLogits.append(try read(ctx, logits, offsetHalves: 0, halves: Self.vocab))
        }

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: config, into: logits, onProgress: { _ in })
        let rowLogits = try #require(
            ctx.device.makeBuffer(
                length: rows * Self.vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared))
        _ = try await runner.runVerifyGreedyRows(
            drafts, startPosition: start, rowLogits: rowLogits)
        let scratch = try runner.ensurePrefillScratch(config: config)
        for row in 0..<rows {
            let got = try read(ctx, scratch.hidden, offsetHalves: row * width, halves: width)
            // Distinct rows guard against comparing two untouched buffers.
            #expect(Set(reference[row]).count > 1, "row \(row): scalar decode left no residual")
            let differing = zip(got, reference[row]).filter { $0 != $1 }.count
            #expect(
                differing == 0,
                "\(weightBits)-bit W=\(rows) row \(row): \(differing) of \(width) residual values differ"
            )
            let gotLogits = try read(
                ctx, rowLogits, offsetHalves: row * Self.vocab, halves: Self.vocab)
            #expect(Set(referenceLogits[row]).count > 1, "row \(row): decode logits are flat")
            let differingLogits = zip(gotLogits, referenceLogits[row]).filter { $0 != $1 }.count
            #expect(
                differingLogits == 0,
                "W=\(rows) row \(row): \(differingLogits) of \(Self.vocab) logits differ")
        }

        // The greedy pass reads the same head into its own buffer and argmaxes
        // each row on the host.
        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0, outputMode: .logits,
            config: config, into: logits, onProgress: { _ in })
        let greedy = try await runner.runVerifyGreedyRows(drafts, startPosition: start)
        #expect(greedy.predictions == referenceLogits.map(argmax))
    }

    /// The W-row head on its own: random residual rows through
    /// `encodeVerifyHeadRows` against scalar decode's `encodeFinalNorm` and
    /// `encodeHeadGEMV` per row, for the 4-bit (int4) and 8-bit (affine) heads.
    @Test(arguments: [(4, 9), (8, 9), (4, 3), (8, 2)] as [(Int, Int)])
    func verifyHeadRowsMatchDecodeHead(weightBits: Int, rows: Int) throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: weightBits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let width = runner.residualWidth
        let D = runner.cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        var generator = MultiRowGEMVFixtures.Generator(seed: 0x4EAD_1000)
        let residual = try MultiRowGEMVFixtures.activations(
            device: ctx.device, count: rows * width, generator: &generator)
        let scratch = try runner.ensurePrefillScratch(config: .production(chunkTokens: 32))
        let expected = try MultiRowGEMVFixtures.zeroed(
            device: ctx.device, halves: rows * Self.vocab)
        let actual = try MultiRowGEMVFixtures.zeroed(
            device: ctx.device, halves: rows * Self.vocab)
        let finalNorm = try runner.model.finalNorm()
        let lm = try runner.model.lmHead()
        try MultiRowGEMVFixtures.run(ctx) { cb in
            try runner.encodeRowCopy(
                cb, from: residual, offset: 0, to: scratch.hidden, offset: 0,
                bytes: rows * width * half)
            try runner.encodeVerifyHeadRows(
                commandBuffer: cb, hidden: scratch.hidden, rows: rows,
                scratch: scratch, logits: actual)
            for row in 0..<rows {
                try runner.encodeRowCopy(
                    cb, from: residual, offset: row * width * half,
                    to: runner.hidden, offset: 0, bytes: width * half)
                try runner.encodeFinalNorm(
                    commandBuffer: cb, out: runner.normed, outOffset: 0,
                    fNorm: finalNorm, d: UInt32(D), eps: 1e-6)
                try runner.encodeHeadGEMV(
                    commandBuffer: cb,
                    weights: lm.buffer, weightsOffset: Int(lm.offset),
                    scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                    x: runner.normed, y: expected, yOffset: row * Self.vocab * half,
                    m: UInt32(Self.vocab), n: UInt32(D))
            }
        }
        let values = expected.contents().assumingMemoryBound(to: UInt16.self)
        #expect(Set(UnsafeBufferPointer(start: values, count: Self.vocab)).count > 1)
        MultiRowGEMVFixtures.expectIdentical(
            expected, actual, halves: rows * Self.vocab, "\(weightBits)-bit head W=\(rows)")
    }

    /// The prewarm compiles every pipeline the pass uses without touching the
    /// GPU, and a pass after it still matches (run by the identity test above).
    @Test func prewarmCompilesTheVerifyPipelines() throws {
        let (dir, _, runner) = try makeRunner(weightBits: 4)
        defer { try? FileManager.default.removeItem(at: dir) }
        try runner.prewarmWideVerifyPipelines()
        try runner.prewarmWideVerifyPipelines(rows: 2...2)
    }
}
