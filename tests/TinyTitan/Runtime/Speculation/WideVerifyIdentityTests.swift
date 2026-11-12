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
/// everything around them. (The toy's head is degenerate -- every logit is
/// equal -- so logits could not tell a wrong row from a right one.)
@Suite struct WideVerifyIdentityTests {
    static let vocab = 1024

    private func makeRunner(weightBits: Int) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try QwenToySynthetic.write(weightBits: weightBits, qsa: false)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(forceLogitsHead: true)
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 64, slots: 1,
            runtimeConfiguration: runtime)
        return (dir, ctx, runner)
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
        for (offset, token) in drafts.enumerated() {
            try await runner.produce(token: token, position: start + offset, into: logits)
            reference.append(try read(ctx, runner.hidden, offsetHalves: 0, halves: width))
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
        }
    }
}
