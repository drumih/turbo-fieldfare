import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// The chunked dense layer against the token path: same weights, same K/V
/// cache, same positions. The toy geometry is too short for the register-
/// resident GEMM (projections fall back to GEMV), so this pins the batching
/// orchestration -- norms over rows, per-token RoPE and causal attention,
/// contiguous K/V writes, flat elementwise stages, residuals -- while the GEMM
/// itself is covered by `AffineQuantGEMMTests`.
@Suite struct K2DenseBatchTests {
    private static func halves(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float16] {
        let base = buffer.contents()
        return (0..<count).map {
            base.load(
                fromByteOffset: offset + $0 * MemoryLayout<Float16>.stride,
                as: Float16.self)
        }
    }

    private static func copy(_ values: [Float16], into buffer: MTLBuffer, offset: Int = 0) {
        values.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(
                    buffer.contents() + offset, base,
                    values.count * MemoryLayout<Float16>.stride)
            }
        }
    }

    private static func values(_ count: Int, salt: Int) -> [Float16] {
        (0..<count).map { index in
            Float16(Float((index * 37 + salt * 11) % 29) / 14 - 1)
        }
    }

    /// A prefill chunk and a decode step have to agree about the cache, not
    /// just about their own outputs: the batch writes rows 0-3 and the token
    /// layer then decodes position 4 from them. Compared against a token-only
    /// run over the same five positions.
    @Test func chunkThenTokenSharesTheCache() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let d = config.hiddenSize
        let chunk = 4
        let decoderLayer = try K2DenseLayer(
            model: model, context: ctx, layer: 0, maxSequence: 8)
        let referenceLayer = try K2DenseLayer(
            model: model, context: ctx, layer: 0, maxSequence: 8)
        let shared = try K2DenseBatch(
            model: model, context: ctx, layer: 0,
            maxSequence: 8, maxTokens: chunk,
            kCache: decoderLayer.kCache, vCache: decoderLayer.vCache)

        guard
            let batchIn = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let batchOut = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let singleIn = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared),
            let singleOut = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared)
        else {
            Issue.record("could not allocate the layer buffers")
            return
        }

        let input = Self.values(chunk * d, salt: 0)
        let last = Self.values(d, salt: 9)

        // Reference: five positions through one token layer.
        var expectedLast: [Float16] = []
        for index in 0..<5 {
            let row = index < chunk ? Array(input[index * d..<(index + 1) * d]) : last
            Self.copy(row, into: singleIn)
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try referenceLayer.encode(
                commandBuffer: cb, hidden: singleIn, out: singleOut,
                position: UInt32(index))
            cb.commit()
            await cb.completed()
            if index == 4 { expectedLast = Self.halves(singleOut, offset: 0, count: d) }
        }

        // The chunk writes 0-3, then the token layer decodes 4 from the cache.
        Self.copy(input, into: batchIn)
        let batchCB = try #require(ctx.queue.makeCommandBuffer())
        try shared.encode(
            commandBuffer: batchCB, hidden: batchIn, out: batchOut,
            basePosition: 0, tokenCount: chunk)
        batchCB.commit()
        await batchCB.completed()
        #expect(batchCB.status == .completed)

        Self.copy(last, into: singleIn)
        let decodeCB = try #require(ctx.queue.makeCommandBuffer())
        try decoderLayer.encode(
            commandBuffer: decodeCB, hidden: singleIn, out: singleOut,
            position: 4)
        decodeCB.commit()
        await decodeCB.completed()
        #expect(decodeCB.status == .completed)

        let actual = Self.halves(singleOut, offset: 0, count: d).map { Float($0) }
        let reference = expectedLast.map { Float($0) }
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
        // The batch must not have allocated a second cache and ignored the
        // shared one: the decode result would then be wrong above, but assert
        // the buffers are the ones handed over.
        #expect(shared.kCache === decoderLayer.kCache)
    }

    @Test func chunkMatchesTheTokenPathAcrossTwoChunks() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let d = config.hiddenSize
        let chunk = 4
        let batch = try K2DenseBatch(
            model: model, context: ctx, layer: 0,
            maxSequence: 8, maxTokens: chunk)
        let token = try K2DenseLayer(
            model: model, context: ctx, layer: 0, maxSequence: 8)

        guard
            let batchIn = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let batchOut = ctx.device.makeBuffer(
                length: chunk * d * 2, options: .storageModeShared),
            let tokenIn = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared),
            let tokenOut = ctx.device.makeBuffer(
                length: d * 2, options: .storageModeShared)
        else {
            Issue.record("could not allocate the layer buffers")
            return
        }

        for base in stride(from: 0, to: 8, by: chunk) {
            let input = Self.values(chunk * d, salt: base)
            Self.copy(input, into: batchIn)

            let batchCB = try #require(ctx.queue.makeCommandBuffer())
            try batch.encode(
                commandBuffer: batchCB, hidden: batchIn, out: batchOut,
                basePosition: base, tokenCount: chunk)
            batchCB.commit()
            await batchCB.completed()
            #expect(batchCB.status == .completed)

            var expected: [Float16] = []
            for index in 0..<chunk {
                let row = Array(
                    input[index * d..<(index + 1) * d])
                Self.copy(row, into: tokenIn)
                let cb = try #require(ctx.queue.makeCommandBuffer())
                try token.encode(
                    commandBuffer: cb, hidden: tokenIn, out: tokenOut,
                    position: UInt32(base + index))
                cb.commit()
                await cb.completed()
                #expect(cb.status == .completed)
                expected += Self.halves(tokenOut, offset: 0, count: d)
            }

            let actual = Self.halves(
                batchOut, offset: 0, count: chunk * d
            ).map { Float($0) }
            let reference = expected.map { Float($0) }
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel) base=\(base)")
        }
    }
}
