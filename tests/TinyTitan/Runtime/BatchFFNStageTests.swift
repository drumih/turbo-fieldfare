import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// T34's per-row FFN pieces: the shared expert and the router must produce a
/// row's numbers from a slice of a batch buffer, so the fused decode loop can
/// keep the FFN per row while the attention stage is batched.
@Suite struct BatchFFNStageTests {
    private let layer = 1
    private let maxContext = 64

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
            slots: 2, runtimeConfiguration: runtime)
        return (dir, runner)
    }

    private func upload(
        _ ctx: MetalContext, values: [Float16], to destination: MTLBuffer
    ) throws {
        let bytes = values.count * 2
        let source = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(source.contents(), base, bytes) }
        }
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        blit.copy(
            from: source, sourceOffset: 0, to: destination,
            destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func download(
        _ ctx: MetalContext, from source: MTLBuffer, bytes: Int
    ) throws -> [Float16] {
        let destination = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        blit.copy(
            from: source, sourceOffset: 0, to: destination,
            destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = destination.contents().bindMemory(to: Float16.self, capacity: bytes / 2)
        return (0..<(bytes / 2)).map { p[$0] }
    }

    private func rowValues(row: Int, d: Int) -> [Float16] {
        (0..<d).map { i in
            Float16(sin(Float(row * 1000 + i) * 0.017) * 0.3)
        }
    }

    @Test func sharedExpertEncodesRowSlices() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize

        // Reference: each row through the decode loop's runner-wide form.
        var reference: [[Float16]] = []
        for row in 0..<2 {
            try upload(ctx, values: rowValues(row: row, d: d), to: runner.routedX)
            let cb = try runner.encodeAndCommitSharedExpert(
                layer: layer, completionClock: nil)
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            reference.append(try download(ctx, from: runner.h1Buf, bytes: d * 2))
        }

        let x = try #require(
            ctx.device.makeBuffer(length: 2 * d * 2, options: .storageModePrivate))
        let y = try #require(
            ctx.device.makeBuffer(length: 2 * d * 2, options: .storageModePrivate))
        var rows: [Float16] = []
        for row in 0..<2 { rows += rowValues(row: row, d: d) }
        try upload(ctx, values: rows, to: x)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        for row in 0..<2 {
            try runner.encodeSharedExpert(
                layer: layer,
                x: x, xOffset: row * d * 2,
                y: y, yOffset: row * d * 2,
                commandBuffer: cb)
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let got = try download(ctx, from: y, bytes: 2 * d * 2)

        for row in 0..<2 {
            for i in 0..<d {
                let value = got[row * d + i]
                let want = reference[row][i]
                #expect(
                    abs(Float(value) - Float(want)) < 1e-3,
                    "row \(row) element \(i): \(value) vs \(want)")
            }
        }
    }

    @Test func routerEncodesRowSlices() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize
        let topK = runner.cfg.topKExperts
        let half = MemoryLayout<Float16>.stride
        let routerW = try runner.model.router(layer: layer)
        let perExpertScale = try runner.requireOnesPerExpertScale()

        // Reference: the decode loop's runner-wide router, one row at a time.
        var refIndices: [[UInt32]] = []
        var refWeights: [[Float16]] = []
        for row in 0..<2 {
            try upload(ctx, values: rowValues(row: row, d: d), to: runner.routedX)
            guard let cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.moe.encodeRouter(
                commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                hidden: runner.routedX,
                effectiveScale: runner.effectiveScaleBuffers[layer],
                perExpertScale: perExpertScale, perExpertScaleOffset: 0,
                outIndices: runner.outIndices, outWeights: runner.outWeights,
                numExperts: UInt32(runner.cfg.numExperts), d: UInt32(d),
                topK: UInt32(topK))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let idx = runner.outIndices.contents().bindMemory(
                to: UInt32.self, capacity: topK)
            let w = runner.outWeights.contents().bindMemory(
                to: Float16.self, capacity: topK)
            refIndices.append((0..<topK).map { idx[$0] })
            refWeights.append((0..<topK).map { w[$0] })
        }

        let x = try #require(
            ctx.device.makeBuffer(length: 2 * d * 2, options: .storageModePrivate))
        var rows: [Float16] = []
        for row in 0..<2 { rows += rowValues(row: row, d: d) }
        try upload(ctx, values: rows, to: x)
        let indices = try #require(
            ctx.device.makeBuffer(
                length: 2 * topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
        let weights = try #require(
            ctx.device.makeBuffer(
                length: 2 * topK * half, options: .storageModeShared))

        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        for row in 0..<2 {
            try runner.moe.encodeRouter(
                commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                hidden: x, hiddenOffset: row * d * 2,
                effectiveScale: runner.effectiveScaleBuffers[layer],
                perExpertScale: perExpertScale, perExpertScaleOffset: 0,
                outIndices: indices,
                outIndicesOffset: row * topK * MemoryLayout<UInt32>.stride,
                outWeights: weights, outWeightsOffset: row * topK * half,
                numExperts: UInt32(runner.cfg.numExperts), d: UInt32(d),
                topK: UInt32(topK))
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let idx = indices.contents().bindMemory(
            to: UInt32.self, capacity: 2 * topK)
        let w = weights.contents().bindMemory(
            to: Float16.self, capacity: 2 * topK)
        for row in 0..<2 {
            for k in 0..<topK {
                #expect(idx[row * topK + k] == refIndices[row][k])
                let got = w[row * topK + k]
                let want = refWeights[row][k]
                #expect(
                    abs(Float(got) - Float(want)) < 1e-3,
                    "row \(row) k \(k): \(got) vs \(want)")
            }
        }
    }
}
