import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// T34: the fused loop keeps GDN layers per row while the attention stage is
/// batched, so the linear-attention decode must read a row slice of the batch
/// input and write a row slice of its output without disturbing the recurrent
/// state of either slot.
@Suite struct BatchLinearAttentionStageTests {
    private let layer = 0
    private let maxContext = 64

    private func makeRunner(
        _ ctx: MetalContext, directory: URL
    ) throws -> RealForwardRunner {
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: .qwenToy())
        let runtime = try RuntimeConfiguration(
            forceLogitsHead: true, kvCachePrecision: .int8)
        return try RealForwardRunner(
            model: model, context: ctx, maxContext: maxContext,
            slots: 2, runtimeConfiguration: runtime)
    }

    private func upload(
        _ ctx: MetalContext, values: [Float16], to destination: MTLBuffer,
        offset: Int = 0
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
            destinationOffset: offset, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func download(
        _ ctx: MetalContext, from source: MTLBuffer, offset: Int = 0, bytes: Int
    ) throws -> [Float16] {
        let destination = try #require(
            ctx.device.makeBuffer(length: bytes, options: .storageModeShared))
        guard let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { throw MetalError.commandEncoderFailed }
        blit.copy(
            from: source, sourceOffset: offset, to: destination,
            destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let p = destination.contents().bindMemory(to: Float16.self, capacity: bytes / 2)
        return (0..<(bytes / 2)).map { p[$0] }
    }

    private func rowValues(slot: Int, position: Int, d: Int) -> [Float16] {
        (0..<d).map { i in
            Float16(
                sin(Float(slot * 10_000 + position * 37 + i) * 0.013) * 0.4)
        }
    }

    /// One step through the per-row entry norm + linear attention, writing
    /// into the runner's own `normed`/`oOut`.
    private func productionStep(
        _ ctx: MetalContext, runner: RealForwardRunner, slot: Int,
        position: Int
    ) throws {
        let d = runner.cfg.hiddenSize
        try upload(
            ctx, values: rowValues(slot: slot, position: position, d: d),
            to: runner.hidden)
        let norm = try runner.model.inputNorm(layer: layer)
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try runner.encodeResidualEntryDecode(
            commandBuffer: cb, hidden: runner.hidden,
            norm: norm, out: runner.normed,
            sublayer: .attention, layer: layer, eps: 1e-6)
        try runner.encodeLinearAttentionDecode(&cb, layer: layer, slot: slot)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
    }

    @Test func linearAttentionAcceptsRowSlices() throws {
        let ctx = try MetalContext()
        let directory = try QwenToySynthetic.write(weightBits: 4)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Two runners keep two independent GDN states: one follows the
        // production path, one follows the row-sliced path.
        let reference = try makeRunner(ctx, directory: directory)
        let sliced = try makeRunner(ctx, directory: directory)
        let d = sliced.cfg.hiddenSize
        let positions = [5, 3]

        for (slot, position) in positions.enumerated() {
            for t in 0..<position {
                try productionStep(
                    ctx, runner: reference, slot: slot, position: t)
                try productionStep(ctx, runner: sliced, slot: slot, position: t)
            }

            // Reference: the production buffer pair for this position.
            try productionStep(
                ctx, runner: reference, slot: slot, position: position)
            let want = try download(ctx, from: reference.oOut, bytes: d * 2)

            // Row-sliced: the same state, input from a slice of the batch
            // normed buffer, output into a slice of the batch attention buffer.
            let scratch = try sliced.ensureBatchedDecodeScratch(rows: 2)
            try upload(
                ctx,
                values: rowValues(slot: slot, position: position, d: d),
                to: scratch.hidden, offset: slot * d * 2)
            let norm = try sliced.model.inputNorm(layer: layer)
            guard var cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try sliced.encodeResidualEntryDecode(
                commandBuffer: cb,
                hidden: scratch.hidden, hiddenOffset: slot * d * 2,
                norm: norm,
                out: scratch.normed, outOffset: slot * d * 2,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try sliced.encodeLinearAttentionDecode(
                &cb, layer: layer, slot: slot,
                input: scratch.normed, inputOffset: slot * d * 2,
                output: scratch.o, outputOffset: slot * d * 2)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            let got = try download(
                ctx, from: scratch.o, offset: slot * d * 2, bytes: d * 2)

            for i in 0..<d {
                #expect(
                    abs(Float(got[i]) - Float(want[i])) < 1e-3,
                    "slot \(slot) element \(i): \(got[i]) vs \(want[i])")
            }
        }
    }
}
