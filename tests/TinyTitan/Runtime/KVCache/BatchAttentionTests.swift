import Foundation
import Metal
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

/// T5: the batched QKV/o_proj stage must produce each row's attention output
/// through that row's own KV slot, matching the per-row gated attention path.
@Suite struct BatchAttentionTests {
    private let layer = 3
    private let maxContext = 64

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

    @Test func batchMatchesPerRowGatedAttention() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize
        let scratch = try runner.ensureBatchedDecodeScratch(rows: 2)
        var hidden = [Float16](repeating: 0, count: 2 * d)
        for row in 0..<2 {
            for i in 0..<d {
                hidden[row * d + i] = Float16(
                    sin(Float(row * 1000 + i) * 0.017) * 0.3)
            }
        }
        try upload(ctx, values: hidden, to: scratch.hidden)
        let rows = [(position: 0, slot: 0), (position: 0, slot: 1)]
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try runner.encodeBatchGatedAttention(
            layer: layer, rows: rows, scratch: scratch, commandBuffer: &cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let batchOut = try download(
            ctx, from: scratch.o, bytes: 2 * d * 2)

        // Reference: the same two rows through the normal per-row path.
        runner.reset()
        var reference: [[Float16]] = []
        for row in 0..<2 {
            let staging = try #require(
                ctx.device.makeBuffer(
                    length: d * 2, options: .storageModeShared))
            hidden.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    memcpy(
                        staging.contents(),
                        base.advanced(by: row * d * 2), d * 2)
                }
            }
            guard let copy = ctx.queue.makeCommandBuffer(),
                let blit = copy.makeBlitCommandEncoder()
            else { throw MetalError.commandEncoderFailed }
            blit.copy(
                from: staging, sourceOffset: 0, to: runner.hidden,
                destinationOffset: 0, size: d * 2)
            blit.endEncoding()
            copy.commit()
            copy.waitUntilCompleted()
            let norm = try runner.model.inputNorm(layer: layer)
            guard var refCB = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.encodeResidualEntryDecode(
                commandBuffer: refCB, hidden: runner.hidden,
                norm: norm, out: runner.normed,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try runner.encodeGatedFullAttentionDecode(
                &refCB, layer: layer, position: 0, slot: row,
                seqLen: 1)
            refCB.commit()
            refCB.waitUntilCompleted()
            #expect(refCB.error == nil)
            reference.append(try download(ctx, from: runner.oOut, bytes: d * 2))
        }
        for row in 0..<2 {
            for i in 0..<d {
                let got = batchOut[row * d + i]
                let want = reference[row][i]
                #expect(
                    abs(Float(got) - Float(want)) < 1e-3,
                    "row \(row) element \(i): \(got) vs \(want)")
            }
        }
    }

    /// T33: the batch stage must stay exact once the rows sit at different
    /// decode positions and read their own multi-token KV slots. The T5 test
    /// only covered position 0/seqLen 1, where every slot holds one row and
    /// the RoPE offset, `seqLen`, and KV read range are all trivial.
    @Test func batchMatchesPerRowAtDecodePositions() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx, slots: 4)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize
        let qDim = runner.cfg.numHeads * runner.cfg.fullHeadDim
        let positions = [17, 9, 23, 5]

        func rowValues(slot: Int, position: Int) -> [Float16] {
            (0..<d).map { i in
                Float16(
                    sin(Float(slot * 10_000 + position * 37 + i) * 0.013) * 0.4)
            }
        }

        func encodePerRow(slot: Int, position: Int, seqLen: UInt32) throws {
            try upload(
                ctx, values: rowValues(slot: slot, position: position),
                to: runner.hidden)
            let norm = try runner.model.inputNorm(layer: layer)
            guard var cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.encodeResidualEntryDecode(
                commandBuffer: cb, hidden: runner.hidden,
                norm: norm, out: runner.normed,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try runner.encodeGatedFullAttentionDecode(
                &cb, layer: layer, position: position, slot: slot,
                seqLen: seqLen)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
        }

        // Build each row's own KV history through the per-row path.
        for (slot, position) in positions.enumerated() {
            for t in 0..<position {
                try encodePerRow(slot: slot, position: t, seqLen: UInt32(t + 1))
            }
        }

        let scratch = try runner.ensureBatchedDecodeScratch(rows: positions.count)
        for (slot, position) in positions.enumerated() {
            try upload(
                ctx, values: rowValues(slot: slot, position: position),
                to: scratch.hidden, offset: slot * d * 2)
        }
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try runner.encodeBatchGatedAttention(
            layer: layer,
            rows: positions.enumerated().map { ($0.element, $0.offset) },
            scratch: scratch, commandBuffer: &cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let batchOut = try download(
            ctx, from: scratch.o, bytes: positions.count * d * 2)
        let batchAttn = try download(
            ctx, from: scratch.attn, bytes: positions.count * qDim * 2)

        for (slot, position) in positions.enumerated() {
            try encodePerRow(
                slot: slot, position: position, seqLen: UInt32(position + 1))
            let reference = try download(ctx, from: runner.oOut, bytes: d * 2)
            let referenceAttn = try download(
                ctx, from: runner.attnOut, bytes: qDim * 2)
            for i in 0..<d {
                let got = batchOut[slot * d + i]
                let want = reference[i]
                #expect(
                    abs(Float(got) - Float(want)) < 1e-3,
                    "row \(slot) element \(i): \(got) vs \(want)")
            }
            for i in 0..<qDim {
                let got = batchAttn[slot * qDim + i]
                let want = referenceAttn[i]
                #expect(
                    abs(Float(got) - Float(want)) < 1e-3,
                    "row \(slot) attn element \(i): \(got) vs \(want)")
            }
        }
    }

    /// T34 step 1: the batch attention block must leave each row's residual
    /// and post-attention-normed MLP input exactly where the per-row layer
    /// leaves them. This is the composition the fused loop needs before its
    /// per-row FFN stage.
    @Test func batchAttentionBlockMatchesPerRow() throws {
        let ctx = try MetalContext()
        let (dir, runner) = try makeRunner(ctx, slots: 2)
        defer { try? FileManager.default.removeItem(at: dir) }
        let d = runner.cfg.hiddenSize
        let positions = [7, 4]

        func rowValues(slot: Int, position: Int) -> [Float16] {
            (0..<d).map { i in
                Float16(
                    sin(Float(slot * 10_000 + position * 37 + i) * 0.013) * 0.4)
            }
        }

        func encodePerRow(slot: Int, position: Int) throws {
            try upload(
                ctx, values: rowValues(slot: slot, position: position),
                to: runner.hidden)
            let norm = try runner.model.inputNorm(layer: layer)
            guard var cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.encodeResidualEntryDecode(
                commandBuffer: cb, hidden: runner.hidden,
                norm: norm, out: runner.normed,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try runner.encodeGatedFullAttentionDecode(
                &cb, layer: layer, position: position, slot: slot,
                seqLen: UInt32(position + 1))
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
        }

        // Build both rows' KV history through the per-row path.
        for (slot, position) in positions.enumerated() {
            for t in 0..<position {
                try encodePerRow(slot: slot, position: t)
            }
        }

        let postAttn = try runner.model.postAttnNorm(layer: layer)

        // Reference: per-row entry norm -> gated attention -> residual exit
        // -> post-attention norm into the MLP input.
        var refHidden: [[Float16]] = []
        var refRouted: [[Float16]] = []
        for (slot, position) in positions.enumerated() {
            try upload(
                ctx, values: rowValues(slot: slot, position: position),
                to: runner.hidden)
            let norm = try runner.model.inputNorm(layer: layer)
            guard var cb = ctx.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try runner.encodeResidualEntryDecode(
                commandBuffer: cb, hidden: runner.hidden,
                norm: norm, out: runner.normed,
                sublayer: .attention, layer: layer, eps: 1e-6)
            try runner.encodeGatedFullAttentionDecode(
                &cb, layer: layer, position: position, slot: slot,
                seqLen: UInt32(position + 1))
            try runner.encodeResidualExitDecode(
                commandBuffer: cb, hidden: runner.hidden,
                delta: runner.oOut, sublayer: .attention, layer: layer)
            try runner.encodeResidualEntryDecode(
                commandBuffer: cb, hidden: runner.hidden,
                norm: postAttn, out: runner.routedX,
                sublayer: .mlp, layer: layer, eps: 1e-6)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(cb.error == nil)
            refHidden.append(try download(ctx, from: runner.hidden, bytes: d * 2))
            refRouted.append(try download(ctx, from: runner.routedX, bytes: d * 2))
        }

        let scratch = try runner.ensureBatchedDecodeScratch(rows: positions.count)
        for (slot, position) in positions.enumerated() {
            try upload(
                ctx, values: rowValues(slot: slot, position: position),
                to: scratch.hidden, offset: slot * d * 2)
        }
        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try runner.encodeBatchAttentionBlock(
            layer: layer,
            rows: positions.enumerated().map { ($0.element, $0.offset) },
            scratch: scratch, postAttnNorm: postAttn,
            commandBuffer: &cb)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let batchHidden = try download(
            ctx, from: scratch.hidden, bytes: positions.count * d * 2)
        let batchRouted = try download(
            ctx, from: scratch.routedX, bytes: positions.count * d * 2)

        for slot in positions.indices {
            for i in 0..<d {
                let hid = batchHidden[slot * d + i]
                let wantHid = refHidden[slot][i]
                #expect(
                    abs(Float(hid) - Float(wantHid)) < 1e-3,
                    "row \(slot) hidden element \(i): \(hid) vs \(wantHid)")
                let routed = batchRouted[slot * d + i]
                let wantRouted = refRouted[slot][i]
                #expect(
                    abs(Float(routed) - Float(wantRouted)) < 1e-3,
                    "row \(slot) routed element \(i): \(routed) vs \(wantRouted)")
            }
        }
    }
}
