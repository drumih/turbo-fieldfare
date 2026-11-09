import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The GGUF hyper-connection against the engine's own implementation on the
/// same inputs: the only difference is that ours reads the up/down projections
/// from the mapped Q8_0 blocks instead of a bf16 arena copy.
@Suite struct GGUFHyperConnectionTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func matchesTheEngineImplementation() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let names = [
            "blk.0.hc_attn_norm.weight", "blk.0.hc_attn_down.weight",
            "blk.0.hc_attn_up.weight", "blk.0.hc_attn_inject.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let mapped = try GGUFMappedShards(model: model, device: context.device)
        let dim = arch.hiddenSize
        let streams = arch.hyperConnections.count
        let wide = dim * streams

        var rng = SeedTree(0x4C4C).key("hc-compare")
        let residual = (0..<wide).map { _ in Float16(rng.uniform(-1, 1)) }
        let blockOut = (0..<dim).map { _ in Float16(rng.uniform(-1, 1)) }
        func makeBuffer(_ values: [Float16]) -> MTLBuffer? {
            let bits = values.map(\.bitPattern)
            return bits.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return nil }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }
        }
        guard
            let engineStreams = makeBuffer(residual),
            let oursStreams = makeBuffer(residual),
            let blockOutBuffer = makeBuffer(blockOut),
            let engineIn = context.device.makeBuffer(length: dim * 2, options: .storageModeShared),
            let oursIn = context.device.makeBuffer(length: dim * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("hc compare buffers") }

        // Engine path.
        func weights(_ name: String) throws -> HyperConnection.Weights {
            let entry = try #require(arena.entries[name])
            return HyperConnection.Weights(
                weights: arena.buffer, weightsOffset: Int(entry.offset),
                scales: arena.buffer, biases: arena.buffer, isBF16: true)
        }
        let engine = try HyperConnection(
            context: context, dim: dim, streams: streams,
            lowRank: arch.hyperConnections.lowRank, weightBits: 4)
        let normEntry = try #require(arena.entries["blk.0.hc_attn_norm.weight"])
        guard let engineCB = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try engine.encodeRead(
            commandBuffer: engineCB, streamsBuffer: engineStreams,
            hcNorm: arena.buffer, hcNormOffset: Int(normEntry.offset),
            down: try weights("blk.0.hc_attn_down.weight"),
            up: try weights("blk.0.hc_attn_up.weight"),
            blockInput: engineIn, eps: 1e-6)
        try engine.encodeWrite(
            commandBuffer: engineCB, streamsBuffer: engineStreams,
            inject: try weights("blk.0.hc_attn_inject.weight"),
            blockOut: blockOutBuffer)
        engineCB.commit()
        engineCB.waitUntilCompleted()
        if let error = engineCB.error { throw MetalError.invalidState("\(error)") }

        // GGUF path.
        let ours = try GGUFHyperConnection(
            context: context, arch: arch, arena: arena, mapped: mapped)
        guard let oursCB = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try ours.encodeRead(
            commandBuffer: oursCB, layer: 0, sublayer: "attn",
            streamsBuffer: oursStreams, out: oursIn)
        try ours.encodeWrite(
            commandBuffer: oursCB, layer: 0, sublayer: "attn",
            streamsBuffer: oursStreams, blockOut: blockOutBuffer)
        oursCB.commit()
        oursCB.waitUntilCompleted()
        if let error = oursCB.error { throw MetalError.invalidState("\(error)") }

        func relative(_ a: MTLBuffer, _ b: MTLBuffer, _ count: Int) -> Double {
            let pa = a.contents().assumingMemoryBound(to: UInt16.self)
            let pb = b.contents().assumingMemoryBound(to: UInt16.self)
            var sqDiff: Double = 0
            var sqRef: Double = 0
            for i in 0..<count {
                let va = Double(Float(Float16(bitPattern: pa[i])))
                let vb = Double(Float(Float16(bitPattern: pb[i])))
                sqDiff += (va - vb) * (va - vb)
                sqRef += vb * vb
            }
            return (sqDiff / max(sqRef, 1e-12)).squareRoot()
        }
        let inError = relative(oursIn, engineIn, dim)
        let streamError = relative(oursStreams, engineStreams, wide)
        print("GGUF hyper-connection vs engine: input \(inError), streams \(streamError)")
        #expect(inError < 5e-3)
        #expect(streamError < 5e-3)
    }
}
