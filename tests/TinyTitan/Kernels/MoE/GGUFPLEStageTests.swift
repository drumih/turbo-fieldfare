import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The PLE stage on the real checkpoint: the rows the hash selects are gathered
/// bit-exactly, and the block injects a finite, non-zero term into the streams.
@Suite struct GGUFPLEStageTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func gathersTheHashedRowsBitExactly() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let layer = 1
        let names = [
            "blk.\(layer).ple_key.weight", "blk.\(layer).ple_value.weight",
            "blk.\(layer).ple_conv1d.weight", "blk.\(layer).ple_norm_key.weight",
            "blk.\(layer).ple_norm_query.weight", "blk.\(layer).ple_norm_conv.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let stage = try GGUFPLEStage(
            context: context, arch: arch, model: model, arena: arena)
        let tokens: [Int32] = [100, 200, 300]
        try stage.gather(context: tokens)
        let gathered = stage.gatheredBits()

        // Independent path: the same rows, decoded directly.
        let setup = try GGUFPLESetup.read(from: model)
        let rows = setup.hash.rows(context: tokens)
        let location = try #require(model.tensor(named: "per_layer_token_embd.weight"))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        let rowBytes = Int(location.tensor.dimensions[0]) / 32 * 18
        var expected: [UInt16] = []
        for row in rows {
            try handle.seek(
                toOffset: shard.dataOffset + location.tensor.offset
                    + UInt64(row) * UInt64(rowBytes))
            let data = try #require(try handle.read(upToCount: rowBytes))
            let values = try GGUFDequant.decode(type: .iq4NL, bytes: [UInt8](data))
            for value in values { expected.append(Float16(value).bitPattern) }
        }
        #expect(gathered == expected)

        // The block must change the streams and keep them finite.
        var rng = SeedTree(0x91E5).key("ple-streams")
        let residual = (0..<(arch.hyperConnections.count * arch.hiddenSize))
            .map { _ in Float16(rng.uniform(-1, 1)) }
        let bits = residual.map(\.bitPattern)
        guard
            let streamsBuffer = bits.withUnsafeBytes({ raw in
                guard let base = raw.baseAddress else { return nil as MTLBuffer? }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("PLE streams") }
        try stage.encode(commandBuffer: cb, streamsBuffer: streamsBuffer)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }
        let after = streamsBuffer.contents().assumingMemoryBound(to: UInt16.self)
        var changed = 0
        var nonFinite = 0
        for index in 0..<residual.count {
            let value = Float(Float16(bitPattern: after[index]))
            if !value.isFinite { nonFinite += 1 }
            if Float16(bitPattern: after[index]) != residual[index] { changed += 1 }
        }
        print(
            "PLE stage: \(changed)/\(residual.count) stream values changed, \(nonFinite) non-finite"
        )
        #expect(nonFinite == 0)
        #expect(changed > residual.count / 2)
    }
}
