import Foundation
import Metal
import Testing

@testable import TinyTitan
import TinyTitanFormat

/// The resident arena: every non-expert tensor decoded to bf16 into one buffer.
/// The assertion is byte-for-byte agreement with the CPU decoders the GPU
/// kernels were already checked against.
@Suite struct GGUFResidentArenaTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func decodesResidentTensorsIntoOneArena() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let names = [
            "blk.3.attn_k.weight",  // Q6_K
            "blk.0.ffn_down_shexp.weight",  // Q8_0
            "blk.0.hc_attn_norm.weight",  // F32
        ]
        let arena = try GGUFResidentArena.build(
            names: names, from: model, context: context)

        #expect(arena.byteCount % SSDAIFormatV1.alignmentBytes == 0)
        for name in names {
            let entry = try #require(arena.entries[name])
            #expect(entry.offset % SSDAIFormatV1.alignmentBytes == 0)
            let location = try #require(model.tensor(named: name))
            #expect(entry.byteCount == location.tensor.elementCount * 2)

            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let raw = try #require(
                try handle.read(upToCount: Int(location.tensor.byteCount)))
            try handle.close()
            let reference = try GGUFDequant.decode(
                type: location.tensor.type, bytes: [UInt8](raw))
            let pointer = arena.buffer.contents()
                .advanced(by: Int(entry.offset))
                .assumingMemoryBound(to: UInt16.self)
            var mismatches = 0
            for index in 0..<reference.count
            where pointer[index] != Quantization.bf16Bits(reference[index]) {
                mismatches += 1
            }
            #expect(mismatches == 0, "\(name): \(mismatches) of \(reference.count) differ")
        }
        print(
            "GGUF resident arena: \(names.count) tensors, "
                + "\(arena.byteCount) bytes (\(String(format: "%.1f", Double(arena.byteCount) / 1e6)) MB)"
        )
    }
}
