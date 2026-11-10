import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The derived fp16 source an ANE prefill sidecar needs: one full-attention
/// layer's six tensors, read from the GGUF's Q6_K blocks without converting the
/// checkpoint. The check is that every written fp16, read back through bf16,
/// is the arena's own bf16 value - so ordering, shape and transposition are all
/// pinned against the decode the decoder uses.
@Suite struct GGUFANEPrefillSourceTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func writesOneFullAttentionLayerAsFP16() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ane-source-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }

        let layer = 3
        let written = try GGUFANEPrefillSource.write(
            layer: layer, from: model, context: context, to: temporary)
        #expect(written["wq"] == 12_288 * 2_560)
        #expect(written["wk"] == 512 * 2_560)
        #expect(written["wv"] == 512 * 2_560)
        #expect(written["wo"] == 2_560 * 6_144)
        #expect(written["q_norm"] == 256 && written["k_norm"] == 256)

        let attributes = try FileManager.default.attributesOfItem(
            atPath: temporary.appendingPathComponent("wq.f16").path)
        #expect((attributes[.size] as? NSNumber)?.intValue == 12_288 * 2_560 * 2)

        let geometry = try JSONDecoder().decode(
            GGUFANEPrefillGeometry.self,
            from: try Data(contentsOf: temporary.appendingPathComponent("geometry.json")))
        #expect(geometry.hidden == 2_560)
        #expect(geometry.numHeads == 24)
        #expect(geometry.numKVHeads == 2)
        #expect(geometry.headDim == 256)
        #expect(geometry.rotaryDim == 64)
        #expect(geometry.chunk == 4_096)

        // Sample wq against the arena's bf16 through the file's fp16.
        let arena = try GGUFResidentArena.build(
            names: ["blk.\(layer).attn_q.weight"], from: model, context: context)
        let entry = try #require(arena.entries["blk.\(layer).attn_q.weight"])
        let source = arena.buffer.contents()
            .advanced(by: Int(entry.offset))
            .assumingMemoryBound(to: UInt16.self)
        let fileData = try Data(contentsOf: temporary.appendingPathComponent("wq.f16"))
        let fileHalves = fileData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }
        #expect(fileHalves.count == Int(entry.rows * entry.columns))
        for index in [0, 1, 2_559, 2_560, 123_456, fileHalves.count - 1] {
            let roundTripped = Quantization.bf16Bits(
                Float(Float16(bitPattern: fileHalves[index])))
            #expect(roundTripped == source[index], "index \(index)")
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func refusesAGDNLayer() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ane-source-refusal-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        #expect(throws: GGUFANEPrefillError.notFullAttention(0)) {
            _ = try GGUFANEPrefillSource.write(
                layer: 0, from: model, context: context, to: temporary)
        }
    }
}
