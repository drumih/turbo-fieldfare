import Foundation
import Testing

@testable import TinyTitan
import TinyTitanFormat

@Suite struct GGUFResidentWriterTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func read(_ url: URL, offset: UInt64, size: UInt64) throws -> [UInt8] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        return [UInt8](try #require(try handle.read(upToCount: Int(size))))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func writesAResidentFileTheEngineCanIndex() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let widths: [String: GGUFResidentWidth] = [
            "blk.0.ffn_gate_shexp.weight": .affine4,
            "blk.0.ffn_gate_inp.weight": .affine8,
            "blk.0.hc_attn_norm.weight": .bf16,
            "blk.0.ssm_a": .bf16,
        ]
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-resident-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let url = temporary.appendingPathComponent("model_weights.bin")
        let result = try GGUFResidentWriter.write(
            names: Array(widths.keys).sorted(), from: model, widths: widths, to: url)

        #expect(result.indexSize % SSDAIFormatV1.alignmentBytes == 0)
        #expect(result.entries.count == 4)
        let index = try ResidentIndexReader.load(fileURL: url)
        #expect(index.entries.count == 4)
        let entry = try #require(index.entries["blk.0.ffn_gate_shexp.weight"])
        #expect(entry.dtype == 0)
        #expect(entry.sizeBytes == 819_200)  // 640 x 2560 at 4 bits
        #expect(entry.scaleSize == 51_200)  // 640 x 40 groups, BF16
        #expect(entry.fileOffset >= index.header.indexSize)
        let eight = try #require(index.entries["blk.0.ffn_gate_inp.weight"])
        #expect(eight.dtype == 0)
        #expect(eight.sizeBytes == 1_310_720)  // 512 x 2560 at 8 bits
        let norm = try #require(index.entries["blk.0.hc_attn_norm.weight"])
        #expect(norm.dtype == 1)
        #expect(norm.sizeBytes == 20_480)
        #expect(norm.scaleSize == 0 && norm.biasSize == 0)

        // The 4-bit payload must decode to what the GGUF row quantises to.
        let packed = try Self.read(url, offset: entry.fileOffset, size: 1_280)
        let scales = try Self.read(url, offset: entry.scaleOffset, size: 80)
        let biases = try Self.read(url, offset: entry.biasOffset, size: 80)
        let scaleBits = (0..<40).map {
            UInt16(scales[$0 * 2]) | (UInt16(scales[$0 * 2 + 1]) << 8)
        }
        let biasBits = (0..<40).map {
            UInt16(biases[$0 * 2]) | (UInt16(biases[$0 * 2 + 1]) << 8)
        }
        let fromDisk = Quantization.dequantizeInt4Affine(
            Quantization.Int4AffineRow(packed: packed, scales: scaleBits, biases: biasBits),
            n: 2_560)
        let reference = try Self.referenceRow(model, row: 0)
        #expect(fromDisk == reference)
    }

    private static func referenceRow(_ model: GGUFModel, row: Int) throws -> [Float] {
        let location = try #require(model.tensor(named: "blk.0.ffn_gate_shexp.weight"))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let raw = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        let values = try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](raw))
        let start = row * 2_560
        return Quantization.dequantizeInt4Affine(
            Quantization.quantizeInt4Affine(Array(values[start..<(start + 2_560)])), n: 2_560)
    }
}
