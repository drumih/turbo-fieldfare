import Foundation
import Testing

@testable import TinyTitan
import TinyTitanFormat

/// Writes one layer of routed experts as a `.ssdai` blob and reads it back the
/// way the runtime does: through `PackedExpertsLayoutReader`, then by offset
/// into the file. The assertion is weight equality — a blob the engine would
/// load must decode to exactly what the writer quantised.
@Suite struct GGUFExpertBlobWriterTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static let stride: UInt64 = 2_768_896  // 2,764,800 padded to 16 KiB
    private static let columns = 2_560
    private static let rows = 640
    private static let groups = 40

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func writesALayerTheRuntimeCanReadBack() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-blob-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        let layoutDirectory = temporary.appendingPathComponent("packed_experts")

        let result = try GGUFExpertBlobWriter.writeLayer(
            layer: 0, from: model, expertCount: 4, into: layoutDirectory,
            fileName: "layer_00.bin")
        #expect(result.expertStride == Self.stride)
        let attributes = try FileManager.default.attributesOfItem(atPath: result.fileURL.path)
        #expect((attributes[.size] as? NSNumber)?.uint64Value == 4 * Self.stride)

        let first = try #require(result.layer.experts.first)
        #expect(first.tensors["gate"]?.offset == 0)
        #expect(first.tensors["gate"]?.size == 819_200)
        #expect(first.tensors["gate"]?.shape == [640, 2_560])
        #expect(first.tensors["gate_scales"]?.offset == 819_200)
        #expect(first.tensors["gate_scales"]?.size == 51_200)
        #expect(first.tensors["gate_biases"]?.offset == 870_400)
        #expect(first.tensors["down"]?.offset == 1_843_200)
        #expect(first.tensors["down"]?.shape == [2_560, 640])
        #expect(first.tensors["down_biases"]?.offset == 2_713_600)
        #expect(first.size == Self.stride)

        try GGUFExpertBlobWriter.writeLayout(
            layers: [result.layer], expertStride: result.expertStride,
            numLayers: 1, expertsPerLayer: 4, into: layoutDirectory)
        let layout = try PackedExpertsLayoutReader.load(directoryURL: temporary)
        #expect(layout.expertStride == Self.stride)
        #expect(layout.layers.count == 1)
        #expect(layout.layers[0].file == "layer_00.bin")
        #expect(layout.layers[0].experts.count == 4)

        // Read the last expert's gate matrix back through the layout record and
        // compare it with the same quantise-then-dequantise applied to the GGUF.
        let expert = layout.layers[0].experts[3]
        let gate = try #require(expert.subTensors["gate"])
        let scales = try #require(expert.subTensors["gate_scales"])
        let biases = try #require(expert.subTensors["gate_biases"])
        let packedBytes = try Self.read(
            result.fileURL, offset: expert.offset + gate.offset, size: gate.size)
        let scaleBytes = try Self.read(
            result.fileURL, offset: expert.offset + scales.offset, size: scales.size)
        let biasBytes = try Self.read(
            result.fileURL, offset: expert.offset + biases.offset, size: biases.size)

        for row in [0, Self.rows - 1] {
            let fromDisk = Self.dequantizeRow(
                packed: packedBytes, scales: scaleBytes, biases: biasBytes, row: row,
                columns: Self.columns)
            let reference = try Self.referenceRow(model, expert: 3, row: row)
            #expect(fromDisk == reference, "row \(row) of expert 3 differs")
        }
    }

    private static func read(_ url: URL, offset: UInt64, size: UInt64) throws -> [UInt8] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        let data = try #require(try handle.read(upToCount: Int(size)))
        return [UInt8](data)
    }

    private static func dequantizeRow(
        packed: [UInt8], scales: [UInt8], biases: [UInt8], row: Int, columns: Int
    ) -> [Float] {
        let packedWidth = columns / 2
        let groupCount = columns / 64
        let packedStart = row * packedWidth
        let auxStart = row * groupCount * 2
        let scaleBits = (0..<groupCount).map {
            UInt16(scales[auxStart + $0 * 2]) | (UInt16(scales[auxStart + $0 * 2 + 1]) << 8)
        }
        let biasBits = (0..<groupCount).map {
            UInt16(biases[auxStart + $0 * 2]) | (UInt16(biases[auxStart + $0 * 2 + 1]) << 8)
        }
        let affine = Quantization.Int4AffineRow(
            packed: Array(packed[packedStart..<(packedStart + packedWidth)]),
            scales: scaleBits,
            biases: biasBits)
        return Quantization.dequantizeInt4Affine(affine, n: columns)
    }

    private static func referenceRow(
        _ model: GGUFModel, expert: Int, row: Int
    ) throws -> [Float] {
        let weights = try GGUFExpertLayout.weights(layer: 0, expert: expert, in: model)
        let handle = try FileHandle(forReadingFrom: model.shards[weights.gate.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: weights.gate.fileOffset)
        let data = try #require(try handle.read(upToCount: Int(weights.gate.byteCount)))
        let values = try GGUFDequant.decode(type: weights.gate.type, bytes: [UInt8](data))
        let start = row * columns
        let quantized = Quantization.quantizeInt4Affine(
            Array(values[start..<(start + columns)]))
        return Quantization.dequantizeInt4Affine(quantized, n: columns)
    }
}
