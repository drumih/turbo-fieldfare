import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2DenseTensorsTests {
    private func write(_ header: [String: Any], data: Data) throws -> URL {
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(headerData)
        file.append(data)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try file.write(to: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        return directory
    }

    private func bf16(_ values: [Float]) -> Data {
        var data = Data()
        for value in values {
            var bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16).littleEndian
            data.append(Data(bytes: &bits, count: 2))
        }
        return data
    }

    @Test func readsAnEmbeddingRowAndNorm() throws {
        let header: [String: Any] = [
            "model.llm.embed.weight": [
                "dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12],
            ],
            "model.llm.embed_norm.weight": [
                "dtype": "BF16", "shape": [3], "data_offsets": [12, 18],
            ],
            "model.llm.norm.weight": ["dtype": "BF16", "shape": [3], "data_offsets": [18, 24]],
        ]
        let values: [Float] = [1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 2, 2]
        let directory = try write(header, data: bf16(values))
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = try Inkling2DenseTensors(directory: directory)
        #expect(tensors.vocabularySize == 2)
        #expect(tensors.hiddenSize == 3)
        #expect(try tensors.embeddingRow(1) == [4, 5, 6])
        #expect(try tensors.embeddingNorm() == [1, 1, 1])
        #expect(try tensors.finalNorm() == [2, 2, 2])
    }

    @Test func rmsNormIsNormalised() {
        let output = Inkling2Ops.rmsNorm([3, 4], weight: [1, 1])
        let norm = (output[0] * output[0] + output[1] * output[1]).squareRoot()
        #expect(abs(norm - sqrtf(2)) < 1e-5)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func readsTheRealEmbeddingRow() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        #expect(tensors.vocabularySize == 201_024)
        let row = try tensors.embeddingRow(0)
        #expect(row.count == 2304)
        #expect(row.allSatisfy { $0.isFinite })
    }
}
