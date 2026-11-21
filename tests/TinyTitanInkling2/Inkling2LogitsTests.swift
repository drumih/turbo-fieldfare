import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2LogitsTests {
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

    @Test func dotProductsTheUnembedRows() throws {
        let header: [String: Any] = [
            "model.llm.embed.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]],
            "model.llm.unembed.weight": [
                "dtype": "BF16", "shape": [2, 2], "data_offsets": [4, 12],
            ],
        ]
        let directory = try write(header, data: bf16([1, 0, 1, 2, 3, 4]))
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = try Inkling2DenseTensors(directory: directory)
        #expect(try tensors.logits([1, 2]) == [5, 11])
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLogitsAreFinite() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        let logits = try tensors.logits(tensors.embeddingRow(0))
        #expect(logits.count == 201_024)
        #expect(logits.allSatisfy { $0.isFinite })
    }
}
