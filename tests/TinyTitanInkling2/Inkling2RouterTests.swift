import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2RouterTests {
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

    private func f32(_ values: [Float]) -> Data {
        var data = Data()
        for value in values {
            var bits = value.bitPattern.littleEndian
            data.append(Data(bytes: &bits, count: 4))
        }
        return data
    }

    @Test func selectsTheHighestScores() throws {
        // 8 experts, hidden 1: gate rows 0 and 7 win; bias 0.
        let gate = bf16([1, 2, 3, 4, 5, 6, 7, 8])
        let bias = f32(Array(repeating: 0, count: 8))
        let header: [String: Any] = [
            "model.llm.embed.weight": ["dtype": "BF16", "shape": [1, 1], "data_offsets": [0, 2]],
            "model.llm.layers.0.mlp.gate.weight": [
                "dtype": "BF16", "shape": [8, 1], "data_offsets": [2, 18],
            ],
            "model.llm.layers.0.mlp.gate.bias": [
                "dtype": "F32", "shape": [8], "data_offsets": [18, 50],
            ],
        ]
        let data = bf16([1]) + gate + bias
        let directory = try write(header, data: data)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = try Inkling2DenseTensors(directory: directory)
        let router = try Inkling2Router(tensors: tensors, layer: 0, topK: 2, routeScale: 8)
        let route = router.route([1])
        #expect(route.indices == [7, 6])
        #expect(abs(route.weights.reduce(0, +) - 8) < 1e-4)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func routesTheRealLayerZero() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        let router = try Inkling2Router(tensors: tensors, layer: 0)
        let route = router.route(try tensors.embeddingRow(0))
        #expect(route.indices.count == 8)
        #expect(Set(route.indices).count == 8)
        #expect(abs(route.weights.reduce(0, +) - 8) < 1e-4)
    }
}
