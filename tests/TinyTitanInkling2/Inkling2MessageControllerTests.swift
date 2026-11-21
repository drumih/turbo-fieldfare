import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2MessageControllerTests {
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

    @Test func zeroGatesReturnTheInput() throws {
        let hidden = 2
        let rank = 1
        let prefix = "model.llm.layers.6.controllers.attn."
        let header: [String: Any] = [
            "model.llm.embed.weight": [
                "dtype": "BF16", "shape": [1, hidden], "data_offsets": [0, 4],
            ],
            prefix + "down_weight": [
                "dtype": "BF16", "shape": [rank, 2 * hidden],
                "data_offsets": [4, 4 + 2 * (2 * hidden)],
            ],
            prefix + "forget_weight": [
                "dtype": "BF16", "shape": [hidden, rank],
                "data_offsets": [4 + 2 * (2 * hidden), 4 + 2 * (2 * hidden) + 2 * hidden],
            ],
            prefix + "gain_weight": [
                "dtype": "BF16", "shape": [hidden, rank],
                "data_offsets": [
                    4 + 2 * (2 * hidden) + 2 * hidden, 4 + 2 * (2 * hidden) + 4 * hidden,
                ],
            ],
        ]
        var body = bf16([1, 0])
        body.append(bf16(Array(repeating: 0, count: 2 * hidden)))
        body.append(bf16(Array(repeating: 0, count: hidden)))
        body.append(bf16(Array(repeating: 0, count: hidden)))
        let directory = try write(header, data: body)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = try Inkling2DenseTensors(directory: directory)
        let controller = try Inkling2MessageController(
            tensors: tensors, layer: 6, name: "attn")
        let output = controller.forward([1, 2], message: [9, 9])
        #expect(abs(output[0] - 1) < 1e-5)
        #expect(abs(output[1] - 2) < 1e-5)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerSixControllerRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        let controller = try Inkling2MessageController(
            tensors: tensors, layer: 6, name: "attn")
        let x = try tensors.embeddingRow(0)
        let output = controller.forward(x, message: x)
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
    }
}
