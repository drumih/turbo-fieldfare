import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2SafetensorsTests {
    private func write(_ header: [String: Any], data: Data) throws -> URL {
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(headerData)
        file.append(data)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    @Test func readsOneTensor() throws {
        let header: [String: Any] = [
            "__metadata__": ["format": "pt"],
            "t": ["dtype": "BF16", "shape": [2], "data_offsets": [0, 4]],
        ]
        let url = try write(header, data: Data([1, 2, 3, 4]))
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try Inkling2Safetensors(url: url)
        #expect(file.tensor("t")?.dtype == .bf16)
        #expect(file.tensor("t")?.shape == [2])
        #expect(try file.data(for: "t") == Data([1, 2, 3, 4]))
        #expect(throws: Inkling2Safetensors.Error.self) {
            try file.data(for: "missing")
        }
    }

    @Test func rejectsAMismatchedRange() throws {
        let header: [String: Any] = [
            "t": ["dtype": "BF16", "shape": [2], "data_offsets": [0, 3]]
        ]
        let url = try write(header, data: Data([1, 2, 3]))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: Inkling2Safetensors.Error.self) {
            try Inkling2Safetensors(url: url)
        }
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func readsTheRealHeader() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let file = try Inkling2Safetensors(
            url: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        let layer0 = try #require(
            file.tensor("model.llm.layers.0.mlp.experts.w13_weight"))
        let layer20 = try #require(
            file.tensor("model.llm.layers.20.mlp.experts.w13_weight"))
        #expect(layer0.dtype == .u8)
        #expect(layer0.shape == [128, 2304, 1152])
        #expect(layer20.dtype == .f8e4m3)
        #expect(layer20.shape == [128, 2304, 2304])
    }
}
