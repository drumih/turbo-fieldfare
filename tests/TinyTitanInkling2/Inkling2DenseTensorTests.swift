import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2DenseTensorTests {
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

    @Test func readsBF16AndF32() throws {
        var bf16 = Data()
        for value in [Float(1), Float(-2.5)] {
            var bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16).littleEndian
            bf16.append(Data(bytes: &bits, count: 2))
        }
        var f32 = Data()
        for value in [Float(3.25)] {
            var bits = value.bitPattern.littleEndian
            f32.append(Data(bytes: &bits, count: 4))
        }
        let header: [String: Any] = [
            "b": ["dtype": "BF16", "shape": [2], "data_offsets": [0, 4]],
            "f": ["dtype": "F32", "shape": [1], "data_offsets": [4, 8]],
        ]
        let url = try write(header, data: bf16 + f32)
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try Inkling2Safetensors(url: url)
        #expect(try file.floatValues("b") == [1, -2.5])
        #expect(try file.floatValues("f") == [3.25])
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func readsARealNorm() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let file = try Inkling2Safetensors(
            url: URL(fileURLWithPath: path)
                .appendingPathComponent("model-00001-of-00001.safetensors"))
        let norm = try file.floatValues("model.llm.layers.0.attn_norm.weight")
        #expect(norm.count == 2304)
        #expect(norm.allSatisfy { $0.isFinite })
    }
}
