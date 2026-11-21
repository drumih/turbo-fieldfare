import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2ExpertStoreTests {
    private struct TensorBlob {
        let dtype: String
        let shape: [Int]
        let bytes: Data
    }

    private func write(_ tensors: [String: TensorBlob], named name: String) throws -> URL {
        var header: [String: Any] = [:]
        var body = Data()
        for (key, tensor) in tensors {
            let start = body.count
            body.append(tensor.bytes)
            header[key] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [start, body.count],
            ]
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(headerData)
        file.append(body)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try file.write(to: directory.appendingPathComponent(name))
        return directory
    }

    private func floatBytes(_ values: [Float]) -> Data {
        var data = Data()
        for value in values {
            var bits = value.bitPattern.littleEndian
            data.append(Data(bytes: &bits, count: 4))
        }
        return data
    }

    private func nvfp4Tensors(layer: Int) -> [String: TensorBlob] {
        let prefix = "model.llm.layers.\(layer).mlp.experts"
        // 2 rows, 16 columns for w13 and 8 for w2; one 16-value block per row.
        var w13 = Data(repeating: 0, count: 32)
        w13[0] = 0x21  // 0.5, 1.0 with scale/outer 1
        var w2 = Data(repeating: 0, count: 16)
        w2[0] = 0x02  // 1.0
        return [
            "\(prefix).w13_weight": TensorBlob(dtype: "U8", shape: [1, 2, 16], bytes: w13),
            "\(prefix).w13_weight_scale": TensorBlob(
                dtype: "F8_E4M3", shape: [1, 2, 2], bytes: Data([0x38, 0x38, 0x38, 0x38])),
            "\(prefix).w13_weight_amax": TensorBlob(
                dtype: "F32", shape: [1], bytes: floatBytes([1])),
            "\(prefix).w13_weight_outer_scale_denominator": TensorBlob(
                dtype: "F32", shape: [1], bytes: floatBytes([1])),
            "\(prefix).w2_weight": TensorBlob(dtype: "U8", shape: [1, 2, 8], bytes: w2),
            "\(prefix).w2_weight_scale": TensorBlob(
                dtype: "F8_E4M3", shape: [1, 2, 1], bytes: Data([0x38, 0x38])),
            "\(prefix).w2_weight_amax": TensorBlob(
                dtype: "F32", shape: [1], bytes: floatBytes([1])),
            "\(prefix).w2_weight_outer_scale_denominator": TensorBlob(
                dtype: "F32", shape: [1], bytes: floatBytes([1])),
        ]
    }

    @Test func loadsOneNVFP4Expert() throws {
        let directory = try write(nvfp4Tensors(layer: 0), named: "model-00001-of-00001.safetensors")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try Inkling2ExpertStore(directory: directory)
        let expert = try store.load(layer: 0, expert: 0)
        #expect(expert.hidden == 2)
        #expect(expert.intermediate == 16)
        #expect(expert.w13.count == 64)
        #expect(expert.w2.count == 32)
        #expect(expert.w13[0] == 0.5)
        #expect(expert.w13[1] == 1.0)
        #expect(expert.w2[0] == 1.0)
    }

    @Test func loadsOneMXFP8Expert() throws {
        let prefix = "model.llm.layers.20.mlp.experts"
        let weights = Data(repeating: 0x38, count: 64)  // 2 rows x 32
        let w2Weights = Data(repeating: 0x38, count: 64)
        let tensors: [String: TensorBlob] = [
            "\(prefix).w13_weight": TensorBlob(
                dtype: "F8_E4M3", shape: [1, 2, 32], bytes: weights),
            "\(prefix).w13_weight_scale": TensorBlob(
                dtype: "U8", shape: [1, 2, 1], bytes: Data([127, 127])),
            "\(prefix).w2_weight": TensorBlob(
                dtype: "F8_E4M3", shape: [1, 2, 32], bytes: w2Weights),
            "\(prefix).w2_weight_scale": TensorBlob(
                dtype: "U8", shape: [1, 2, 1], bytes: Data([127, 127])),
        ]
        let directory = try write(tensors, named: "model-00001-of-00001.safetensors")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try Inkling2ExpertStore(directory: directory)
        let expert = try store.load(layer: 20, expert: 0)
        #expect(expert.intermediate == 32)
        #expect(expert.w13.count == 64)
        #expect(expert.w13.allSatisfy { $0 == 1 })
        #expect(expert.w2.allSatisfy { $0 == 1 })
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func loadsARealMXFP8Expert() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let store = try Inkling2ExpertStore(directory: URL(fileURLWithPath: path))
        let expert = try store.load(layer: 20, expert: 0)
        #expect(expert.w13.count == 2304 * 2304)
        #expect(expert.w2.count == 2304 * 1152)
        #expect(expert.w13.allSatisfy { $0.isFinite })
        #expect(expert.w2.allSatisfy { $0.isFinite })
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func loadsARealExpert() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let store = try Inkling2ExpertStore(directory: URL(fileURLWithPath: path))
        let expert = try store.load(layer: 0, expert: 0)
        #expect(expert.w13.count == 2304 * 2304)
        #expect(expert.w2.count == 2304 * 1152)
        #expect(expert.w13.allSatisfy { $0.isFinite })
        #expect(expert.w2.allSatisfy { $0.isFinite })
        // Independent Python dequant of the same bytes: low nibble first,
        // E4M3 block scale times amax[expert]/denominator.
        let w13Head: [Float] = [
            0.009372413, -0.056234475, 0.018744825, -0.018744825, -0.018744825,
        ]
        let w2Head: [Float] = [
            -0.010069003, 0.030207010, -0.040276013, 0.010069003, -0.010069003,
        ]
        for (index, expected) in w13Head.enumerated() {
            #expect(abs(expert.w13[index] - expected) < 1e-6)
        }
        for (index, expected) in w2Head.enumerated() {
            #expect(abs(expert.w2[index] - expected) < 1e-6)
        }
        #expect(abs(expert.w13.prefix(100).reduce(0, +) - (-0.012075993)) < 1e-5)
        #expect(abs(expert.w2.prefix(100).reduce(0, +) - 0.027330152) < 1e-5)
    }
}
