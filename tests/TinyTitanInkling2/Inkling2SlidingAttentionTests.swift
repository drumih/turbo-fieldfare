import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2SlidingAttentionTests {
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

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func gpuAttentionMatchesCPU() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(
            directory: URL(fileURLWithPath: path))
        let cpu = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let gpu = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4, gpu: try Inkling2GPUExpert())
        let x = try tensors.embeddingRow(0)
        let cpuOutput = try cpu.forward(x, tokenCount: 1).values
        let gpuOutput = try gpu.forward(x, tokenCount: 1).values
        var maxDifference: Float = 0
        for index in 0..<cpuOutput.count {
            maxDifference = max(maxDifference, abs(cpuOutput[index] - gpuOutput[index]))
        }
        #expect(maxDifference < 1e-3)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerZeroRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let x = try tensors.embeddingRow(0)
        let result = try attention.forward(x, tokenCount: 1)
        #expect(result.values.count == 2304)
        #expect(result.values.allSatisfy { $0.isFinite })
        #expect(result.state.k.count == 1024 * 3)
        #expect(result.state.v.count == 1024 * 3)
    }

    @Test func oneTokenRuns() throws {
        let hidden = 2
        let heads = 1
        let dim = 2
        let prefix = "model.llm.layers.0.attn."
        var header: [String: Any] = [
            "model.llm.embed.weight": ["dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]]
        ]
        var body = bf16([1, 0])
        func add(_ name: String, shape: [Int], values: [Float]) {
            let start = body.count
            let data = bf16(values)
            body.append(data)
            header[name] = [
                "dtype": "BF16", "shape": shape, "data_offsets": [start, body.count],
            ]
        }
        add(prefix + "wq_du.weight", shape: [2, 2], values: [1, 0, 0, 1])
        add(prefix + "wk_dv.weight", shape: [2, 2], values: [1, 0, 0, 1])
        add(prefix + "wv_dv.weight", shape: [2, 2], values: [1, 0, 0, 1])
        add(prefix + "wr_du.weight", shape: [1, 2], values: [0, 0])
        add(prefix + "gate_attn.weight", shape: [2, 2], values: [0, 0, 0, 0])
        add(prefix + "attn_sink_proj.weight", shape: [1, 2], values: [0, 0])
        add(prefix + "wo_ud.weight", shape: [2, 2], values: [1, 0, 0, 1])
        add(prefix + "q_l2_clip.scale", shape: [2], values: [1, 1])
        add(prefix + "rel_l2_clip.scale", shape: [1], values: [1])
        add(prefix + "rel_logits_proj.proj", shape: [1, 2], values: [0, 0])
        add(prefix + "k_sconv.weight", shape: [2, 1, 4], values: Array(repeating: 0, count: 8))
        add(prefix + "v_sconv.weight", shape: [2, 1, 4], values: Array(repeating: 0, count: 8))
        let directory = try write(header, data: body)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensors = try Inkling2DenseTensors(directory: directory)
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: heads, kvHeads: 1, headDim: dim,
            relativeDim: 1, extent: 2, sliding: 2, hidden: hidden, kernelSize: 4)
        let result = try attention.forward([1, 0], tokenCount: 1)
        #expect(result.values.count == 2)
        #expect(result.values.allSatisfy { $0.isFinite })
    }
}
