import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

@Suite struct GGUFExpertLayoutTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func tensor(
        _ name: String, dims: [UInt64], type: GGUFType, offset: UInt64
    ) -> GGUFTensor {
        let elements = dims.reduce(UInt64(1)) { $0 * $1 }
        return GGUFTensor(
            name: name,
            dimensions: dims,
            type: type,
            offset: offset,
            elementCount: elements,
            byteCount: elements / type.blockSize * type.blockBytes)
    }

    /// A synthetic layer with four experts: gate/up at IQ2_S (52,480 bytes per
    /// expert) and down at IQ4_NL (92,160), laid out end to end with no padding.
    /// The literals below are hand-computed from the block geometry, not read
    /// back from the type under test.
    private static func syntheticModel(expertCount: Int = 4, gateExpertExtent: Int? = nil)
        -> GGUFModel
    {
        let extent = UInt64(gateExpertExtent ?? expertCount)
        let experts = UInt64(expertCount)
        let gate = tensor(
            "blk.0.ffn_gate_exps.weight", dims: [256, 640, extent], type: .iq2S, offset: 0)
        let up = tensor(
            "blk.0.ffn_up_exps.weight", dims: [256, 640, extent], type: .iq2S,
            offset: gate.byteCount)
        let down = tensor(
            "blk.0.ffn_down_exps.weight", dims: [256, 640, experts], type: .iq4NL,
            offset: gate.byteCount + up.byteCount)
        let tensors = [gate, up, down]
        let shard = GGUFShard(
            url: URL(fileURLWithPath: "/nonexistent/toy.gguf"),
            version: 3,
            alignment: 32,
            metadata: [:],
            tensors: tensors,
            dataOffset: 1_000)
        var locations: [String: GGUFTensorLocation] = [:]
        for tensor in tensors {
            locations[tensor.name] = GGUFTensorLocation(shardIndex: 0, tensor: tensor)
        }
        return GGUFModel(
            shards: [shard],
            tensors: locations,
            metadata: [
                "general.architecture": .string("toy"),
                "toy.expert_count": .uint32(UInt32(expertCount)),
            ])
    }

    @Test func addressesEachExpertAtItsOwnContiguousSlice() throws {
        let model = Self.syntheticModel()
        let weights = try GGUFExpertLayout.weights(layer: 0, expert: 3, in: model)
        #expect(weights.gate.byteCount == 52_480)
        #expect(weights.gate.fileOffset == 1_000 + 3 * 52_480)
        #expect(weights.up.byteCount == 52_480)
        #expect(weights.up.fileOffset == 1_000 + 209_920 + 3 * 52_480)
        #expect(weights.down.type == .iq4NL)
        #expect(weights.down.byteCount == 92_160)
        #expect(weights.down.fileOffset == 1_000 + 419_840 + 3 * 92_160)
        #expect(weights.totalBytes == 197_120)
    }

    /// The slices must tile the tensor exactly; a writer that padded between
    /// experts would move every expert after the first and this catches it.
    @Test func slicesTileTheTensorExactly() throws {
        let model = Self.syntheticModel()
        let slices = try GGUFExpertLayout.slices(layer: 0, in: model)
        #expect(slices.gate * 4 == 209_920)
        #expect(slices.up * 4 == 209_920)
        #expect(slices.down * 4 == 368_640)
        let last = try GGUFExpertLayout.weights(layer: 0, expert: 3, in: model)
        #expect(last.gate.fileOffset + last.gate.byteCount == 1_000 + 209_920)
    }

    @Test func refusesAnExpertPastTheLayer() throws {
        #expect(throws: GGUFExpertLayoutError.expertOutOfRange(layer: 0, expert: 4, count: 4)) {
            _ = try GGUFExpertLayout.weights(layer: 0, expert: 4, in: Self.syntheticModel())
        }
    }

    @Test func refusesATensorWhoseExpertExtentDisagrees() throws {
        let model = Self.syntheticModel(gateExpertExtent: 8)
        #expect(
            throws: GGUFExpertLayoutError.expertCountMismatch(
                tensor: "blk.0.ffn_gate_exps.weight", found: 8, expected: 4)
        ) {
            _ = try GGUFExpertLayout.weights(layer: 0, expert: 0, in: model)
        }
    }

    @Test func refusesATensorWhoseBytesAreNotAWholeNumberOfSlices() throws {
        let model = Self.syntheticModel()
        let broken = GGUFTensor(
            name: "blk.0.ffn_gate_exps.weight",
            dimensions: [256, 640, 4],
            type: .iq2S,
            offset: 0,
            elementCount: 655_360,
            byteCount: 209_920 - 82)
        var tensors = model.tensors
        tensors[broken.name] = GGUFTensorLocation(shardIndex: 0, tensor: broken)
        let corrupted = GGUFModel(
            shards: model.shards, tensors: tensors, metadata: model.metadata)
        #expect(throws: GGUFExpertLayoutError.self) {
            _ = try GGUFExpertLayout.weights(layer: 0, expert: 0, in: corrupted)
        }
    }

    /// The real checkpoint: 48 layers, 512 experts, and the layout facts this
    /// component exists to prove — 4,096 slices tile 50.29 GB of experts with no
    /// gaps, and the widening types are where the inventory said they are.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func addressesTheInstalledCheckpointsExperts() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        #expect(try GGUFExpertLayout.expertsPerLayer(in: model) == 512)

        for layer in 0..<48 {
            let slices = try GGUFExpertLayout.slices(layer: layer, in: model)
            let first = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
            let last = try GGUFExpertLayout.weights(layer: layer, expert: 511, in: model)
            for (suffix, slice, firstRange, lastRange) in [
                ("ffn_gate_exps.weight", slices.gate, first.gate, last.gate),
                ("ffn_up_exps.weight", slices.up, first.up, last.up),
                ("ffn_down_exps.weight", slices.down, first.down, last.down),
            ] {
                let location = try #require(
                    model.tensor(named: "blk.\(layer).\(suffix)"))
                let dataStart = model.shards[location.shardIndex].dataOffset
                #expect(firstRange.fileOffset == dataStart + location.tensor.offset)
                #expect(
                    lastRange.fileOffset + slice
                        == dataStart + location.tensor.offset + location.tensor.byteCount)
            }
        }

        #expect(try GGUFExpertLayout.weights(layer: 0, expert: 0, in: model).gate.type == .iq2S)
        #expect(try GGUFExpertLayout.weights(layer: 8, expert: 0, in: model).down.type == .q2_0)
        #expect(try GGUFExpertLayout.weights(layer: 14, expert: 0, in: model).gate.type == .iq3XXS)
        #expect(try GGUFExpertLayout.weights(layer: 47, expert: 0, in: model).gate.type == .iq4XS)

        // Decode the last expert's first gate block from the computed offset:
        // the addressing has to land inside the tensor, on a real block.
        let weights = try GGUFExpertLayout.weights(layer: 14, expert: 511, in: model)
        let handle = try FileHandle(forReadingFrom: model.shards[weights.gate.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: weights.gate.fileOffset)
        let block = try #require(try handle.read(upToCount: Int(weights.gate.type.blockBytes)))
        let values = try GGUFDequant.decode(type: weights.gate.type, bytes: [UInt8](block))
        #expect(values.count == Int(weights.gate.type.blockSize))
        #expect(values.allSatisfy { $0.isFinite })
        #expect(values.contains { $0 != 0 })
    }
}
