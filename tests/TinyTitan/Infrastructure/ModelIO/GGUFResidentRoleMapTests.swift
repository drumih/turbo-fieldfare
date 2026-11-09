import Foundation
import Testing

@testable import TinyTitan

@Suite struct GGUFResidentRoleMapTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func mapsEveryResidentTensorAndChecksItsShape() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let counts = try GGUFResidentRoleMap.validate(model: model, arch: .qwen38FlashNext)

        #expect(counts.values.reduce(0, +) == 1_080)
        // The two traps, counted: 36 GDN layers carry qkv/z, 12 full-attention
        // layers carry separate q/k/v with the gate fused into q.
        #expect(counts["gdnQKV"] == 36)
        #expect(counts["gdnZ"] == 36)
        #expect(counts["qProj"] == 12)
        #expect(counts["kProj"] == 12)
        #expect(counts["vProj"] == 12)
        #expect(counts["oProj"] == 12)
        #expect(counts["qNorm"] == 12)
        #expect(counts["kNorm"] == 12)
        #expect(counts["router"] == 48)
        #expect(counts["sharedExpertGate"] == 48)
        #expect(counts["hcAttnUp"] == 48)
        #expect(counts["indexerQ"] == 12)
        #expect(counts["gdnConv"] == 36)
        #expect(counts["embedding"] == 1)
        #expect(counts["lmHead"] == 1)
        #expect(counts["pleTable"] == 1)
        print(
            "GGUF resident roles: \(counts.count) roles over \(counts.values.reduce(0, +)) tensors")
    }

    /// A role whose dims disagree must fail here, not after a 100 GB install.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func refusesATensorWhoseShapeDisagrees() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let name = "blk.0.attn_qkv.weight"
        let location = try #require(model.tensor(named: name))
        let original = location.tensor
        var dimensions = original.dimensions
        dimensions[1] -= 1
        let broken = GGUFTensor(
            name: original.name,
            dimensions: dimensions,
            type: original.type,
            offset: original.offset,
            elementCount: original.elementCount,
            byteCount: original.byteCount)
        var tensors = model.tensors
        tensors[name] = GGUFTensorLocation(shardIndex: location.shardIndex, tensor: broken)
        let corrupted = GGUFModel(
            shards: model.shards, tensors: tensors, metadata: model.metadata)
        #expect(throws: GGUFResidentRoleError.self) {
            _ = try GGUFResidentRoleMap.validate(model: corrupted, arch: .qwen38FlashNext)
        }
    }
}
