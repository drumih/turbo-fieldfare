import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

@Suite struct PackedExpertsLayoutTests {

    /// Hand-write a tiny layout.json with one layer, two experts, two
    /// sub-tensors each. Returns the directory URL.
    static func writeToyLayout() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-layout-test-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        let root: [String: Any] = [
            "expertStride": 16384,
            "numLayers": 1,
            "expertsPerLayer": 2,
            "layers": [
                [
                    "layer": 0,
                    "file": "layer_00.bin",
                    "experts": [
                        [
                            "expert": 0,
                            "offset": 0,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4,
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1],
                                ],
                            ],
                        ],
                        [
                            "expert": 1,
                            "physicalRank": 1,
                            "offset": 16384,
                            "size": 16384,
                            "tensors": [
                                "gate": [
                                    "offset": 0,
                                    "size": 4096,
                                    "dtype": "U32",
                                    "shape": [64, 64],
                                    "bits": 4,
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1],
                                ],
                            ],
                        ],
                    ],
                ]
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        try data.write(to: exp.appendingPathComponent("layout.json"))
        return dir
    }

    @Test func decodesToyLayout() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        #expect(layout.expertStride == 16384)
        #expect(layout.numLayers == 1)
        #expect(layout.expertsPerLayer == 2)
        #expect(layout.layers.count == 1)
        let exp0 = try layout.expert(layer: 0, expert: 0)
        #expect(exp0.offset == 0)
        #expect(exp0.size == 16384)
        let gate = try #require(exp0.subTensors["gate"])
        #expect(gate.offset == 0)
        #expect(exp0.subTensors["gate_scales"]?.offset == 4096)
        let exp1 = try layout.expert(layer: 0, expert: 1)
        #expect(exp1.offset == 16384)
        #expect(exp1.expert == 1)
    }

    @Test func missingLayoutJsonThrowsMissingFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-no-layout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            if case ModelError.missingFile = error { return true }
            return false
        }
    }

    /// Rewrite the toy `layout.json` with one field of one expert's record
    /// changed, leaving every other expert exactly as `writeToyLayout()` wrote
    /// it. The point is that the defect lives only in a *later* expert: the
    /// reference expert at index 0 stays valid, so a validator that checked
    /// only the first record of each layer would accept these.
    private static func mutateExpertField(
        _ dir: URL, layer: Int, expert: Int, field: String, to value: Any
    ) throws {
        let url = dir.appendingPathComponent("packed_experts/layout.json")
        var root = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        var layers = try #require(root["layers"] as? [[String: Any]])
        var experts = try #require(layers[layer]["experts"] as? [[String: Any]])
        experts[expert][field] = value
        layers[layer]["experts"] = experts
        root["layers"] = layers
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: url)
    }

    /// The same, for one field of one sub-tensor inside an expert's blob.
    private static func mutateSubTensorField(
        _ dir: URL, layer: Int, expert: Int, tensor: String, field: String, to value: Any
    ) throws {
        let url = dir.appendingPathComponent("packed_experts/layout.json")
        var root = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
        var layers = try #require(root["layers"] as? [[String: Any]])
        var experts = try #require(layers[layer]["experts"] as? [[String: Any]])
        var tensors = try #require(experts[expert]["tensors"] as? [String: [String: Any]])
        tensors[tensor]?[field] = value
        experts[expert]["tensors"] = tensors
        layers[layer]["experts"] = experts
        root["layers"] = layers
        try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
            .write(to: url)
    }

    private static func rejects(_ dir: URL, _ phrase: String) throws {
        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains(phrase)
            }
            return false
        }
    }

    @Test func laterExpertBlobOffsetRejectsEvenWithValidFirstExpert() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Expert 1 at 20,480 with a 16,384-byte blob ends at 36,864, past the
        // layer's 2 * 16,384 capacity. Offsets are compact since the tiered
        // format, but a range may not leave the layer.
        try Self.mutateExpertField(dir, layer: 0, expert: 1, field: "offset", to: 20_480)
        try Self.rejects(dir, "ends past the layer's capacity")
    }

    @Test func laterExpertBlobSizeRejectsEvenWithValidFirstExpert() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        // A blob larger than one slot is still refused; a smaller one is the
        // tiered format's normal case (see MixedExpertWidthLayoutTests).
        try Self.mutateExpertField(dir, layer: 0, expert: 1, field: "size", to: 32_768)
        try Self.rejects(dir, "is not within 1...16384")
    }

    @Test func laterExpertTensorRangePastBlobRejects() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 4096 + 20000 runs past the 16384-byte expert blob. The streamer would
        // read the next expert's payload as this one's gate weights.
        try Self.mutateSubTensorField(
            dir, layer: 0, expert: 1, tensor: "gate", field: "size", to: 20_000)
        try Self.rejects(dir, "range exceeds expert blob")
    }

    @Test func laterExpertOverlappingTensorRangesRejects() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        // gate_scales declared at offset 0 aliases gate's packed weights.
        try Self.mutateSubTensorField(
            dir, layer: 0, expert: 1, tensor: "gate_scales", field: "offset", to: 0)
        try Self.rejects(dir, "overlapping ranges")
    }

    @Test func oversizedLayoutRejectsBeforeDecode() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layoutURL =
            dir
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        try Data(repeating: 0x20, count: 64).write(to: layoutURL)

        #expect {
            _ = try PackedExpertsLayoutReader.load(
                directoryURL: dir,
                maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }
}
