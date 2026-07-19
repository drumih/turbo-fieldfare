import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct PackedExpertsLayoutTests {

    /// Hand-write a tiny layout.json with one layer, two experts, two
    /// sub-tensors each. Returns the directory URL.
    static func writeToyLayout() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-layout-test-\(UUID().uuidString)")
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
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
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
                                    "bits": 4
                                ],
                                "gate_scales": [
                                    "offset": 4096,
                                    "size": 256,
                                    "dtype": "BF16",
                                    "shape": [64, 1]
                                ],
                            ],
                        ],
                    ],
                ],
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
        let exp0 = layout.expert(layer: 0, expert: 0)
        #expect(exp0.offset == 0)
        #expect(exp0.size == 16384)
        let gate = try #require(exp0.subTensors["gate"])
        #expect(gate.offset == 0)
        #expect(exp0.subTensors["gate_scales"]?.offset == 4096)
        let exp1 = layout.expert(layer: 0, expert: 1)
        #expect(exp1.offset == 16384)
        #expect(exp1.expert == 1)
    }

    @Test func missingLayoutJsonThrowsMissingFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-no-layout-\(UUID().uuidString)")
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

    @Test func oversizedLayoutRejectsBeforeDecode() throws {
        let dir = try Self.writeToyLayout()
        defer { try? FileManager.default.removeItem(at: dir) }
        let layoutURL = dir
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        try Data(repeating: 0x20, count: 64).write(to: layoutURL)

        #expect {
            _ = try PackedExpertsLayoutReader.load(directoryURL: dir,
                                                   maxBytes: 16)
        } throws: { error in
            if case ModelError.indexCorrupt(let detail) = error {
                return detail.contains("metadata cap")
            }
            return false
        }
    }
}
