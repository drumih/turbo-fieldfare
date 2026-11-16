import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// A tiered install keeps hot experts at 4-bit and cold experts at 2-bit in the
/// same layer file. `layout.json` already describes that shape — every expert
/// has its own `offset`/`size` and every tensor its own `bits` — so the runtime
/// must decode it without assuming one `expertStride` per layer.
///
/// These tests pin the parse half of the cold tier: the reader must carry the
/// per-expert size and width through, because the streamer will read `size`
/// bytes and the MoE dispatch will pick a pipeline by `bits`.
@Suite struct MixedExpertWidthLayoutTests {

    private static func writeMixedLayout() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-mixed-layout-\(UUID().uuidString)")
        let expertsDir = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(
            at: expertsDir, withIntermediateDirectories: true)

        // Expert 0 is 4-bit and 16 KiB; expert 1 is 2-bit and 8 KiB. The layer
        // stride remains the 4-bit maximum so a slot can hold either.
        func expert(_ id: Int, offset: Int, size: Int, bits: Int) -> [String: Any] {
            let weightBytes = size / 2
            let tail = size - weightBytes
            return [
                "expert": id,
                "offset": offset,
                "size": size,
                "tensors": [
                    "gate": [
                        "offset": 0,
                        "size": weightBytes,
                        "dtype": "U32",
                        "shape": [64, 64],
                        "bits": bits,
                    ],
                    "gate_scales": [
                        "offset": weightBytes,
                        "size": tail,
                        "dtype": "BF16",
                        "shape": [64, 1],
                    ],
                ],
            ]
        }

        let root: [String: Any] = [
            "expertStride": 16384,
            "numLayers": 1,
            "expertsPerLayer": 2,
            "layers": [
                [
                    "layer": 0,
                    "file": "layer_00.bin",
                    "experts": [
                        expert(0, offset: 0, size: 16384, bits: 4),
                        expert(1, offset: 16384, size: 8192, bits: 2),
                    ],
                ]
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        try data.write(to: expertsDir.appendingPathComponent("layout.json"))
        return dir
    }

    @Test
    func mixedWidthsSurviveDecoding() throws {
        let dir = try Self.writeMixedLayout()
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)

        let hot = try layout.expert(layer: 0, expert: 0)
        let cold = try layout.expert(layer: 0, expert: 1)

        #expect(hot.size == 16384)
        #expect(hot.subTensors["gate"]?.bits == 4)
        #expect(cold.size == 8192)
        #expect(cold.subTensors["gate"]?.bits == 2)
        // The layer stride stays the 4-bit maximum, so a slot is large enough
        // for either tier and the cold tier simply fills half of it.
        #expect(layout.expertStride == 16384)
    }

    @Test
    func coldExpertOffsetIsAbsoluteWithinTheLayer() throws {
        let dir = try Self.writeMixedLayout()
        let layout = try PackedExpertsLayoutReader.load(directoryURL: dir)
        let cold = try layout.expert(layer: 0, expert: 1)
        #expect(cold.offset == 16384)
        #expect(cold.offset + cold.size == 16384 + 8192)
    }
}
