import Foundation
import Testing

@testable import TinyTitan

/// Measures how fast the host decoder reads the two expert formats, in the
/// shape the engine would call it: one whole expert slice per call.
///
/// This is a measurement, not a gate. The assertion is correctness (every value
/// finite, the right count); the rate is printed so a decision can cite it.
/// Run it in **release** — a debug build has no optimisations and would
/// overstate the case against decoding on the host:
///
///     TINYTITAN_GGUF_MODEL=models/qwen3.8-flash-next-unsloth-iq3xxs \
///       swift test -c release --no-parallel --filter GGUFDecodeMeasurement
@Suite struct GGUFDecodeMeasurementTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func slice(_ model: GGUFModel, layer: Int) throws -> ([UInt8], GGUFType) {
        // Expert 0's gate matrix is one contiguous slice (see GGUFExpertLayout).
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
        let shard = model.shards[weights.gate.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: weights.gate.fileOffset)
        let data = try #require(try handle.read(upToCount: Int(weights.gate.byteCount)))
        return ([UInt8](data), weights.gate.type)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func measuresExpertSliceDecodeRate() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let iterations = 50

        for layer in [0, 14] {
            let (bytes, type) = try Self.slice(model, layer: layer)
            let elementsPerSlice = Int(type.blockSize) * (bytes.count / Int(type.blockBytes))
            let expected = elementsPerSlice

            let start = DispatchTime.now().uptimeNanoseconds
            var decoded = 0
            for _ in 0..<iterations {
                let values = try GGUFDequant.decode(type: type, bytes: bytes)
                #expect(values.count == expected)
                #expect(values.allSatisfy { $0.isFinite })
                decoded += values.count
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let rate = Double(decoded) / elapsed / 1e9
            print(
                "GGUF host decode \(type.name): \(String(format: "%.3f", rate)) G elements/s "
                    + "(\(elementsPerSlice) per slice, \(iterations) slices in "
                    + "\(String(format: "%.3f", elapsed)) s)"
            )
        }
    }
}
