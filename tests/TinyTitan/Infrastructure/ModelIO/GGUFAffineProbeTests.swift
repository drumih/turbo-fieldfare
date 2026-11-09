import Foundation
import Testing

@testable import TinyTitan

/// One layer's worth of the conversion question: how large do the routed experts
/// become as TinyTitan's affine 4-bit (group 64, BF16 scale+bias), and how much
/// error does re-quantising add on top of the IQ quantisation already in the
/// file?
///
/// The error is weight-space, not model quality: it says what the conversion
/// does to the numbers, not what the model then answers. The real bar for a
/// conversion is KLD/top-1 against a GGUF run, which needs the model executing
/// somewhere; this probe is the cheap first filter. Sizes are exact rather than
/// sampled — affine bytes are a function of the element count
/// (`elements/2 + elements/64*4`), so every expert tensor can be counted without
/// decoding it.
///
///     TINYTITAN_GGUF_MODEL=models/qwen3.8-flash-next-unsloth-iq3xxs \
///       swift test -c release --no-parallel --filter GGUFAffineProbe
@Suite struct GGUFAffineProbeTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Probe {
        let label: String
        let type: GGUFType
        let weights: Int
        let rmse: Float
        let maxAbs: Float
        let relativeRms: Float
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func probesExpertRequantisationAndSizesTheModel() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))

        let cases = [
            (0, "ffn_gate_exps.weight"),
            (0, "ffn_down_exps.weight"),
            (14, "ffn_gate_exps.weight"),
            (47, "ffn_gate_exps.weight"),
        ]
        for (layer, suffix) in cases {
            let probe = try Self.probe(model, layer: layer, suffix: suffix)
            print(
                String(
                    format:
                        "GGUF affine probe blk.%d %@ (%@): %d weights, rmse %.6g, max %.6g, rel %.4f",
                    layer, suffix, probe.type.name, probe.weights, probe.rmse,
                    probe.maxAbs, probe.relativeRms))
            #expect(probe.rmse > 0)
            #expect(probe.relativeRms < 0.25)
        }

        var iqBytes: UInt64 = 0
        var elements: UInt64 = 0
        for (name, location) in model.tensors where name.hasSuffix("_exps.weight") {
            iqBytes += location.tensor.byteCount
            elements += location.tensor.elementCount
        }
        // Affine 4-bit group 64: 0.5 B of nibbles + 2 scales + 2 biases per 64.
        let affineBytes = elements / 64 * 36
        // Every tensor's own size is summed, so the widest layers (14+: IQ3_XXS
        // gate/up; 47: IQ4_XS) are accounted for rather than extrapolated from
        // layer 0. Interpolated, not printf: an Int through %d truncates at
        // 32 bits and 120.8 G weights do not fit.
        print(
            "GGUF affine probe: \(elements) expert weights, "
                + String(
                    format: "IQ %.2f GB -> affine %.2f GB (%.2fx)",
                    Double(iqBytes) / 1e9, Double(affineBytes) / 1e9,
                    Double(affineBytes) / Double(iqBytes)))
        #expect(affineBytes > iqBytes)
    }

    private static func probe(_ model: GGUFModel, layer: Int, suffix: String) throws -> Probe {
        let name = "blk.\(layer).\(suffix)"
        let location = try #require(model.tensor(named: name))
        let tensor = location.tensor
        let weights = try GGUFExpertLayout.weights(layer: layer, expert: 0, in: model)
        let range = Self.range(suffix: suffix, weights: weights)
        let handle = try FileHandle(forReadingFrom: model.shards[range.shardIndex].url)
        defer { try? handle.close() }
        try handle.seek(toOffset: range.fileOffset)
        let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
        let values = try GGUFDequant.decode(type: range.type, bytes: [UInt8](data))

        let rowLength = Int(tensor.dimensions[0])
        #expect(rowLength % Quantization.groupSize == 0)
        #expect(values.count % rowLength == 0)
        var sqError: Double = 0
        var sqReference: Double = 0
        var maxAbs: Float = 0
        for rowStart in stride(from: 0, to: values.count, by: rowLength) {
            let row = Array(values[rowStart..<(rowStart + rowLength)])
            let restored = Quantization.dequantizeInt4Affine(
                Quantization.quantizeInt4Affine(row), n: rowLength)
            for index in 0..<rowLength {
                let error = restored[index] - row[index]
                sqError += Double(error * error)
                sqReference += Double(row[index] * row[index])
                maxAbs = max(maxAbs, abs(error))
            }
        }
        let count = Double(values.count)
        let rmse = Float((sqError / count).squareRoot())
        let reference = Float((sqReference / count).squareRoot())
        return Probe(
            label: name,
            type: range.type,
            weights: values.count,
            rmse: rmse,
            maxAbs: maxAbs,
            relativeRms: rmse / max(reference, .leastNormalMagnitude))
    }

    private static func range(suffix: String, weights: GGUFExpertWeights) -> GGUFTensorRange {
        switch suffix {
        case "ffn_gate_exps.weight": return weights.gate
        case "ffn_up_exps.weight": return weights.up
        default: return weights.down
        }
    }
}
