import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// The fitted next-layer guess file: the loader finds each target layer's
/// matrix where `fit_guess.py` writes it, and the router GEMV computes the
/// dequantized product from it with the unit effective scale.
@Suite struct NextLayerGuessWeightsTests {
    private static let layers = 3
    private static let experts = 128
    private static let dimension = 2816
    private static let groups = dimension / Quantization.groupSize

    private struct Layer {
        var weights: [UInt8]
        var scales: [UInt16]
        var biases: [UInt16]
    }

    private static func randomLayers(seed: UInt64) -> [Layer] {
        var rng = SplitMix64(seed: seed)
        return (1..<layers).map { _ in
            Layer(weights: (0..<experts * dimension).map { _ in UInt8(rng.next() & 0xFF) },
                  scales: (0..<experts * groups).map { _ in
                      Quantization.bf16Bits(rng.uniform(1e-4, 1e-3)) },
                  biases: (0..<experts * groups).map { _ in
                      Quantization.bf16Bits(rng.uniform(-0.1, 0.1)) })
        }
    }

    private static func write(_ layers: [Layer], header fields: [UInt32]? = nil,
                              dropLast: Int = 0) throws -> URL {
        var data = Data(NextLayerGuessWeights.magic)
        let header = fields ?? [UInt32(Self.layers), UInt32(experts), UInt32(dimension),
                                UInt32(Quantization.groupSize), 1, UInt32(Self.layers - 1)]
        for v in header { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        data.append(Data(count: NextLayerGuessWeights.headerBytes - data.count))
        for layer in layers {
            data.append(contentsOf: layer.weights)
            layer.scales.withUnsafeBytes { data.append(contentsOf: $0) }
            layer.biases.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        data.removeLast(dropLast)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("next-layer-guess-\(UUID().uuidString).bin")
        try data.write(to: url)
        return url
    }

    private static func load(_ url: URL, device: MTLDevice) throws -> NextLayerGuessWeights {
        try NextLayerGuessWeights(fileURL: url, numLayers: layers, numExperts: experts,
                                  hiddenSize: dimension, device: device)
    }

    @Test func guessScoresMatchDequantizedReference() throws {
        let layers = Self.randomLayers(seed: 0x6E57_1A7E)
        let url = try Self.write(layers)
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let guess = try Self.load(url, device: context.device)
        let moe = try MoE(context: context)

        var rng = SplitMix64(seed: 0x0B5E_55ED)
        let hidden = (0..<Self.dimension).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuffer = try #require(context.device.makeBuffer(
            bytes: hidden, length: hidden.count * 2, options: .storageModeShared))
        let out = try #require(context.device.makeBuffer(
            length: Self.experts * 4, options: .storageModeShared))

        for target in 1..<Self.layers {
            let m = guess.matrix(targetLayer: target)
            let cb = try #require(context.queue.makeCommandBuffer())
            moe.encodeRouterLogitsGemma4(commandBuffer: cb,
                weights: guess.buffer, weightsOffset: m.weightsOffset,
                scales: guess.buffer, scalesOffset: m.scalesOffset,
                biases: guess.buffer, biasesOffset: m.biasesOffset,
                hidden: hiddenBuffer, effectiveScale: guess.unitScale,
                outLogits: out, numExperts: UInt32(Self.experts), d: UInt32(Self.dimension))
            cb.commit()
            cb.waitUntilCompleted()

            let layer = layers[target - 1]
            let got = out.contents().bindMemory(to: Float.self, capacity: Self.experts)
            var worst: Float = 0
            var largest: Float = 0
            for e in 0..<Self.experts {
                var ref: Float = 0
                for i in 0..<Self.dimension {
                    let g = e * Self.groups + i / Quantization.groupSize
                    let w = Quantization.bf16ToFloat(layer.scales[g])
                        * Float(layer.weights[e * Self.dimension + i])
                        + Quantization.bf16ToFloat(layer.biases[g])
                    ref += w * Float(hidden[i])
                }
                worst = max(worst, abs(got[e] - ref))
                largest = max(largest, abs(ref))
            }
            #expect(largest > 0.1)
            #expect(worst <= 1e-4 * largest)
        }
    }

    @Test func rejectsFileForAnotherModel() throws {
        let device = try MetalContext().device
        let layers = Self.randomLayers(seed: 1)
        let wrongHeader = try Self.write(layers, header: [UInt32(Self.layers), 64,
            UInt32(Self.dimension * 2), UInt32(Quantization.groupSize), 1, UInt32(Self.layers - 1)])
        let truncated = try Self.write(layers, dropLast: 1)
        defer {
            try? FileManager.default.removeItem(at: wrongHeader)
            try? FileManager.default.removeItem(at: truncated)
        }
        #expect(throws: ModelError.self) { _ = try Self.load(wrongHeader, device: device) }
        #expect(throws: ModelError.self) { _ = try Self.load(truncated, device: device) }
    }
}
