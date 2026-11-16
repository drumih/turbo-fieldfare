import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// K2-Horizon's MoVA value step end to end: bf16 router GEMV, sigmoid + bias
/// selection over 4 of N experts, four 4-bit value GEMVs, then SiLU before the
/// weighted sum. The reference is the CPU formula from `K2Model+Attention`,
/// with every buffer rounded the way the kernel reads it.
@Suite struct MoVAValueStageTests {
    private static let hidden = 128
    private static let valueDim = 64
    private static let expertCount = 8
    private static let scaling: Float = 2.5

    private struct Expert {
        let weights: [UInt8]  // row-major packed nibbles
        let scales: [UInt16]
        let biases: [UInt16]
    }

    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base,
                length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    private static func dequantRow(
        _ expert: Expert, row: Int
    ) -> [Float] {
        let columns = hidden
        let groups = columns / Quantization.groupSize
        var out = [Float](repeating: 0, count: columns)
        for column in 0..<columns {
            let byte = expert.weights[row * (columns / 2) + column / 2]
            let q = (column & 1) == 0 ? byte & 0x0F : byte >> 4
            let group = column / Quantization.groupSize
            let scale = Quantization.bf16ToFloat(
                expert.scales[row * groups + group])
            let bias = Quantization.bf16ToFloat(
                expert.biases[row * groups + group])
            out[column] = Float(q) * scale + bias
        }
        return out
    }

    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    private static func reference(
        x: [Float],
        router: [UInt16],  // [experts][hidden] bf16
        routerBias: [Float],
        experts: [Expert],
        topK: Int,
        scaling: Float
    ) -> [Float] {
        let xHalf = x.map { Float(Float16($0)) }
        let logits = (0..<expertCount).map { e in
            (0..<hidden).reduce(Float(0)) { sum, d in
                sum + Quantization.bf16ToFloat(router[e * hidden + d]) * xHalf[d]
            }
        }
        let scores = logits.map(sigmoid)
        let ranks = scores.indices.map { scores[$0] + routerBias[$0] }
        let order = ranks.indices.sorted {
            ranks[$0] == ranks[$1] ? $0 < $1 : ranks[$0] > ranks[$1]
        }
        let chosen = Array(order.prefix(topK))
        let total = chosen.reduce(Float(0)) { $0 + scores[$1] }
        let weights = chosen.map { Float(Float16(scores[$0] / total * scaling)) }
        var out = [Float](repeating: 0, count: valueDim)
        for (slot, expertID) in chosen.enumerated() {
            let expert = experts[expertID]
            for row in 0..<valueDim {
                let weightsRow = dequantRow(expert, row: row)
                let dot = zip(weightsRow, xHalf).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let rounded = Float(Float16(dot))
                out[row] += weights[slot] * (rounded / (1 + exp(-rounded)))
            }
        }
        return out.map { Float(Float16($0)) }
    }

    @Test func fourOfEightExpertsMatchTheCPUFormula() throws {
        var rng = SeedTree(0x40A_177).key("mova-stage")
        let ctx = try MetalContext()

        let x = (0..<Self.hidden).map { _ in rng.uniform(-2, 2) }
        let routerFloats = (0..<(Self.expertCount * Self.hidden)).map { _ in
            rng.uniform(-0.1, 0.1)
        }
        let routerBits = routerFloats.map { Quantization.bf16Bits($0) }
        let biasFloats = (0..<Self.expertCount).map { _ in rng.uniform(-1, 1) }
        let biasBits = biasFloats.map { Quantization.bf16Bits($0) }

        var experts: [Expert] = []
        for _ in 0..<Self.expertCount {
            var packed: [UInt8] = []
            var scales: [UInt16] = []
            var biases: [UInt16] = []
            for _ in 0..<Self.valueDim {
                let row = (0..<Self.hidden).map { _ in rng.uniform(-1, 1) }
                let quantized = Quantization.quantizeInt4Affine(row)
                packed += quantized.packed
                scales += quantized.scales
                biases += quantized.biases
            }
            experts.append(Expert(weights: packed, scales: scales, biases: biases))
        }

        let stage = try MoVAValueStage(
            context: ctx, weightBits: 4,
            hiddenSize: Self.hidden, valueDim: Self.valueDim,
            maxExperts: Self.expertCount)
        let xBuf = try #require(Self.makeBuffer(ctx, x.map { Float16($0) }))
        let routerBuf = try #require(Self.makeBuffer(ctx, routerBits))
        let biasBuf = try #require(Self.makeBuffer(ctx, biasBits))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: Self.valueDim))

        var bindings: [MoVAValueStage.ExpertWeights] = []
        for expert in experts {
            let weightBuf = try #require(Self.makeBuffer(ctx, expert.weights))
            let scaleBuf = try #require(Self.makeBuffer(ctx, expert.scales))
            let biasValueBuf = try #require(Self.makeBuffer(ctx, expert.biases))
            bindings.append(
                MoVAValueStage.ExpertWeights(
                    weights: weightBuf, scales: scaleBuf, biases: biasValueBuf))
        }

        let routerCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeRouter(
            commandBuffer: routerCB, hidden: xBuf,
            routerWeight: routerBuf, routerBias: biasBuf,
            numExperts: Self.expertCount, scaling: Self.scaling)
        routerCB.commit()
        routerCB.waitUntilCompleted()

        let selected = stage.selectedExpertIDs()
        let valueCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeValues(
            commandBuffer: valueCB, hidden: xBuf, experts: bindings, out: outBuf)
        valueCB.commit()
        valueCB.waitUntilCompleted()

        let expected = Self.reference(
            x: x, router: routerBits,
            routerBias: biasBits.map { Quantization.bf16ToFloat($0) },
            experts: experts, topK: 4, scaling: Self.scaling)
        let actual = Fp16Buffer.read(outBuf, count: Self.valueDim)
        let relError = RelError.compute(actual: actual, reference: expected)
        #expect(relError < Tolerance.fp16Reduction, "relErr=\(relError) ids=\(selected)")
    }

    /// The selection must be the router's, not slot order: a stage that fed
    /// experts 0-3 to the combine would still produce a plausible vector.
    @Test func selectedIdsFollowTheRouter() throws {
        var rng = SeedTree(0x5E1_EC7).key("mova-ids")
        let ctx = try MetalContext()
        let x = (0..<Self.hidden).map { _ in rng.uniform(-1, 1) }
        // Expert 7's router row points along x, so it must rank first.
        var router = [UInt16](repeating: 0, count: Self.expertCount * Self.hidden)
        for d in 0..<Self.hidden {
            router[7 * Self.hidden + d] = Quantization.bf16Bits(x[d] * 2)
        }
        let bias = [UInt16](repeating: 0, count: Self.expertCount)
        let stage = try MoVAValueStage(
            context: ctx, hiddenSize: Self.hidden, valueDim: Self.valueDim,
            maxExperts: Self.expertCount)
        let xBuf = try #require(Self.makeBuffer(ctx, x.map { Float16($0) }))
        let routerBuf = try #require(Self.makeBuffer(ctx, router))
        let biasBuf = try #require(Self.makeBuffer(ctx, bias))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeRouter(
            commandBuffer: cb, hidden: xBuf,
            routerWeight: routerBuf, routerBias: biasBuf,
            numExperts: Self.expertCount, scaling: Self.scaling)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(stage.selectedExpertIDs().first == 7)
    }
}
