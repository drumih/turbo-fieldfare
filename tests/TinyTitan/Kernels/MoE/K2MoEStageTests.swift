import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// K2's sparse expert block on synthetic weights: bf16 router, sigmoid top-k
/// selection with a ranking-only bias, the selected SwiGLU experts and the
/// always-on shared expert. The reference is the CPU formula with the same
/// 4-bit packing the runtime reads.
@Suite struct K2MoEStageTests {
    private static let hidden = 128
    private static let width = 64
    private static let experts = 4
    private static let topK = 2
    private static let scaling: Float = 2.5

    private struct Packed {
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
    }

    private static func quantize(_ rows: [[Float]]) -> Packed {
        var packed: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in rows {
            let q = Quantization.quantizeInt4Affine(row)
            packed += q.packed
            scales += q.scales
            biases += q.biases
        }
        return Packed(packed: packed, scales: scales, biases: biases)
    }

    private static func dequant(_ packed: Packed, rows: Int, columns: Int) -> [[Float]] {
        let groups = columns / Quantization.groupSize
        return (0..<rows).map { row in
            (0..<columns).map { column in
                let byte = packed.packed[row * (columns / 2) + column / 2]
                let q = (column & 1) == 0 ? byte & 0x0F : byte >> 4
                let group = column / Quantization.groupSize
                let scale = Quantization.bf16ToFloat(
                    packed.scales[row * groups + group])
                let bias = Quantization.bf16ToFloat(
                    packed.biases[row * groups + group])
                return Float(q) * scale + bias
            }
        }
    }

    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base, length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    private static func makeTensor(
        _ ctx: MetalContext, rows: Int, columns: Int, values: [[Float]]
    ) throws -> (TensorView, Packed) {
        let packed = quantize(values)
        let weightBytes = packed.packed.count
        let scaleBytes = packed.scales.count * 2
        let biasBytes = packed.biases.count * 2
        guard
            let buffer = ctx.device.makeBuffer(
                length: weightBytes + scaleBytes + biasBytes,
                options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("moe test tensor") }
        let base = buffer.contents()
        packed.packed.withUnsafeBytes { source in
            if let address = source.baseAddress {
                memcpy(base, address, weightBytes)
            }
        }
        packed.scales.withUnsafeBytes { source in
            if let address = source.baseAddress {
                memcpy(base.advanced(by: weightBytes), address, scaleBytes)
            }
        }
        packed.biases.withUnsafeBytes { source in
            if let address = source.baseAddress {
                memcpy(base.advanced(by: weightBytes + scaleBytes), address, biasBytes)
            }
        }
        let view = TensorView(
            buffer: buffer, offset: 0, length: UInt64(weightBytes),
            scaleOffset: UInt64(weightBytes), scaleLength: UInt64(scaleBytes),
            biasOffset: UInt64(weightBytes + scaleBytes), biasLength: UInt64(biasBytes),
            shape: (UInt32(rows), UInt32(columns), 0, 0), dtype: 0)
        return (view, packed)
    }

    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    @Test func selectedExpertsAndSharedMatchTheReference() throws {
        var rng = SeedTree(0x30E_1A2B).key("k2-moe-stage")
        let ctx = try MetalContext()
        let x = (0..<Self.hidden).map { _ in Float16(rng.uniform(-1, 1)) }
        let xFloat = x.map(Float.init)

        let routerValues = (0..<Self.experts).map { _ in
            (0..<Self.hidden).map { _ in rng.uniform(-0.2, 0.2) }
        }
        let routerFlat = routerValues.flatMap { $0 }
        let routerBF16 = routerFlat.map { Quantization.bf16Bits($0) }
        let biasValues = (0..<Self.experts).map { _ in rng.uniform(-1, 1) }
        let biasBF16 = biasValues.map { Quantization.bf16Bits($0) }

        var expertViews: [K2MoEStage.ExpertWeights] = []
        var expertValues: [(gate: [[Float]], up: [[Float]], down: [[Float]])] = []
        for _ in 0..<Self.experts {
            let gate = (0..<Self.width).map { _ in
                (0..<Self.hidden).map { _ in rng.uniform(-0.05, 0.05) }
            }
            let up = (0..<Self.width).map { _ in
                (0..<Self.hidden).map { _ in rng.uniform(-0.05, 0.05) }
            }
            let down = (0..<Self.hidden).map { _ in
                (0..<Self.width).map { _ in rng.uniform(-0.05, 0.05) }
            }
            let (gateView, gatePacked) = try Self.makeTensor(
                ctx, rows: Self.width, columns: Self.hidden, values: gate)
            let (upView, upPacked) = try Self.makeTensor(
                ctx, rows: Self.width, columns: Self.hidden, values: up)
            let (downView, downPacked) = try Self.makeTensor(
                ctx, rows: Self.hidden, columns: Self.width, values: down)
            expertViews.append(
                K2MoEStage.ExpertWeights(gate: gateView, up: upView, down: downView))
            expertValues.append(
                (
                    Self.dequant(gatePacked, rows: Self.width, columns: Self.hidden),
                    Self.dequant(upPacked, rows: Self.width, columns: Self.hidden),
                    Self.dequant(downPacked, rows: Self.hidden, columns: Self.width)
                ))
        }
        // Shared expert reuses expert 0's weights and is passed explicitly.
        let shared = expertViews[0]

        let stage = try K2MoEStage(
            context: ctx, weightBits: 4,
            hiddenSize: Self.hidden, expertWidth: Self.width,
            numExperts: Self.experts, topK: Self.topK)
        let xBuf = try #require(Self.makeBuffer(ctx, x))
        let routerBuf = try #require(Self.makeBuffer(ctx, routerBF16))
        let biasBuf = try #require(Self.makeBuffer(ctx, biasBF16))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: Self.hidden))

        let routerCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeRouter(
            commandBuffer: routerCB, hidden: xBuf,
            routerWeight: routerBuf, routerBias: biasBuf,
            scaling: Self.scaling)
        routerCB.commit()
        routerCB.waitUntilCompleted()
        let ids = stage.selectedExpertIDs()

        let valueCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeExperts(
            commandBuffer: valueCB, hidden: xBuf,
            experts: ids.map { expertViews[$0] }, shared: shared, out: outBuf)
        valueCB.commit()
        valueCB.waitUntilCompleted()
        let actual = Fp16Buffer.read(outBuf, count: Self.hidden)

        // The fused two-dispatch form has to produce the same block.
        let fusedBuf = try #require(Fp16Buffer.make(ctx.device, count: Self.hidden))
        let fusedCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeFusedExperts(
            commandBuffer: fusedCB, hidden: xBuf,
            experts: ids.map { expertViews[$0] }, shared: shared, out: fusedBuf)
        fusedCB.commit()
        fusedCB.waitUntilCompleted()
        let fused = Fp16Buffer.read(fusedBuf, count: Self.hidden)
        let fusedVsStaged = RelError.compute(actual: fused, reference: actual)
        #expect(fusedVsStaged < Tolerance.fp16Reduction, "relErr=\(fusedVsStaged)")

        // Reference router.
        let logits = (0..<Self.experts).map { expert in
            (0..<Self.hidden).reduce(Float(0)) {
                $0 + Quantization.bf16ToFloat(routerBF16[expert * Self.hidden + $1])
                    * xFloat[$1]
            }
        }
        let scores = logits.map(Self.sigmoid)
        let ranks = scores.indices.map {
            scores[$0] + Quantization.bf16ToFloat(biasBF16[$0])
        }
        let order = ranks.indices.sorted {
            ranks[$0] == ranks[$1] ? $0 < $1 : ranks[$0] > ranks[$1]
        }
        let chosen = Array(order.prefix(Self.topK))
        #expect(ids == chosen)
        let total = chosen.reduce(Float(0)) { $0 + scores[$1] }
        let weights = chosen.map { Float(Float16(scores[$0] / total * Self.scaling)) }

        func swiGLU(_ index: Int) -> [Float] {
            let expert = expertValues[index]
            var act = [Float](repeating: 0, count: Self.width)
            for row in 0..<Self.width {
                let g = zip(expert.gate[row], xFloat).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let u = zip(expert.up[row], xFloat).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let gh = Float(Float16(g))
                let uh = Float(Float16(u))
                act[row] = Float(Float16((gh / (1 + exp(-gh))) * uh))
            }
            return (0..<Self.hidden).map { row in
                let value = zip(expert.down[row], act).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                return Float(Float16(value))
            }
        }

        var expected = swiGLU(0)  // shared expert
        for (slot, expert) in chosen.enumerated() {
            let routed = swiGLU(expert)
            for i in 0..<Self.hidden {
                expected[i] = Float(Float16(expected[i] + weights[slot] * routed[i]))
            }
        }
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel) ids=\(ids)")
        let fusedRel = RelError.compute(actual: fused, reference: expected)
        #expect(fusedRel < Tolerance.fp16Reduction, "fused relErr=\(fusedRel)")
    }
}
