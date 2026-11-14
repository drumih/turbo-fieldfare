import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
import TinyTitanCore

/// The MoE stage on the sparse toy install's real packed experts: fetch a
/// layer's blobs through the streaming cache, convert them to per-role views
/// from the layout, and compare the routed output with a reference dequantized
/// from the same bytes.
@Suite struct K2MoEStageInstallTests {
    private static func makeBuffer<T>(_ ctx: MetalContext, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return nil }
            return ctx.device.makeBuffer(
                bytes: base, length: ptr.count * MemoryLayout<T>.stride,
                options: .storageModeShared)
        }
    }

    private static func dequant(
        _ view: TensorView, rows: Int, columns: Int
    ) -> [[Float]] {
        let bits = Int(view.length) * 8 / (rows * columns)
        precondition(bits == 4 || bits == 8)
        let groups = columns / Quantization.groupSize
        let base = view.buffer.contents()
        return (0..<rows).map { row in
            (0..<columns).map { column in
                let wordOffset = (row * columns + column) * bits / 32
                let shift = ((row * columns + column) * bits) % 32
                let word = base.load(
                    fromByteOffset: Int(view.offset) + wordOffset * 4,
                    as: UInt32.self)
                var value = word >> UInt32(shift)
                if shift + bits > 32 {
                    let next = base.load(
                        fromByteOffset: Int(view.offset) + (wordOffset + 1) * 4,
                        as: UInt32.self)
                    value |= next << UInt32(32 - shift)
                }
                let q = value & ((1 << UInt32(bits)) - 1)
                let group = column / Quantization.groupSize
                let scale = base.load(
                    fromByteOffset: Int(view.scaleOffset) + (row * groups + group) * 2,
                    as: UInt16.self)
                let bias = base.load(
                    fromByteOffset: Int(view.biasOffset) + (row * groups + group) * 2,
                    as: UInt16.self)
                return Float(q) * Quantization.bf16ToFloat(scale)
                    + Quantization.bf16ToFloat(bias)
            }
        }
    }

    private static func bf16(_ view: TensorView, count: Int) -> [Float] {
        let base = view.buffer.contents()
        return (0..<count).map {
            Quantization.bf16ToFloat(
                base.load(
                    fromByteOffset: Int(view.offset) + $0 * 2, as: UInt16.self))
        }
    }

    @Test func packedExpertsRunThroughTheStage() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let layer = config.numLayers - 1
        let plan = try #require(
            try model.planRoutedExperts(layer: layer, experts: [0]))
        let blobs = try await model.fetchRoutedExperts(plan: plan)
        #expect(blobs.count == 1)

        var experts: [K2MoEStage.ExpertWeights] = []
        for (index, id) in plan.experts.enumerated() {
            let entry = try model.packedExpertsLayout.expert(
                layer: layer, expert: id)
            experts.append(
                try K2MoEStage.expertWeights(blob: blobs[index], entry: entry))
        }
        let shared = K2MoEStage.ExpertWeights(
            gate: try model.k2SharedGate(layer: layer),
            up: try model.k2SharedUp(layer: layer),
            down: try model.k2SharedDown(layer: layer))

        let hiddenSize = config.hiddenSize
        let width = config.moeIntermediateSize
        var rng = SeedTree(0x1EAF_5001).key("k2-moe-install")
        let x = (0..<hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBuf = try #require(Self.makeBuffer(ctx, x))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: hiddenSize))
        let stage = try K2MoEStage(
            context: ctx, weightBits: 4,
            hiddenSize: hiddenSize, expertWidth: width,
            numExperts: config.numExperts, topK: config.topKExperts)

        let routerView = try model.k2MoeRouter(layer: layer)
        let biasView = try model.k2MoeRouterBias(layer: layer)
        let routerCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeRouter(
            commandBuffer: routerCB, hidden: xBuf,
            routerWeight: routerView.buffer,
            routerWeightOffset: Int(routerView.offset),
            routerBias: biasView.buffer,
            routerBiasOffset: Int(biasView.offset),
            scaling: Float(config.routerScaling))
        routerCB.commit()
        await routerCB.completed()
        let ids = stage.selectedExpertIDs()

        let valueCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeExperts(
            commandBuffer: valueCB, hidden: xBuf,
            experts: experts, shared: shared, out: outBuf)
        valueCB.commit()
        await valueCB.completed()
        let actual = Fp16Buffer.read(outBuf, count: hiddenSize)

        // The fused form binds the same fetched blobs through argument
        // buffers, so this also checks the layout's sub-tensor offsets are
        // addressable that way.
        let fusedBuf = try #require(Fp16Buffer.make(ctx.device, count: hiddenSize))
        let fusedCB = try #require(ctx.queue.makeCommandBuffer())
        try stage.encodeFusedExperts(
            commandBuffer: fusedCB, hidden: xBuf,
            experts: experts, shared: shared, out: fusedBuf)
        fusedCB.commit()
        await fusedCB.completed()
        let fused = Fp16Buffer.read(fusedBuf, count: hiddenSize)
        let fusedVsStaged = RelError.compute(actual: fused, reference: actual)
        #expect(fusedVsStaged < Tolerance.fp16Reduction, "relErr=\(fusedVsStaged)")

        let xFloat = x.map(Float.init)
        let router = Self.bf16(routerView, count: config.numExperts * hiddenSize)
        let bias = Self.bf16(biasView, count: config.numExperts)
        let logits = (0..<config.numExperts).map { expert in
            (0..<hiddenSize).reduce(Float(0)) {
                $0 + router[expert * hiddenSize + $1] * xFloat[$1]
            }
        }
        let scores = logits.map { 1 / (1 + exp(-$0)) }
        let ranks = scores.indices.map { scores[$0] + bias[$0] }
        let order = ranks.indices.sorted {
            ranks[$0] == ranks[$1] ? $0 < $1 : ranks[$0] > ranks[$1]
        }
        let chosen = Array(order.prefix(config.topKExperts))
        #expect(ids == chosen)
        let total = chosen.reduce(Float(0)) { $0 + scores[$1] }
        let weights = chosen.map {
            Float(Float16(scores[$0] / total * Float(config.routerScaling)))
        }

        func swiGLU(_ weights: K2MoEStage.ExpertWeights) -> [Float] {
            let gate = Self.dequant(weights.gate, rows: width, columns: hiddenSize)
            let up = Self.dequant(weights.up, rows: width, columns: hiddenSize)
            let down = Self.dequant(weights.down, rows: hiddenSize, columns: width)
            var act = [Float](repeating: 0, count: width)
            for row in 0..<width {
                let g = zip(gate[row], xFloat).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let u = zip(up[row], xFloat).reduce(Float(0)) { $0 + $1.0 * $1.1 }
                let gh = Float(Float16(g))
                let uh = Float(Float16(u))
                act[row] = Float(Float16((gh / (1 + exp(-gh))) * uh))
            }
            return (0..<hiddenSize).map { row in
                Float(Float16(zip(down[row], act).reduce(Float(0)) { $0 + $1.0 * $1.1 }))
            }
        }

        var expected = swiGLU(shared)
        for slot in chosen.indices {
            let routed = swiGLU(experts[slot])
            for i in 0..<hiddenSize {
                expected[i] = Float(Float16(expected[i] + weights[slot] * routed[i]))
            }
        }
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
