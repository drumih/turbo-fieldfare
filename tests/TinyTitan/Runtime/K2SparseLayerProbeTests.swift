import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

/// One K2 sparse layer on the GPU against the CPU formulas: MoVA values from
/// four resident experts, one-token gated attention, then the bf16 MoE router,
/// the fetched routed expert and the shared expert. The toy install's resident
/// tensors are the 0.01 constant; the routed expert is dequantized from the
/// same fetched bytes the layer used.
@Suite struct K2SparseLayerProbeTests {
    private static let eps: Float = 1e-6
    private static let weight: Float = 0.01

    private static func groupNorm(_ x: [Float], groups: Int) -> [Float] {
        let groupDim = x.count / groups
        var out = [Float](repeating: 0, count: x.count)
        for group in 0..<groups {
            let range = (group * groupDim)..<((group + 1) * groupDim)
            let mean = range.reduce(Float(0)) { $0 + x[$1] * x[$1] } / Float(groupDim)
            let inv = 1 / (mean + eps).squareRoot()
            for i in range { out[i] = Float(Float16(x[i] * inv)) }
        }
        return out
    }

    private static func gemv(_ x: [Float], rows: Int) -> [Float] {
        let dot = x.reduce(Float(0), +)
        return (0..<rows).map { _ in Float(Float16(dot * weight)) }
    }

    private static func softplus(_ x: Float) -> Float {
        let beta = Float(log(2.0))
        let s = beta * x
        return (s > 0 ? s + log(1 + exp(-s)) : log(1 + exp(s))) / beta
    }

    private static func dequant(_ view: TensorView, rows: Int, columns: Int) -> [[Float]] {
        let bits = Int(view.length) * 8 / (rows * columns)
        let groups = columns / Quantization.groupSize
        let base = view.buffer.contents()
        return (0..<rows).map { row in
            (0..<columns).map { column in
                let index = row * columns + column
                let wordOffset = index * bits / 32
                let shift = (index * bits) % 32
                let word = base.load(
                    fromByteOffset: Int(view.offset) + wordOffset * 4, as: UInt32.self)
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

    private static func swiGLU(
        gate: [[Float]], up: [[Float]], down: [[Float]], x: [Float]
    ) -> [Float] {
        let width = gate.count
        var act = [Float](repeating: 0, count: width)
        for row in 0..<width {
            let g = zip(gate[row], x).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            let u = zip(up[row], x).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            let gh = Float(Float16(g))
            let uh = Float(Float16(u))
            act[row] = Float(Float16((gh / (1 + exp(-gh))) * uh))
        }
        return (0..<x.count).map { row in
            Float(Float16(zip(down[row], act).reduce(Float(0)) { $0 + $1.0 * $1.1 }))
        }
    }

    @Test func sparseLayerMatchesTheReference() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let layer = config.numLayers - 1
        let probe = try K2SparseLayerProbe(
            model: model, context: ctx, layer: layer)

        var rng = SeedTree(0x5A17_0001).key("k2-sparse-probe")
        let hidden = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let actual = try await probe.run(
            queue: ctx.queue, hidden: hidden, position: 0)

        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let width = config.moeIntermediateSize
        let x = hidden.map(Float.init)
        let normed = Self.groupNorm(x, groups: config.normGroups)
        let gate = Self.gemv(normed, rows: queryDim)
        let dot = normed.reduce(Float(0), +)
        // MoVA: the router is bf16 1.0 with bias 1.0, so every expert ties and
        // top-4 takes all four; each value expert is the 0.01 constant.
        let expertValue = Float(Float16(dot * Self.weight))
        let silu = Float(Float16(expertValue / (1 + exp(-expertValue))))
        let value = [Float](repeating: Float(Float16(silu * 2.5)), count: kvDim)
        var attn = [Float](repeating: 0, count: queryDim)
        let headsPerKV = config.numHeads / config.numFullKVHeads
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            for i in 0..<config.fullHeadDim {
                let index = head * config.fullHeadDim + i
                attn[index] = Float(
                    Float16(
                        value[kvHead * config.fullHeadDim + i]
                            * Self.softplus(gate[index])))
            }
        }
        let o = Self.gemv(attn, rows: d)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(x[i] + o[i])) }
        let normed2 = Self.groupNorm(h1, groups: config.normGroups)

        // The routed expert is the fetched blob; dequantize it from the same
        // plan the layer used.
        let plan = try #require(try model.planRoutedExperts(layer: layer, experts: [0]))
        let blobs = try await model.fetchRoutedExperts(plan: plan)
        let entry = try model.packedExpertsLayout.expert(layer: layer, expert: 0)
        let weights = try K2MoEStage.expertWeights(blob: blobs[0], entry: entry)
        let expertOut = Self.swiGLU(
            gate: Self.dequant(weights.gate, rows: width, columns: d),
            up: Self.dequant(weights.up, rows: width, columns: d),
            down: Self.dequant(weights.down, rows: d, columns: width),
            x: normed2)
        let sharedOut = Self.swiGLU(
            gate: Self.dequant(try model.k2SharedGate(layer: layer), rows: width, columns: d),
            up: Self.dequant(try model.k2SharedUp(layer: layer), rows: width, columns: d),
            down: Self.dequant(try model.k2SharedDown(layer: layer), rows: d, columns: width),
            x: normed2)

        // Router logits tie, so expert 0 wins the lower-id tie-break and the
        // single selected score normalizes to 1, times the 2.5 scale.
        let expertWeight = Float(Float16(2.5))
        let expected = (0..<d).map {
            Float(Float16(h1[$0] + sharedOut[$0] + expertWeight * expertOut[$0]))
        }
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }

    @Test func secondTokenAttendsOverTheSparseCache() async throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let layer = config.numLayers - 1
        let probe = try K2SparseLayerProbe(
            model: model, context: ctx, layer: layer, maxSequence: 2)
        var rng = SeedTree(0x5A17_0002).key("k2-sparse-cache")
        let first = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let second = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        _ = try await probe.run(queue: ctx.queue, hidden: first, position: 0)
        let actual = try await probe.run(queue: ctx.queue, hidden: second, position: 1)

        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let width = config.moeIntermediateSize
        let headsPerKV = config.numHeads / config.numFullKVHeads
        let scale = Float(config.attentionScale)

        struct State {
            let q: [Float]
            let k: [Float]
            let v: [Float]
            let gate: [Float]
            let hidden: [Float]
        }
        func state(_ hidden: [Float], position: Int) -> State {
            let normed = Self.groupNorm(hidden, groups: config.normGroups)
            let q = Self.gemv(normed, rows: queryDim)
            let k = Self.gemv(normed, rows: kvDim)
            let gate = Self.gemv(normed, rows: queryDim)
            let dot = normed.reduce(Float(0), +)
            let expertValue = Float(Float16(dot * Self.weight))
            let silu = Float(Float16(expertValue / (1 + exp(-expertValue))))
            let v = [Float](repeating: Float(Float16(silu * 2.5)), count: kvDim)
            let rotary = config.fullHeadDim * Int(config.partialRotaryFactor)
            let ropeQ = RopeRef.applyNeox(
                input: q, numTokens: 1, numHeads: config.numHeads,
                headDim: config.fullHeadDim, rotatedPairs: rotary / 2,
                position: position, theta: Float(config.fullRopeTheta))
            let ropeK = RopeRef.applyNeox(
                input: k, numTokens: 1, numHeads: config.numFullKVHeads,
                headDim: config.fullHeadDim, rotatedPairs: rotary / 2,
                position: position, theta: Float(config.fullRopeTheta))
            return State(
                q: ropeQ.map { Float(Float16($0)) },
                k: ropeK.map { Float(Float16($0)) },
                v: v, gate: gate, hidden: hidden)
        }
        let states = [
            state(first.map(Float.init), position: 0),
            state(second.map(Float.init), position: 1),
        ]
        let current = states[1]
        var attn = [Float](repeating: 0, count: queryDim)
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            let q = Array(
                current.q[(head * config.fullHeadDim)..<((head + 1) * config.fullHeadDim)])
            var scores = [Float](repeating: 0, count: states.count)
            for (index, state) in states.enumerated() {
                let k = Array(
                    state.k[(kvHead * config.fullHeadDim)..<((kvHead + 1) * config.fullHeadDim)])
                scores[index] = zip(q, k).reduce(Float(0)) { $0 + $1.0 * $1.1 } * scale
            }
            let maximum = scores.max() ?? 0
            let exps = scores.map { exp($0 - maximum) }
            let total = exps.reduce(Float(0), +)
            for (index, state) in states.enumerated() {
                let p = exps[index] / total
                let v = Array(
                    state.v[(kvHead * config.fullHeadDim)..<((kvHead + 1) * config.fullHeadDim)])
                for i in 0..<config.fullHeadDim {
                    attn[head * config.fullHeadDim + i] += p * v[i]
                }
            }
            for i in 0..<config.fullHeadDim {
                let index = head * config.fullHeadDim + i
                attn[index] = Float(Float16(attn[index] * Self.softplus(current.gate[index])))
            }
        }
        let o = Self.gemv(attn, rows: d)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(current.hidden[i] + o[i])) }
        let normed2 = Self.groupNorm(h1, groups: config.normGroups)

        let plan = try #require(try model.planRoutedExperts(layer: layer, experts: [0]))
        let blobs = try await model.fetchRoutedExperts(plan: plan)
        let entry = try model.packedExpertsLayout.expert(layer: layer, expert: 0)
        let weights = try K2MoEStage.expertWeights(blob: blobs[0], entry: entry)
        let expertOut = Self.swiGLU(
            gate: Self.dequant(weights.gate, rows: width, columns: d),
            up: Self.dequant(weights.up, rows: width, columns: d),
            down: Self.dequant(weights.down, rows: d, columns: width),
            x: normed2)
        let sharedOut = Self.swiGLU(
            gate: Self.dequant(try model.k2SharedGate(layer: layer), rows: width, columns: d),
            up: Self.dequant(try model.k2SharedUp(layer: layer), rows: width, columns: d),
            down: Self.dequant(try model.k2SharedDown(layer: layer), rows: d, columns: width),
            x: normed2)
        let expertWeight = Float(Float16(2.5))
        let expected = (0..<d).map {
            Float(Float16(h1[$0] + sharedOut[$0] + expertWeight * expertOut[$0]))
        }
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
