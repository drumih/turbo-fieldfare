import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// K2 dense-layer arithmetic on Metal: grouped norm, q/k/v plus the separate
/// gate projection, RoPE, GQA over a plain K/V cache, softplus gate, o_proj and
/// the SwiGLU MLP. The synthetic install's weights are the known 0.01 constant,
/// so the reference is a direct transcription of the CPU formulas.
@Suite struct K2DenseLayerTests {
    private static let eps: Float = 1e-6
    private static let weight: Float = 0.01

    private struct LayerState {
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let gate: [Float]
        let normed: [Float]
    }

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
        let value = s > 0 ? s + log(1 + exp(-s)) : log(1 + exp(s))
        return value / beta
    }

    private static func state(
        hidden: [Float], position: Int, config: ArchConfig
    ) -> LayerState {
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let normed = groupNorm(hidden, groups: config.normGroups)
        let q = gemv(normed, rows: queryDim)
        let k = gemv(normed, rows: kvDim)
        let v = gemv(normed, rows: kvDim)
        let gate = gemv(normed, rows: queryDim)
        let rotary = config.fullHeadDim * Int(config.partialRotaryFactor)
        let ropeQ = RopeRef.applyNeox(
            input: q, numTokens: 1, numHeads: config.numHeads,
            headDim: config.fullHeadDim, rotatedPairs: rotary / 2,
            position: position, theta: Float(config.fullRopeTheta))
        let ropeK = RopeRef.applyNeox(
            input: k, numTokens: 1, numHeads: config.numFullKVHeads,
            headDim: config.fullHeadDim, rotatedPairs: rotary / 2,
            position: position, theta: Float(config.fullRopeTheta))
        return LayerState(
            q: ropeQ.map { Float(Float16($0)) },
            k: ropeK.map { Float(Float16($0)) },
            v: v, gate: gate, normed: normed)
    }

    private static func finish(
        hidden: [Float], states: [LayerState], config: ArchConfig
    ) -> [Float] {
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let ffn = config.intermediateSize
        let scale = Float(config.attentionScale)
        let headsPerKV = config.numHeads / config.numFullKVHeads
        let current = states[states.count - 1]

        var attn = [Float](repeating: 0, count: queryDim)
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            let q = Array(
                current.q[(head * config.fullHeadDim)..<((head + 1) * config.fullHeadDim)])
            var scores = [Float](repeating: 0, count: states.count)
            for (position, state) in states.enumerated() {
                let k = Array(
                    state.k[(kvHead * config.fullHeadDim)..<((kvHead + 1) * config.fullHeadDim)])
                scores[position] = zip(q, k).reduce(Float(0)) { $0 + $1.0 * $1.1 } * scale
            }
            let maximum = scores.max() ?? 0
            let exps = scores.map { exp($0 - maximum) }
            let total = exps.reduce(Float(0), +)
            for (position, state) in states.enumerated() {
                let p = exps[position] / total
                let v = Array(
                    state.v[(kvHead * config.fullHeadDim)..<((kvHead + 1) * config.fullHeadDim)])
                for i in 0..<config.fullHeadDim {
                    attn[head * config.fullHeadDim + i] += p * v[i]
                }
            }
            for i in 0..<config.fullHeadDim {
                let index = head * config.fullHeadDim + i
                attn[index] = Float(Float16(attn[index] * softplus(current.gate[index])))
            }
        }
        let o = gemv(attn, rows: d)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(hidden[i] + o[i])) }
        let normed2 = groupNorm(h1, groups: config.normGroups)
        let up = gemv(normed2, rows: ffn)
        let gate = gemv(normed2, rows: ffn)
        var act = [Float](repeating: 0, count: ffn)
        for i in 0..<ffn {
            let g = gate[i]
            act[i] = Float(Float16((g / (1 + exp(-g))) * up[i]))
        }
        let down = gemv(act, rows: d)
        return (0..<d).map { Float(Float16(h1[$0] + down[$0])) }
    }

    @Test func oneTokenMatchesTheReference() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let layer = try K2DenseLayer(model: model, context: ctx, layer: 0)

        var rng = SeedTree(0x2DE_5EED).key("k2-dense-one")
        let hidden = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let hiddenBuf = try #require(Fp16Buffer.make(ctx.device, halves: hidden))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, count: config.hiddenSize))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try layer.encode(commandBuffer: cb, hidden: hiddenBuf, out: outBuf, position: 0)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: config.hiddenSize)
        let state = Self.state(
            hidden: hidden.map(Float.init), position: 0, config: config)
        let expected = Self.finish(
            hidden: hidden.map(Float.init), states: [state], config: config)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }

    @Test func secondTokenAttendsOverTheCache() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let layer = try K2DenseLayer(
            model: model, context: ctx, layer: 0, maxSequence: 4)

        var rng = SeedTree(0xCAC_4E11).key("k2-dense-cache")
        let first = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let second = (0..<config.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let firstBuf = try #require(Fp16Buffer.make(ctx.device, halves: first))
        let secondBuf = try #require(Fp16Buffer.make(ctx.device, halves: second))
        let scratch = try #require(Fp16Buffer.make(ctx.device, count: config.hiddenSize))

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try layer.encode(commandBuffer: cb, hidden: firstBuf, out: scratch, position: 0)
        try layer.encode(commandBuffer: cb, hidden: secondBuf, out: scratch, position: 1)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(scratch, count: config.hiddenSize)
        let states = [
            Self.state(hidden: first.map(Float.init), position: 0, config: config),
            Self.state(hidden: second.map(Float.init), position: 1, config: config),
        ]
        let expected = Self.finish(
            hidden: second.map(Float.init), states: states, config: config)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
