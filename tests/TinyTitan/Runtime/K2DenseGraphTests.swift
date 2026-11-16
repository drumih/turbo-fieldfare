import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
@testable import TinyTitanCore

@testable import TinyTitan

/// A full dense-only token through the GPU graph on the synthetic install:
/// embedding → three K2 layers (each with its own cached attention) → final
/// norm → untied head. The reference chains the same formulas used by the
/// single-layer tests.
@Suite struct K2DenseGraphTests {
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

    private struct LayerState {
        let q: [Float]
        let k: [Float]
        let v: [Float]
        let gate: [Float]
    }

    private static func state(
        _ hidden: [Float], position: Int, config: ArchConfig
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
            v: v, gate: gate)
    }

    private static func finish(
        _ hidden: [Float], states: [LayerState], config: ArchConfig
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
        let mlpGate = gemv(normed2, rows: ffn)
        var act = [Float](repeating: 0, count: ffn)
        for i in 0..<ffn {
            let g = mlpGate[i]
            act[i] = Float(Float16((g / (1 + exp(-g))) * up[i]))
        }
        let down = gemv(act, rows: d)
        return (0..<d).map { Float(Float16(h1[$0] + down[$0])) }
    }

    private static func layer(_ hidden: [Float], config: ArchConfig) -> [Float] {
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let ffn = config.intermediateSize
        let normed = groupNorm(hidden, groups: config.normGroups)
        let q = gemv(normed, rows: queryDim)
        let v = gemv(normed, rows: kvDim)
        let gate = gemv(normed, rows: queryDim)
        var attn = [Float](repeating: 0, count: queryDim)
        let headsPerKV = config.numHeads / config.numFullKVHeads
        for head in 0..<config.numHeads {
            let kvHead = head / headsPerKV
            for i in 0..<config.fullHeadDim {
                let index = head * config.fullHeadDim + i
                let value = v[kvHead * config.fullHeadDim + i]
                // One token: attention is the value, then the gate.
                attn[index] = Float(Float16(value * softplus(gate[index])))
            }
        }
        _ = q
        let o = gemv(attn, rows: d)
        var h1 = [Float](repeating: 0, count: d)
        for i in 0..<d { h1[i] = Float(Float16(hidden[i] + o[i])) }
        let normed2 = groupNorm(h1, groups: config.normGroups)
        let up = gemv(normed2, rows: ffn)
        let mlpGate = gemv(normed2, rows: ffn)
        var act = [Float](repeating: 0, count: ffn)
        for i in 0..<ffn {
            let g = mlpGate[i]
            act[i] = Float(Float16((g / (1 + exp(-g))) * up[i]))
        }
        let down = gemv(act, rows: d)
        return (0..<d).map { Float(Float16(h1[$0] + down[$0])) }
    }

    @Test func oneTokenThroughTheWholeDenseGraph() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let graph = try K2DenseGraph(model: model, context: ctx, maxSequence: 2)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try graph.encode(commandBuffer: cb, tokenID: 7, position: 0)
        cb.commit()
        cb.waitUntilCompleted()

        // Embedding table: packed code 1, scale 0.01.
        var hidden = [Float](repeating: 0.01, count: config.hiddenSize)
        for _ in 0..<config.numLayers { hidden = Self.layer(hidden, config: config) }
        let normed = Self.groupNorm(hidden, groups: config.normGroups)
        let dot = normed.reduce(Float(0), +) * Self.weight
        let expected = [Float](
            repeating: Float16(dot) == 0 ? 0 : Float(Float16(dot)),
            count: config.vocabSize)
        let actual = graph.readLogits()
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }

    @Test func secondTokenUsesEveryLayersCache() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        let graph = try K2DenseGraph(model: model, context: ctx, maxSequence: 2)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try graph.encode(commandBuffer: cb, tokenID: 7, position: 0)
        try graph.encode(commandBuffer: cb, tokenID: 8, position: 1)
        cb.commit()
        cb.waitUntilCompleted()

        var caches = [[Self.LayerState]](repeating: [], count: config.numLayers)
        var hidden = [Float](repeating: 0.01, count: config.hiddenSize)
        for layer in 0..<config.numLayers {
            let state = Self.state(hidden, position: 0, config: config)
            caches[layer] = [state]
            hidden = Self.finish(hidden, states: [state], config: config)
        }
        hidden = [Float](repeating: 0.01, count: config.hiddenSize)
        for layer in 0..<config.numLayers {
            let state = Self.state(hidden, position: 1, config: config)
            caches[layer].append(state)
            hidden = Self.finish(hidden, states: caches[layer], config: config)
        }
        let normed = Self.groupNorm(hidden, groups: config.normGroups)
        let dot = normed.reduce(Float(0), +) * Self.weight
        let expected = [Float](repeating: Float(Float16(dot)), count: config.vocabSize)
        let actual = graph.readLogits()
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16Reduction, "relErr=\(rel)")
    }
}
