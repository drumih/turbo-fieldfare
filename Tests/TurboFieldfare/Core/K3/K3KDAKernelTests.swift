import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Drives the KDA decode kernels (`k3_kda_conv`, `k3_kda_step`,
/// `k3_kda_onorm`) against `K3KDAReference` (a straight port of the
/// fla-verified k3_ops.c) on small shapes, replaying multi-token sequences
/// token by token through the in-place conv state and recurrent state.
@Suite struct K3KDAKernelTests {

    // MARK: helpers

    private static func makeF32Buffer(_ device: MTLDevice, _ values: [Float])
        -> MTLBuffer?
    {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
    }

    private static func readF32Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    private struct Weights {
        var conv: [Float]      // [3][P][4]
        var aLog: [Float]      // [H]
        var onorm: [Float]     // [D]
    }

    private struct Token {
        var xq, xk, xv: [Float16]
        var z: [Float]
        var betaLogits: [Float]
        var gate: [Float16]
    }

    private static func makeWeights(p: Int, h: Int, d: Int, rng: inout SplitMix64)
        -> Weights
    {
        Weights(
            conv: (0..<(3 * p * 4)).map { _ in rng.uniform(-0.5, 0.5) },
            // exp(A_log) ~ uniform(1, 16), matching the HF init distribution.
            aLog: (0..<h).map { _ in rng.uniform(0, 2.8) },
            onorm: (0..<d).map { _ in rng.uniform(0.5, 1.5) })
    }

    private static func makeToken(p: Int, h: Int, rng: inout SplitMix64,
                                  zRange: ClosedRange<Float> = -2.0...2.0)
        -> Token
    {
        Token(
            xq: (0..<p).map { _ in Float16(rng.uniform(-1.0, 1.0)) },
            xk: (0..<p).map { _ in Float16(rng.uniform(-1.0, 1.0)) },
            xv: (0..<p).map { _ in Float16(rng.uniform(-1.0, 1.0)) },
            z: (0..<p).map { _ in rng.uniform(zRange.lowerBound,
                                              zRange.upperBound) },
            betaLogits: (0..<h).map { _ in rng.uniform(-2.0, 2.0) },
            gate: (0..<p).map { _ in Float16(rng.uniform(-1.5, 1.5)) })
    }

    /// GPU replay: returns the per-token fp16 output and the final state S.
    private static func runGPU(weights: Weights, tokens: [Token],
                               numHeads: Int, headDim: Int) throws
        -> (outputs: [[Float]], state: [Float], convOut: [[Float]])
    {
        let P = numHeads * headDim
        let ctx = try MetalContext()
        let kda = try K3KDA(context: ctx)

        guard let wBuf = makeF32Buffer(ctx.device, weights.conv),
              let aLogBuf = makeF32Buffer(ctx.device, weights.aLog),
              let onormBuf = makeF32Buffer(ctx.device, weights.onorm),
              let convStates = ctx.device.makeBuffer(
                length: 3 * P * K3KDA.convHist * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let state = ctx.device.makeBuffer(
                length: numHeads * headDim * headDim * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let qOut = makeF32Buffer(ctx.device, [Float](repeating: 0, count: P)),
              let kOut = makeF32Buffer(ctx.device, [Float](repeating: 0, count: P)),
              let vOut = makeF32Buffer(ctx.device, [Float](repeating: 0, count: P)),
              let stepOut = makeF32Buffer(ctx.device, [Float](repeating: 0, count: P)),
              let finalOut = Fp16Buffer.make(ctx.device, count: P) else {
            Issue.record("Failed to allocate buffers")
            return ([], [], [])
        }
        // Zero-init both states (fresh sequence = zeroed history).
        memset(convStates.contents(), 0,
               3 * P * K3KDA.convHist * MemoryLayout<Float>.stride)
        memset(state.contents(), 0,
               numHeads * headDim * headDim * MemoryLayout<Float>.stride)

        var outputs: [[Float]] = []
        var convOuts: [[Float]] = []
        for token in tokens {
            guard let xq = Fp16Buffer.make(ctx.device, halves: token.xq),
                  let xk = Fp16Buffer.make(ctx.device, halves: token.xk),
                  let xv = Fp16Buffer.make(ctx.device, halves: token.xv),
                  let zBuf = makeF32Buffer(ctx.device, token.z),
                  let betaBuf = makeF32Buffer(ctx.device, token.betaLogits),
                  let gateBuf = Fp16Buffer.make(ctx.device, halves: token.gate),
                  let cmd = ctx.queue.makeCommandBuffer() else {
                Issue.record("Failed to allocate token buffers")
                return ([], [], [])
            }
            kda.encodeConv(commandBuffer: cmd,
                           xq: xq, xk: xk, xv: xv,
                           weights: wBuf, convStates: convStates,
                           qOut: qOut, kOut: kOut, vOut: vOut,
                           channels: UInt32(P))
            kda.encodeStep(commandBuffer: cmd,
                           state: state, q: qOut, k: kOut, v: vOut,
                           z: zBuf, betaLogits: betaBuf, aLog: aLogBuf,
                           o: stepOut,
                           numHeads: UInt32(numHeads), headDim: UInt32(headDim))
            kda.encodeOutputNorm(commandBuffer: cmd,
                                 o: stepOut, gate: gateBuf, weight: onormBuf,
                                 out: finalOut, eps: 1e-5,
                                 numHeads: UInt32(numHeads),
                                 headDim: UInt32(headDim))
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)
            outputs.append(Fp16Buffer.read(finalOut, count: P))
            convOuts.append(readF32Buffer(qOut, count: P))
        }
        return (outputs,
                readF32Buffer(state, count: numHeads * headDim * headDim),
                convOuts)
    }

    /// Reference replay of the same token stream.
    private static func runReference(weights: Weights, tokens: [Token],
                                     numHeads: Int, headDim: Int)
        -> (outputs: [[Float]], state: [Float], convQ: [[Float]])
    {
        let P = numHeads * headDim
        var convState = [Float](repeating: 0, count: 3 * P * K3KDA.convHist)
        var state = [Float](repeating: 0, count: numHeads * headDim * headDim)
        var outputs: [[Float]] = []
        var convQ: [[Float]] = []
        for token in tokens {
            var xqState = Array(convState[0..<(P * 3)])
            var xkState = Array(convState[(P * 3)..<(2 * P * 3)])
            var xvState = Array(convState[(2 * P * 3)..<(3 * P * 3)])
            let w = weights.conv
            let q = K3KDAReference.convStep(
                x: token.xq.map { Float($0) },
                weights: Array(w[0..<(P * 4)]), state: &xqState)
            let k = K3KDAReference.convStep(
                x: token.xk.map { Float($0) },
                weights: Array(w[(P * 4)..<(2 * P * 4)]), state: &xkState)
            let v = K3KDAReference.convStep(
                x: token.xv.map { Float($0) },
                weights: Array(w[(2 * P * 4)..<(3 * P * 4)]), state: &xvState)
            convState.replaceSubrange(0..<(P * 3), with: xqState)
            convState.replaceSubrange((P * 3)..<(2 * P * 3), with: xkState)
            convState.replaceSubrange((2 * P * 3)..<(3 * P * 3), with: xvState)
            convQ.append(q)

            let o = K3KDAReference.step(
                state: &state, q: q, k: k, v: v,
                z: token.z, betaLogits: token.betaLogits, aLog: weights.aLog,
                numHeads: numHeads, headDim: headDim)
            outputs.append(K3KDAReference.outputNorm(
                o: o, gate: token.gate.map { Float($0) },
                weight: weights.onorm, eps: 1e-5,
                numHeads: numHeads, headDim: headDim))
        }
        return (outputs, state, convQ)
    }

    // MARK: tests

    /// Conv + SiLU in isolation, 12 tokens through the rolling state —
    /// pins the history layout (oldest first, tap 3 = current input).
    @Test func convMatchesReferenceMultiToken() throws {
        let (h, d) = (4, 16)
        let p = h * d
        var rng = SeedTree(0xE1).key("k3-kda-conv")
        let weights = Self.makeWeights(p: p, h: h, d: d, rng: &rng)
        let tokens = (0..<12).map { _ in Self.makeToken(p: p, h: h, rng: &rng) }
        let (_, _, gpuConv) = try Self.runGPU(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let (_, _, refConv) = Self.runReference(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        for (t, pair) in zip(gpuConv, refConv).enumerated() {
            let rel = RelError.compute(actual: pair.0, reference: pair.1)
            #expect(rel < Tolerance.identity, "token \(t): conv rel=\(rel)")
        }
    }

    /// Full KDA stage chain per token, plus the final recurrent state —
    /// validates the state-update-then-output order over a 12-token replay.
    @Test func stepAndOnormMatchReferenceMultiToken() throws {
        let (h, d) = (4, 16)
        let p = h * d
        var rng = SeedTree(0xE2).key("k3-kda-step")
        let weights = Self.makeWeights(p: p, h: h, d: d, rng: &rng)
        let tokens = (0..<12).map { _ in Self.makeToken(p: p, h: h, rng: &rng) }
        let (gpuOut, gpuState, _) = try Self.runGPU(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let (refOut, refState, _) = Self.runReference(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        for (t, pair) in zip(gpuOut, refOut).enumerated() {
            let rel = RelError.compute(actual: pair.0, reference: pair.1)
            let maxAbs = RelError.maxAbsDiff(pair.0, pair.1)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "token \(t): out rel=\(rel) maxAbs=\(maxAbs)")
        }
        let relS = RelError.compute(actual: gpuState, reference: refState)
        #expect(relS < Tolerance.fp16Reduction, "final S rel=\(relS)")
    }

    /// Decay rails: extreme z drives sigmoid to saturation, so alpha hits
    /// exp(-5) (z -> +inf) and 1 (z -> -inf). The reference asserts the
    /// bounds directly; the GPU step must track it exactly at the rails.
    @Test func decayGateRails() throws {
        let (h, d) = (4, 16)
        let p = h * d
        var rng = SeedTree(0xE3).key("k3-kda-rails")
        let weights = Self.makeWeights(p: p, h: h, d: d, rng: &rng)
        var tokens = (0..<3).map { _ in Self.makeToken(p: p, h: h, rng: &rng) }
        // Token 1: huge positive z (alpha = e^-5); token 2: huge negative
        // (alpha = 1).
        for i in 0..<p { tokens[1].z[i] = 100 + Float(i % 7) }
        for i in 0..<p { tokens[2].z[i] = -100 - Float(i % 5) }

        let alpha1 = K3KDAReference.decayAlpha(
            z: tokens[1].z, aLog: weights.aLog, numHeads: h, headDim: d)
        let alpha2 = K3KDAReference.decayAlpha(
            z: tokens[2].z, aLog: weights.aLog, numHeads: h, headDim: d)
        let minAlpha = exp(Float(-5))
        #expect(alpha1.allSatisfy { abs($0 - minAlpha) < 1e-6 },
                "saturated positive z should give alpha == e^-5")
        #expect(alpha2.allSatisfy { abs($0 - 1) < 1e-6 },
                "saturated negative z should give alpha == 1")

        let (gpuOut, gpuState, _) = try Self.runGPU(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let (refOut, refState, _) = Self.runReference(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        for (t, pair) in zip(gpuOut, refOut).enumerated() {
            let rel = RelError.compute(actual: pair.0, reference: pair.1)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "token \(t): out rel=\(rel)")
        }
        let relS = RelError.compute(actual: gpuState, reference: refState)
        #expect(relS < Tolerance.fp16Reduction, "final S rel=\(relS)")
    }

    /// One token from zeroed state: conv is a pure 1-tap + SiLU and S is
    /// the rank-one write — the minimal contract a streaming integration
    /// relies on.
    @Test func singleTokenFromZeroState() throws {
        let (h, d) = (2, 16)
        let p = h * d
        var rng = SeedTree(0xE4).key("k3-kda-single")
        let weights = Self.makeWeights(p: p, h: h, d: d, rng: &rng)
        let tokens = [Self.makeToken(p: p, h: h, rng: &rng)]
        let (gpuOut, gpuState, _) = try Self.runGPU(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let (refOut, refState, _) = Self.runReference(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let rel = RelError.compute(actual: gpuOut[0], reference: refOut[0])
        #expect(rel < Tolerance.fp16ChainedReduction, "out rel=\(rel)")
        let relS = RelError.compute(actual: gpuState, reference: refState)
        #expect(relS < Tolerance.fp16Reduction, "S rel=\(relS)")
    }

    /// Different head geometry (D = 32) to shake out dimension handling.
    @Test func stepHeadDim32() throws {
        let (h, d) = (3, 32)
        let p = h * d
        var rng = SeedTree(0xE5).key("k3-kda-d32")
        let weights = Self.makeWeights(p: p, h: h, d: d, rng: &rng)
        let tokens = (0..<6).map { _ in Self.makeToken(p: p, h: h, rng: &rng) }
        let (gpuOut, gpuState, _) = try Self.runGPU(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        let (refOut, refState, _) = Self.runReference(
            weights: weights, tokens: tokens, numHeads: h, headDim: d)
        for (t, pair) in zip(gpuOut, refOut).enumerated() {
            let rel = RelError.compute(actual: pair.0, reference: pair.1)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "token \(t): out rel=\(rel)")
        }
        let relS = RelError.compute(actual: gpuState, reference: refState)
        #expect(relS < Tolerance.fp16Reduction, "final S rel=\(relS)")
    }
}
