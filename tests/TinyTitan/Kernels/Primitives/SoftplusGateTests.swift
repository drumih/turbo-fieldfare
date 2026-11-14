import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
import TinyTitanCore

/// K2-Horizon's attention output gate is `softplus(x; beta = ln 2)`, not the
/// sigmoid the Qwen families use. The two curves are both smooth, positive and
/// bounded, so reading one as the other is a wrong model that answers fluently;
/// this pins the checkpoint's formula.
@Suite struct SoftplusGateTests {
    private static func reference(_ out: [Float], _ gate: [Float]) -> [Float] {
        let beta = Float(log(2.0))
        return zip(out, gate).map { value, g in
            let s = beta * g
            let sp = s > 0 ? s + log(1 + exp(-s)) : log(1 + exp(s))
            return value * (sp / beta)
        }
    }

    private static func run(gate: [Float], out: [Float]) throws -> [Float] {
        let ctx = try MetalContext()
        let kernel = try Elementwise(context: ctx)
        let gateHalf = gate.map { Float16($0) }
        let outHalf = out.map { Float16($0) }
        let gateBuf = try #require(Fp16Buffer.make(ctx.device, halves: gateHalf))
        let outBuf = try #require(Fp16Buffer.make(ctx.device, halves: outHalf))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeSoftplusGateMul(
            commandBuffer: cb, out: outBuf, gate: gateBuf, count: gate.count)
        cb.commit()
        cb.waitUntilCompleted()
        return Fp16Buffer.read(outBuf, count: out.count)
    }

    @Test func matchesTheCheckpointFormulaAtBothTails() throws {
        var rng = SeedTree(0x5F7_1CE).key("k2-softplus-gate")
        let gate = (0..<256).map { _ in rng.uniform(-6, 6) }
        let out = (0..<256).map { _ in rng.uniform(-2, 2) }
        // The reference reads the same rounded inputs the kernel does.
        let gateHalf = gate.map { Float($0) }
        let outHalf = out.map { Float($0) }
        let expected = Self.reference(
            outHalf.map { Float(Float16($0)) },
            gateHalf.map { Float(Float16($0)) })
        let actual = try Self.run(gate: gate, out: out)
        let relError = RelError.compute(actual: actual, reference: expected)
        #expect(relError < Tolerance.fp16Reduction, "relErr=\(relError)")
    }

    /// At large positive `x` the base-2 softplus approaches `x`; at large
    /// negative it approaches 0. A naive `log(1 + exp(s))` overflows the
    /// positive side, so the tails are the part worth pinning.
    @Test func isStableAtBothExtremes() throws {
        let gate: [Float] = [-60, -20, -8, 0, 8, 20, 60]
        let actual = try Self.run(gate: gate, out: [Float](repeating: 1, count: gate.count))
        let expected = Self.reference(
            [Float](repeating: 1, count: gate.count),
            gate.map { Float(Float16($0)) })
        for (index, pair) in zip(actual, expected).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= max(1e-3, abs(pair.1) * 5e-3),
                "index \(index): actual \(pair.0) expected \(pair.1)")
        }
        #expect(actual[0] >= 0 && actual[0] < 1e-3)
        #expect(abs(actual[6] - 60) < 0.2)
    }
}
