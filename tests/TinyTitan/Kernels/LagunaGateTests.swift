import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// Laguna applies one plain softplus gate per head, broadcast across that
/// head's dimensions, before o_proj. K2's kernel is a different function
/// (softplus with beta = ln 2), so the per-head form has its own test against
/// the formula rather than borrowing that one's.
@Suite struct LagunaGateTests {
    private static func softplus(_ x: Float) -> Float {
        x > 0 ? x + log(1 + exp(-x)) : log(1 + exp(x))
    }

    @Test func multipliesEachHeadByItsOwnSoftplus() throws {
        let ctx = try MetalContext()
        let elementwise = try Elementwise(context: ctx)
        let heads = 3
        let headDim = 8
        let count = heads * headDim
        let values = (0..<count).map { Float($0) / Float(count) + 0.25 }
        let gates: [Float] = [0, 1, -1]
        let out = try #require(Fp16Buffer.make(ctx.device, values: values))
        let gate = try #require(Fp16Buffer.make(ctx.device, values: gates))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try elementwise.encodeSoftplusGatePerHeadMul(
            commandBuffer: cb, out: out, gate: gate,
            count: count, headDim: headDim)
        cb.commit()
        cb.waitUntilCompleted()
        let got = Fp16Buffer.read(out, count: count)
        for index in 0..<count {
            let head = index / headDim
            let want = values[index] * Self.softplus(gates[head])
            #expect(abs(got[index] - want) < 2e-3, "element \(index)")
        }
    }
}
