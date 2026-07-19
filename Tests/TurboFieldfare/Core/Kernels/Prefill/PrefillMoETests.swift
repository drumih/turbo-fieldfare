import Metal
import Testing

@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillMoETests {
    @Test func tokenMajorReduceMatchesCPUReference() throws {
        let rows = 3
        let topK = 8
        let d = 64
        var rng = SplitMix64(seed: 0x4D0E_2026)
        let partials = (0..<(rows * topK * d)).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let weights = (0..<(rows * topK)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let sentinel = [Float16](repeating: Float16(-19.0), count: rows * d)
        let expected = Self.cpuReduce(routePartials: partials,
                                      routeWeights: weights,
                                      rows: rows,
                                      topK: topK,
                                      d: d)

        let ctx = try MetalContext()
        let prefill = try PrefillMoE(context: ctx)
        guard let partialsBuf = Fp16Buffer.make(ctx.device, halves: partials),
              let weightsBuf = Fp16Buffer.make(ctx.device, halves: weights),
              let h2Buf = Fp16Buffer.make(ctx.device, halves: sentinel),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("alloc failed")
            return
        }
        prefill.encodeReduceTokenMajor(commandBuffer: cb,
                                       routePartials: partialsBuf,
                                       routeWeights: weightsBuf,
                                       h2: h2Buf,
                                       queryCount: UInt32(rows),
                                       topK: UInt32(topK),
                                       d: UInt32(d))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let got = Fp16Buffer.readHalf(h2Buf, count: rows * d)
        Self.assertClose(got, expected, tolerance: 2e-3)
    }

    private static func cpuReduce(routePartials: [Float16],
                                  routeWeights: [Float16],
                                  rows: Int,
                                  topK: Int,
                                  d: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: rows * d)
        for token in 0..<rows {
            for column in 0..<d {
                var accumulator: Float = 0
                for rank in 0..<topK {
                    let partialIndex = (token * topK + rank) * d + column
                    accumulator += Float(routeWeights[token * topK + rank])
                        * Float(routePartials[partialIndex])
                }
                out[token * d + column] = Float16(accumulator)
            }
        }
        return out
    }

    private static func assertClose(_ actual: [Float16],
                                    _ expected: [Float16],
                                    tolerance: Float) {
        let maxAbsoluteDifference = zip(actual, expected)
            .map { abs(Float($0) - Float($1)) }
            .max() ?? 0
        #expect(maxAbsoluteDifference <= tolerance, "maxAbsDiff=\(maxAbsoluteDifference)")
    }
}
