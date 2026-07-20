import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillAttentionTests {
    private struct Fixture {
        var q: [Float]
        var k: [Float]
        var v: [Float]
        var qStride: Int
        var kvStride: Int
        var oStride: Int
        var headDim: Int
        var qHeads: Int
        var kvHeads: Int
        var start: Int
        var chunk: Int
        var kvValid: Int
        var window: Int
        var scale: Float
    }

    @Test func prefillAttentionMatchesCPUReferenceFullAndSWA() throws {
        let cases: [(label: String, start: Int, chunk: Int, window: Int)] = [
            ("full-origin", 0, 3, 0),
            ("full-offset", 4, 3, 0),
            ("swa-inside-window", 2, 4, 16),
            ("swa-truncated", 7, 4, 5),
        ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(start: c.start,
                                           chunk: c.chunk,
                                           window: c.window,
                                           seed: 0xA510 + UInt64(index))
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test func prefillAttentionRingCapacityMatchesLinearReference() throws {
        let fixture = Self.makeFixture(start: 20,
                                       chunk: 4,
                                       window: 8,
                                       seed: 0xA611,
                                       headDim: 32,
                                       qHeads: 4,
                                       kvHeads: 2)
        let ringCapacity = 16
        var kRing = [Float](repeating: 0, count: ringCapacity * fixture.kvStride)
        var vRing = [Float](repeating: 0, count: ringCapacity * fixture.kvStride)
        for p in 0..<fixture.kvValid {
            let dst = (p % ringCapacity) * fixture.kvStride
            let src = p * fixture.kvStride
            kRing.replaceSubrange(dst..<(dst + fixture.kvStride),
                                  with: fixture.k[src..<(src + fixture.kvStride)])
            vRing.replaceSubrange(dst..<(dst + fixture.kvStride),
                                  with: fixture.v[src..<(src + fixture.kvStride)])
        }

        var ringFixture = fixture
        ringFixture.k = kRing
        ringFixture.v = vRing
        let actual = try Self.runKernel(ringFixture, kvRingCapacity: UInt32(ringCapacity))
        let reference = Self.reference(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "ring prefill maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "ring prefill rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func prefillAttentionMasksFutureChunkTokens() throws {
        var fixture = Self.makeFixture(start: 5, chunk: 4, window: 0, seed: 0xA620)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        for row in fixture.start..<fixture.kvValid {
            for kvh in 0..<fixture.kvHeads {
                for d in 0..<fixture.headDim {
                    let base = row * fixture.kvStride + kvh * fixture.headDim + d
                    fixture.k[base] = Float(row - fixture.start + 1) * 2.0 + Float(d) * 0.125
                    fixture.v[base] = Float(row - fixture.start + 1) * -2.0 - Float(d) * 0.125
                }
            }
        }

        let actual = try Self.runKernel(fixture)
        let reference = Self.reference(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "future mask maxAbs=\(maxAbs) rel=\(rel) qPerKV=\(qPerKV)")
        #expect(rel <= 2e-2, "future mask rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func prefillAttentionProductionDimsBoundedVisibility() throws {
        let cases: [(label: String, start: Int, chunk: Int, window: Int, headDim: Int, qHeads: Int, kvHeads: Int)] = [
            ("swa-current-key-at-1023", 1023, 1, 1, 256, 16, 8),
            ("swa-current-key-at-1024", 1024, 1, 1, 256, 16, 8),
            ("swa-current-key-at-4095", 4095, 1, 1, 256, 16, 8),
            ("full-origin", 0, 1, 0, 512, 16, 2),
            ("full-short-gqa", 3, 1, 0, 512, 16, 2),
        ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(start: c.start,
                                           chunk: c.chunk,
                                           window: c.window,
                                           seed: 0xA730 + UInt64(index),
                                           headDim: c.headDim,
                                           qHeads: c.qHeads,
                                           kvHeads: c.kvHeads)
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test func prefillAttentionTiledProductionBoundarySmoke() throws {
        let cases: [(label: String, start: Int, chunk: Int, window: Int, headDim: Int, qHeads: Int, kvHeads: Int)] = [
            ("swa-production-window-1024", 1023, 4, 1024, 256, 16, 8),
            ("full-production-gqa", 31, 8, 0, 512, 16, 2),
        ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(start: c.start,
                                           chunk: c.chunk,
                                           window: c.window,
                                           seed: 0xA840 + UInt64(index),
                                           headDim: c.headDim,
                                           qHeads: c.qHeads,
                                           kvHeads: c.kvHeads)
            try Self.runAndCompare(fixture, label: c.label)
        }
    }


    private static func makeFixture(start: Int,
                                    chunk: Int,
                                    window: Int,
                                    seed: UInt64,
                                    headDim: Int = 8,
                                    qHeads: Int = 4,
                                    kvHeads: Int = 2) -> Fixture {
        let qStride = qHeads * headDim + 3
        let kvStride = kvHeads * headDim + 5
        let oStride = qHeads * headDim + 7
        let kvValid = start + chunk
        var rng = SeedTree(seed).key("prefill-attn-start\(start)-chunk\(chunk)-window\(window)")
        var q = [Float](repeating: 0, count: chunk * qStride)
        var k = [Float](repeating: 0, count: kvValid * kvStride)
        var v = [Float](repeating: 0, count: kvValid * kvStride)

        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    q[t * qStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        for pos in 0..<kvValid {
            for h in 0..<kvHeads {
                for d in 0..<headDim {
                    k[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                    v[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }

        return Fixture(q: q, k: k, v: v,
                       qStride: qStride, kvStride: kvStride, oStride: oStride,
                       headDim: headDim, qHeads: qHeads, kvHeads: kvHeads,
                       start: start, chunk: chunk, kvValid: kvValid,
                       window: window, scale: 1.0)
    }

    private static func runAndCompare(_ fixture: Fixture, label: String) throws {
        let actual = try Self.runKernel(fixture)
        let reference = Self.reference(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "\(label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "\(label) rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func runKernel(_ fixture: Fixture,
                                  kvRingCapacity: UInt32 = 0) throws -> [Float] {
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(context: ctx)
        let qPrefix = 17
        let kPrefix = 19
        let vPrefix = 23
        let oPrefix = 29
        let outCount = oPrefix + fixture.chunk * fixture.oStride

        guard let qBuf = Fp16Buffer.make(ctx.device,
                                         values: [Float](repeating: 0, count: qPrefix) + fixture.q),
              let kBuf = Fp16Buffer.make(ctx.device,
                                         values: [Float](repeating: 0, count: kPrefix) + fixture.k),
              let vBuf = Fp16Buffer.make(ctx.device,
                                         values: [Float](repeating: 0, count: vPrefix) + fixture.v),
              let outBuf = Fp16Buffer.make(ctx.device, count: outCount) else {
            Issue.record("alloc failed")
            return []
        }

        let params = PrefillAttentionParams(
            startPosition: UInt32(fixture.start),
            queryCount: UInt32(fixture.chunk),
            headDim: UInt32(fixture.headDim),
            numQHeads: UInt32(fixture.qHeads),
            numKVHeads: UInt32(fixture.kvHeads),
            kvValidCount: UInt32(fixture.kvValid),
            slidingWindow: UInt32(fixture.window),
            kvTokenStrideElements: UInt32(fixture.kvStride),
            qTokenStrideElements: UInt32(fixture.qStride),
            oTokenStrideElements: UInt32(fixture.oStride),
            scale: fixture.scale)

        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeCausal(commandBuffer: cb,
                             q: qBuf,
                             qOffset: qPrefix * MemoryLayout<Float16>.size,
                             k: kBuf,
                             kOffset: kPrefix * MemoryLayout<Float16>.size,
                             v: vBuf,
                             vOffset: vPrefix * MemoryLayout<Float16>.size,
                             out: outBuf,
                             outOffset: oPrefix * MemoryLayout<Float16>.size,
                             params: params,
                             kvRingCapacity: kvRingCapacity)
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(outBuf, count: outCount)
        var compact = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        for t in 0..<fixture.chunk {
            for h in 0..<fixture.qHeads {
                for d in 0..<fixture.headDim {
                    compact[(t * fixture.qHeads + h) * fixture.headDim + d] =
                        out[oPrefix + t * fixture.oStride + h * fixture.headDim + d]
                }
            }
        }
        return compact
    }



    private static func reference(_ fixture: Fixture) -> [Float] {
        var out = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        for t in 0..<fixture.chunk {
            let absQ = fixture.start + t
            let first: Int
            if fixture.window == 0 {
                first = 0
            } else {
                first = max(0, absQ + 1 - fixture.window)
            }
            let last = min(fixture.kvValid, absQ + 1)
            for qh in 0..<fixture.qHeads {
                let kvh = qh / qPerKV
                var scores: [Float] = []
                scores.reserveCapacity(last - first)
                for key in first..<last {
                    var score: Float = 0
                    for d in 0..<fixture.headDim {
                        let qv = fixture.q[t * fixture.qStride + qh * fixture.headDim + d]
                        let kv = fixture.k[key * fixture.kvStride + kvh * fixture.headDim + d]
                        score += qv * kv
                    }
                    scores.append(score * fixture.scale)
                }
                let maxScore = scores.max() ?? -.infinity
                var denom: Float = 0
                for score in scores {
                    denom += Foundation.exp(score - maxScore)
                }
                for d in 0..<fixture.headDim {
                    var acc: Float = 0
                    for (i, key) in (first..<last).enumerated() {
                        let w = Foundation.exp(scores[i] - maxScore)
                        acc += w * fixture.v[key * fixture.kvStride + kvh * fixture.headDim + d]
                    }
                    out[(t * fixture.qHeads + qh) * fixture.headDim + d] = denom > 0 ? acc / denom : 0
                }
            }
        }
        return out
    }
}
