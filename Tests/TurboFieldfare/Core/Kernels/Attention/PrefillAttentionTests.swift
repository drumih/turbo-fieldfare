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

    @Test func prefillTurboQuantPackedAttentionMatchesMaterializedDefaultKV() throws {
        try Self.runTurboQuantPackedAndCompare()
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

    private static func runTurboQuantPackedAndCompare() throws {
        let headDim = 256
        let qHeads = 4
        let kvHeads = 2
        let start = 5
        let chunk = 3
        let kvValid = start + chunk
        let window = 6
        let qStride = qHeads * headDim
        let kvStride = kvHeads * headDim
        let oStride = qStride
        let layer: UInt32 = 2
        let rotationSeed: UInt32 = 0xA11CE

        var rng = SeedTree(0xA910).key("prefill-tq-k4v4")
        let q = (0..<(chunk * qStride)).map { _ in rng.uniform(-0.35, 0.35) }
        let k = (0..<(kvValid * kvStride)).map { _ in rng.uniform(-0.35, 0.35) }
        let v = (0..<(kvValid * kvStride)).map { _ in rng.uniform(-0.35, 0.35) }

        let keyLayout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                                headDim: headDim,
                                                numKVHeads: kvHeads)
        let valueLayout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                                  headDim: headDim,
                                                  numKVHeads: kvHeads)

        let ctx = try MetalContext()
        let prefill = try PrefillAttention(context: ctx)
        let quant = try TurboQuantQuant(context: ctx)

        guard let qBuf = Fp16Buffer.make(ctx.device, values: q),
              let kBuf = Fp16Buffer.make(ctx.device, values: k),
              let vBuf = Fp16Buffer.make(ctx.device, values: v),
              let outBuf = Fp16Buffer.make(ctx.device, count: chunk * oStride),
              let keyCache = ctx.device.makeBuffer(length: kvValid * keyLayout.bytesPerToken,
                                                   options: .storageModeShared),
              let valueCache = ctx.device.makeBuffer(length: kvValid * valueLayout.bytesPerToken,
                                                     options: .storageModeShared) else {
            Issue.record("alloc failed")
            return
        }

        let whtParams = TurboQuantWHTParams(numHeads: UInt32(kvHeads),
                                            layer: layer,
                                            rotationSeed: rotationSeed,
                                            applyRotation: true)
        let keyWrite = TurboQuantKVWriteParams(d: UInt32(headDim),
                                               numHeads: UInt32(kvHeads),
                                               roleLayout: keyLayout)
        let valueWrite = TurboQuantKVWriteParams(d: UInt32(headDim),
                                                 numHeads: UInt32(kvHeads),
                                                 roleLayout: valueLayout)
        let keyBulk = TurboQuantKVBulkWriteParams(kv: keyWrite,
                                                  tokenCount: kvValid,
                                                  dstTokenBase: 0,
                                                  cacheTokenCapacity: kvValid,
                                                  sourceTokenStrideElements: kvStride)
        let valueBulk = TurboQuantKVBulkWriteParams(kv: valueWrite,
                                                    tokenCount: kvValid,
                                                    dstTokenBase: 0,
                                                    cacheTokenCapacity: kvValid,
                                                    sourceTokenStrideElements: kvStride)
        let prefillParams = PrefillAttentionParams(
            startPosition: UInt32(start),
            queryCount: UInt32(chunk),
            headDim: UInt32(headDim),
            numQHeads: UInt32(qHeads),
            numKVHeads: UInt32(kvHeads),
            kvValidCount: UInt32(kvValid),
            slidingWindow: UInt32(window),
            kvTokenStrideElements: UInt32(kvStride),
            qTokenStrideElements: UInt32(qStride),
            oTokenStrideElements: UInt32(oStride),
            scale: 1.0)
        let tqParams = PrefillTurboQuantAttentionParams(prefill: prefillParams,
                                                        layer: layer,
                                                        rotationSeed: rotationSeed,
                                                        keyLayout: keyLayout,
                                                        valueLayout: valueLayout)

        let cb = ctx.queue.makeCommandBuffer()!
        try quant.encodeKVWriteWHTBulk(commandBuffer: cb,
                                       x: kBuf,
                                       cache: keyCache,
                                       params: keyBulk,
                                       whtParams: whtParams)
        try quant.encodeKVWriteWHTBulk(commandBuffer: cb,
                                       x: vBuf,
                                       cache: valueCache,
                                       params: valueBulk,
                                       whtParams: whtParams)
        prefill.encodeTurboQuantCausal(commandBuffer: cb,
                                       q: qBuf,
                                       keyCache: keyCache,
                                       valueCache: valueCache,
                                       out: outBuf,
                                       params: tqParams)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: chunk * oStride)
        let keyBytes = Array(UnsafeBufferPointer(
            start: keyCache.contents().assumingMemoryBound(to: UInt8.self),
            count: keyCache.length))
        let valueBytes = Array(UnsafeBufferPointer(
            start: valueCache.contents().assumingMemoryBound(to: UInt8.self),
            count: valueCache.length))
        let materializedKey = TurboQuantRef.materializePackedCache(
            keyBytes,
            tokenCount: kvValid,
            headDim: headDim,
            numHeads: kvHeads,
            bytesPerHead: keyLayout.bytesPerHead,
            bytesPerToken: keyLayout.bytesPerToken,
            packedOffset: keyLayout.packedOffsetPerHead,
            scaleOffset: keyLayout.scaleOffsetPerHead,
            layer: layer,
            rotationSeed: rotationSeed,
            applyRotation: true)
        let materializedValue = TurboQuantRef.materializePackedCache(
            valueBytes,
            tokenCount: kvValid,
            headDim: headDim,
            numHeads: kvHeads,
            bytesPerHead: valueLayout.bytesPerHead,
            bytesPerToken: valueLayout.bytesPerToken,
            packedOffset: valueLayout.packedOffsetPerHead,
            scaleOffset: valueLayout.scaleOffsetPerHead,
            layer: layer,
            rotationSeed: rotationSeed,
            applyRotation: true)
        let refFixture = Fixture(q: q,
                                 k: materializedKey,
                                 v: materializedValue,
                                 qStride: qStride,
                                 kvStride: kvStride,
                                 oStride: oStride,
                                 headDim: headDim,
                                 qHeads: qHeads,
                                 kvHeads: kvHeads,
                                 start: start,
                                 chunk: chunk,
                                 kvValid: kvValid,
                                 window: window,
                                 scale: 1.0)
        let reference = Self.reference(refFixture)
        let compact = Self.compactOutput(actual,
                                         chunk: chunk,
                                         qHeads: qHeads,
                                         headDim: headDim,
                                         oStride: oStride)
        let maxAbs = RelError.maxAbsDiff(compact, reference)
        let rel = RelError.compute(actual: compact, reference: reference)
        #expect(maxAbs <= 3e-2, "k4v4 packed maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 3e-2, "k4v4 packed rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func compactOutput(_ out: [Float],
                                      chunk: Int,
                                      qHeads: Int,
                                      headDim: Int,
                                      oStride: Int) -> [Float] {
        var compact = [Float](repeating: 0, count: chunk * qHeads * headDim)
        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    compact[(t * qHeads + h) * headDim + d] =
                        out[t * oStride + h * headDim + d]
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
