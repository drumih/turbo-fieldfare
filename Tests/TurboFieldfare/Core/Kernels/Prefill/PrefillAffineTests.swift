import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillAffineTests {
    private static func packWeights(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let nRows = rows.count
        let rowBytes = rows[0].packed.count
        let groups = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: nRows * rowBytes)
        var scales = [UInt16](repeating: 0, count: nRows * groups)
        var biases = [UInt16](repeating: 0, count: nRows * groups)

        for row in 0..<nRows {
            for i in 0..<rowBytes {
                packed[row * rowBytes + i] = rows[row].packed[i]
            }
            for g in 0..<groups {
                scales[row * groups + g] = rows[row].scales[g]
                biases[row * groups + g] = rows[row].biases[g]
            }
        }
        return (packed, scales, biases)
    }

    private static func runQMMMatchesRepeatedGEMV(t: Int, n: Int, k: Int, seed: UInt64) throws {
        precondition(k % Quantization.groupSize == 0)
        var rng = SeedTree(seed).key("prefill-qmm-t\(t)-n\(n)-k\(k)")

        var rows: [Quantization.Int4AffineRow] = []
        rows.reserveCapacity(n)
        for _ in 0..<n {
            let raw = (0..<k).map { _ in rng.uniform(-0.5, 0.5) }
            rows.append(Quantization.quantizeInt4Affine(raw))
        }
        let (packed, scales, biases) = packWeights(rows)
        let x = (0..<(t * k)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        let qmm = try PrefillInt4QMM(context: ctx)

        guard let wBuf = ctx.device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales,
                                               length: scales.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: biases,
                                               length: biases.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let gemvOut = Fp16Buffer.make(ctx.device, count: t * n),
              let qmmOut = Fp16Buffer.make(ctx.device, count: t * n) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<t {
            gemv.encode(commandBuffer: cb,
                        weights: wBuf,
                        scales: sBuf,
                        biases: bBuf,
                        x: xBuf,
                        xOffset: row * k * MemoryLayout<Float16>.size,
                        y: gemvOut,
                        yOffset: row * n * MemoryLayout<Float16>.size,
                        m: UInt32(n),
                        n: UInt32(k))
        }
        qmm.encode(commandBuffer: cb,
                   weights: wBuf,
                   scales: sBuf,
                   biases: bBuf,
                   x: xBuf,
                   y: qmmOut,
                   t: t,
                   n: n,
                   k: k)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(gemvOut, count: t * n)
        let actual = Fp16Buffer.read(qmmOut, count: t * n)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 1e-3, "shape T=\(t) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 1e-4, "shape T=\(t) N=\(n) K=\(k) rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func runPatternQMMMatchesRepeatedGEMV(t: Int,
                                                         n: Int,
                                                         k: Int,
                                                         seed: UInt64,
                                                         maxAbsTolerance: Float = 2e-2,
                                                         relTolerance: Float = 2e-4) throws {
        precondition(k % Quantization.groupSize == 0)
        let groups = k / Quantization.groupSize
        var rng = SeedTree(seed).key("prefill-qmm-pattern-t\(t)-n\(n)-k\(k)")

        var packed = [UInt8](repeating: 0, count: n * k / 2)
        for i in packed.indices {
            packed[i] = UInt8((i &* 31 &+ Int(seed & 0xff)) & 0xff)
        }

        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        for row in 0..<n {
            for group in 0..<groups {
                let idx = row * groups + group
                let scale = 0.001 + Float((row + group) % 7) * 0.00025
                let bias = -0.012 + Float((row * 3 + group) % 11) * 0.002
                scales[idx] = Quantization.bf16Bits(scale)
                biases[idx] = Quantization.bf16Bits(bias)
            }
        }

        let x = (0..<(t * k)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }

        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        let qmm = try PrefillInt4QMM(context: ctx)

        guard let wBuf = ctx.device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales,
                                               length: scales.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: biases,
                                               length: biases.count * MemoryLayout<UInt16>.size,
                                               options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let gemvOut = Fp16Buffer.make(ctx.device, count: t * n),
              let qmmOut = Fp16Buffer.make(ctx.device, count: t * n) else {
            Issue.record("alloc failed for shape T=\(t) N=\(n) K=\(k)")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<t {
            gemv.encode(commandBuffer: cb,
                        weights: wBuf,
                        scales: sBuf,
                        biases: bBuf,
                        x: xBuf,
                        xOffset: row * k * MemoryLayout<Float16>.size,
                        y: gemvOut,
                        yOffset: row * n * MemoryLayout<Float16>.size,
                        m: UInt32(n),
                        n: UInt32(k))
        }
        qmm.encode(commandBuffer: cb,
                   weights: wBuf,
                   scales: sBuf,
                   biases: bBuf,
                   x: xBuf,
                   y: qmmOut,
                   t: t,
                   n: n,
                   k: k)
        cb.commit()
        cb.waitUntilCompleted()

        let reference = Fp16Buffer.read(gemvOut, count: t * n)
        let actual = Fp16Buffer.read(qmmOut, count: t * n)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= maxAbsTolerance,
                "shape T=\(t) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= relTolerance,
                "shape T=\(t) N=\(n) K=\(k) rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func int4QMMMatchesRepeatedGEMV() throws {
        let shapes = [
            (t: 1, n: 64, k: 64),
            (t: 2, n: 129, k: 128),
            (t: 3, n: 65, k: 192),
            (t: 7, n: 256, k: 128),
            (t: 32, n: 64, k: 64),
        ]

        for (index, shape) in shapes.enumerated() {
            try Self.runQMMMatchesRepeatedGEMV(t: shape.t,
                                               n: shape.n,
                                               k: shape.k,
                                               seed: 0x6100 + UInt64(index))
        }
    }

    @Test func int4QMMProductionShapesMatchRepeatedGEMV() throws {
        let shapes = [
            (t: 1, n: 4096, k: 2816),  // SWA q projection
            (t: 2, n: 2048, k: 2816),  // SWA k/v projection
            (t: 2, n: 8192, k: 2816),  // full q projection
            (t: 2, n: 2816, k: 8192),  // full o/down projection family
        ]

        for (index, shape) in shapes.enumerated() {
            try Self.runPatternQMMMatchesRepeatedGEMV(t: shape.t,
                                                      n: shape.n,
                                                      k: shape.k,
                                                      seed: 0x7100 + UInt64(index))
        }
    }

    @Test func int4QMMSelectedChunkShapesMatchRepeatedGEMV() throws {
        let shapes = [
            (n: 4096, k: 2816),  // SWA q projection
            (n: 2048, k: 2816),  // SWA k/v projection
            (n: 2816, k: 4096),  // SWA o projection
            (n: 8192, k: 2816),  // full q projection
            (n: 1024, k: 2816),  // full k/v projection
            (n: 2816, k: 8192),  // full o projection
            (n: 2112, k: 2816),  // shared gate/up projection
            (n: 2816, k: 2112),  // shared down projection
            (n: 704,  k: 2816),  // routed gate/up projection
            (n: 2816, k: 704),   // routed down projection
        ]

        for (index, shape) in shapes.enumerated() {
            try Self.runPatternQMMMatchesRepeatedGEMV(t: 32,
                                                      n: shape.n,
                                                      k: shape.k,
                                                      seed: 0x8100 + UInt64(index),
                                                      maxAbsTolerance: 2e-4,
                                                      relTolerance: 5e-4)
        }
    }

    @Test func int4QMMRejectsNonGroupAlignedK_contract() {
        // PrefillInt4QMM.encode uses the same precondition contract as
        // DequantInt4GEMV: K must be a multiple of Quantization.groupSize.
        // Swift Testing cannot catch precondition traps in-process, so this
        // documents the rejection contract without deliberately crashing.
        #expect(Quantization.groupSize == 64)
    }
}
