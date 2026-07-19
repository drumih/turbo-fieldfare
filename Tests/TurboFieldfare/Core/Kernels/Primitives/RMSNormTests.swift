import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `rmsnorm` kernel against the Accelerate-based
/// `RmsNormRef`. The reference uses `vDSP_svesq` + `vDSP_vmul` + `vDSP_vsmul`
/// (Accelerate's pipeline). The kernel uses per-thread block reduction with
/// `simd_sum`. Different summation tree, different rounding chain — bugs in
/// either are unlikely to match the other.
///
/// The reference is itself unit-tested in `RMSNormReferenceTests`.
@Suite struct RMSNormTests {

    private static let eps: Float = 1e-6

    // Every Gemma 4 norm weight is BF16. The reference is
    // `RmsNormRef.apply` over weights that have been round-tripped through
    // BF16 (matching what the kernel reads from device memory).

    private static func runAndCompareBF16W(d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("rmsnorm-bf16w-d\(d)")
        let xFp32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let wFp32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }

        let xFp16 = xFp32.map { Float16($0) }
        let xRef  = xFp16.map { Float($0) }
        let wBits = wFp32.map { Quantization.bf16Bits($0) }
        let wRef  = wBits.map { Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: d),
              let wBuf = ctx.device.makeBuffer(length: wBits.count * 2,
                                                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBF16W(commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
                           d: UInt32(d), eps: eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: xRef, weight: wRef, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "BF16W D=\(d): relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }

    @Test func rmsNorm_bf16w_d256() throws {
        try Self.runAndCompareBF16W(d: 256, seed: 0xC3)
    }
    @Test func rmsNorm_bf16w_d2816() throws {
        try Self.runAndCompareBF16W(d: 2816, seed: 0xD4)
    }

    // MARK: - No-scale variant
    //
    // v_norm and the MoE router's internal norm omit the learnable weight:
    // y[i] = x[i] * rsqrt(mean(x^2) + eps). Reference is `RmsNormRef.apply`
    // with a unit weight buffer.

    private static func runAndCompareNoScale(d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("rmsnorm-noscale-d\(d)")
        let xFp32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef  = xFp16.map { Float($0) }
        let unitW = [Float](repeating: 1.0, count: d)

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeNoScale(commandBuffer: cb, x: xBuf, out: yBuf,
                             d: UInt32(d), eps: eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: xRef, weight: unitW, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "no-scale D=\(d): relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }

    @Test func rmsNorm_noScale_d256() throws {
        try Self.runAndCompareNoScale(d: 256, seed: 0xE5)
    }
    @Test func rmsNorm_noScale_d2816() throws {
        try Self.runAndCompareNoScale(d: 2816, seed: 0xF6)
    }

    // MARK: - Per-head dispatch coverage
    //
    // `RealForwardRunner.produce` applies Q/K/V norms per head by looping
    // `for h in 0..<numHeads { rms.encodeBF16W(x, xOffset: h * headDim * 2,
    // ..., out, outOffset: h * headDim * 2, d: headDim, ...) }`. An
    // off-by-one in the offset multiplier (e.g. forgetting the FP16
    // size-of factor) would silently swap or overlap per-head data.
    //
    // This test fills a [numHeads, headDim] buffer with each head set to
    // the constant `Float16(h + 1)`. After per-head norm with unit BF16
    // weights, each head's RMS is `(h+1)`, so the kernel output is
    // `(h+1) / RMS = 1.0` everywhere within the head's range. Sign
    // preservation (sign-of-input matches sign-of-output) catches a swap
    // that smuggles wrong-head values into the wrong slice.

    @Test func rmsNorm_perHead_offsetsCoverEachHeadInIsolation() throws {
        let numHeads = 16
        let headDim  = 256
        let total    = numHeads * headDim

        // Per-head constant: head h is all (h+1). RMS of a constant c is |c|,
        // so y = c / |c| * unitW = sign(c). Pick c = h + 1 > 0 → y = 1.
        var x = [Float16](repeating: 0, count: total)
        for h in 0..<numHeads {
            let c = Float16(h + 1)
            for k in 0..<headDim {
                x[h * headDim + k] = c
            }
        }
        let wBits = [UInt16](repeating: Quantization.bf16Bits(1.0), count: headDim)

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let wBuf = ctx.device.makeBuffer(length: wBits.count * 2,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        // Mirror the runner's per-head dispatch loop literally.
        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = ctx.queue.makeCommandBuffer()!
        for h in 0..<numHeads {
            kernel.encodeBF16W(commandBuffer: cb,
                               x: xBuf, xOffset: h * headBytes,
                               weight: wBuf, weightOffset: 0,
                               out: xBuf, outOffset: h * headBytes,
                               d: UInt32(headDim), eps: 1e-6)
        }
        cb.commit(); cb.waitUntilCompleted()

        let out = Fp16Buffer.read(xBuf, count: total)
        for h in 0..<numHeads {
            for k in 0..<headDim {
                let v = Float(out[h * headDim + k])
                #expect(abs(v - 1.0) < 1e-2,
                        "head \(h) idx \(k): expected ~1.0, got \(v) — per-head offset bug?")
            }
        }
    }

    /// Mirror of the K/V branch using the no-scale variant dispatched for V.
    /// Each head's output should normalize to ±1.
    @Test func rmsNorm_noScale_perHead_offsetsCoverEachHeadInIsolation() throws {
        let numKVL  = 8
        let headDim = 256
        let total   = numKVL * headDim
        var x = [Float16](repeating: 0, count: total)
        for h in 0..<numKVL {
            let c = Float16(h + 1)
            for k in 0..<headDim { x[h * headDim + k] = c }
        }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x) else {
            Issue.record("alloc failed"); return
        }

        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = ctx.queue.makeCommandBuffer()!
        for h in 0..<numKVL {
            kernel.encodeNoScale(commandBuffer: cb,
                                 x: xBuf, xOffset: h * headBytes,
                                 out: xBuf, outOffset: h * headBytes,
                                 d: UInt32(headDim), eps: 1e-6)
        }
        cb.commit(); cb.waitUntilCompleted()

        let out = Fp16Buffer.read(xBuf, count: total)
        for h in 0..<numKVL {
            for k in 0..<headDim {
                let v = Float(out[h * headDim + k])
                #expect(abs(v - 1.0) < 1e-2,
                        "no-scale head \(h) idx \(k): \(v) — per-head offset bug?")
            }
        }
    }

    // MARK: - Batched per-head
    //
    // The coalesced `encodeBF16WPerHead` / `encodeNoScalePerHead` must produce
    // bytewise-equal output to the per-head loop they replace — same
    // rms_block_inv per head, just all heads in one dispatch. Random per-head
    // data so a cross-head bleed (wrong head offset in the grid) is caught.

    @Test func rmsNorm_bf16wPerHead_matchesLoop() throws {
        let numHeads = 16, headDim = 256, total = numHeads * headDim
        var rng = SeedTree(0x2B1).key("rmsnorm-perhead-bf16w")
        let x = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wBits = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(length: headDim * 2, options: .storageModeShared),
              let loopOut = Fp16Buffer.make(ctx.device, halves: x),
              let batchOut = Fp16Buffer.make(ctx.device, halves: x) else {
            Issue.record("alloc failed"); return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: headDim)
        for i in 0..<headDim { wPtr[i] = wBits[i] }
        let headBytes = headDim * MemoryLayout<Float16>.size

        let cb = ctx.queue.makeCommandBuffer()!
        for h in 0..<numHeads {
            kernel.encodeBF16W(commandBuffer: cb, x: loopOut, xOffset: h * headBytes,
                               weight: wBuf, out: loopOut, outOffset: h * headBytes,
                               d: UInt32(headDim), eps: 1e-6)
        }
        kernel.encodeBF16WPerHead(commandBuffer: cb, x: batchOut, weight: wBuf, out: batchOut,
                                  headDim: UInt32(headDim), numHeads: numHeads, eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()

        let a = Fp16Buffer.read(loopOut, count: total)
        let b = Fp16Buffer.read(batchOut, count: total)
        #expect(a == b, "batched per-head BF16W must match the per-head loop bytewise")
    }

    @Test func rmsNorm_noScalePerHead_matchesLoop() throws {
        let numKVL = 8, headDim = 512, total = numKVL * headDim
        var rng = SeedTree(0x2C2).key("rmsnorm-perhead-noscale")
        let x = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let loopOut = Fp16Buffer.make(ctx.device, halves: x),
              let batchOut = Fp16Buffer.make(ctx.device, halves: x) else {
            Issue.record("alloc failed"); return
        }
        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = ctx.queue.makeCommandBuffer()!
        for h in 0..<numKVL {
            kernel.encodeNoScale(commandBuffer: cb, x: loopOut, xOffset: h * headBytes,
                                 out: loopOut, outOffset: h * headBytes,
                                 d: UInt32(headDim), eps: 1e-6)
        }
        kernel.encodeNoScalePerHead(commandBuffer: cb, x: batchOut, out: batchOut,
                                    headDim: UInt32(headDim), numHeads: numKVL, eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()

        let a = Fp16Buffer.read(loopOut, count: total)
        let b = Fp16Buffer.read(batchOut, count: total)
        #expect(a == b, "batched per-head no-scale must match the per-head loop bytewise")
    }
}
