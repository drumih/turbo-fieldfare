import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// Compares the Metal `rmsnorm` kernel against the Accelerate-based
/// `RmsNormRef`. The reference uses `vDSP_svesq` + `vDSP_vmul` + `vDSP_vsmul`
/// (Accelerate's pipeline). The kernel uses per-thread block reduction with
/// `simd_sum`. Different summation tree, different rounding chain — bugs in
/// either are unlikely to match the other.
///
/// The reference is itself unit-tested in `RMSNormReferenceTests`.
@Suite struct RMSNormTests {

    private static let eps: Float = 1e-6

    // Every Qwen norm weight is BF16. The reference is
    // `RmsNormRef.apply` over weights that have been round-tripped through
    // BF16 (matching what the kernel reads from device memory).

    private static func runAndCompareBF16W(d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("rmsnorm-bf16w-d\(d)")
        let xFp32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let wFp32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }

        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }
        let wBits = wFp32.map { Quantization.bf16Bits($0) }
        let wRef = wBits.map { Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
            let yBuf = Fp16Buffer.make(ctx.device, count: d),
            let wBuf = ctx.device.makeBuffer(
                length: wBits.count * 2,
                options: .storageModeShared)
        else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeBF16W(
            commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
            d: UInt32(d), eps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: xRef, weight: wRef, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(
            relErr < Tolerance.fp16Reduction,
            "BF16W D=\(d): relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }

    @Test func rmsNorm_bf16w_d256() throws {
        try Self.runAndCompareBF16W(d: 256, seed: 0xC3)
    }
    @Test func rmsNorm_bf16w_d2048() throws {
        try Self.runAndCompareBF16W(d: 2048, seed: 0xD4)
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
        let xRef = xFp16.map { Float($0) }
        let unitW = [Float](repeating: 1.0, count: d)

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
            let yBuf = Fp16Buffer.make(ctx.device, count: d)
        else {
            Issue.record("alloc failed")
            return
        }
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeNoScale(
            commandBuffer: cb, x: xBuf, out: yBuf,
            d: UInt32(d), eps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: xRef, weight: unitW, eps: eps)
        let actual = Fp16Buffer.read(yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(
            relErr < Tolerance.fp16Reduction,
            "no-scale D=\(d): relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }

    @Test func rmsNorm_noScale_d256() throws {
        try Self.runAndCompareNoScale(d: 256, seed: 0xE5)
    }
    @Test func rmsNorm_noScale_d2048() throws {
        try Self.runAndCompareNoScale(d: 2048, seed: 0xF6)
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
        let headDim = 256
        let total = numHeads * headDim

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
            let wBuf = ctx.device.makeBuffer(
                length: wBits.count * 2,
                options: .storageModeShared)
        else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        // Mirror the runner's per-head dispatch loop literally.
        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for h in 0..<numHeads {
            try kernel.encodeBF16W(
                commandBuffer: cb,
                x: xBuf, xOffset: h * headBytes,
                weight: wBuf, weightOffset: 0,
                out: xBuf, outOffset: h * headBytes,
                d: UInt32(headDim), eps: 1e-6)
        }
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(xBuf, count: total)
        for h in 0..<numHeads {
            for k in 0..<headDim {
                let v = Float(out[h * headDim + k])
                #expect(
                    abs(v - 1.0) < 1e-2,
                    "head \(h) idx \(k): expected ~1.0, got \(v) — per-head offset bug?")
            }
        }
    }

    /// Mirror of the K/V branch using the no-scale variant dispatched for V.
    /// Each head's output should normalize to ±1.
    @Test func rmsNorm_noScale_perHead_offsetsCoverEachHeadInIsolation() throws {
        let numKVL = 2
        let headDim = 256
        let total = numKVL * headDim
        var x = [Float16](repeating: 0, count: total)
        for h in 0..<numKVL {
            let c = Float16(h + 1)
            for k in 0..<headDim { x[h * headDim + k] = c }
        }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x) else {
            Issue.record("alloc failed")
            return
        }

        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for h in 0..<numKVL {
            try kernel.encodeNoScale(
                commandBuffer: cb,
                x: xBuf, xOffset: h * headBytes,
                out: xBuf, outOffset: h * headBytes,
                d: UInt32(headDim), eps: 1e-6)
        }
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(xBuf, count: total)
        for h in 0..<numKVL {
            for k in 0..<headDim {
                let v = Float(out[h * headDim + k])
                #expect(
                    abs(v - 1.0) < 1e-2,
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
        let numHeads = 16
        let headDim = 256
        let total = numHeads * headDim
        var rng = SeedTree(0x2B1).key("rmsnorm-perhead-bf16w")
        let x = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let wBits = (0..<headDim).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(length: headDim * 2, options: .storageModeShared),
            let loopOut = Fp16Buffer.make(ctx.device, halves: x),
            let batchOut = Fp16Buffer.make(ctx.device, halves: x)
        else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: headDim)
        for i in 0..<headDim { wPtr[i] = wBits[i] }
        let headBytes = headDim * MemoryLayout<Float16>.size

        let cb = try #require(ctx.queue.makeCommandBuffer())
        for h in 0..<numHeads {
            try kernel.encodeBF16W(
                commandBuffer: cb, x: loopOut, xOffset: h * headBytes,
                weight: wBuf, out: loopOut, outOffset: h * headBytes,
                d: UInt32(headDim), eps: 1e-6)
        }
        try kernel.encodeBF16WPerHead(
            commandBuffer: cb, x: batchOut, weight: wBuf, out: batchOut,
            headDim: UInt32(headDim), numHeads: numHeads, eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()

        let a = Fp16Buffer.read(loopOut, count: total)
        let b = Fp16Buffer.read(batchOut, count: total)
        #expect(a == b, "batched per-head BF16W must match the per-head loop bytewise")
    }

    @Test func rmsNorm_noScalePerHead_matchesLoop() throws {
        let numKVL = 2
        let headDim = 256
        let total = numKVL * headDim
        var rng = SeedTree(0x2C2).key("rmsnorm-perhead-noscale")
        let x = (0..<total).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let loopOut = Fp16Buffer.make(ctx.device, halves: x),
            let batchOut = Fp16Buffer.make(ctx.device, halves: x)
        else {
            Issue.record("alloc failed")
            return
        }
        let headBytes = headDim * MemoryLayout<Float16>.size
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for h in 0..<numKVL {
            try kernel.encodeNoScale(
                commandBuffer: cb, x: loopOut, xOffset: h * headBytes,
                out: loopOut, outOffset: h * headBytes,
                d: UInt32(headDim), eps: 1e-6)
        }
        try kernel.encodeNoScalePerHead(
            commandBuffer: cb, x: batchOut, out: batchOut,
            headDim: UInt32(headDim), numHeads: numKVL, eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()

        let a = Fp16Buffer.read(loopOut, count: total)
        let b = Fp16Buffer.read(batchOut, count: total)
        #expect(a == b, "batched per-head no-scale must match the per-head loop bytewise")
    }
}

/// Grouped RMSNorm: `numGroups` contiguous groups, each against its own weight
/// slice. Qwen3.8-Flash-Next normalizes 4 residual streams of 2560 with one
/// `[10240]` weight, so the distinction from the shared-weight per-head form
/// is the whole point -- sharing would apply stream 0's scale to all four.
@Suite struct RMSNormGroupedTests {
    private static let eps: Float = 1e-6

    private static func run(groupDim: Int, groups: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("rmsnorm-grouped-\(groupDim)x\(groups)")
        let total = groupDim * groups
        let xFp32 = (0..<total).map { _ in rng.uniform(-1.0, 1.0) }
        // Distinct scale per group, so a shared-weight implementation cannot
        // pass by coincidence.
        let wFp32 = (0..<total).map { i in
            rng.uniform(0.5, 1.5) * (1.0 + Float(i / groupDim))
        }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }
        let wBits = wFp32.map { Quantization.bf16Bits($0) }
        let wRef = wBits.map { Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
            let yBuf = Fp16Buffer.make(ctx.device, count: total),
            let wBuf = ctx.device.makeBuffer(
                length: wBits.count * 2,
                options: .storageModeShared)
        else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeBF16WGrouped(
            commandBuffer: cb, x: xBuf, weight: wBuf,
            out: yBuf, groupDim: UInt32(groupDim),
            numGroups: groups, eps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        // Reference: each group normalized independently against its own slice.
        var ref: [Float] = []
        for g in 0..<groups {
            let lo = g * groupDim
            let hi = lo + groupDim
            ref += RmsNormRef.apply(
                x: Array(xRef[lo..<hi]),
                weight: Array(wRef[lo..<hi]), eps: eps)
        }
        let actual = Fp16Buffer.read(yBuf, count: total)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(
            relErr < Tolerance.fp16Reduction,
            "grouped \(groupDim)x\(groups): relErr=\(relErr)")
    }

    @Test("The production shape: 4 streams of 2560")
    func hyperConnectionShape() throws {
        try Self.run(groupDim: 2560, groups: 4, seed: 0x11CE)
    }

    /// K2-Horizon's layer norm is two groups of 1280 inside one 2560-wide
    /// hidden state, the other production user of this kernel.
    @Test("K2-Horizon's shape: 2 groups of 1280")
    func k2HorizonShape() throws {
        try Self.run(groupDim: 1280, groups: 2, seed: 0x66BC)
    }

    @Test("Smaller shapes")
    func smallShapes() throws {
        try Self.run(groupDim: 256, groups: 3, seed: 0x22DE)
        try Self.run(groupDim: 512, groups: 2, seed: 0x33EF)
    }

    @Test("One group equals the plain kernel")
    func singleGroupMatchesPlain() throws {
        try Self.run(groupDim: 2048, groups: 1, seed: 0x44FA)
    }

    @Test("Groups really are independent, not sharing one weight slice")
    func groupsUseTheirOwnWeights() throws {
        // Identical data in both groups but different weights: a shared-weight
        // implementation would produce identical halves, which this rejects.
        let groupDim = 256
        let groups = 2
        var rng = SeedTree(0x55AB).key("grouped-independence")
        let half = (0..<groupDim).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xFp16 = half + half
        let wBits =
            (0..<groupDim).map { _ in Quantization.bf16Bits(1.0) }
            + (0..<groupDim).map { _ in Quantization.bf16Bits(2.0) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
            let yBuf = Fp16Buffer.make(ctx.device, count: groupDim * groups),
            let wBuf = ctx.device.makeBuffer(
                length: wBits.count * 2,
                options: .storageModeShared)
        else {
            Issue.record("alloc failed")
            return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try kernel.encodeBF16WGrouped(
            commandBuffer: cb, x: xBuf, weight: wBuf,
            out: yBuf, groupDim: UInt32(groupDim),
            numGroups: groups, eps: Self.eps)
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(yBuf, count: groupDim * groups)
        // Group 1's weight is 2x group 0's, so its output must be 2x.
        for i in 0..<groupDim {
            let a = out[i]
            let b = out[groupDim + i]
            if abs(a) > 1e-3 {
                #expect(
                    abs(b / a - 2.0) < 0.05,
                    "group weights not applied independently at \(i)")
            }
        }
    }
}
