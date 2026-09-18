import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `attention` kernel against `AttentionRef`, which
/// materializes the full attention matrix per Q head via `vDSP_dotpr` and
/// then softmaxes/outputs. The kernel runs FlashAttention-style tiled
/// online softmax; this reference does no online merging. Different code
/// shape, different summation order.
@Suite struct AttentionTests {

    private enum Mode: CustomStringConvertible {
        case swa(window: Int)
        case full
        var description: String {
            switch self {
            case .swa(let w): return "swa(window=\(w))"
            case .full:       return "full"
            }
        }
    }

    @Test func attentionSplitGeometry_reportsEffectiveDispatchShape() throws {
        let swa = Attention.splitGeometry(numQHeads: 16,
                                          numKVHeads: 8,
                                          seqLen: 1536,
                                          kvStart: 512,
                                          preferGQASWA: true)
        #expect(swa.effectiveLength == 1024)
        #expect(swa.numChunks == 16)
        #expect(swa.chunkLength == 64)
        #expect(swa.partialThreadgroups == 128)
        #expect(swa.useSWAGroupedPartial)

        let full = Attention.splitGeometry(numQHeads: 16,
                                           numKVHeads: 2,
                                           seqLen: 1536,
                                           kvStart: 0,
                                           preferGQASWA: false)
        #expect(full.effectiveLength == 1536)
        #expect(full.numChunks == 16)
        #expect(full.chunkLength == 96)
        #expect(full.partialThreadgroups == 256)
        #expect(!full.useSWAGroupedPartial)
    }

    // MARK: - Gemma 4 scale=1.0 path

    /// Gemma 4 uses an attention scale of 1.0. Verify the kernel honours the
    /// runtime scale argument by computing the same forward with
    /// scale=1.0 in both kernel and reference, and confirming the output
    /// differs from the default rsqrt(head_dim) scale path.
    @Test func attentionSWA_unitScale_matchesReference_andDiffersFromDefault() throws {
        let headDim = 64, numQHeads = 4, numKVHeads = 2
        let seqLen = 8, window = 8
        let qCount = numQHeads * headDim
        let kvCount = seqLen * numKVHeads * headDim

        var rng = SeedTree(0x501).key("attn-scale1-swa")
        let qFp32 = (0..<qCount).map { _ in rng.uniform(-0.5, 0.5) }
        let kFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let vFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let qFp16 = qFp32.map { Float16($0) }
        let kFp16 = kFp32.map { Float16($0) }
        let vFp16 = vFp32.map { Float16($0) }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
              let outScaled = Fp16Buffer.make(ctx.device, count: qCount),
              let outUnit   = Fp16Buffer.make(ctx.device, count: qCount) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSWA(commandBuffer: cb,
                         q: qBuf, k: kBuf, v: vBuf, out: outScaled,
                         headDim: UInt32(headDim), numQHeads: UInt32(numQHeads),
                         numKVHeads: UInt32(numKVHeads),
                         seqLen: UInt32(seqLen), window: UInt32(window))
        kernel.encodeSWA(commandBuffer: cb,
                         q: qBuf, k: kBuf, v: vBuf, out: outUnit,
                         headDim: UInt32(headDim), numQHeads: UInt32(numQHeads),
                         numKVHeads: UInt32(numKVHeads),
                         seqLen: UInt32(seqLen), window: UInt32(window),
                         scale: 1.0)
        cb.commit(); cb.waitUntilCompleted()

        let kernelUnit = Fp16Buffer.read(outUnit, count: qCount)
        let qRef = qFp16.map { Float($0) }
        let kRef = kFp16.map { Float($0) }
        let vRef = vFp16.map { Float($0) }
        let refUnit = AttentionRef.apply(
            q: qRef, k: kRef, v: vRef,
            headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
            seqLen: seqLen, window: window, scale: 1.0)
        let rel = RelError.compute(actual: kernelUnit, reference: refUnit)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "scale=1.0 SWA kernel vs ref rel=\(rel)")

        // Regression guard: scale=1.0 must produce different numbers from the
        // default rsqrt(head_dim) scale on the same input.
        let kernelScaled = Fp16Buffer.read(outScaled, count: qCount)
        #expect(kernelUnit != kernelScaled,
                "scale=1.0 produced identical output to rsqrt(head_dim) — runtime arg ignored?")
    }

    private static func runAndCompare(
        headDim: Int,
        numQHeads: Int,
        numKVHeads: Int,
        seqLen: Int,
        mode: Mode,
        shareKV: Bool = false,
        useGroupedVector: Bool = true,
        seed: UInt64,
        tolerance: Float = Tolerance.fp16ChainedReduction
    ) throws {
        var rng = SeedTree(seed).key(
            "attn-h\(numQHeads)-kv\(numKVHeads)-d\(headDim)-T\(seqLen)-kv=\(shareKV)"
        )

        let qCount = numQHeads * headDim
        let kvCount = seqLen * numKVHeads * headDim

        let qFp32 = (0..<qCount).map { _ in rng.uniform(-0.5, 0.5) }
        let kFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let vFp32: [Float] = shareKV
            ? kFp32
            : (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }

        let qFp16 = qFp32.map { Float16($0) }
        let kFp16 = kFp32.map { Float16($0) }
        let vFp16 = vFp32.map { Float16($0) }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let outBuf = Fp16Buffer.make(ctx.device, count: qCount) else {
            Issue.record("Failed to allocate buffers"); return
        }
        let vBuf: MTLBuffer
        if shareKV {
            vBuf = kBuf
        } else {
            guard let b = Fp16Buffer.make(ctx.device, halves: vFp16) else {
                Issue.record("Failed to allocate V buffer"); return
            }
            vBuf = b
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        switch mode {
        case .swa(let window):
            kernel.encodeSWA(commandBuffer: cmd,
                             q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(numQHeads),
                             numKVHeads: UInt32(numKVHeads),
                             seqLen: UInt32(seqLen),
                             window: UInt32(window))
        case .full:
            kernel.encodeFull(commandBuffer: cmd,
                              q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                              headDim: UInt32(headDim),
                              numQHeads: UInt32(numQHeads),
                              numKVHeads: UInt32(numKVHeads),
                              seqLen: UInt32(seqLen), useGroupedVector: useGroupedVector)
        }
        cmd.commit()
        cmd.waitUntilCompleted()

        let qRef = qFp16.map { Float($0) }
        let kRef = kFp16.map { Float($0) }
        let vRef = vFp16.map { Float($0) }
        let window: Int? = {
            if case .swa(let w) = mode { return w }
            return nil
        }()
        let ref = AttentionRef.apply(
            q: qRef, k: kRef, v: vRef,
            headDim: headDim, numQHeads: numQHeads,
            numKVHeads: numKVHeads, seqLen: seqLen, window: window
        )
        let actual = Fp16Buffer.read(outBuf, count: qCount)

        let rel = RelError.compute(actual: actual, reference: ref)
        let passed = rel < tolerance
        if !passed {
            print("attention check failed: mode=\(mode) headDim=\(headDim) " +
                  "Hq=\(numQHeads) Hkv=\(numKVHeads) T=\(seqLen) rel=\(rel)")
        }
        #expect(passed)
    }

    // SWA ---------------------------------------------------------------------

    @Test func attentionSWA_smallShape() throws {
        try Self.runAndCompare(headDim: 64, numQHeads: 4, numKVHeads: 2,
                               seqLen: 128, mode: .swa(window: 64), seed: 0x171)
    }

    @Test func attentionSWA_shorterThanWindow() throws {
        try Self.runAndCompare(headDim: 64, numQHeads: 4, numKVHeads: 2,
                               seqLen: 32, mode: .swa(window: 128), seed: 0x172)
    }

    @Test func attentionSWA_realShape() throws {
        try Self.runAndCompare(headDim: 256, numQHeads: 16, numKVHeads: 8,
                               seqLen: 256, mode: .swa(window: 128), seed: 0x173)
    }

    @Test func attentionSWA_ringCapacityMatchesLinearReference() throws {
        let headDim = 64
        let numQHeads = 4
        let numKVHeads = 2
        let seqLen = 40
        let window = 16
        let ringCapacity = 24
        let qCount = numQHeads * headDim
        let kvStride = numKVHeads * headDim
        let kvCount = seqLen * kvStride
        let ringCount = ringCapacity * kvStride

        var rng = SeedTree(0x174).key("attn-swa-ring")
        let qFp32 = (0..<qCount).map { _ in rng.uniform(-0.5, 0.5) }
        let kFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let vFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }

        var kRing = [Float](repeating: 0, count: ringCount)
        var vRing = [Float](repeating: 0, count: ringCount)
        for p in 0..<seqLen {
            let dst = (p % ringCapacity) * kvStride
            let src = p * kvStride
            kRing.replaceSubrange(dst..<(dst + kvStride),
                                  with: kFp32[src..<(src + kvStride)])
            vRing.replaceSubrange(dst..<(dst + kvStride),
                                  with: vFp32[src..<(src + kvStride)])
        }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp32.map { Float16($0) }),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kRing.map { Float16($0) }),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vRing.map { Float16($0) }),
              let outBuf = Fp16Buffer.make(ctx.device, count: qCount) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSWA(commandBuffer: cb,
                         q: qBuf,
                         k: kBuf,
                         v: vBuf,
                         out: outBuf,
                         headDim: UInt32(headDim),
                         numQHeads: UInt32(numQHeads),
                         numKVHeads: UInt32(numKVHeads),
                         seqLen: UInt32(seqLen),
                         window: UInt32(window),
                         ringCapacity: UInt32(ringCapacity))
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: qCount)
        let ref = AttentionRef.apply(q: qFp32.map { Float(Float16($0)) },
                                     k: kFp32.map { Float(Float16($0)) },
                                     v: vFp32.map { Float(Float16($0)) },
                                     headDim: headDim,
                                     numQHeads: numQHeads,
                                     numKVHeads: numKVHeads,
                                     seqLen: seqLen,
                                     window: window)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "ring SWA kernel vs linear ref rel=\(rel)")
    }

    @Test func attentionSWA_ringCapacityMatchesLinearKernelAtWrapRealShape() throws {
        let headDim = 256
        let numQHeads = 16
        let numKVHeads = 8
        let seqLen = 1187
        let window = 1024
        let ringCapacity = 1152
        let qCount = numQHeads * headDim
        let kvStride = numKVHeads * headDim
        let kvCount = seqLen * kvStride
        let ringCount = ringCapacity * kvStride

        var rng = SeedTree(0x181).key("attn-swa-ring-real-wrap")
        let qFp16 = (0..<qCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
        let kFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
        let vFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }

        var kRing = [Float16](repeating: 0, count: ringCount)
        var vRing = [Float16](repeating: 0, count: ringCount)
        for p in 0..<seqLen {
            let dst = (p % ringCapacity) * kvStride
            let src = p * kvStride
            kRing.replaceSubrange(dst..<(dst + kvStride),
                                  with: kFp16[src..<(src + kvStride)])
            vRing.replaceSubrange(dst..<(dst + kvStride),
                                  with: vFp16[src..<(src + kvStride)])
        }

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kLinearBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vLinearBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
              let kRingBuf = Fp16Buffer.make(ctx.device, halves: kRing),
              let vRingBuf = Fp16Buffer.make(ctx.device, halves: vRing),
              let linearOut = Fp16Buffer.make(ctx.device, count: qCount),
              let ringOut = Fp16Buffer.make(ctx.device, count: qCount) else {
            Issue.record("alloc failed")
            return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSWA(commandBuffer: cb,
                         q: qBuf,
                         k: kLinearBuf,
                         v: vLinearBuf,
                         out: linearOut,
                         headDim: UInt32(headDim),
                         numQHeads: UInt32(numQHeads),
                         numKVHeads: UInt32(numKVHeads),
                         seqLen: UInt32(seqLen),
                         window: UInt32(window),
                         ringCapacity: 0)
        kernel.encodeSWA(commandBuffer: cb,
                         q: qBuf,
                         k: kRingBuf,
                         v: vRingBuf,
                         out: ringOut,
                         headDim: UInt32(headDim),
                         numQHeads: UInt32(numQHeads),
                         numKVHeads: UInt32(numKVHeads),
                         seqLen: UInt32(seqLen),
                         window: UInt32(window),
                         ringCapacity: UInt32(ringCapacity))
        cb.commit()
        cb.waitUntilCompleted()

        let linear = Fp16Buffer.read(linearOut, count: qCount)
        let ring = Fp16Buffer.read(ringOut, count: qCount)
        let rel = RelError.compute(actual: ring, reference: linear)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "ring SWA kernel vs linear SWA kernel rel=\(rel)")
    }


    // Full --------------------------------------------------------------------

    @Test func attentionFull_smallShape() throws {
        try Self.runAndCompare(headDim: 64, numQHeads: 8, numKVHeads: 1,
                               seqLen: 128, mode: .full, seed: 0x174)
    }

    @Test func attentionFull_realShape() throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: 128, mode: .full, seed: 0x175)
    }



    /// Generic aliasing coverage. Gemma 4 runtime uses distinct post-norm K/V.
    @Test func attentionFull_kvShared_smallShape() throws {
        try Self.runAndCompare(headDim: 64, numQHeads: 8, numKVHeads: 1,
                               seqLen: 128, mode: .full, shareKV: true,
                               seed: 0x176)
    }

    @Test func attentionFull_kvShared_realShape() throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: 128, mode: .full, shareKV: true,
                               seed: 0x177)
    }

    // Grouped vector split-KV (D512 full attention) ---------------------------

    /// These are tolerance checks against the independent FP32 reference.
    /// GroupedAttentionIdentityTests separately checks D-1a against production.
    ///
    /// 17 is the case that matters most: 16 chunks over 17 keys gives
    /// chunkLen 2, so chunk 9 onward start past the range end and must
    /// contribute (-inf, 0, 0) rather than garbage.
    @Test(arguments: [1, 3, 5, 17, 31, 32, 33, 64, 128, 1_024, 4_096])
    func attentionFull_groupedVector_matchesReference(seqLen: Int) throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: seqLen, mode: .full,
                               useGroupedVector: true,
                               seed: 0x9E1 &+ UInt64(seqLen))
    }

    /// K and V diverge through separate per-head norms and RoPE in the real
    /// runtime; aliasing them would hide a V-indexing bug.
    @Test func attentionFull_groupedVector_sharedKV() throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: 320, mode: .full, shareKV: true,
                               useGroupedVector: true, seed: 0x9E2)
    }

    /// Production scale is 1.0; the reference convention is 1/sqrt(head_dim).
    /// The kernel applies the scale after the dot, as the exact kernel does,
    /// so both must land on the reference.
    @Test func attentionFull_groupedVector_defaultScale() throws {
        try Self.runAndCompare(headDim: 512, numQHeads: 16, numKVHeads: 2,
                               seqLen: 96, mode: .full,
                               useGroupedVector: true, seed: 0x9E3)
    }

    @Test func attentionFull_groupedVector_isDeterministic() throws {
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        var rng = SeedTree(0x9E4).key("grouped-vec-determinism")
        let qCount = 16 * 512
        let kvCount = 700 * 2 * 512
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k),
              let vBuf = Fp16Buffer.make(ctx.device, halves: v) else {
            Issue.record("Failed to allocate buffers"); return
        }
        func run() -> [Float] {
            guard let out = Fp16Buffer.make(ctx.device, count: qCount),
                  let cb = ctx.queue.makeCommandBuffer() else { return [] }
            kernel.encodeFull(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf, out: out,
                              headDim: 512, numQHeads: 16, numKVHeads: 2,
                              seqLen: 700, scale: 1.0, useGroupedVector: true)
            cb.commit(); cb.waitUntilCompleted()
            return Fp16Buffer.read(out, count: qCount)
        }
        #expect(run() == run())
    }

    /// One threadgroup per (kv_head, chunk) instead of one per (q_head, chunk):
    /// 8x fewer partial threadgroups for the same split.
    @Test func attentionFull_groupedVector_dispatchesPerKVHead() throws {
        let grouped = Attention.splitGeometry(headDim: 512, numQHeads: 16,
                                              numKVHeads: 2, seqLen: 4_096,
                                              kvStart: 0, preferGQASWA: false,
                                              forceGroupedVector: true)
        let exact = Attention.splitGeometry(headDim: 512, numQHeads: 16,
                                            numKVHeads: 2, seqLen: 4_096,
                                            kvStart: 0, preferGQASWA: false)
        #expect(grouped.useFullGroupedVectorPartial)
        #expect(!exact.useFullGroupedVectorPartial)
        #expect(grouped.numChunks == exact.numChunks)
        #expect(grouped.partialThreadgroups == 2 * grouped.numChunks)
        #expect(exact.partialThreadgroups == 16 * exact.numChunks)
    }

    /// The kernel indexes 16 Q heads over 2 KV heads with 16 components per
    /// lane. Any other shape must fall back to the exact split kernel.
    @Test func attentionFull_groupedVector_rejectsOtherShapes() throws {
        let wrongHeadDim = Attention.splitGeometry(headDim: 256, numQHeads: 16,
                                                   numKVHeads: 8, seqLen: 1_024,
                                                   kvStart: 0, preferGQASWA: false,
                                                   forceGroupedVector: true)
        let wrongGQA = Attention.splitGeometry(headDim: 512, numQHeads: 16,
                                               numKVHeads: 4, seqLen: 1_024,
                                               kvStart: 0, preferGQASWA: false,
                                               forceGroupedVector: true)
        #expect(!wrongHeadDim.useFullGroupedVectorPartial)
        #expect(!wrongGQA.useFullGroupedVectorPartial)
    }

    /// The FP32 reference holds ~0.8 GB of `[Float]` at these lengths, which
    /// the 8 GB dev Mac cannot afford, so the long-context correctness pin is
    /// against the exact split kernel instead: same inputs, same combine, a
    /// different pass-1 reduction. Without this the kernel is only covered to
    /// 4,096 while it is measured at 65,536.
    @Test(arguments: [16_384, 65_536])
    func attentionFull_groupedVector_matchesExactKernelAtLongContext(seqLen: Int) throws {
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        var rng = SeedTree(0x9E5).key("grouped-vec-long-\(seqLen)")
        let qCount = 16 * 512
        let kvCount = seqLen * 2 * 512
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k),
              let vBuf = Fp16Buffer.make(ctx.device, halves: v) else {
            Issue.record("Failed to allocate buffers"); return
        }
        func run(grouped: Bool) -> [Float] {
            guard let out = Fp16Buffer.make(ctx.device, count: qCount),
                  let cb = ctx.queue.makeCommandBuffer() else { return [] }
            kernel.encodeFull(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf, out: out,
                              headDim: 512, numQHeads: 16, numKVHeads: 2,
                              seqLen: UInt32(seqLen), scale: 1.0,
                              useGroupedVector: grouped)
            cb.commit(); cb.waitUntilCompleted()
            return Fp16Buffer.read(out, count: qCount)
        }
        let rel = RelError.compute(actual: run(grouped: true),
                                   reference: run(grouped: false))
        if rel >= Tolerance.fp16ChainedReduction {
            print("grouped-vec vs exact at seqLen=\(seqLen): rel=\(rel)")
        }
        #expect(rel < Tolerance.fp16ChainedReduction)
    }

    /// Every other case draws from U(-0.5, 0.5), which keeps scores near zero
    /// and barely exercises the online-softmax rescale. At production scale 1.0
    /// over 512 dims, real scores span a far wider range: here U(-3, 3) inputs
    /// give scores around +/-200, so `alpha = exp(m_run - m_new)` underflows to
    /// zero whenever a late key takes the running max and the accumulated
    /// output must be discarded rather than blended. A kernel that dropped the
    /// rescale entirely still passes the small-amplitude cases.
    @Test(arguments: [64, 1_024, 4_096])
    func attentionFull_groupedVector_wideScoreRange(seqLen: Int) throws {
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        var rng = SeedTree(0x9E6).key("grouped-vec-wide-\(seqLen)")
        let qCount = 16 * 512
        let kvCount = seqLen * 2 * 512
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-3.0, 3.0)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-3.0, 3.0)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k),
              let vBuf = Fp16Buffer.make(ctx.device, halves: v) else {
            Issue.record("Failed to allocate buffers"); return
        }
        func run(grouped: Bool) -> [Float] {
            guard let out = Fp16Buffer.make(ctx.device, count: qCount),
                  let cb = ctx.queue.makeCommandBuffer() else { return [] }
            kernel.encodeFull(commandBuffer: cb, q: qBuf, k: kBuf, v: vBuf, out: out,
                              headDim: 512, numQHeads: 16, numKVHeads: 2,
                              seqLen: UInt32(seqLen), scale: 1.0,
                              useGroupedVector: grouped)
            cb.commit(); cb.waitUntilCompleted()
            return Fp16Buffer.read(out, count: qCount)
        }
        let reference = AttentionRef.apply(q: q.map(Float.init), k: k.map(Float.init),
                                           v: v.map(Float.init),
                                           headDim: 512, numQHeads: 16, numKVHeads: 2,
                                           seqLen: seqLen, window: nil, scale: 1.0)
        let grouped = run(grouped: true)
        let exact = run(grouped: false)
        let relGrouped = RelError.compute(actual: grouped, reference: reference)
        let relExact = RelError.compute(actual: exact, reference: reference)
        #expect(relGrouped < Tolerance.fp16ChainedReduction,
                "grouped rel=\(relGrouped) exact rel=\(relExact) at seqLen=\(seqLen)")
    }
}
