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
                              seqLen: UInt32(seqLen))
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

    @Test func attentionSWA_turboQuantPackedKV_matchesMaterializedReference() throws {
        try Self.runTurboQuantPackedKVAndCompare(
            headDim: 256,
            numQHeads: 4,
            numKVHeads: 2,
            seqLen: 16,
            mode: .swa(window: 8),
            seed: 0x178)
    }

    private static func runTurboQuantPackedKVAndCompare(
        headDim: Int,
        numQHeads: Int,
        numKVHeads: Int,
        seqLen: Int,
        mode: Mode,
        seed: UInt64
    ) throws {
        let window: Int?
        let kvStart: UInt32
        switch mode {
        case .swa(let w):
            window = w
            kvStart = UInt32(seqLen > w ? seqLen - w : 0)
        case .full:
            window = nil
            kvStart = 0
        }
        let qCount = numQHeads * headDim
        let kvCount = seqLen * numKVHeads * headDim
        let pairs = seqLen * numKVHeads
        let layer: UInt32 = 3
        let rotationSeed: UInt32 = 0xC0FFEE

        var rng = SeedTree(seed).key("attn-tq-kv-\(mode)-d\(headDim)")
        let qFp32 = (0..<qCount).map { _ in rng.uniform(-0.5, 0.5) }
        let kFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let vFp32 = (0..<kvCount).map { _ in rng.uniform(-0.5, 0.5) }
        let qFp16 = qFp32.map { Float16($0) }
        let kFp16 = kFp32.map { Float16($0) }
        let vFp16 = vFp32.map { Float16($0) }

        let keyLayout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                                headDim: headDim,
                                                numKVHeads: numKVHeads)
        let valueLayout = TurboQuantKVLayout.role(mode: .k4v4NormCorrected,
                                                  headDim: headDim,
                                                  numKVHeads: numKVHeads)

        let ctx = try MetalContext()
        let quant = try TurboQuantQuant(context: ctx)
        let attention = try Attention(context: ctx)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
              let outPackedSplit = Fp16Buffer.make(ctx.device, count: qCount),
              let keyCache = ctx.device.makeBuffer(length: keyLayout.bytesPerToken * seqLen,
                                                   options: .storageModeShared),
              let valueCache = ctx.device.makeBuffer(length: valueLayout.bytesPerToken * seqLen,
                                                     options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let whtParams = TurboQuantWHTParams(numHeads: UInt32(numKVHeads),
                                            layer: layer,
                                            rotationSeed: rotationSeed,
                                            applyRotation: true)
        let keyWrite = TurboQuantKVWriteParams(d: UInt32(headDim),
                                               numHeads: UInt32(numKVHeads),
                                               roleLayout: keyLayout)
        let valueWrite = TurboQuantKVWriteParams(d: UInt32(headDim),
                                                 numHeads: UInt32(numKVHeads),
                                                 roleLayout: valueLayout)
        let attentionParams = AttentionTurboQuantKVParams(
            headDim: UInt32(headDim),
            numQHeads: UInt32(numQHeads),
            numKVHeads: UInt32(numKVHeads),
            seqLen: UInt32(seqLen),
            kvStart: kvStart,
            scale: 1.0,
            layer: layer,
            rotationSeed: rotationSeed,
            keyLayout: keyLayout,
            valueLayout: valueLayout)

        let cb = ctx.queue.makeCommandBuffer()!
        quant.encodeKVWriteWHT(commandBuffer: cb,
                               x: kBuf,
                               cache: keyCache,
                               params: keyWrite,
                               whtParams: whtParams,
                               pairs: pairs)
        quant.encodeKVWriteWHT(commandBuffer: cb,
                               x: vBuf,
                               cache: valueCache,
                               params: valueWrite,
                               whtParams: whtParams,
                               pairs: pairs)
        attention.encodeTurboQuantSplit(commandBuffer: cb,
                                        q: qBuf,
                                        keyCache: keyCache,
                                        valueCache: valueCache,
                                        out: outPackedSplit,
                                        params: attentionParams)
        cb.commit()
        cb.waitUntilCompleted()

        let actualSplit = Fp16Buffer.read(outPackedSplit, count: qCount)
        let keyBytes = Array(UnsafeBufferPointer(
            start: keyCache.contents().assumingMemoryBound(to: UInt8.self),
            count: keyCache.length))
        let valueBytes = Array(UnsafeBufferPointer(
            start: valueCache.contents().assumingMemoryBound(to: UInt8.self),
            count: valueCache.length))
        let materializedKey = TurboQuantRef.materializePackedCache(
            keyBytes,
            tokenCount: seqLen,
            headDim: headDim,
            numHeads: numKVHeads,
            bytesPerHead: keyLayout.bytesPerHead,
            bytesPerToken: keyLayout.bytesPerToken,
            packedOffset: keyLayout.packedOffsetPerHead,
            scaleOffset: keyLayout.scaleOffsetPerHead,
            layer: layer,
            rotationSeed: rotationSeed,
            applyRotation: true)
        let materializedValue = TurboQuantRef.materializePackedCache(
            valueBytes,
            tokenCount: seqLen,
            headDim: headDim,
            numHeads: numKVHeads,
            bytesPerHead: valueLayout.bytesPerHead,
            bytesPerToken: valueLayout.bytesPerToken,
            packedOffset: valueLayout.packedOffsetPerHead,
            scaleOffset: valueLayout.scaleOffsetPerHead,
            layer: layer,
            rotationSeed: rotationSeed,
            applyRotation: true)
        let ref = AttentionRef.apply(q: qFp16.map { Float($0) },
                                     k: materializedKey,
                                     v: materializedValue,
                                     headDim: headDim,
                                     numQHeads: numQHeads,
                                     numKVHeads: numKVHeads,
                                     seqLen: seqLen,
                                     window: window,
                                     scale: 1.0)
        let splitRel = RelError.compute(actual: actualSplit, reference: ref)
        #expect(splitRel < 0.02, "packed TurboQuant split attention rel=\(splitRel)")
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

    @Test func attentionFull_mlxGeometryV2_distinctKV_stripeBoundaries() throws {
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)

        let headDim = 512
        let numQHeads = 16
        let numKVHeads = 2
        let qCount = numQHeads * headDim
        for seqLen in [1, 31, 32, 33, 257] {
            var rng = SeedTree(UInt64(0x5200 + seqLen)).key("mlx-v2-distinct-kv")
            let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            let kvCount = seqLen * numKVHeads * headDim
            let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
                  let kBuf = Fp16Buffer.make(ctx.device, halves: k),
                  let vBuf = Fp16Buffer.make(ctx.device, halves: v),
                  let outA = Fp16Buffer.make(ctx.device, count: qCount),
                  let outB = Fp16Buffer.make(ctx.device, count: qCount) else {
                Issue.record("allocation failed at seqLen=\(seqLen)")
                return
            }

            for output in [outA, outB] {
                guard let cb = ctx.queue.makeCommandBuffer() else {
                    Issue.record("command buffer failed")
                    return
                }
                kernel.encodeFull(commandBuffer: cb,
                                  q: qBuf, k: kBuf, v: vBuf, out: output,
                                  headDim: UInt32(headDim),
                                  numQHeads: UInt32(numQHeads),
                                  numKVHeads: UInt32(numKVHeads),
                                  seqLen: UInt32(seqLen),
                                  scale: 1.0)
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw error }
            }

            let actualA = Fp16Buffer.read(outA, count: qCount)
            let actualB = Fp16Buffer.read(outB, count: qCount)
            #expect(actualA == actualB, "candidate output changed at seqLen=\(seqLen)")
            let reference = AttentionRef.apply(q: q.map(Float.init),
                                               k: k.map(Float.init),
                                               v: v.map(Float.init),
                                               headDim: headDim,
                                               numQHeads: numQHeads,
                                               numKVHeads: numKVHeads,
                                               seqLen: seqLen,
                                               window: nil,
                                               scale: 1.0)
            let rel = RelError.compute(actual: actualA, reference: reference)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "candidate rel=\(rel) at seqLen=\(seqLen)")
        }
    }

    @Test func attentionFull_turboQuantPackedKV_matchesMaterializedReference() throws {
        try Self.runTurboQuantPackedKVAndCompare(
            headDim: 512,
            numQHeads: 8,
            numKVHeads: 1,
            seqLen: 16,
            mode: .full,
            seed: 0x179)
    }

    @Test func attentionFull_turboQuantPackedKV_realHeadShape_matchesMaterializedReference() throws {
        try Self.runTurboQuantPackedKVAndCompare(
            headDim: 512,
            numQHeads: 16,
            numKVHeads: 2,
            seqLen: 16,
            mode: .full,
            seed: 0x17A)
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
}
