import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
import TinyTitanCore

/// attention_decode_partial_simd (simdgroup-per-key pass 1) against the
/// serial pass 1 and the FP32 reference, under a sparse key selection --
/// the regime the goldens never reach (short prompts stay inside the dense
/// window, where the serial kernel still runs). Qwen3.8's full-attention
/// geometry: 24 Q heads, 2 KV heads, head_dim 256, int8 KV.
@Suite(.serialized) struct SimdPartialAttentionTests {

    private static func run(seqLen: Int, keepFraction: Double, seed: UInt64)
        throws -> (simd: [Float], serial: [Float], reference: [Float])
    {
        let config = ArchConfig.qwen38FlashNext
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let qCount = numQHeads * headDim
        let rowElements = numKVHeads * headDim
        let kvCount = seqLen * rowElements
        var rng = SeedTree(seed).key("simd-partial-\(seqLen)")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        // V biased positive so the outputs are O(0.5), not a near-zero
        // mean that makes a relative metric meaningless.
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(0.0, 1.0)) }
        var keep = (0..<seqLen).map { _ in
            rng.uniform(0.0, 1.0) < Float(keepFraction) ? UInt8(1) : UInt8(0)
        }
        keep[seqLen - 1] = 1  // the query's own position is always kept

        let context = try MetalContext()
        let cache = try KVCacheManager(
            device: context.device, config: config,
            maxContext: seqLen, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        let attention = try Attention(context: context)
        guard let qBuffer = Fp16Buffer.make(context.device, halves: q),
            let kBuffer = Fp16Buffer.make(context.device, halves: k),
            let vBuffer = Fp16Buffer.make(context.device, halves: v),
            let keepBuffer = context.device.makeBuffer(
                bytes: keep, length: seqLen,
                options: .storageModeShared),
            let outSimd = Fp16Buffer.make(context.device, count: qCount),
            let outSerial = Fp16Buffer.make(context.device, count: qCount)
        else {
            throw MetalError.bufferAllocationFailed("test buffers")
        }
        let keyView = cache.keyView(layer: 3, validTokenCount: seqLen)
        let valueView = cache.valueView(layer: 3, validTokenCount: seqLen)
        guard let fill = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try quantizer.encode(
            commandBuffer: fill, source: kBuffer,
            sourceTokenStrideElements: rowElements,
            destination: keyView, tokenCount: seqLen,
            elementCount: rowElements)
        try quantizer.encode(
            commandBuffer: fill, source: vBuffer,
            sourceTokenStrideElements: rowElements,
            destination: valueView, tokenCount: seqLen,
            elementCount: rowElements)
        fill.commit()
        fill.waitUntilCompleted()

        for (useSimd, out) in [(true, outSimd), (false, outSerial)] {
            attention.simdPartialOverride = useSimd
            // First pass is the correctness run; the timed pass repeats the
            // same encode so the kernel cost is visible in the test log.
            for pass in 0..<2 {
                guard let cb = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                let reps = pass == 0 ? 1 : 10
                for _ in 0..<reps {
                    try attention.encodeFull(
                        commandBuffer: cb, q: qBuffer,
                        k: keyView.buffer, v: valueView.buffer,
                        out: out,
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(numQHeads),
                        numKVHeads: UInt32(numKVHeads),
                        seqLen: UInt32(seqLen),
                        kvFormat: keyView,
                        keepMask: keepBuffer)
                }
                cb.commit()
                cb.waitUntilCompleted()
                #expect(cb.error == nil)
                if pass == 1 {
                    let us = (cb.gpuEndTime - cb.gpuStartTime) * 1_000_000 / Double(reps)
                    print(
                        "simd_partial_timing seqLen=\(seqLen) simd=\(useSimd) per_call_us=\(Int(us))"
                    )
                }
            }
        }

        // Reference over the kept keys only: masking a key out is the same as
        // its absence.
        let kept = (0..<seqLen).filter { keep[$0] == 1 }
        var kc = [Float]()
        var vc = [Float]()
        kc.reserveCapacity(kept.count * rowElements)
        vc.reserveCapacity(kept.count * rowElements)
        for p in kept {
            for e in 0..<rowElements {
                kc.append(Float(k[p * rowElements + e]))
                vc.append(Float(v[p * rowElements + e]))
            }
        }
        let reference = AttentionRef.apply(
            q: q.map(Float.init), k: kc, v: vc,
            headDim: headDim, numQHeads: numQHeads,
            numKVHeads: numKVHeads, seqLen: kept.count)
        return (
            Fp16Buffer.read(outSimd, count: qCount),
            Fp16Buffer.read(outSerial, count: qCount),
            reference
        )
    }

    @Test func sparseSelectionMatchesSerialAndReference() throws {
        let r = try Self.run(seqLen: 4096, keepFraction: 0.55, seed: 0x51AD)
        let simdRel = RelError.compute(actual: r.simd, reference: r.reference)
        let serialRel = RelError.compute(actual: r.serial, reference: r.reference)
        let crossRel = RelError.compute(actual: r.simd, reference: r.serial)
        print(
            "simd_partial_err sparse simdRel=\(simdRel) serialRel=\(serialRel) crossRel=\(crossRel) maxAbs=\(RelError.maxAbsDiff(r.simd, r.reference))"
        )
        #expect(simdRel < 0.02, "simd vs reference rel=\(simdRel)")
        #expect(serialRel < 0.02, "serial vs reference rel=\(serialRel)")
        #expect(crossRel < 0.01, "simd vs serial rel=\(crossRel)")
    }

    @Test func denseMaskMatchesReference() throws {
        let r = try Self.run(seqLen: 700, keepFraction: 1.0, seed: 0x51AE)
        let simdRel = RelError.compute(actual: r.simd, reference: r.reference)
        let crossRel = RelError.compute(actual: r.simd, reference: r.serial)
        #expect(simdRel < 0.02, "simd vs reference rel=\(simdRel)")
        #expect(crossRel < 0.01, "simd vs serial rel=\(crossRel)")
    }

    /// Chunks with no kept key at all must not poison the merge.
    @Test func sparseWithEmptyChunks() throws {
        let r = try Self.run(seqLen: 2048, keepFraction: 0.02, seed: 0x51AF)
        let simdRel = RelError.compute(actual: r.simd, reference: r.reference)
        #expect(simdRel < 0.02, "simd vs reference rel=\(simdRel)")
        #expect(!r.simd.contains { $0.isNaN }, "NaN in simd output")
    }
}
