import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport
import TurboFieldfareFormat

/// Compares the Metal `k3_mxfp4_*` kernels against `K3MXFP4Reference`
/// (canonical `QuantizationMXFP4` group decode + `vDSP_dotpr` over the
/// bulk-dequantized matrix — independent staging and summation order).
///
/// MXFP4 fixtures are generated as random BYTES, not quantized floats:
/// every nibble pattern is meaningful E2M1 data, so random bytes exercise
/// the full decode table, and scale bytes are drawn from a moderate
/// exponent band with occasional 0xFF (zeroed group) and 0x00 (subnormal
/// scale) sentinels.
@Suite struct K3MXFP4KernelTests {

    // MARK: helpers

    private static func makeF32Buffer(_ device: MTLDevice, _ values: [Float])
        -> MTLBuffer?
    {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
    }

    private static func readF32Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    /// Random MXFP4 matrix bytes. Scale bytes sit in 2^-7...2^5 so row dots
    /// stay in a sane fp16 range; ~5 % of groups are zeroed via the 0xFF
    /// sentinel and a rare 0x00 exercises the subnormal decode.
    private static func randomMXFP4Matrix(m: Int, n: Int, rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt8])
    {
        var packed = [UInt8](repeating: 0, count: m * n / 2)
        for i in packed.indices {
            packed[i] = UInt8(truncatingIfNeeded: rng.next())
        }
        var scales = [UInt8](repeating: 0, count: m * n / 32)
        for i in scales.indices {
            let roll = rng.next() % 97
            if roll == 0 {
                scales[i] = 0x00
            } else if roll <= 5 {
                scales[i] = 0xFF
            } else {
                scales[i] = UInt8(120 + rng.next() % 13)
            }
        }
        return (packed, scales)
    }

    private static func runGemv(m: Int, n: Int, seed: UInt64,
                                f32: Bool, weightByteOffset: Int = 0) throws {
        precondition(n % 32 == 0)
        var rng = SeedTree(seed).key("k3-mxfp4-gemv-m\(m)-n\(n)-f32\(f32)-off\(weightByteOffset)")
        let (packed, scales) = randomMXFP4Matrix(m: m, n: n, rng: &rng)
        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }

        let ctx = try MetalContext()
        let kernel = try K3MXFP4GEMV(context: ctx)

        var paddedPacked = [UInt8](repeating: 0, count: packed.count + weightByteOffset)
        for i in packed.indices { paddedPacked[weightByteOffset + i] = packed[i] }

        guard let wBuf = ctx.device.makeBuffer(
                bytes: paddedPacked, length: paddedPacked.count,
                options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count, options: .storageModeShared) else {
            Issue.record("Failed to allocate weight buffers"); return
        }

        let reference = K3MXFP4Reference.matvec(
            packedWeights: packed, scales: scales,
            rows: m, columns: n,
            vector: f32 ? xFp32 : xFp32.map { Float(Float16($0)) })

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }

        let actual: [Float]
        if f32 {
            guard let xBuf = makeF32Buffer(ctx.device, xFp32),
                  let yBuf = ctx.device.makeBuffer(
                    length: m * MemoryLayout<Float>.stride,
                    options: .storageModeShared) else {
                Issue.record("Failed to allocate f32 buffers"); return
            }
            kernel.encodeF32(commandBuffer: cmd,
                             weights: wBuf, weightsOffset: weightByteOffset,
                             scales: sBuf, x: xBuf, y: yBuf,
                             m: UInt32(m), n: UInt32(n))
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)
            actual = readF32Buffer(yBuf, count: m)
        } else {
            let xFp16 = xFp32.map { Float16($0) }
            guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
                  let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
                Issue.record("Failed to allocate f16 buffers"); return
            }
            kernel.encode(commandBuffer: cmd,
                          weights: wBuf, weightsOffset: weightByteOffset,
                          scales: sBuf, x: xBuf, y: yBuf,
                          m: UInt32(m), n: UInt32(n))
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)
            actual = Fp16Buffer.read(yBuf, count: m)
        }

        let rel = RelError.compute(actual: actual, reference: reference)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        // FP32 in/out: only the summation order differs from vDSP.
        // FP16 in/out: output rounding dominates, the usual single-reduction
        // bar.
        let tolerance = f32 ? Tolerance.identity : Tolerance.fp16Reduction
        #expect(rel < tolerance,
                "M=\(m) N=\(n) f32=\(f32): rel=\(rel) maxAbs=\(maxAbs)")
    }

    // MARK: GEMV

    @Test func gemvF16_m64_n128() throws {
        try Self.runGemv(m: 64, n: 128, seed: 0xA1, f32: false)
    }

    @Test func gemvF16_m8_n32_singleGroupPerRow() throws {
        try Self.runGemv(m: 8, n: 32, seed: 0xA2, f32: false)
    }

    @Test func gemvF16_m130_n288_rowCountNotMultipleOfEight() throws {
        try Self.runGemv(m: 130, n: 288, seed: 0xA3, f32: false)
    }

    /// Binds the packed weights at a 2-aligned-but-not-4-aligned offset;
    /// the kernel's `ushort` row loads must stay correct (a `uint` load
    /// would be misaligned), mirroring the resident-layout regression test
    /// on the int4 GEMV.
    @Test func gemvF16_weightsAt2AlignedNot4AlignedOffset() throws {
        try Self.runGemv(m: 64, n: 128, seed: 0xA4, f32: false, weightByteOffset: 2)
    }

    @Test func gemvF32_m64_n128() throws {
        try Self.runGemv(m: 64, n: 128, seed: 0xA5, f32: true)
    }

    /// Canonical K3 expert shapes, exercising the function-constant
    /// specialized pipelines: w1/w3 are 3072x3584, w2 is 3584x3072.
    @Test func gemvF32_canonicalW1Shape() throws {
        try Self.runGemv(m: 3072, n: 3584, seed: 0xA6, f32: true)
    }

    @Test func gemvF16_canonicalW2Shape() throws {
        try Self.runGemv(m: 3584, n: 3072, seed: 0xA7, f32: false)
    }

    @Test(arguments: [(16, 32), (64, 96), (40, 160), (256, 224)] as [(Int, Int)])
    func gemvSweep(m: Int, n: Int) throws {
        let seed = UInt64(m) &* 0x9E37 &+ UInt64(n)
        try Self.runGemv(m: m, n: n, seed: seed, f32: m % 2 == 0)
    }

    // MARK: bit-exact dequant

    /// `k3_mxfp4_dequant` must reproduce `QuantizationMXFP4`'s decode
    /// bit-for-bit — including the 0xFF scale byte zeroing its group and the
    /// 0x00 subnormal scale. The K3 shader library is compiled with fast
    /// math off precisely so this holds.
    @Test func dequantBitExactIncludingSentinelScales() throws {
        let m = 8
        let n = 128
        var rng = SeedTree(0xB1).key("k3-mxfp4-dequant-bitexact")
        var packed = [UInt8](repeating: 0, count: m * n / 2)
        for i in packed.indices {
            packed[i] = UInt8(truncatingIfNeeded: rng.next())
        }
        var scales = [UInt8](repeating: 0, count: m * n / 32)
        for i in scales.indices {
            scales[i] = UInt8(truncatingIfNeeded: rng.next())
        }
        // Force the sentinel and edge scale bytes: 0xFF (zero group), 0x00
        // (subnormal 2^-127), 127 (1.0), 1 (min normal), 254 (max).
        scales[0] = 0xFF
        scales[1] = 0x00
        scales[2] = 127
        scales[3] = 1
        scales[4] = 254
        scales[5] = 0xFF

        let reference = K3MXFP4Reference.dequantize(
            packedWeights: packed, scales: scales, rows: m, columns: n)

        let ctx = try MetalContext()
        let kernel = try K3MXFP4Dequant(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count, options: .storageModeShared),
              let outBuf = ctx.device.makeBuffer(
                length: m * n * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, scales: sBuf, out: outBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let actual = Self.readF32Buffer(outBuf, count: m * n)
        #expect(actual.count == reference.count)
        var mismatches = 0
        var firstMismatch = -1
        for i in actual.indices {
            if actual[i].bitPattern != reference[i].bitPattern {
                mismatches += 1
                if firstMismatch < 0 { firstMismatch = i }
            }
        }
        let detail = firstMismatch >= 0
            ? "gpu=0x\(String(actual[firstMismatch].bitPattern, radix: 16)) "
                + "ref=0x\(String(reference[firstMismatch].bitPattern, radix: 16))"
            : "none"
        #expect(mismatches == 0,
                "bit mismatches: \(mismatches), first at flat index \(firstMismatch): \(detail)")
    }

    /// Direct group-level cross-check of the reference itself against the
    /// canonical decoder, so a broken wrapper can't masquerade as a kernel
    /// bug (or vice versa).
    @Test func referenceDequantizeMatchesCanonicalGroups() {
        let columns = 64
        var rng = SeedTree(0xB2).key("k3-mxfp4-reference-selfcheck")
        let packed = (0..<(columns / 2)).map { _ in
            UInt8(truncatingIfNeeded: rng.next())
        }
        var out: [Float] = []
        for group in 0..<(columns / 32) {
            let slice = Array(packed[(group * 16)..<((group + 1) * 16)])
            let scale = UInt8(truncatingIfNeeded: rng.next())
            out.append(contentsOf: QuantizationMXFP4.dequantizeMxfp4Group(
                packed: slice, scale: scale))
            let viaReference = K3MXFP4Reference.dequantize(
                packedWeights: slice, scales: [scale], rows: 1, columns: 32)
            #expect(viaReference == Array(out[(group * 32)...]))
        }
    }
}
