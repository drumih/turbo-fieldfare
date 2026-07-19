import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `dequant_int4_gemv` kernel against `DequantInt4GemvRef`,
/// which bulk-dequantizes each row to FP32 and dots with `vDSP_dotpr`. The
/// kernel interleaves nibble unpack + per-group scale + bias + FMA inside one
/// inner loop per thread. Independent code shape, independent FP summation
/// order.
///
/// The reference is itself unit-tested in `DequantInt4GemvRefTests`.
@Suite struct DequantInt4GEMVTests {

    private static func packWeights(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let m = rows.count
        let n2 = rows[0].packed.count        // N/2
        let s  = rows[0].scales.count        // N/groupSize
        var packed = [UInt8](repeating: 0, count: m * n2)
        var scales = [UInt16](repeating: 0, count: m * s)
        var biases = [UInt16](repeating: 0, count: m * s)
        for row in 0..<m {
            for i in 0..<n2 { packed[row * n2 + i] = rows[row].packed[i] }
            for i in 0..<s  {
                scales[row * s + i] = rows[row].scales[i]
                biases[row * s + i] = rows[row].biases[i]
            }
        }
        return (packed, scales, biases)
    }

    /// `weightByteOffset` pads the packed-weight buffer by that many bytes and
    /// binds it at that offset, exercising the kernel at a live, non-zero base.
    /// A value of 2 reproduces the resident-layout case (BF16 scale/bias regions
    /// leave a 2-aligned-but-not-4-aligned weight offset) — the exact alignment
    /// the vectorized `ushort` load depends on and the offset-0 parity tests
    /// never touch (R2).
    private static func runAndCompare(m: Int, n: Int, seed: UInt64,
                                      weightByteOffset: Int = 0) throws {
        precondition(n % Quantization.groupSize == 0)
        // Keep the offset-zero fixtures stable when adding non-zero offset cases.
        let baseKey = "int4-gemv-kernel-m\(m)-n\(n)"
        let rngKey = weightByteOffset == 0 ? baseKey : "\(baseKey)-off\(weightByteOffset)"
        var rng = SeedTree(seed).key(rngKey)

        var rows: [Quantization.Int4AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
            rows.append(Quantization.quantizeInt4Affine(raw))
        }
        let (packed, scales, biases) = packWeights(rows)

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef  = xFp16.map { Float($0) }

        let ctx    = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)

        // Copy the packed rows at +weightByteOffset inside a padded buffer so the
        // kernel reads them at a non-zero base, not the natural buffer start.
        var paddedPacked = [UInt8](repeating: 0, count: packed.count + weightByteOffset)
        for i in 0..<packed.count { paddedPacked[weightByteOffset + i] = packed[i] }

        guard let wBuf = ctx.device.makeBuffer(
                bytes: paddedPacked, length: paddedPacked.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, weightsOffset: weightByteOffset,
                      scales: sBuf, biases: bBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()

        let ref = DequantInt4GemvRef.apply(weightRows: rows, x: xRef, n: n)
        let actual = Fp16Buffer.read(yBuf, count: m)

        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16Reduction,
                "M=\(m) N=\(n): rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func gemv_d128_n128() throws {
        try Self.runAndCompare(m: 128, n: 128, seed: 0xC1)
    }

    @Test func gemv_m1024_n2816() throws {
        try Self.runAndCompare(m: 1024, n: 2816, seed: 0xC2)
    }

    @Test func gemv_m64_n64() throws {
        try Self.runAndCompare(m: 64, n: 64, seed: 0xC3)
    }

    /// Binds the packed weights at a 2-aligned-but-NOT-4-aligned byte offset
    /// (+2), reproducing the resident layout where BF16 scale/bias regions leave
    /// the weight offset only 2-aligned. The vectorized kernel's `ushort` load
    /// must stay correct here; a `uint` load would be misaligned. N=2816 forces
    /// the vectorized loop rather than only the scalar tail.
    @Test func gemv_weightsAt2AlignedNot4AlignedOffset() throws {
        try Self.runAndCompare(m: 128, n: 2816, seed: 0xC4, weightByteOffset: 2)
    }

    @Test(arguments: [64, 65, 128, 129] as [Int],
                     OffByMultiples.multiplesOfGroup.filter { $0 <= 512 })
    func gemv_sweep(m: Int, n: Int) throws {
        let seed: UInt64 = UInt64(m) &* 0x9E37 &+ UInt64(n)
        try Self.runAndCompare(m: m, n: n, seed: seed)
    }
}
