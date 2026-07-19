import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `dequant_int8_gemv` kernel against
/// `DequantInt8GemvRef`, which bulk-dequantizes each row to FP32 and dots
/// with `vDSP_dotpr`. Reference is unit-tested in `DequantInt8GemvRefTests`.
///
/// Covers:
///   * Router shape  — M=128, N=2816
///   * lm_head sub-shape — M=1024, N=2816 (real lm_head is M=262144)
@Suite struct DequantInt8GEMVTests {

    private static func runAndCompare(m: Int, n: Int, seed: UInt64) throws -> Float {
        precondition(n % Quantization.groupSize == 0)
        let groupsPerRow = n / Quantization.groupSize
        var rng = SeedTree(seed).key("int8-gemv-m\(m)-n\(n)")

        var rows: [Quantization.Int8AffineRow] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            rows.append(Quantization.quantizeInt8Affine(raw))
        }

        var packed = [UInt8](repeating: 0, count: m * n)
        var scales = [UInt16](repeating: 0, count: m * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: m * groupsPerRow)
        for row in 0..<m {
            for i in 0..<n { packed[row * n + i] = rows[row].packed[i] }
            for g in 0..<groupsPerRow {
                scales[row * groupsPerRow + g] = rows[row].scales[g]
                biases[row * groupsPerRow + g] = rows[row].biases[g]
            }
        }

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantInt8GEMV(context: ctx)

        guard let wBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count,
                options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return .infinity
        }
        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return .infinity
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, scales: sBuf, biases: bBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()

        let ref = DequantInt8GemvRef.apply(weightRows: rows, x: xRef, n: n)
        let actual = Fp16Buffer.read(yBuf, count: m)
        return RelError.compute(actual: actual, reference: ref)
    }

    @Test func gemv_router_shape() throws {
        let rel = try Self.runAndCompare(m: 128, n: 2816, seed: 0x151)
        #expect(rel < Tolerance.fp16Reduction, "router rel=\(rel)")
    }

    @Test func gemv_lmHeadSubShape() throws {
        let rel = try Self.runAndCompare(m: 1024, n: 2816, seed: 0x152)
        #expect(rel < Tolerance.fp16Reduction, "lm_head sub-shape rel=\(rel)")
    }

    @Test(arguments: [128, 256, 1024] as [Int],
                     OffByMultiples.multiplesOfGroup.filter { $0 <= 512 })
    func gemv_sweep(m: Int, n: Int) throws {
        let rel = try Self.runAndCompare(m: m, n: n, seed: UInt64(m * 1000 + n))
        #expect(rel < Tolerance.fp16Reduction, "M=\(m) N=\(n) rel=\(rel)")
    }
}
