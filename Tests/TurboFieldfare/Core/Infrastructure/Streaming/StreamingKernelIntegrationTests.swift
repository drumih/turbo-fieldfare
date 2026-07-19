import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// End-to-end: write a synthetic .gturbo blob containing one "expert" worth of
/// affine-quantized weights (packed nibbles + BF16 scales + BF16 biases), open
/// it through `PreadExpertStreamer`, load it into a bounded cache slot, run
/// `dequant_int4_gemv`, and compare against an FP32 reference computed on the
/// CPU.
///
/// This is the smoke test for the streaming → kernel data path on the affine
/// decode hot path. No real Gemma weights, no model — just proof that the
/// pipe works end-to-end with the new layout.
@Suite struct StreamingKernelIntegrationTests {

    /// Layout for a single "expert blob" used in this test.
    ///   M = 64  output rows
    ///   N = 128 input columns          (2 groups of 64 per row)
    /// Packed:   M * N/2                 = 4096 bytes
    /// Scales:   M * N/groupSize * 2 B   =  256 bytes BF16
    /// Biases:   M * N/groupSize * 2 B   =  256 bytes BF16
    /// We page-align each of the three regions so MTLBuffer offsets are legal.
    private struct Sizes {
        static let M = 64
        static let N = 128
        static let groupsPerRow = N / Quantization.groupSize  // 2
        static let packedBytes  = M * (N / 2)                 // 4096
        static let scalesBytes  = M * groupsPerRow * MemoryLayout<UInt16>.size  // 256
        static let biasesBytes  = M * groupsPerRow * MemoryLayout<UInt16>.size  // 256
    }

    /// Build the in-memory bytes that we'll write to the fake .gturbo.
    /// Layout inside one expert blob (page-aligned externally):
    ///   [0,        packedBytes)              packed nibbles, row-major
    ///   [scalesOff, scalesOff + scalesBytes) BF16 scales, row-major
    ///   [biasesOff, biasesOff + biasesBytes) BF16 biases, row-major
    private static func buildExpertBlob(weightsFp32: [[Float]],
                                        pageSize: Int)
        -> (blob: [UInt8], scalesOffset: Int, biasesOffset: Int, blobSize: Int)
    {
        precondition(weightsFp32.count == Sizes.M)
        precondition(weightsFp32[0].count == Sizes.N)

        var packed = [UInt8]( repeating: 0, count: Sizes.packedBytes)
        var scales = [UInt16](repeating: 0, count: Sizes.M * Sizes.groupsPerRow)
        var biases = [UInt16](repeating: 0, count: Sizes.M * Sizes.groupsPerRow)
        for m in 0..<Sizes.M {
            let q = Quantization.quantizeInt4Affine(weightsFp32[m])
            for i in 0..<q.packed.count { packed[m * (Sizes.N / 2) + i] = q.packed[i] }
            for g in 0..<Sizes.groupsPerRow {
                scales[m * Sizes.groupsPerRow + g] = q.scales[g]
                biases[m * Sizes.groupsPerRow + g] = q.biases[g]
            }
        }

        let scalesOffset = roundUp(Sizes.packedBytes, to: pageSize)
        let biasesOffset = roundUp(scalesOffset + Sizes.scalesBytes, to: pageSize)
        let blobSize     = roundUp(biasesOffset + Sizes.biasesBytes, to: pageSize)

        var blob = [UInt8](repeating: 0, count: blobSize)
        blob.withUnsafeMutableBufferPointer { ptr in
            _ = memcpy(ptr.baseAddress!, packed, Sizes.packedBytes)
            scales.withUnsafeBufferPointer { sptr in
                _ = memcpy(ptr.baseAddress!.advanced(by: scalesOffset),
                           sptr.baseAddress!, Sizes.scalesBytes)
            }
            biases.withUnsafeBufferPointer { bptr in
                _ = memcpy(ptr.baseAddress!.advanced(by: biasesOffset),
                           bptr.baseAddress!, Sizes.biasesBytes)
            }
        }
        return (blob, scalesOffset, biasesOffset, blobSize)
    }

    private static func roundUp(_ n: Int, to pageSize: Int) -> Int {
        (n + pageSize - 1) / pageSize * pageSize
    }

    private static func cpuReferenceGemv(weightsFp32: [[Float]], x: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: Sizes.M)
        for m in 0..<Sizes.M {
            let q = Quantization.quantizeInt4Affine(weightsFp32[m])
            let w = Quantization.dequantizeInt4Affine(q, n: Sizes.N)
            var acc: Float = 0
            for n in 0..<Sizes.N { acc += w[n] * x[n] }
            y[m] = acc
        }
        return y
    }

    @Test func endToEnd_streamFromFile_runKernel_matchesCpuReference() throws {
        // ----- Setup: random weights + input -----
        var rng = SeedTree(0x51A7_1002).key("streaming-kernel-integration")
        var weights = [[Float]](repeating: [], count: Sizes.M)
        for m in 0..<Sizes.M {
            weights[m] = (0..<Sizes.N).map { _ in rng.uniform(-1.0, 1.0) }
        }
        let xFp32: [Float] = (0..<Sizes.N).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16: [Float16] = xFp32.map { Float16($0) }

        // ----- Build fake .gturbo on disk -----
        let pageSize = Int(getpagesize())
        let (blob, scalesOffset, biasesOffset, blobSize) =
            Self.buildExpertBlob(weightsFp32: weights, pageSize: pageSize)

        let headerSize = pageSize
        let fileSize   = headerSize + blobSize
        var fileBytes  = [UInt8](repeating: 0, count: fileSize)
        fileBytes.replaceSubrange(headerSize..<(headerSize + blob.count), with: blob)

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbo-fieldfare-smoke-\(UUID().uuidString).bin")
        try Data(fileBytes).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let layout = StreamLayout(
            path: tmp.path,
            streamOffset: UInt64(headerSize),
            streamSize:   UInt64(blobSize),
            expertsPerLayer: 1,
            expertStride: UInt64(blobSize)
        )
        let ctx = try MetalContext()
        let streamer = try PreadExpertStreamer(
            layout: layout,
            device: ctx.device,
            slotCount: 1)
        let kernel = try DequantInt4GEMV(context: ctx)
        let expert = try streamer.loadExpert(layer: 0, expert: 0)
        let weightsBuf = expert.buffer

        guard let xBuf = ctx.device.makeBuffer(length: Sizes.N * MemoryLayout<Float16>.size,
                                               options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(length: Sizes.M * MemoryLayout<Float16>.size,
                                               options: .storageModeShared) else {
            Issue.record("Failed to allocate x / y buffers")
            return
        }
        xBuf.contents().withMemoryRebound(to: Float16.self, capacity: Sizes.N) { dst in
            for i in 0..<Sizes.N { dst[i] = xFp16[i] }
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: weightsBuf, weightsOffset: Int(expert.offset),
                      scales: weightsBuf, scalesOffset: Int(expert.offset) + scalesOffset,
                      biases: weightsBuf, biasesOffset: Int(expert.offset) + biasesOffset,
                      x: xBuf, y: yBuf,
                      m: UInt32(Sizes.M), n: UInt32(Sizes.N))
        cmd.commit()
        cmd.waitUntilCompleted()

        let yKernel: [Float] = (0..<Sizes.M).map { i in
            Float(yBuf.contents().load(fromByteOffset: i * MemoryLayout<Float16>.size,
                                       as: Float16.self))
        }
        let yRef = Self.cpuReferenceGemv(weightsFp32: weights, x: xFp32)

        var maxAbsDiff: Float = 0
        var refNorm: Float = 0
        for i in 0..<Sizes.M {
            maxAbsDiff = max(maxAbsDiff, abs(yKernel[i] - yRef[i]))
            refNorm    = max(refNorm, abs(yRef[i]))
        }
        let relErr = maxAbsDiff / max(refNorm, 1e-6)

        #expect(relErr < 5e-3,
                "kernel vs CPU ref relErr=\(relErr) maxAbsDiff=\(maxAbsDiff) refNorm=\(refNorm)")

    }
}
