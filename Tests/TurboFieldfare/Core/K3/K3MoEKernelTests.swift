import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport
import TurboFieldfareFormat

/// Drives the fused K3 LatentMoE kernels (`k3_moe_phase1_situ`,
/// `k3_moe_phase2_reduce`) against `K3MoEReference` on small shapes, with
/// the expert pool in ONE buffer plus a UInt64 slot-offset table, exactly
/// like the production binding. Also pins the canonical expert stride math
/// (17,547,264 bytes) without allocating a real expert.
@Suite struct K3MoEKernelTests {

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

    /// One random expert blob laid out per `K3ExpertSubtensorOffsets.canonical`.
    /// Packed sub-tensors take any byte (every nibble pattern is valid E2M1);
    /// scale bytes sit in 2^-7...2^5 with occasional 0xFF (zeroed group) and
    /// 0x00 (subnormal) sentinels.
    private static func randomBlob(dLatent: Int, intermediate: Int,
                                   offsets: K3ExpertSubtensorOffsets,
                                   rng: inout SplitMix64) -> [UInt8] {
        let size = Int(K3ExpertSubtensorOffsets.canonicalBlobSize(
            dLatent: UInt32(dLatent), intermediate: UInt32(intermediate)))
        var blob = [UInt8](repeating: 0, count: size)
        let packedBytes = dLatent * intermediate / 2
        let scaleBytes = dLatent * intermediate / 32
        for region in [offsets.w1PackedOff, offsets.w2PackedOff,
                       offsets.w3PackedOff] {
            for i in 0..<packedBytes {
                blob[Int(region) + i] = UInt8(truncatingIfNeeded: rng.next())
            }
        }
        for region in [offsets.w1ScalesOff, offsets.w2ScalesOff,
                       offsets.w3ScalesOff] {
            for i in 0..<scaleBytes {
                let roll = rng.next() % 97
                if roll == 0 {
                    blob[Int(region) + i] = 0x00
                } else if roll <= 5 {
                    blob[Int(region) + i] = 0xFF
                } else {
                    blob[Int(region) + i] = UInt8(120 + rng.next() % 13)
                }
            }
        }
        return blob
    }

    private struct Fixture {
        var dLatent: Int
        var intermediate: Int
        var topK: Int
        var poolCount: Int
        var blobStride: Int
        var selectedSlots: [Int]      // pool slots, shuffled (non-sorted)
        var offsets: K3ExpertSubtensorOffsets
        var pool: [UInt8]             // ALL pool experts, stride `blobStride`
        var selectedBlobs: [[UInt8]]  // in selection order
        var xLatFp16: [Float16]
        var xLatFp32: [Float]         // Float(Float16(...)) — kernel-exact
        var weights: [Float]          // renormalized, like the router emits
        var referenceH: [Float]       // [topK * intermediate]
        var referenceY: [Float]       // [dLatent]
    }

    private static func makeFixture(seed: UInt64, dLatent: Int,
                                    intermediate: Int, topK: Int,
                                    poolCount: Int) -> Fixture {
        var rng = SeedTree(seed).key(
            "k3-moe-d\(dLatent)-f\(intermediate)-k\(topK)-p\(poolCount)")
        let offsets = K3ExpertSubtensorOffsets.canonical(
            dLatent: UInt32(dLatent), intermediate: UInt32(intermediate))
        let blobStride = Int(K3ExpertSubtensorOffsets.canonicalBlobSize(
            dLatent: UInt32(dLatent), intermediate: UInt32(intermediate)))
        var pool: [UInt8] = []
        for _ in 0..<poolCount {
            pool.append(contentsOf: randomBlob(
                dLatent: dLatent, intermediate: intermediate,
                offsets: offsets, rng: &rng))
        }
        // Selection: shuffled distinct pool slots — the offsets table must
        // honor non-contiguous, non-sorted selections.
        let selectedSlots = Array((0..<poolCount).shuffled(using: &rng).prefix(topK))
        let selectedBlobs = selectedSlots.map {
            Array(pool[($0 * blobStride)..<(($0 + 1) * blobStride)])
        }

        let xLatFp16 = (0..<dLatent).map { _ in Float16(rng.uniform(-0.75, 0.75)) }
        let xLatFp32 = xLatFp16.map { Float($0) }
        var rawWeights = (0..<topK).map { _ in rng.uniform(0.05, 1.0) }
        let sum = rawWeights.reduce(0, +)
        for i in rawWeights.indices { rawWeights[i] /= sum }

        var referenceH = [Float](repeating: 0, count: topK * intermediate)
        for (slot, blob) in selectedBlobs.enumerated() {
            let (h, _) = K3MoEReference.expertForward(
                blob: blob, offsets: offsets, xLat: xLatFp32,
                dLatent: dLatent, intermediate: intermediate)
            referenceH.replaceSubrange(
                (slot * intermediate)..<((slot + 1) * intermediate), with: h)
        }
        let referenceY = K3MoEReference.apply(
            xLat: xLatFp32, blobs: selectedBlobs, offsets: offsets,
            routingWeights: rawWeights,
            dLatent: dLatent, intermediate: intermediate)

        return Fixture(dLatent: dLatent, intermediate: intermediate,
                       topK: topK, poolCount: poolCount, blobStride: blobStride,
                       selectedSlots: selectedSlots, offsets: offsets,
                       pool: pool, selectedBlobs: selectedBlobs,
                       xLatFp16: xLatFp16, xLatFp32: xLatFp32,
                       weights: rawWeights,
                       referenceH: referenceH, referenceY: referenceY)
    }

    /// Allocates the one-experts-buffer + offset-table binding (offsets =
    /// pool slot x stride, as in the packed_experts region) and runs both
    /// phases in a single command buffer, like production decode.
    private static func runGPU(fixture: Fixture) throws -> (h: [Float], y: [Float]) {
        let ctx = try MetalContext()
        let moe = try K3MoE(context: ctx)

        let slotOffsets = fixture.selectedSlots.map {
            UInt64($0 * fixture.blobStride)
        }
        guard let experts = ctx.device.makeBuffer(
                bytes: fixture.pool, length: fixture.pool.count,
                options: .storageModeShared),
              let slots = slotOffsets.withUnsafeBufferPointer({
                  ctx.device.makeBuffer(
                    bytes: $0.baseAddress!,
                    length: slotOffsets.count * MemoryLayout<UInt64>.stride,
                    options: .storageModeShared)
              }),
              let xLat = Fp16Buffer.make(ctx.device, halves: fixture.xLatFp16),
              let hBuf = ctx.device.makeBuffer(
                length: fixture.topK * fixture.intermediate
                    * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let wBuf = makeF32Buffer(ctx.device, fixture.weights),
              let yBuf = ctx.device.makeBuffer(
                length: fixture.dLatent * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers")
            return ([], [])
        }

        moe.encodePhase1(commandBuffer: cmd,
                         experts: experts, slotOffsets: slots,
                         xLat: xLat, h: hBuf,
                         subtensorOffsets: fixture.offsets,
                         dLatent: UInt32(fixture.dLatent),
                         intermediate: UInt32(fixture.intermediate),
                         topK: UInt32(fixture.topK))
        moe.encodePhase2(commandBuffer: cmd,
                         experts: experts, slotOffsets: slots,
                         h: hBuf, routingWeights: wBuf, yLat: yBuf,
                         subtensorOffsets: fixture.offsets,
                         dLatent: UInt32(fixture.dLatent),
                         intermediate: UInt32(fixture.intermediate),
                         topK: UInt32(fixture.topK))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        return (readF32Buffer(hBuf, count: fixture.topK * fixture.intermediate),
                readF32Buffer(yBuf, count: fixture.dLatent))
    }

    // MARK: kernel vs reference

    @Test func phase1SiTUMatchesReference() throws {
        let fixture = Self.makeFixture(
            seed: 0xC1, dLatent: 64, intermediate: 96, topK: 3, poolCount: 8)
        let (h, _) = try Self.runGPU(fixture: fixture)
        let rel = RelError.compute(actual: h, reference: fixture.referenceH)
        let maxAbs = RelError.maxAbsDiff(h, fixture.referenceH)
        #expect(rel < Tolerance.fp16Reduction,
                "phase1 h: rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func phase1And2LatentMoEMatchesReference() throws {
        let fixture = Self.makeFixture(
            seed: 0xC2, dLatent: 64, intermediate: 96, topK: 3, poolCount: 8)
        let (_, y) = try Self.runGPU(fixture: fixture)
        let rel = RelError.compute(actual: y, reference: fixture.referenceY)
        let maxAbs = RelError.maxAbsDiff(y, fixture.referenceY)
        #expect(rel < Tolerance.fp16Reduction,
                "y_lat: rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Full canonical k=16 selection over a 16-expert pool.
    @Test func latentMoE_topK16() throws {
        let fixture = Self.makeFixture(
            seed: 0xC3, dLatent: 32, intermediate: 64, topK: 16, poolCount: 16)
        let (h, y) = try Self.runGPU(fixture: fixture)
        let relH = RelError.compute(actual: h, reference: fixture.referenceH)
        let relY = RelError.compute(actual: y, reference: fixture.referenceY)
        #expect(relH < Tolerance.fp16Reduction, "phase1 h: rel=\(relH)")
        #expect(relY < Tolerance.fp16Reduction, "y_lat: rel=\(relY)")
    }

    /// Minimal shapes: two groups per w1/w3 row, one per w2 row, one expert.
    @Test func latentMoE_singleExpertMinimalGroups() throws {
        let fixture = Self.makeFixture(
            seed: 0xC4, dLatent: 64, intermediate: 32, topK: 1, poolCount: 2)
        let (_, y) = try Self.runGPU(fixture: fixture)
        let rel = RelError.compute(actual: y, reference: fixture.referenceY)
        #expect(rel < Tolerance.fp16Reduction, "y_lat: rel=\(rel)")
    }

    // MARK: canonical layout

    /// The canonical K3 expert stride math, checked against the format
    /// profile's pinned constant — no 17.5 MB allocation, offsets only.
    @Test func canonicalExpertLayoutMatchesFormatProfile() {
        let offsets = K3ExpertSubtensorOffsets.canonical(dLatent: 3584,
                                                         intermediate: 3072)
        #expect(offsets.w1PackedOff == 0)
        #expect(offsets.w1ScalesOff == 5_505_024)
        #expect(offsets.w2PackedOff == 5_849_088)
        #expect(offsets.w2ScalesOff == 11_354_112)
        #expect(offsets.w3PackedOff == 11_698_176)
        #expect(offsets.w3ScalesOff == 17_203_200)
        let size = K3ExpertSubtensorOffsets.canonicalBlobSize(dLatent: 3584,
                                                              intermediate: 3072)
        #expect(size == 17_547_264)
        #expect(size == KimiK3FormatProfile.expertStride)
        #expect(UInt64(offsets.w3ScalesOff) + 3072 * 3584 / 32 == size)
        // The blob is a multiple of the 16 KB v2 alignment, which is what
        // makes experts directly streamable.
        #expect(size % 16_384 == 0)
        #expect(GTurboFormatV2.mxfp4ExpertTensorNames == [
            "w1_packed", "w1_scales", "w2_packed", "w2_scales",
            "w3_packed", "w3_scales",
        ])
    }
}
