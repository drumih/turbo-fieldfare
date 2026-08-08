import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Drives the MLA absorbed-decode kernels (`k3_mla_kvb_expand`,
/// `k3_mla_absorb_q`, `k3_mla_attn_decode_{partial,combine}`,
/// `k3_mla_out_project`, `k3_mla_cache_append`) against
/// `K3MLAReference.attention` — the NAIVE expanded eager form, which is the
/// ground truth for the absorbed math. Small shapes: H=4, L=64, R=8,
/// N=V=16, contexts up to 70 tokens (split-KV with 16 chunks).
@Suite struct K3MLAKernelTests {

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

    private static let H = 4
    private static let L = 64
    private static let R = 8
    private static let N = 16
    private static let V = 16
    private static let rowLen = L + R
    private static let capacity = 80
    private static let scale = 1 / Float(24.0).squareRoot()  // (N+R)^-0.5

    private struct Fixture {
        var kvB: [Float16]        // [H*(N+V)][L]
        var normWeight: [Float]   // [L]
        var kvATokens: [[Float16]] // 70 raw kv_a rows
        var q: [Float16]          // [H][N+R]
        var gate: [Float16]       // [H*V]
    }

    private static func makeFixture(seed: UInt64, qScale: Float = 1) -> Fixture {
        var rng = SeedTree(seed).key("k3-mla-h\(H)-l\(L)-q\(qScale)")
        return Fixture(
            kvB: (0..<(H * (N + V) * L)).map { _ in
                Float16(rng.uniform(-0.2, 0.2)) },
            normWeight: (0..<L).map { _ in rng.uniform(0.5, 1.5) },
            kvATokens: (0..<70).map { _ in
                (0..<rowLen).map { _ in Float16(rng.uniform(-1.5, 1.5)) } },
            q: (0..<(H * (N + R))).map { _ in
                Float16(qScale * rng.uniform(-1.0, 1.0)) },
            gate: (0..<(H * V)).map { _ in Float16(rng.uniform(-1.5, 1.5)) })
    }

    /// Builds the GPU cache (all 70 appends in one command buffer) and
    /// returns (context, mla, cache buffer, the fixture's GPU-side buffers).
    private static func buildCache(_ fixture: Fixture) throws
        -> (K3MLA, MTLBuffer)
    {
        let ctx = try MetalContext()
        let mla = try K3MLA(context: ctx)
        guard let cache = ctx.device.makeBuffer(
                length: capacity * rowLen * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let normBuf = makeF32Buffer(ctx.device, fixture.normWeight),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate cache buffers")
            return (mla, ctx.device.makeBuffer(length: 16, options: .storageModeShared)!)
        }
        for (t, kvA) in fixture.kvATokens.enumerated() {
            guard let kvABuf = Fp16Buffer.make(ctx.device, halves: kvA) else {
                Issue.record("Failed to allocate kvA buffer")
                return (mla, cache)
            }
            mla.encodeCacheAppend(commandBuffer: cmd, cache: cache,
                                  position: UInt32(t), kvA: kvABuf,
                                  normWeight: normBuf, eps: 1e-5,
                                  latent: UInt32(L), rope: UInt32(R))
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        return (mla, cache)
    }

    /// One absorbed decode over rows [0, seqLen): expand + absorb_q +
    /// split-KV + out_project. Returns the gated fp16 output [H*V].
    private static func decode(_ mla: K3MLA, cache: MTLBuffer,
                               fixture: Fixture, seqLen: Int) throws -> [Float] {
        let device = cache.device
        guard let kvBBuf = Fp16Buffer.make(device, halves: fixture.kvB),
              let kTBuf = Fp16Buffer.make(device, count: H * L * N),
              let vBuf = Fp16Buffer.make(device, count: H * V * L),
              let qBuf = Fp16Buffer.make(device, halves: fixture.q),
              let qAbsBuf = device.makeBuffer(
                length: H * rowLen * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let outLatBuf = device.makeBuffer(
                length: H * L * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let gateBuf = Fp16Buffer.make(device, halves: fixture.gate),
              let outBuf = Fp16Buffer.make(device, count: H * V),
              let cmd = device.makeCommandQueue()?.makeCommandBuffer() else {
            Issue.record("Failed to allocate decode buffers")
            return []
        }
        mla.encodeKVBExpand(commandBuffer: cmd, kvB: kvBBuf, kT: kTBuf, v: vBuf,
                            numHeads: UInt32(H), latent: UInt32(L),
                            nope: UInt32(N), vHead: UInt32(V))
        mla.encodeAbsorbQ(commandBuffer: cmd, kT: kTBuf, q: qBuf, qAbs: qAbsBuf,
                          numHeads: UInt32(H), latent: UInt32(L),
                          rope: UInt32(R), nope: UInt32(N))
        mla.encodeAttnDecode(commandBuffer: cmd, cache: cache, qAbs: qAbsBuf,
                             outLat: outLatBuf, seqLen: UInt32(seqLen),
                             numHeads: UInt32(H), latent: UInt32(L),
                             rope: UInt32(R), scale: Self.scale)
        mla.encodeOutProject(commandBuffer: cmd, v: vBuf, outLat: outLatBuf,
                             gate: gateBuf, out: outBuf,
                             numHeads: UInt32(H), latent: UInt32(L),
                             vHead: UInt32(V))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        return Fp16Buffer.read(outBuf, count: H * V)
    }

    /// The cache rows as fp32-of-fp16, for the reference.
    private static func cacheRowsFp32(_ cache: MTLBuffer, tokens: Int)
        -> [[Float]]
    {
        let halves = Fp16Buffer.readHalf(cache, count: tokens * rowLen)
        return (0..<tokens).map { t in
            Array(halves[(t * rowLen)..<((t + 1) * rowLen)]).map { Float($0) }
        }
    }

    private static func reference(_ fixture: Fixture, cache: MTLBuffer,
                                  tokens: Int) -> [Float] {
        K3MLAReference.attention(
            kvB: fixture.kvB.map { Float($0) },
            q: fixture.q.map { Float($0) },
            cache: cacheRowsFp32(cache, tokens: tokens),
            gate: fixture.gate.map { Float($0) },
            scale: Self.scale,
            numHeads: H, latent: L, rope: R, nope: N, vHead: V)
    }

    // MARK: tests

    /// Cache append: row content is [normed latent | raw rope] in fp16.
    @Test func cacheAppendMatchesReference() throws {
        let (L, R) = (Self.L, Self.R)
        let fixture = Self.makeFixture(seed: 0xF1)
        let (_, cache) = try Self.buildCache(fixture)
        let rows = Self.cacheRowsFp32(cache, tokens: 70)
        for t in [0, 1, 7, 69] {
            let expected = K3MLAReference.cacheRow(
                kvA: fixture.kvATokens[t].map { Float($0) },
                normWeight: fixture.normWeight, eps: 1e-5,
                latent: L, rope: R)
            // The kernel stores fp16; the reference is fp32 — the gap is
            // fp16 output rounding (max ~2^-11 relative).
            let rel = RelError.compute(actual: rows[t], reference: expected)
            #expect(rel < Tolerance.fp16Reduction, "row \(t): rel=\(rel)")
        }
    }

    /// The expand + absorb_q layout: qAbs must equal the direct fp32
    /// computation q~[h][l] = sum_i kvB[h][i][l] * q_nope[h][i], with the
    /// rope part passed through.
    @Test func absorbQMatchesDirectComputation() throws {
        let (H, L, R, N, V, rowLen) = (Self.H, Self.L, Self.R, Self.N, Self.V,
                                       Self.rowLen)
        let fixture = Self.makeFixture(seed: 0xF2)
        let (mla, _) = try Self.buildCache(fixture)
        let device = try MetalContext().device
        guard let kvBBuf = Fp16Buffer.make(device, halves: fixture.kvB),
              let kTBuf = Fp16Buffer.make(device, count: H * L * N),
              let vBuf = Fp16Buffer.make(device, count: H * V * L),
              let qBuf = Fp16Buffer.make(device, halves: fixture.q),
              let qAbsBuf = device.makeBuffer(
                length: H * rowLen * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let cmd = device.makeCommandQueue()?.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers")
            return
        }
        mla.encodeKVBExpand(commandBuffer: cmd, kvB: kvBBuf, kT: kTBuf, v: vBuf,
                            numHeads: UInt32(H), latent: UInt32(L),
                            nope: UInt32(N), vHead: UInt32(V))
        mla.encodeAbsorbQ(commandBuffer: cmd, kT: kTBuf, q: qBuf, qAbs: qAbsBuf,
                          numHeads: UInt32(H), latent: UInt32(L),
                          rope: UInt32(R), nope: UInt32(N))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let qAbs = Self.readF32Buffer(qAbsBuf, count: H * rowLen)
        let kvB = fixture.kvB.map { Float($0) }
        let q = fixture.q.map { Float($0) }
        var expected = [Float](repeating: 0, count: H * rowLen)
        for h in 0..<H {
            for l in 0..<L {
                var acc: Float = 0
                for i in 0..<N {
                    acc += kvB[(h * (N + V) + i) * L + l] * q[h * (N + R) + i]
                }
                expected[h * rowLen + l] = acc
            }
            for r in 0..<R {
                expected[h * rowLen + L + r] = q[h * (N + R) + N + r]
            }
        }
        let rel = RelError.compute(actual: qAbs, reference: expected)
        let maxAbs = RelError.maxAbsDiff(qAbs, expected)
        #expect(rel < Tolerance.identity, "qAbs rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Full absorbed decode vs the naive expanded reference at three cache
    /// depths: 1 (single chunk), 5, and 70 tokens (16 split-KV chunks of
    /// 5, two of them empty-tail).
    @Test func decodeMatchesReferenceAtMultipleDepths() throws {
        let fixture = Self.makeFixture(seed: 0xF3)
        let (mla, cache) = try Self.buildCache(fixture)
        for tokens in [1, 5, 70] {
            let actual = try Self.decode(mla, cache: cache,
                                         fixture: fixture, seqLen: tokens)
            let expected = Self.reference(fixture, cache: cache, tokens: tokens)
            let rel = RelError.compute(actual: actual, reference: expected)
            let maxAbs = RelError.maxAbsDiff(actual, expected)
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "T=\(tokens): rel=\(rel) maxAbs=\(maxAbs)")
        }
    }

    /// Softmax stability: q scaled by 40 makes scores span hundreds; the
    /// online softmax (and the reference) both subtract the running max.
    @Test func decodeSoftmaxStabilityLargeScores() throws {
        let fixture = Self.makeFixture(seed: 0xF4, qScale: 40)
        let (mla, cache) = try Self.buildCache(fixture)
        let actual = try Self.decode(mla, cache: cache,
                                     fixture: fixture, seqLen: 70)
        #expect(actual.allSatisfy { $0.isFinite }, "output must stay finite")
        let expected = Self.reference(fixture, cache: cache, tokens: 70)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "large scores: rel=\(rel)")
    }

    /// The expand plane layouts, checked element-for-element against the
    /// documented gather (k transposed per head, v row-copied).
    @Test func kvbExpandLayout() throws {
        let (H, L, N, V) = (Self.H, Self.L, Self.N, Self.V)
        let fixture = Self.makeFixture(seed: 0xF5)
        let (mla, _) = try Self.buildCache(fixture)
        let device = try MetalContext().device
        guard let kvBBuf = Fp16Buffer.make(device, halves: fixture.kvB),
              let kTBuf = Fp16Buffer.make(device, count: H * L * N),
              let vBuf = Fp16Buffer.make(device, count: H * V * L),
              let cmd = device.makeCommandQueue()?.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers")
            return
        }
        mla.encodeKVBExpand(commandBuffer: cmd, kvB: kvBBuf, kT: kTBuf, v: vBuf,
                            numHeads: UInt32(H), latent: UInt32(L),
                            nope: UInt32(N), vHead: UInt32(V))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let kT = Fp16Buffer.readHalf(kTBuf, count: H * L * N)
        let v = Fp16Buffer.readHalf(vBuf, count: H * V * L)
        for h in 0..<H {
            for l in 0..<L {
                for i in 0..<N {
                    #expect(kT[(h * L + l) * N + i]
                            == fixture.kvB[(h * (N + V) + i) * L + l])
                }
            }
            for j in 0..<V {
                for l in 0..<L {
                    #expect(v[(h * V + j) * L + l]
                            == fixture.kvB[(h * (N + V) + N + j) * L + l])
                }
            }
        }
    }
}
