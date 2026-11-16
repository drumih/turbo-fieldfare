import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// Compares the Metal `logit_softcap_softmax` kernel against
/// `LogitSoftcapSoftmaxRef`, an Accelerate-based two-pass reference (apply
/// softcap → find max → subtract → exp → divide by sum). The kernel does a
/// single-pass online safe softmax with the softcap fused into the (m, d)
/// running update. Different code paths through the same math.
@Suite struct LogitSoftcapSoftmaxTests {

    private static let softcap: Float = 30.0

    private static func runKernel(logitsFp16: [Float16], v: Int, softcap: Float) throws -> [Float] {
        let ctx = try MetalContext()
        let kernel = try LogitSoftcapSoftmax(context: ctx)

        guard let inBuf = Fp16Buffer.make(ctx.device, halves: logitsFp16),
            let outBuf = Fp16Buffer.make(ctx.device, count: v),
            let cmd = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("Failed to allocate Metal resources")
            return []
        }
        try kernel.encode(
            commandBuffer: cmd,
            logits: inBuf, probs: outBuf,
            v: UInt32(v), softcap: softcap)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Fp16Buffer.read(outBuf, count: v)
    }

    /// The cross-SIMD merge publishes `final_m` / `final_inv_d` through
    /// threadgroup memory. Seeding those defaults from every thread instead of
    /// one races the real write: a late default store leaves final_m = -inf and
    /// final_inv_d = 0, and the normalize loop then emits exp(z - -inf) * 0 =
    /// NaN for the entire row.
    ///
    /// Repeat over vocab sizes that span the SIMD-group counts (one group, a
    /// partial group, the full eight) so the merge path is exercised with
    /// varying numbers of contributing groups, and assert the invariant every
    /// consumer depends on: finite probabilities that sum to one.
    @Test func probabilitiesStayFiniteAndNormalizedAcrossSimdGroupCounts() throws {
        for v in [1, 31, 32, 33, 64, 200, 256, 257, 1024, 4096] {
            var rng = SeedTree(0xF17E).key("softcap-finite-\(v)")
            let logits = (0..<v).map { _ in Float16(rng.uniform(-40.0, 40.0)) }
            for attempt in 0..<8 {
                let probs = try Self.runKernel(logitsFp16: logits, v: v, softcap: 30.0)
                #expect(
                    probs.allSatisfy { $0.isFinite },
                    "non-finite probability at v=\(v) attempt=\(attempt)")
                let total = probs.reduce(0, +)
                #expect(
                    abs(total - 1.0) < 2e-2,
                    "probabilities sum to \(total), not 1, at v=\(v) attempt=\(attempt)")
            }
        }
    }

    @Test func randomLogits_matchesReference() throws {
        let v = 2048
        var rng = SeedTree(0x131).key("softcap-softmax-random")
        let logitsFp32 = (0..<v).map { _ in rng.uniform(-50.0, 50.0) }
        let logitsFp16 = logitsFp32.map { Float16($0) }

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: Self.softcap)
        let cpu = LogitSoftcapSoftmaxRef.apply(
            x: logitsFp16.map { Float($0) }, softcap: Self.softcap
        )

        // Probabilities at the extreme tail are near FP16 subnormal; use the
        // bounded-relative form so we don't blow up on c~0.
        let rel = RelError.boundedRel(actual: gpu, reference: cpu, absFloor: 1e-4)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")

        // Probability axioms.
        let sum = gpu.reduce(0, +)
        #expect(abs(sum - 1.0) < Tolerance.fp16Reduction, "sum=\(sum)")
    }

    /// All logits identical → uniform 1/V distribution.
    @Test func uniformLogits_producesUniformProbs() throws {
        let v = 2048
        let logitsFp16 = [Float16](repeating: Float16(0.1234), count: v)

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: Self.softcap)
        let expected: Float = 1.0 / Float(v)
        for i in 0..<v {
            #expect(
                abs(gpu[i] - expected) < 5e-4,
                "i=\(i) g=\(gpu[i]) expected=\(expected)")
        }
    }

    /// One logit far above the softcap, rest at zero. Softcap must pin the
    /// outlier — kernel must not let the un-capped 1000.0 leak through and
    /// overflow exp. Small softcap (5.0) so saturated prob is < 1.0.
    @Test func singleHugeLogit_isSoftcapped() throws {
        let v = 2048
        let smallSoftcap: Float = 5.0
        var logits = [Float](repeating: 0, count: v)
        logits[42] = 1000.0
        let logitsFp16 = logits.map { Float16($0) }

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: smallSoftcap)
        let cpu = LogitSoftcapSoftmaxRef.apply(
            x: logitsFp16.map { Float($0) }, softcap: smallSoftcap
        )
        let g = gpu[42]
        let c = cpu[42]
        #expect(c < 0.5, "softcap reference should not saturate: c=\(c)")
        #expect(abs(g - c) / c < Tolerance.fp16Reduction, "g=\(g) c=\(c)")
    }

    /// A single NaN logit used to take the whole row with it: `tanh(NaN)` is
    /// NaN, the running sum went NaN, and every probability came out NaN, which
    /// the sampler can only answer with its in-range fallback. `softcap_value`
    /// now folds NaN to -inf, leaving the softmax over the finite logits. This
    /// fails on the old code at the first `isFinite` assertion.
    ///
    /// Both softcap settings are covered because the fold has to precede the
    /// `softcap <= 0` early return: with capping disabled there is no `tanh` to
    /// swallow the NaN, which is exactly the Qwen 3.6 production setting.
    @Test func oneNaNLogitDoesNotPoisonTheRow() throws {
        let v = 1024
        var logits = [Float16](repeating: Float16(-30.0), count: v)
        logits[0] = .nan
        for softcap in [Float(0), 30.0] {
            let probs = try Self.runKernel(logitsFp16: logits, v: v, softcap: softcap)
            #expect(
                probs.allSatisfy { $0.isFinite },
                "softcap \(softcap): non-finite probability")
            let total = probs.reduce(0, +)
            #expect(
                abs(total - 1.0) < 2e-2,
                "softcap \(softcap): probabilities sum to \(total)")
            #expect(probs[0] == 0, "softcap \(softcap): the NaN entry kept mass")
        }
    }

    /// A row where *every* logit is NaN cannot be rescued by folding: there is
    /// no finite value left, so the kernel writes a defined all-zero row
    /// (instead of `exp(-inf - -inf) * 0 = NaN`) and publishes a non-finite row
    /// max for the caller to report. Both front-ends are checked, because a
    /// silent NaN row is what the sampler's in-range fallback turns into an
    /// endless run of token 0.
    @Test func anAllNaNRowIsEmptyAndReportsNoRowMax() throws {
        let ctx = try MetalContext()
        let v = 512
        let single = try LogitSoftcapSoftmax(context: ctx)
        let tiled = try LogitSoftcapSoftmaxTiled(context: ctx, vocab: v)
        for useTiled in [false, true] {
            guard
                let logits = Fp16Buffer.make(
                    ctx.device,
                    halves: [Float16](repeating: .nan, count: v)),
                let probs = Fp16Buffer.make(ctx.device, count: v),
                let cb = ctx.queue.makeCommandBuffer()
            else {
                Issue.record("Metal resource allocation failed")
                return
            }
            if useTiled {
                try tiled.encode(
                    commandBuffer: cb, logits: logits, probs: probs,
                    v: UInt32(v), softcap: 30.0)
            } else {
                try single.encode(
                    commandBuffer: cb, logits: logits, probs: probs,
                    v: UInt32(v), softcap: 30.0)
            }
            cb.commit()
            cb.waitUntilCompleted()
            let row = Fp16Buffer.read(probs, count: v)
            #expect(
                row.allSatisfy { $0 == 0 },
                "tiled=\(useTiled): an all-NaN row must be empty, not NaN")
            let rowMax = useTiled ? tiled.rowMax : single.rowMax
            #expect(
                !rowMax.isFinite,
                "tiled=\(useTiled): row max \(rowMax) should be non-finite")
        }
    }

    /// The tiled front-end must agree with the single-threadgroup original.
    ///
    /// Online-softmax rescaling is associative in exact arithmetic, so the two
    /// differ only by fp32 reduction order; at the production vocabulary that
    /// is far below fp16 storage precision. This is the gate that lets the
    /// tiled form serve production sampling.
    @Test func tiledMatchesSingleThreadgroupAcrossVocabularies() throws {
        let ctx = try MetalContext()
        let single = try LogitSoftcapSoftmax(context: ctx)
        for vocab in [248_320, 4_096, 4_097, 1_003] {
            let tiled = try LogitSoftcapSoftmaxTiled(context: ctx, vocab: vocab)
            guard
                let logits = ctx.device.makeBuffer(
                    length: vocab * MemoryLayout<Float16>.stride,
                    options: .storageModeShared),
                let a = ctx.device.makeBuffer(
                    length: vocab * MemoryLayout<Float16>.stride,
                    options: .storageModeShared),
                let b = ctx.device.makeBuffer(
                    length: vocab * MemoryLayout<Float16>.stride,
                    options: .storageModeShared)
            else { throw MetalError.noDevice }
            let src = logits.contents().bindMemory(
                to: Float16.self,
                capacity: vocab)
            var state: UInt64 = 0x9E37_79B9_7F4A_7C15
            for index in 0..<vocab {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let unit = Float(state >> 40) / Float(1 << 24)
                src[index] = Float16(unit * 24 - 12)
            }
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try single.encode(
                commandBuffer: cb, logits: logits, probs: a,
                v: UInt32(vocab), softcap: 0)
            try tiled.encode(
                commandBuffer: cb, logits: logits, probs: b,
                v: UInt32(vocab), softcap: 0)
            cb.commit()
            cb.waitUntilCompleted()
            let pa = a.contents().bindMemory(to: Float16.self, capacity: vocab)
            let pb = b.contents().bindMemory(to: Float16.self, capacity: vocab)
            var worst: Float = 0
            var sumTiled: Float = 0
            for index in 0..<vocab {
                worst = max(worst, abs(Float(pa[index]) - Float(pb[index])))
                sumTiled += Float(pb[index])
            }
            #expect(
                worst < 1e-5,
                "vocab \(vocab): worst |single - tiled| = \(worst)")
            #expect(
                abs(sumTiled - 1) < 5e-2,
                "vocab \(vocab): tiled probabilities sum to \(sumTiled)")
        }
    }
}
