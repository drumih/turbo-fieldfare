import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// The PLE block's kernels, against CPU references derived from the reference
/// implementation (`mlx_qwen4exp/ple.py`).
@Suite("PLE block kernels")
struct PLEBlockTests {
    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    @Test("Per-stream score is a scaled dot product")
    func streamScore() throws {
        let dim = 256
        let streams = 4
        var rng = SeedTree(0x9E11).key("ple-score")
        let key = (0..<dim * streams).map { _ in Float16(rng.uniform(-1, 1)) }
        let query = (0..<dim * streams).map { _ in Float16(rng.uniform(-1, 1)) }

        let ctx = try MetalContext()
        let e = try Elementwise(context: ctx)
        guard let kb = Fp16Buffer.make(ctx.device, halves: key),
            let qb = Fp16Buffer.make(ctx.device, halves: query),
            let ob = Fp16Buffer.make(ctx.device, count: streams)
        else {
            Issue.record("alloc")
            return
        }
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try e.encodePLEStreamScore(
            commandBuffer: cb, key: kb, query: qb,
            out: ob, dim: dim, streams: streams)
        cb.commit()
        cb.waitUntilCompleted()

        let got = Fp16Buffer.read(ob, count: streams)
        for s in 0..<streams {
            var acc: Float = 0
            for d in 0..<dim {
                acc += Float(key[s * dim + d]) * Float(query[s * dim + d])
            }
            let want = acc / Float(dim).squareRoot()
            #expect(abs(got[s] - want) < max(0.02, abs(want) * 0.02), "stream \(s)")
        }
    }

    @Test("The gate takes a SIGNED square root, and floors the magnitude")
    func signedSqrtGate() throws {
        // Sign must survive the root: dropping it makes every gate >= 0.5 and
        // the block can then only ever add, never subtract.
        let values: [Float16] = [-9, -1, -1e-9, 0, 1e-9, 1, 9].map(Float16.init)
        let ctx = try MetalContext()
        let e = try Elementwise(context: ctx)
        guard let xb = Fp16Buffer.make(ctx.device, halves: values),
            let ob = Fp16Buffer.make(ctx.device, count: values.count)
        else {
            Issue.record("alloc")
            return
        }
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try e.encodePLESignedSqrtGate(
            commandBuffer: cb, x: xb, out: ob,
            count: values.count)
        cb.commit()
        cb.waitUntilCompleted()

        let got = Fp16Buffer.read(ob, count: values.count)
        for (i, v) in values.enumerated() {
            let x = Float(v)
            let m = max(abs(x), 1e-6).squareRoot()
            let want = Self.sigmoid(x < 0 ? -m : m)
            #expect(abs(got[i] - want) < 5e-3, "x=\(x)")
        }
        // Negative inputs must gate below 0.5, positive above.
        #expect(got[0] < 0.5)
        #expect(got[values.count - 1] > 0.5)
    }

    @Test("Broadcast scale applies each stream's own gate")
    func broadcastScale() throws {
        let dim = 128
        let streams = 4
        var rng = SeedTree(0xAB12).key("ple-broadcast")
        let value = (0..<dim).map { _ in Float16(rng.uniform(-1, 1)) }
        let gate: [Float16] = [0.25, 0.5, 1.0, 2.0].map(Float16.init)

        let ctx = try MetalContext()
        let e = try Elementwise(context: ctx)
        guard let vb = Fp16Buffer.make(ctx.device, halves: value),
            let gb = Fp16Buffer.make(ctx.device, halves: gate),
            let ob = Fp16Buffer.make(ctx.device, count: dim * streams)
        else {
            Issue.record("alloc")
            return
        }
        let cb = try #require(ctx.queue.makeCommandBuffer())
        try e.encodePLEBroadcastScale(
            commandBuffer: cb, value: vb, gate: gb,
            out: ob, dim: dim, streams: streams)
        cb.commit()
        cb.waitUntilCompleted()

        let got = Fp16Buffer.read(ob, count: dim * streams)
        for s in 0..<streams {
            for d in 0..<dim {
                let want = Float(value[d]) * Float(gate[s])
                #expect(abs(got[s * dim + d] - want) < 5e-3, "s=\(s) d=\(d)")
            }
        }
    }

    @Test("Dilated conv: tap k reads (K-1-k)*dilation positions back")
    func dilatedConvTapOrder() throws {
        // The tap order is the part worth pinning: reversing it still yields
        // smooth plausible output. One channel, an impulse weight on a single
        // tap, and a ramp input make the read position directly observable.
        let C = 1
        let T = 4
        let K = 4
        let dil = 3
        let history = (K - 1) * dil  // 9
        let rows = history + T
        // xpad row r holds value r, so an output identifies which row it read.
        let xpad = (0..<rows * C).map { Float16($0) }

        let ctx = try MetalContext()
        let e = try Elementwise(context: ctx)
        for tap in 0..<K {
            // The conv taps are a BF16 checkpoint tensor, so the test has to
            // hand the kernel BF16 -- feeding it FP16 would compare the
            // kernel against a weight it will never see.
            var w = [Float](repeating: 0, count: C * K)
            w[tap] = 1  // isolate one tap
            let wBits = w.map { Quantization.bf16Bits($0) }
            guard let xb = Fp16Buffer.make(ctx.device, halves: xpad),
                let wb = ctx.device.makeBuffer(
                    length: wBits.count * 2,
                    options: .storageModeShared),
                let ob = Fp16Buffer.make(ctx.device, count: T * C)
            else {
                Issue.record("alloc")
                return
            }
            try wBits.withUnsafeBufferPointer { buffer in
                wb.contents().copyMemory(
                    from: try #require(buffer.baseAddress),
                    byteCount: wBits.count * 2)
            }
            let cb = try #require(ctx.queue.makeCommandBuffer())
            try e.encodePLEDilatedConv(
                commandBuffer: cb, xpad: xb, weight: wb,
                out: ob, channels: C, tokens: T,
                kernelSize: K, dilation: dil)
            cb.commit()
            cb.waitUntilCompleted()
            let got = Fp16Buffer.read(ob, count: T * C)
            for t in 0..<T {
                // Kernel reads xpad[t + tap*dilation]; in chunk coordinates
                // that is (K-1-tap)*dilation back from this token's own row.
                let readRow = Float(t + tap * dil)
                let want = readRow / (1 + expf(-readRow))  // silu
                #expect(
                    abs(got[t] - want) < max(0.05, want * 0.02),
                    "tap \(tap) t \(t): read row should be \(readRow)")
            }
        }
    }

    @Test("Production geometry: K=4, dilation=ngram_size=3, history=9")
    func productionGeometry() {
        let cfg = ArchConfig.qwen38FlashNext
        #expect(cfg.ple.convKernelSize == 4)
        // The dilation IS the n-gram size, not an independent constant.
        #expect(cfg.ple.ngramSize == 3)
        let history = (cfg.ple.convKernelSize - 1) * cfg.ple.ngramSize
        #expect(history == 9)
    }
}
