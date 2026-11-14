import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

/// Whether this host has an Apple10-family GPU, read once. The tile-boundary
/// TensorOps test is gated on it so a hosted runner without that family
/// *records a skip*: the kernel is never dispatched there, and a test that
/// returned early instead would report a pass for a path it never ran. Run the
/// suite on Apple10 hardware before changing the TensorOps path.
private let apple10TensorOpsAvailable: Bool = {
    guard let device = MTLCreateSystemDefaultDevice() else { return false }
    return device.supportsFamily(.apple10)
}()

@Suite struct PrefillAttentionTests {
    private typealias Fixture = PrefillAttentionRef.Inputs

    @Test func prefillAttentionMatchesCPUReferenceFullAndSWA() throws {
        let cases: [(label: String, start: Int, chunk: Int, window: Int)] = [
            ("full-origin", 0, 3, 0),
            ("full-offset", 4, 3, 0),
            ("swa-inside-window", 2, 4, 16),
            ("swa-truncated", 7, 4, 5),
        ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(
                start: c.start,
                chunk: c.chunk,
                window: c.window,
                seed: 0xA510 + UInt64(index))
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test func prefillAttentionRingCapacityMatchesLinearReference() throws {
        let fixture = Self.makeFixture(
            start: 20,
            chunk: 4,
            window: 8,
            seed: 0xA611,
            headDim: 32,
            qHeads: 4,
            kvHeads: 2)
        let ringCapacity = 16
        var kRing = [Float](repeating: 0, count: ringCapacity * fixture.kvStride)
        var vRing = [Float](repeating: 0, count: ringCapacity * fixture.kvStride)
        for p in 0..<fixture.kvValid {
            let dst = (p % ringCapacity) * fixture.kvStride
            let src = p * fixture.kvStride
            kRing.replaceSubrange(
                dst..<(dst + fixture.kvStride),
                with: fixture.k[src..<(src + fixture.kvStride)])
            vRing.replaceSubrange(
                dst..<(dst + fixture.kvStride),
                with: fixture.v[src..<(src + fixture.kvStride)])
        }

        var ringFixture = fixture
        ringFixture.k = kRing
        ringFixture.v = vRing
        let actual = try Self.runKernel(ringFixture, kvRingCapacity: UInt32(ringCapacity))
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "ring prefill maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "ring prefill rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func prefillAttentionMasksFutureChunkTokens() throws {
        var fixture = Self.makeFixture(start: 5, chunk: 4, window: 0, seed: 0xA620)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        for row in fixture.start..<fixture.kvValid {
            for kvh in 0..<fixture.kvHeads {
                for d in 0..<fixture.headDim {
                    let base = row * fixture.kvStride + kvh * fixture.headDim + d
                    fixture.k[base] = Float(row - fixture.start + 1) * 2.0 + Float(d) * 0.125
                    fixture.v[base] = Float(row - fixture.start + 1) * -2.0 - Float(d) * 0.125
                }
            }
        }

        let actual = try Self.runKernel(fixture)
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "future mask maxAbs=\(maxAbs) rel=\(rel) qPerKV=\(qPerKV)")
        #expect(rel <= 2e-2, "future mask rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func prefillAttentionProductionDimsBoundedVisibility() throws {
        let cases:
            [(
                label: String, start: Int, chunk: Int, window: Int, headDim: Int, qHeads: Int,
                kvHeads: Int
            )] = [
                ("swa-current-key-at-1023", 1023, 1, 1, 256, 16, 8),
                ("swa-current-key-at-1024", 1024, 1, 1, 256, 16, 8),
                ("swa-current-key-at-4095", 4095, 1, 1, 256, 16, 8),
                ("full-origin", 0, 1, 0, 512, 16, 2),
                ("full-short-gqa", 3, 1, 0, 512, 16, 2),
            ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(
                start: c.start,
                chunk: c.chunk,
                window: c.window,
                seed: 0xA730 + UInt64(index),
                headDim: c.headDim,
                qHeads: c.qHeads,
                kvHeads: c.kvHeads)
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test func prefillAttentionTiledProductionBoundarySmoke() throws {
        let cases:
            [(
                label: String, start: Int, chunk: Int, window: Int, headDim: Int, qHeads: Int,
                kvHeads: Int
            )] = [
                ("swa-production-window-1024", 1023, 4, 1024, 256, 16, 8),
                ("full-production-gqa", 31, 8, 0, 512, 16, 2),
            ]

        for (index, c) in cases.enumerated() {
            let fixture = Self.makeFixture(
                start: c.start,
                chunk: c.chunk,
                window: c.window,
                seed: 0xA840 + UInt64(index),
                headDim: c.headDim,
                qHeads: c.qHeads,
                kvHeads: c.kvHeads)
            try Self.runAndCompare(fixture, label: c.label)
        }
    }

    @Test(
        .enabled(if: apple10TensorOpsAvailable),
        arguments: [
            1,
            63, 64, 65,
            127, 128, 129,
            255, 256, 257,
            1_023, 1_024, 1_025,
        ])
    func tensorOps2DFullAttentionMatchesReferenceAtTileBoundaries(_ visibleKeys: Int) throws {
        let fixture = Self.makeFixture(
            start: visibleKeys - 1,
            chunk: 1,
            window: 0,
            seed: 0xA870 + UInt64(visibleKeys),
            headDim: 512,
            qHeads: 16,
            kvHeads: 2)
        let candidate = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DValidityV2)
        let repeated = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DValidityV2)
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(candidate, reference)
        let rel = RelError.compute(actual: candidate, reference: reference)
        #expect(
            candidate == repeated,
            "TensorOps 2D full attention is not byte-stable at \(visibleKeys) keys")
        #expect(
            maxAbs <= 2e-2,
            "TensorOps 2D maxAbs=\(maxAbs) rel=\(rel) keys=\(visibleKeys)")
        #expect(
            rel <= 2e-2,
            "TensorOps 2D rel=\(rel) maxAbs=\(maxAbs) keys=\(visibleKeys)")
    }

    @Test func preferredTensorOpsPathUsesSafeHardwareFallback() throws {
        let context = try MetalContext()
        let fixture = Self.makeFixture(
            start: 128,
            chunk: 1,
            window: 0,
            seed: 0xA872,
            headDim: 512,
            qHeads: 16,
            kvHeads: 2)
        let preferred = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DPreferred)
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(preferred, reference)
        let rel = RelError.compute(actual: preferred, reference: reference)
        #expect(
            maxAbs <= 2e-2,
            "preferred TensorOps maxAbs=\(maxAbs) rel=\(rel)")
        #expect(
            rel <= 2e-2,
            "preferred TensorOps rel=\(rel) maxAbs=\(maxAbs)")
        if !context.device.supportsFamily(.apple10) {
            let baseline = try Self.runKernel(fixture, path: .causalTiled)
            #expect(preferred == baseline)
        }
    }

    // MARK: - K0: capability seam

    /// The live probe answers for the host; a test must be able to ask for a
    /// machine that does not exist here. These four tests drive every policy
    /// branch from synthetic capabilities, so they pass on M1-M6 alike.
    @Test func syntheticApple10CapabilityBuildsTheTensorPipeline() throws {
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(
            context: ctx,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: { _ in
                try ctx.pipeline("attention_prefill_causal_tiled")
            })
        #expect(prefill.tensorOpsPipelineBuilt)
        #expect(prefill.tensorOpsDecision.path == .tensorOps)
        #expect(prefill.tensorOpsDecision.reason.contains("Apple10"))
    }

    @Test func offNeverBuildsTheTensorPipelineEvenOnApple10() throws {
        let ctx = try MetalContext()
        var builderCalls = 0
        let prefill = try PrefillAttention(
            context: ctx,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .off,
            tensorOpsPipelineBuilder: { _ in
                builderCalls += 1
                return try ctx.pipeline("attention_prefill_causal_tiled")
            })
        #expect(builderCalls == 0)
        #expect(!prefill.tensorOpsPipelineBuilt)
        #expect(prefill.tensorOpsDecision.path == .fallback)
        #expect(prefill.tensorOpsDecision.reason.contains("off"))
        let fixture = Self.makeFixture(
            start: 16,
            chunk: 2,
            window: 0,
            seed: 0xA881,
            headDim: 512,
            qHeads: 16,
            kvHeads: 2)
        let baseline = try Self.runKernel(fixture, path: .causalTiled)
        let off = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DPreferred,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .off)
        #expect(off == baseline)
    }

    @Test func onFallsBackWhenPipelineCreationFails() throws {
        struct InjectedFailure: Error {}
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(
            context: ctx,
            capabilities: Self.capabilities(apple10: false),
            tensorOpsMode: .on,
            tensorOpsPipelineBuilder: { _ in throw InjectedFailure() })
        #expect(!prefill.tensorOpsPipelineBuilt)
        #expect(prefill.tensorOpsDecision.path == .fallback)
        #expect(prefill.tensorOpsDecision.reason.contains("pipeline creation failed"))
    }

    /// The M3 path is the no-capability path: the tensor kernel is not built
    /// and the preferred request produces exactly the scalar tiled output.
    @Test func syntheticNonApple10KeepsTheScalarPrefillPath() throws {
        let fixture = Self.makeFixture(
            start: 32,
            chunk: 2,
            window: 0,
            seed: 0xA880,
            headDim: 512,
            qHeads: 16,
            kvHeads: 2)
        let baseline = try Self.runKernel(fixture, path: .causalTiled)
        let preferred = try Self.runKernel(
            fixture,
            path: .fullTensorOps2DPreferred,
            capabilities: Self.capabilities(apple10: false),
            tensorOpsMode: .auto)
        #expect(preferred == baseline)
    }

    private static func capabilities(apple10: Bool) -> HardwareCapabilities {
        HardwareCapabilities(
            chipModel: "MacTest,1",
            gpuName: "Synthetic GPU",
            gpuFamily: apple10 ? .apple10 : .apple9,
            supportsApple10TensorOps: apple10,
            metal4Available: true,
            anePresent: true,
            physicalMemoryBytes: 24 << 30,
            osVersion: "26.0.0")
    }

    private static func makeFixture(
        start: Int,
        chunk: Int,
        window: Int,
        seed: UInt64,
        headDim: Int = 8,
        qHeads: Int = 4,
        kvHeads: Int = 2
    ) -> Fixture {
        let qStride = qHeads * headDim + 3
        let kvStride = kvHeads * headDim + 5
        let oStride = qHeads * headDim + 7
        let kvValid = start + chunk
        var rng = SeedTree(seed).key("prefill-attn-start\(start)-chunk\(chunk)-window\(window)")
        var q = [Float](repeating: 0, count: chunk * qStride)
        var k = [Float](repeating: 0, count: kvValid * kvStride)
        var v = [Float](repeating: 0, count: kvValid * kvStride)

        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    q[t * qStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        for pos in 0..<kvValid {
            for h in 0..<kvHeads {
                for d in 0..<headDim {
                    k[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                    v[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }

        return Fixture(
            q: q, k: k, v: v,
            qStride: qStride, kvStride: kvStride, oStride: oStride,
            headDim: headDim, qHeads: qHeads, kvHeads: kvHeads,
            start: start, chunk: chunk, kvValid: kvValid,
            window: window, scale: 1.0)
    }

    private static func runAndCompare(_ fixture: Fixture, label: String) throws {
        let actual = try Self.runKernel(fixture)
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "\(label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2, "\(label) rel=\(rel) maxAbs=\(maxAbs)")
    }

    private static func runKernel(
        _ fixture: Fixture,
        kvRingCapacity: UInt32 = 0,
        path: RuntimePrefillAttentionPath = .causalTiled,
        capabilities: HardwareCapabilities? = nil,
        tensorOpsMode: TensorOpsMode? = nil,
        pipelineBuilder: ((String) throws -> MTLComputePipelineState)? = nil
    ) throws -> [Float] {
        let ctx = try MetalContext()
        let prefill = try PrefillAttention(
            context: ctx,
            capabilities: capabilities,
            tensorOpsMode: tensorOpsMode,
            tensorOpsPipelineBuilder: pipelineBuilder)
        let qPrefix = 17
        let kPrefix = 19
        let vPrefix = 23
        let oPrefix = 29
        let outCount = oPrefix + fixture.chunk * fixture.oStride

        guard
            let qBuf = Fp16Buffer.make(
                ctx.device,
                values: [Float](repeating: 0, count: qPrefix) + fixture.q),
            let kBuf = Fp16Buffer.make(
                ctx.device,
                values: [Float](repeating: 0, count: kPrefix) + fixture.k),
            let vBuf = Fp16Buffer.make(
                ctx.device,
                values: [Float](repeating: 0, count: vPrefix) + fixture.v),
            let outBuf = Fp16Buffer.make(ctx.device, count: outCount)
        else {
            Issue.record("alloc failed")
            return []
        }

        let params = PrefillAttentionParams(
            startPosition: UInt32(fixture.start),
            queryCount: UInt32(fixture.chunk),
            headDim: UInt32(fixture.headDim),
            numQHeads: UInt32(fixture.qHeads),
            numKVHeads: UInt32(fixture.kvHeads),
            kvValidCount: UInt32(fixture.kvValid),
            slidingWindow: UInt32(fixture.window),
            kvTokenStrideElements: UInt32(fixture.kvStride),
            qTokenStrideElements: UInt32(fixture.qStride),
            oTokenStrideElements: UInt32(fixture.oStride),
            scale: fixture.scale)

        let cb = try #require(ctx.queue.makeCommandBuffer())
        try prefill.encodeCausal(
            commandBuffer: cb,
            q: qBuf,
            qOffset: qPrefix * MemoryLayout<Float16>.size,
            k: kBuf,
            kOffset: kPrefix * MemoryLayout<Float16>.size,
            v: vBuf,
            vOffset: vPrefix * MemoryLayout<Float16>.size,
            out: outBuf,
            outOffset: oPrefix * MemoryLayout<Float16>.size,
            params: params,
            kvRingCapacity: kvRingCapacity,
            path: path)
        cb.commit()
        cb.waitUntilCompleted()

        let out = Fp16Buffer.read(outBuf, count: outCount)
        var compact = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        for t in 0..<fixture.chunk {
            for h in 0..<fixture.qHeads {
                for d in 0..<fixture.headDim {
                    compact[(t * fixture.qHeads + h) * fixture.headDim + d] =
                        out[oPrefix + t * fixture.oStride + h * fixture.headDim + d]
                }
            }
        }
        return compact
    }

}
