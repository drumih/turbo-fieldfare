import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Validates the gated-DeltaNet kernels against a straight-line CPU reference
/// implementing the mlx-vlm gated_delta math: causal depthwise conv + SiLU,
/// per-head no-weight q/k RMS norm with folded delta-rule scales, the gated
/// delta recurrence, and the gated output norm. Also checks that a prefill
/// chunk of T rows matches T sequential decode steps through the same GPU
/// kernels (state carry, conv tail carry).
@Suite struct GDNKernelTests {

    // Small but structurally faithful shape: C = 2*Hk*Dk + Hv*Dv = 256.
    private static let cfg = LinearAttentionConfig(
        numKHeads: 2, numVHeads: 4, keyHeadDim: 32, valueHeadDim: 32,
        convKernelSize: 4)

    // MARK: - Helpers

    private static func bf16(_ x: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: x.bitPattern >> 16)
    }

    private static func bf16Value(_ x: Float) -> Float {
        Float(bitPattern: UInt32(bf16(x)) << 16)
    }

    private static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

    private static func softplus(_ x: Float) -> Float {
        x > 20 ? x : logf(1 + expf(x))
    }

    private static func makeBF16Buffer(
        _ device: MTLDevice,
        values: [Float]
    ) -> MTLBuffer? {
        let bits = values.map { bf16($0) }
        return device.makeBuffer(
            bytes: bits,
            length: bits.count * 2,
            options: .storageModeShared)
    }

    private static func readHalves(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    /// CPU model of one layer's GDN chain over `rows` sequential tokens.
    /// Inputs are the raw projection outputs per row. Returns per-row gated
    /// outputs plus the final conv tail and delta state.
    private typealias Reference = GDNReference

    private struct Fixture {
        let convW: [Float]
        let aLog: [Float]
        let dtBias: [Float]
        let normW: [Float]
        let qkvRows: [[Float]]
        let aRows: [[Float]]
        let bRows: [[Float]]
        let zRows: [[Float]]

        init(rows: Int, seed: UInt64) {
            let cfg = GDNKernelTests.cfg
            var rng = SeedTree(seed).key("gdn-fixture-\(rows)")
            self.convW = (0..<(cfg.qkvDim * cfg.convKernelSize)).map { _ in
                GDNKernelTests.bf16Value(rng.uniform(-0.4, 0.4))
            }
            self.aLog = (0..<cfg.numVHeads).map { _ in
                GDNKernelTests.bf16Value(rng.uniform(-1.0, 1.5))
            }
            self.dtBias = (0..<cfg.numVHeads).map { _ in
                GDNKernelTests.bf16Value(rng.uniform(-0.5, 0.5))
            }
            self.normW = (0..<cfg.valueHeadDim).map { _ in
                GDNKernelTests.bf16Value(rng.uniform(0.5, 1.5))
            }
            self.qkvRows = (0..<rows).map { _ in
                (0..<cfg.qkvDim).map { _ in Float(Float16(rng.uniform(-1.0, 1.0))) }
            }
            self.aRows = (0..<rows).map { _ in
                (0..<cfg.numVHeads).map { _ in Float(Float16(rng.uniform(-1.0, 1.0))) }
            }
            self.bRows = (0..<rows).map { _ in
                (0..<cfg.numVHeads).map { _ in Float(Float16(rng.uniform(-1.0, 1.0))) }
            }
            self.zRows = (0..<rows).map { _ in
                (0..<cfg.valueHeadDim * cfg.numVHeads).map { _ in
                    Float(Float16(rng.uniform(-1.0, 1.0)))
                }
            }
        }
    }

    /// Drives the GPU decode chain (conv → qk norm → delta → gated norm) for
    /// one token and returns the gated output.
    private static func gpuDecodeStep(
        ctx: MetalContext, gdn: GDN,
        fixture: Fixture, row: Int,
        tail: MTLBuffer, state: MTLBuffer,
        convW: MTLBuffer, aLog: MTLBuffer,
        dtBias: MTLBuffer, normW: MTLBuffer,
        convOut: MTLBuffer, yBuf: MTLBuffer,
        outBuf: MTLBuffer,
        tailOffset: Int = 0,
        stateOffset: Int = 0
    ) throws -> [Float] {
        let cfg = Self.cfg
        guard
            let qkv = Fp16Buffer.make(ctx.device, halves: fixture.qkvRows[row].map { Float16($0) }),
            let aProj = Fp16Buffer.make(ctx.device, halves: fixture.aRows[row].map { Float16($0) }),
            let bProj = Fp16Buffer.make(ctx.device, halves: fixture.bRows[row].map { Float16($0) }),
            let zBuf = Fp16Buffer.make(ctx.device, halves: fixture.zRows[row].map { Float16($0) })
        else {
            throw MetalError.noDevice
        }
        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        try gdn.encodeConvDecode(
            commandBuffer: cb, tail: tail, tailOffset: tailOffset,
            qkv: qkv,
            convWeight: convW, convWeightOffset: 0,
            out: convOut)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut)
        try gdn.encodeDeltaStepDecode(
            commandBuffer: cb, convOut: convOut,
            aProj: aProj, bProj: bProj,
            aLog: aLog, aLogOffset: 0,
            dtBias: dtBias, dtBiasOffset: 0,
            state: state, stateOffset: stateOffset, y: yBuf)
        try gdn.encodeGatedNorm(
            commandBuffer: cb, y: yBuf, z: zBuf,
            weight: normW, weightOffset: 0, out: outBuf)
        cb.commit()
        cb.waitUntilCompleted()
        return readHalves(outBuf, count: cfg.valueDim)
    }

    @Test func decodeChainMatchesReference() throws {
        let cfg = Self.cfg
        let rows = 6
        let fixture = Fixture(rows: rows, seed: 0x51D)
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)

        var reference = Reference(
            cfg: cfg, convW: fixture.convW,
            aLog: fixture.aLog, dtBias: fixture.dtBias,
            normW: fixture.normW)

        let tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * 2
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim
        guard let tail = ctx.device.makeBuffer(length: tailBytes, options: .storageModeShared),
            let state = ctx.device.makeBuffer(length: stateCount * 4, options: .storageModeShared),
            let convW = Self.makeBF16Buffer(ctx.device, values: fixture.convW),
            let aLog = Self.makeBF16Buffer(ctx.device, values: fixture.aLog),
            let dtBias = Self.makeBF16Buffer(ctx.device, values: fixture.dtBias),
            let normW = Self.makeBF16Buffer(ctx.device, values: fixture.normW),
            let convOut = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let yBuf = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let outBuf = Fp16Buffer.make(ctx.device, count: cfg.valueDim)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }
        memset(tail.contents(), 0, tailBytes)
        memset(state.contents(), 0, stateCount * 4)

        for row in 0..<rows {
            let got = try Self.gpuDecodeStep(
                ctx: ctx, gdn: gdn, fixture: fixture, row: row,
                tail: tail, state: state, convW: convW, aLog: aLog,
                dtBias: dtBias, normW: normW, convOut: convOut,
                yBuf: yBuf, outBuf: outBuf)
            let want = reference.step(
                qkvRaw: fixture.qkvRows[row],
                a: fixture.aRows[row],
                b: fixture.bRows[row],
                z: fixture.zRows[row])
            for i in 0..<cfg.valueDim {
                let tolerance = max(2e-2, abs(want[i]) * 4e-2)
                #expect(
                    abs(got[i] - want[i]) <= tolerance,
                    "row \(row) element \(i): got \(got[i]), want \(want[i])")
            }
        }

        // Final recurrent state should match too (fp32, looser accumulation).
        let statePtr = state.contents().bindMemory(to: Float.self, capacity: stateCount)
        var maxStateErr: Float = 0
        for i in 0..<stateCount {
            maxStateErr = max(maxStateErr, abs(statePtr[i] - reference.state[i]))
        }
        #expect(maxStateErr <= 5e-2, "state divergence \(maxStateErr)")
    }

    /// A slot's recurrent state and conv tail live at a byte offset in a shared
    /// buffer. The decode kernels must bind that offset: if they wrote at offset
    /// 0 instead, two batched sequences would share one recurrence and both
    /// outputs would be silently wrong rather than failing.
    @Test func decodeStepHonoursSlotOffsets() throws {
        let cfg = Self.cfg
        let fixture = Fixture(rows: 1, seed: 0x5107)
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)

        let tailSlot = (cfg.convKernelSize - 1) * cfg.qkvDim * 2
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim
        let stateSlot = stateCount * 4
        guard
            let twoTail = ctx.device.makeBuffer(
                length: 2 * tailSlot,
                options: .storageModeShared),
            let twoState = ctx.device.makeBuffer(
                length: 2 * stateSlot,
                options: .storageModeShared),
            let oneTail = ctx.device.makeBuffer(
                length: tailSlot,
                options: .storageModeShared),
            let oneState = ctx.device.makeBuffer(
                length: stateSlot,
                options: .storageModeShared),
            let convW = Self.makeBF16Buffer(ctx.device, values: fixture.convW),
            let aLog = Self.makeBF16Buffer(ctx.device, values: fixture.aLog),
            let dtBias = Self.makeBF16Buffer(ctx.device, values: fixture.dtBias),
            let normW = Self.makeBF16Buffer(ctx.device, values: fixture.normW),
            let convOut = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let yBuf = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let outBuf = Fp16Buffer.make(ctx.device, count: cfg.valueDim)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }
        memset(twoTail.contents(), 0, 2 * tailSlot)
        memset(twoState.contents(), 0, 2 * stateSlot)
        memset(oneTail.contents(), 0, tailSlot)
        memset(oneState.contents(), 0, stateSlot)

        // Same step twice: once in a lone slot at offset 0, once in slot 1 of a
        // two-slot buffer. Both start from zero, so the results must be equal.
        let reference = try Self.gpuDecodeStep(
            ctx: ctx, gdn: gdn, fixture: fixture, row: 0,
            tail: oneTail, state: oneState, convW: convW, aLog: aLog,
            dtBias: dtBias, normW: normW, convOut: convOut,
            yBuf: yBuf, outBuf: outBuf)
        let slotted = try Self.gpuDecodeStep(
            ctx: ctx, gdn: gdn, fixture: fixture, row: 0,
            tail: twoTail, state: twoState, convW: convW, aLog: aLog,
            dtBias: dtBias, normW: normW, convOut: convOut,
            yBuf: yBuf, outBuf: outBuf,
            tailOffset: tailSlot, stateOffset: stateSlot)

        for i in 0..<cfg.valueDim {
            #expect(
                abs(slotted[i] - reference[i]) <= 1e-6,
                "slot 1 output \(i): \(slotted[i]) vs \(reference[i])")
        }
        let twoStatePtr = twoState.contents().bindMemory(
            to: Float.self,
            capacity: 2 * stateCount)
        let oneStatePtr = oneState.contents().bindMemory(
            to: Float.self,
            capacity: stateCount)
        for i in 0..<stateCount {
            #expect(
                twoStatePtr[stateCount + i] == oneStatePtr[i],
                "slot 1 state \(i) differs from the lone-slot state")
            #expect(
                twoStatePtr[i] == 0,
                "slot 0 state \(i) was written by slot 1's step")
        }
    }

    @Test func prefillChunkMatchesSequentialDecode() throws {
        let cfg = Self.cfg
        let rows = 7
        let fixture = Fixture(rows: rows, seed: 0xBEEF)
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)

        let tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * 2
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim

        // --- Sequential decode path.
        guard let tailA = ctx.device.makeBuffer(length: tailBytes, options: .storageModeShared),
            let stateA = ctx.device.makeBuffer(length: stateCount * 4, options: .storageModeShared),
            let convW = Self.makeBF16Buffer(ctx.device, values: fixture.convW),
            let aLog = Self.makeBF16Buffer(ctx.device, values: fixture.aLog),
            let dtBias = Self.makeBF16Buffer(ctx.device, values: fixture.dtBias),
            let normW = Self.makeBF16Buffer(ctx.device, values: fixture.normW),
            let convOutA = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let yA = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let outA = Fp16Buffer.make(ctx.device, count: cfg.valueDim)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }
        memset(tailA.contents(), 0, tailBytes)
        memset(stateA.contents(), 0, stateCount * 4)

        var decodeOutputs: [[Float]] = []
        for row in 0..<rows {
            decodeOutputs.append(
                try Self.gpuDecodeStep(
                    ctx: ctx, gdn: gdn, fixture: fixture, row: row,
                    tail: tailA, state: stateA, convW: convW, aLog: aLog,
                    dtBias: dtBias, normW: normW, convOut: convOutA,
                    yBuf: yA, outBuf: outA))
        }

        // --- Prefill path over the same rows in one chunk.
        let qkvFlat = fixture.qkvRows.flatMap { $0.map { Float16($0) } }
        let aFlat = fixture.aRows.flatMap { $0.map { Float16($0) } }
        let bFlat = fixture.bRows.flatMap { $0.map { Float16($0) } }
        let zFlat = fixture.zRows.flatMap { $0.map { Float16($0) } }
        guard let tailB = ctx.device.makeBuffer(length: tailBytes, options: .storageModeShared),
            let stateB = ctx.device.makeBuffer(length: stateCount * 4, options: .storageModeShared),
            let qkvRows = Fp16Buffer.make(ctx.device, halves: qkvFlat),
            let aRows = Fp16Buffer.make(ctx.device, halves: aFlat),
            let bRows = Fp16Buffer.make(ctx.device, halves: bFlat),
            let zRows = Fp16Buffer.make(ctx.device, halves: zFlat),
            let convOutB = Fp16Buffer.make(ctx.device, count: rows * cfg.qkvDim),
            let yB = Fp16Buffer.make(ctx.device, count: rows * cfg.valueDim),
            let outB = Fp16Buffer.make(ctx.device, count: rows * cfg.valueDim)
        else {
            Issue.record("Failed to allocate buffers")
            return
        }
        memset(tailB.contents(), 0, tailBytes)
        memset(stateB.contents(), 0, stateCount * 4)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("no command buffer")
            return
        }
        try gdn.encodeConvPrefill(
            commandBuffer: cb, tail: tailB, qkvRows: qkvRows,
            convWeight: convW, convWeightOffset: 0,
            out: convOutB, rows: rows)
        try gdn.encodeConvTailUpdate(
            commandBuffer: cb, tail: tailB,
            qkvRows: qkvRows, rows: rows)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOutB, rows: rows)
        try gdn.encodeDeltaStepPrefill(
            commandBuffer: cb, convOut: convOutB,
            aProj: aRows, bProj: bRows,
            aLog: aLog, aLogOffset: 0,
            dtBias: dtBias, dtBiasOffset: 0,
            state: stateB, y: yB, rows: rows)
        try gdn.encodeGatedNorm(
            commandBuffer: cb, y: yB, z: zRows,
            weight: normW, weightOffset: 0,
            out: outB, rows: rows)
        cb.commit()
        cb.waitUntilCompleted()

        let prefillOut = Self.readHalves(outB, count: rows * cfg.valueDim)
        for row in 0..<rows {
            for i in 0..<cfg.valueDim {
                let got = prefillOut[row * cfg.valueDim + i]
                let want = decodeOutputs[row][i]
                // Bit-identical, not merely close: the wide verify
                // (docs/plan-wide-verify.md) rolls GDN back to an accepted row
                // by re-stepping this prefill path, and its rows must equal
                // scalar decode's. Measured exact on 2026-11-12.
                #expect(
                    got == want,
                    "row \(row) element \(i): prefill \(got), decode \(want)")
            }
        }

        // Conv tails and states must agree between the two paths.
        let tailAH = Self.readHalves(tailA, count: (cfg.convKernelSize - 1) * cfg.qkvDim)
        let tailBH = Self.readHalves(tailB, count: (cfg.convKernelSize - 1) * cfg.qkvDim)
        for i in 0..<tailAH.count {
            #expect(
                tailAH[i] == tailBH[i],
                "conv tail mismatch at \(i)")
        }
        let stateAPtr = stateA.contents().bindMemory(to: Float.self, capacity: stateCount)
        let stateBPtr = stateB.contents().bindMemory(to: Float.self, capacity: stateCount)
        var maxErr: Float = 0
        for i in 0..<stateCount {
            maxErr = max(maxErr, abs(stateAPtr[i] - stateBPtr[i]))
        }
        #expect(maxErr == 0, "state divergence \(maxErr)")
    }

    @Test func speculativeCheckpointMatchesFirstConfirmedRow() throws {
        let cfg = Self.cfg
        let fixture = Fixture(rows: 2, seed: 0xC11E)
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)
        let tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * 2
        let stateCount = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim
        let qkvFlat = fixture.qkvRows.flatMap { $0.map { Float16($0) } }
        let aFlat = fixture.aRows.flatMap { $0.map { Float16($0) } }
        let bFlat = fixture.bRows.flatMap { $0.map { Float16($0) } }

        guard
            let tail = ctx.device.makeBuffer(
                length: tailBytes,
                options: .storageModeShared),
            let state = ctx.device.makeBuffer(
                length: stateCount * 4,
                options: .storageModeShared),
            let checkpointTail = ctx.device.makeBuffer(
                length: tailBytes,
                options: .storageModeShared),
            let checkpointState = ctx.device.makeBuffer(
                length: stateCount * 4,
                options: .storageModeShared),
            let referenceTail = ctx.device.makeBuffer(
                length: tailBytes,
                options: .storageModeShared),
            let referenceState = ctx.device.makeBuffer(
                length: stateCount * 4,
                options: .storageModeShared),
            let qkvRows = Fp16Buffer.make(ctx.device, halves: qkvFlat),
            let aRows = Fp16Buffer.make(ctx.device, halves: aFlat),
            let bRows = Fp16Buffer.make(ctx.device, halves: bFlat),
            let convW = Self.makeBF16Buffer(ctx.device, values: fixture.convW),
            let aLog = Self.makeBF16Buffer(ctx.device, values: fixture.aLog),
            let dtBias = Self.makeBF16Buffer(ctx.device, values: fixture.dtBias),
            let normW = Self.makeBF16Buffer(ctx.device, values: fixture.normW),
            let convOut = Fp16Buffer.make(ctx.device, count: 2 * cfg.qkvDim),
            let y = Fp16Buffer.make(ctx.device, count: 2 * cfg.valueDim),
            let referenceConvOut = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let referenceY = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let referenceOut = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let cb = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("Failed to allocate speculative checkpoint fixture")
            return
        }
        for buffer in [
            tail, state, checkpointTail, checkpointState,
            referenceTail, referenceState,
        ] {
            memset(buffer.contents(), 0, buffer.length)
        }

        try gdn.encodeConvPrefill(
            commandBuffer: cb, tail: tail, qkvRows: qkvRows,
            convWeight: convW, convWeightOffset: 0,
            out: convOut, rows: 2)
        try gdn.encodeConvTailCheckpoint(
            commandBuffer: cb, tail: tail,
            qkvRows: qkvRows, checkpoint: checkpointTail)
        try gdn.encodeConvTailUpdate(
            commandBuffer: cb, tail: tail,
            qkvRows: qkvRows, rows: 2)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut, rows: 2)
        try gdn.encodeDeltaStepPrefill(
            commandBuffer: cb, convOut: convOut,
            aProj: aRows, bProj: bRows,
            aLog: aLog, aLogOffset: 0,
            dtBias: dtBias, dtBiasOffset: 0,
            state: state, checkpointState: checkpointState,
            y: y, rows: 2)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            Issue.record("Speculative checkpoint command failed: \(error)")
            return
        }

        _ = try Self.gpuDecodeStep(
            ctx: ctx, gdn: gdn, fixture: fixture, row: 0,
            tail: referenceTail, state: referenceState, convW: convW,
            aLog: aLog, dtBias: dtBias, normW: normW,
            convOut: referenceConvOut, yBuf: referenceY, outBuf: referenceOut)

        let gotTail = Self.readHalves(
            checkpointTail,
            count: tailBytes / MemoryLayout<Float16>.stride)
        let expectedTail = Self.readHalves(
            referenceTail,
            count: tailBytes / MemoryLayout<Float16>.stride)
        #expect(gotTail == expectedTail)

        let gotState = checkpointState.contents()
            .bindMemory(to: Float.self, capacity: stateCount)
        let expectedState = referenceState.contents()
            .bindMemory(to: Float.self, capacity: stateCount)
        var maxStateError: Float = 0
        for index in 0..<stateCount {
            maxStateError = max(
                maxStateError,
                abs(gotState[index] - expectedState[index]))
        }
        #expect(
            maxStateError <= 1e-6,
            "checkpoint state divergence \(maxStateError)")
    }

    // MARK: - Fused input projection

    /// One INT4 projection packed as a single resident-style buffer holding
    /// `[pad | weights | scales | biases]`, mirroring the repacker's layout
    /// (2-byte but not 4-byte aligned sub-tensor offsets).
    private struct PackedProjection {
        let view: TensorView

        init(
            device: MTLDevice, rows: Int, n: Int, weightPad: Int,
            rng: inout SplitMix64
        ) throws {
            let packedPerRow = n / 2
            let groups = n / Quantization.groupSize
            var weights = [UInt8](repeating: 0, count: rows * packedPerRow)
            var scales = [UInt16](repeating: 0, count: rows * groups)
            var biases = [UInt16](repeating: 0, count: rows * groups)
            for row in 0..<rows {
                let values = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
                let q = Quantization.quantizeInt4Affine(values)
                for i in 0..<packedPerRow {
                    weights[row * packedPerRow + i] = q.packed[i]
                }
                for i in 0..<groups {
                    scales[row * groups + i] = q.scales[i]
                    biases[row * groups + i] = q.biases[i]
                }
            }
            let weightsOffset = weightPad
            let scaleOffset = weightsOffset + weights.count
            let biasOffset = scaleOffset + scales.count * 2
            let total = biasOffset + biases.count * 2
            var bytes = [UInt8](repeating: 0, count: total)
            bytes.replaceSubrange(
                weightsOffset..<(weightsOffset + weights.count),
                with: weights)
            scales.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(scaleOffset..<(scaleOffset + raw.count), with: raw)
            }
            biases.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(biasOffset..<(biasOffset + raw.count), with: raw)
            }
            let buffer = try #require(
                device.makeBuffer(
                    bytes: bytes, length: total,
                    options: .storageModeShared))
            self.view = TensorView(
                buffer: buffer,
                offset: UInt64(weightsOffset),
                length: UInt64(weights.count),
                scaleOffset: UInt64(scaleOffset),
                scaleLength: UInt64(scales.count * 2),
                biasOffset: UInt64(biasOffset),
                biasLength: UInt64(biases.count * 2),
                shape: (UInt32(rows), UInt32(n), 1, 1),
                dtype: 0)
        }
    }

    /// The fused four-way input projection must be bit-identical to the four
    /// separate INT4 GEMVs it replaces — greedy decode output depends on it.
    private static func expectFusedInProjMatchesSeparateGEMVs(
        cfg: LinearAttentionConfig,
        hiddenSize: Int,
        weightPad: Int,
        specialize: Bool,
        seed: UInt64
    ) throws {
        var rng = SplitMix64(seed: seed)
        let ctx = try MetalContext()
        let gdn = try GDN(
            context: ctx, config: cfg,
            specializedHiddenSize: specialize ? hiddenSize : nil)
        let gemv = try DequantInt4GEMV(context: ctx)

        let qkv = try PackedProjection(
            device: ctx.device, rows: cfg.qkvDim,
            n: hiddenSize, weightPad: weightPad, rng: &rng)
        let z = try PackedProjection(
            device: ctx.device, rows: cfg.valueDim,
            n: hiddenSize, weightPad: weightPad, rng: &rng)
        let a = try PackedProjection(
            device: ctx.device, rows: cfg.numVHeads,
            n: hiddenSize, weightPad: weightPad, rng: &rng)
        let b = try PackedProjection(
            device: ctx.device, rows: cfg.numVHeads,
            n: hiddenSize, weightPad: weightPad, rng: &rng)
        let x = (0..<hiddenSize).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
            let qkvRef = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let zRef = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let aRef = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
            let bRef = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
            let qkvGot = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
            let zGot = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
            let aGot = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
            let bGot = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
            let cb = ctx.queue.makeCommandBuffer()
        else {
            Issue.record("Failed to allocate buffers")
            return
        }

        for (proj, out, rows) in [
            (qkv, qkvRef, cfg.qkvDim), (z, zRef, cfg.valueDim),
            (a, aRef, cfg.numVHeads), (b, bRef, cfg.numVHeads),
        ] {
            let v = proj.view
            try gemv.encode(
                commandBuffer: cb,
                weights: v.buffer, weightsOffset: Int(v.offset),
                scales: v.buffer, scalesOffset: Int(v.scaleOffset),
                biases: v.buffer, biasesOffset: Int(v.biasOffset),
                x: xBuf, y: out,
                m: UInt32(rows), n: UInt32(hiddenSize))
        }
        try gdn.encodeInputProjections(
            commandBuffer: cb, x: xBuf,
            qkv: qkv.view, qkvOut: qkvGot,
            z: z.view, zOut: zGot,
            a: a.view, aOut: aGot,
            b: b.view, bOut: bGot,
            hiddenSize: hiddenSize)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            Issue.record("Command buffer failed: \(error)")
            return
        }

        func rawBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
            Array(
                UnsafeBufferPointer(
                    start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                    count: count * MemoryLayout<Float16>.size))
        }
        #expect(
            rawBytes(qkvRef, count: cfg.qkvDim) == rawBytes(qkvGot, count: cfg.qkvDim),
            "qkv projection differs")
        #expect(
            rawBytes(zRef, count: cfg.valueDim) == rawBytes(zGot, count: cfg.valueDim),
            "z projection differs")
        #expect(
            rawBytes(aRef, count: cfg.numVHeads) == rawBytes(aGot, count: cfg.numVHeads),
            "a projection differs")
        #expect(
            rawBytes(bRef, count: cfg.numVHeads) == rawBytes(bGot, count: cfg.numVHeads),
            "b projection differs")
    }

    @Test func fusedInputProjectionMatchesSeparateGEMVs() throws {
        try Self.expectFusedInProjMatchesSeparateGEMVs(
            cfg: Self.cfg, hiddenSize: 128, weightPad: 0, specialize: false,
            seed: 0x1D_0001)
    }

    @Test func fusedInputProjectionMatchesSeparateGEMVs_specialized() throws {
        try Self.expectFusedInProjMatchesSeparateGEMVs(
            cfg: Self.cfg, hiddenSize: 256, weightPad: 0, specialize: true,
            seed: 0x1D_0002)
    }

    /// 2-byte-but-not-4-byte-aligned weight offsets (the repacker's guarantee)
    /// with a row total that is not a multiple of the 8 rows per threadgroup,
    /// so the trailing threadgroup runs partly out of range.
    @Test func fusedInputProjectionMatchesSeparateGEMVs_offsetAndRagged() throws {
        let ragged = LinearAttentionConfig(
            numKHeads: 1, numVHeads: 2,
            keyHeadDim: 32, valueHeadDim: 32,
            convKernelSize: 4)
        // 128 + 64 + 2 + 2 = 196 rows -> 25 threadgroups, last one 4/8 idle.
        try Self.expectFusedInProjMatchesSeparateGEMVs(
            cfg: ragged, hiddenSize: 192, weightPad: 2, specialize: true,
            seed: 0x1D_0003)
    }

    @Test func shortChunkTailCarry() throws {
        // T < convKernelSize - 1 exercises the ordered-shift path in
        // gdn_conv_tail_update: feed 2 single-row chunks then compare the tail
        // with a one-chunk run of both rows.
        let cfg = Self.cfg
        let fixture = Fixture(rows: 2, seed: 0x7A11)
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)
        let tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * 2

        func runChunks(_ chunkRows: [[Int]]) throws -> [Float] {
            guard
                let tail = ctx.device.makeBuffer(
                    length: tailBytes,
                    options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            memset(tail.contents(), 0, tailBytes)
            for chunk in chunkRows {
                let flat = chunk.flatMap { fixture.qkvRows[$0].map { Float16($0) } }
                guard let rowsBuf = Fp16Buffer.make(ctx.device, halves: flat),
                    let cb = ctx.queue.makeCommandBuffer()
                else {
                    throw MetalError.noDevice
                }
                try gdn.encodeConvTailUpdate(
                    commandBuffer: cb, tail: tail,
                    qkvRows: rowsBuf, rows: chunk.count)
                cb.commit()
                cb.waitUntilCompleted()
            }
            return Self.readHalves(tail, count: (cfg.convKernelSize - 1) * cfg.qkvDim)
        }

        let split = try runChunks([[0], [1]])
        let joined = try runChunks([[0, 1]])
        for i in 0..<split.count {
            #expect(abs(split[i] - joined[i]) <= 1e-3, "tail mismatch at \(i)")
        }
    }
}
