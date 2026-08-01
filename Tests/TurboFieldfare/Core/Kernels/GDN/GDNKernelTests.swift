import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

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

    private static func makeBF16Buffer(_ device: MTLDevice,
                                       values: [Float]) -> MTLBuffer? {
        let bits = values.map { bf16($0) }
        return device.makeBuffer(bytes: bits,
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
    private struct Reference {
        let cfg: LinearAttentionConfig
        let convW: [Float]          // [C, K], bf16-representable
        let aLog: [Float]           // [Hv]
        let dtBias: [Float]         // [Hv]
        let normW: [Float]          // [Dv]

        var tail: [[Float]]         // K-1 rows of C
        var state: [Float]          // [Hv, Dv, Dk]

        init(cfg: LinearAttentionConfig, convW: [Float], aLog: [Float],
             dtBias: [Float], normW: [Float]) {
            self.cfg = cfg
            self.convW = convW
            self.aLog = aLog
            self.dtBias = dtBias
            self.normW = normW
            self.tail = Array(repeating: [Float](repeating: 0, count: cfg.qkvDim),
                              count: cfg.convKernelSize - 1)
            self.state = [Float](repeating: 0,
                                 count: cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim)
        }

        mutating func step(qkvRaw: [Float], a: [Float], b: [Float],
                           z: [Float]) -> [Float] {
            let C = cfg.qkvDim
            let K = cfg.convKernelSize
            let Hk = cfg.numKHeads
            let Hv = cfg.numVHeads
            let Dk = cfg.keyHeadDim
            let Dv = cfg.valueHeadDim

            // Conv + SiLU (fp32; row order [tail..., current]).
            var conv = [Float](repeating: 0, count: C)
            for ch in 0..<C {
                var acc = qkvRaw[ch] * convW[ch * K + (K - 1)]
                for j in 0..<(K - 1) {
                    acc += tail[j][ch] * convW[ch * K + j]
                }
                conv[ch] = Float(Float16(silu(acc)))
            }
            tail.removeFirst()
            tail.append(qkvRaw.map { Float(Float16($0)) })

            // q/k norm with folded scales (per key head).
            var normed = conv
            for headIndex in 0..<(2 * Hk) {
                let isQ = headIndex < Hk
                let head = isQ ? headIndex : headIndex - Hk
                let base = (isQ ? 0 : Hk * Dk) + head * Dk
                var sumsq: Float = 0
                for i in 0..<Dk { sumsq += conv[base + i] * conv[base + i] }
                let invRms = 1 / sqrtf(sumsq / Float(Dk) + 1e-6)
                let scale = isQ ? (1 / Float(Dk)) : (1 / sqrtf(Float(Dk)))
                for i in 0..<Dk {
                    normed[base + i] = Float(Float16(conv[base + i] * invRms * scale))
                }
            }

            // Delta recurrence per value head.
            var y = [Float](repeating: 0, count: Hv * Dv)
            for h in 0..<Hv {
                let hk = h / (Hv / Hk)
                let qBase = hk * Dk
                let kBase = Hk * Dk + hk * Dk
                let vBase = 2 * Hk * Dk + h * Dv
                let g = expf(-expf(aLog[h]) * GDNKernelTests.softplus(a[h] + dtBias[h]))
                let beta = 1 / (1 + expf(-b[h]))
                for dv in 0..<Dv {
                    let srow = (h * Dv + dv) * Dk
                    var kv: Float = 0
                    for i in 0..<Dk {
                        state[srow + i] *= g
                        kv += state[srow + i] * normed[kBase + i]
                    }
                    let delta = (normed[vBase + dv] - kv) * beta
                    var out: Float = 0
                    for i in 0..<Dk {
                        state[srow + i] += normed[kBase + i] * delta
                        out += state[srow + i] * normed[qBase + i]
                    }
                    y[h * Dv + dv] = Float(Float16(out))
                }
            }

            // Gated output norm.
            var gated = [Float](repeating: 0, count: Hv * Dv)
            for h in 0..<Hv {
                let base = h * Dv
                var sumsq: Float = 0
                for i in 0..<Dv { sumsq += y[base + i] * y[base + i] }
                let invRms = 1 / sqrtf(sumsq / Float(Dv) + 1e-6)
                for i in 0..<Dv {
                    let normedY = y[base + i] * invRms * normW[i]
                    gated[base + i] = normedY * GDNKernelTests.silu(z[base + i])
                }
            }
            return gated
        }
    }

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
    private static func gpuDecodeStep(ctx: MetalContext, gdn: GDN,
                                      fixture: Fixture, row: Int,
                                      tail: MTLBuffer, state: MTLBuffer,
                                      convW: MTLBuffer, aLog: MTLBuffer,
                                      dtBias: MTLBuffer, normW: MTLBuffer,
                                      convOut: MTLBuffer, yBuf: MTLBuffer,
                                      outBuf: MTLBuffer) throws -> [Float] {
        let cfg = Self.cfg
        guard let qkv = Fp16Buffer.make(ctx.device, halves: fixture.qkvRows[row].map { Float16($0) }),
              let aProj = Fp16Buffer.make(ctx.device, halves: fixture.aRows[row].map { Float16($0) }),
              let bProj = Fp16Buffer.make(ctx.device, halves: fixture.bRows[row].map { Float16($0) }),
              let zBuf = Fp16Buffer.make(ctx.device, halves: fixture.zRows[row].map { Float16($0) })
        else {
            throw MetalError.noDevice
        }
        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        gdn.encodeConvDecode(commandBuffer: cb, tail: tail, qkv: qkv,
                             convWeight: convW, convWeightOffset: 0,
                             out: convOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut)
        gdn.encodeDeltaStepDecode(commandBuffer: cb, convOut: convOut,
                                  aProj: aProj, bProj: bProj,
                                  aLog: aLog, aLogOffset: 0,
                                  dtBias: dtBias, dtBiasOffset: 0,
                                  state: state, y: yBuf)
        gdn.encodeGatedNorm(commandBuffer: cb, y: yBuf, z: zBuf,
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

        var reference = Reference(cfg: cfg, convW: fixture.convW,
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
              let outBuf = Fp16Buffer.make(ctx.device, count: cfg.valueDim) else {
            Issue.record("Failed to allocate buffers"); return
        }
        memset(tail.contents(), 0, tailBytes)
        memset(state.contents(), 0, stateCount * 4)

        for row in 0..<rows {
            let got = try Self.gpuDecodeStep(
                ctx: ctx, gdn: gdn, fixture: fixture, row: row,
                tail: tail, state: state, convW: convW, aLog: aLog,
                dtBias: dtBias, normW: normW, convOut: convOut,
                yBuf: yBuf, outBuf: outBuf)
            let want = reference.step(qkvRaw: fixture.qkvRows[row],
                                      a: fixture.aRows[row],
                                      b: fixture.bRows[row],
                                      z: fixture.zRows[row])
            for i in 0..<cfg.valueDim {
                let tolerance = max(2e-2, abs(want[i]) * 4e-2)
                #expect(abs(got[i] - want[i]) <= tolerance,
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
              let outA = Fp16Buffer.make(ctx.device, count: cfg.valueDim) else {
            Issue.record("Failed to allocate buffers"); return
        }
        memset(tailA.contents(), 0, tailBytes)
        memset(stateA.contents(), 0, stateCount * 4)

        var decodeOutputs: [[Float]] = []
        for row in 0..<rows {
            decodeOutputs.append(try Self.gpuDecodeStep(
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
              let outB = Fp16Buffer.make(ctx.device, count: rows * cfg.valueDim) else {
            Issue.record("Failed to allocate buffers"); return
        }
        memset(tailB.contents(), 0, tailBytes)
        memset(stateB.contents(), 0, stateCount * 4)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("no command buffer"); return
        }
        gdn.encodeConvPrefill(commandBuffer: cb, tail: tailB, qkvRows: qkvRows,
                              convWeight: convW, convWeightOffset: 0,
                              out: convOutB, rows: rows)
        gdn.encodeConvTailUpdate(commandBuffer: cb, tail: tailB,
                                 qkvRows: qkvRows, rows: rows)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: convOutB, rows: rows)
        gdn.encodeDeltaStepPrefill(commandBuffer: cb, convOut: convOutB,
                                   aProj: aRows, bProj: bRows,
                                   aLog: aLog, aLogOffset: 0,
                                   dtBias: dtBias, dtBiasOffset: 0,
                                   state: stateB, y: yB, rows: rows)
        gdn.encodeGatedNorm(commandBuffer: cb, y: yB, z: zRows,
                            weight: normW, weightOffset: 0,
                            out: outB, rows: rows)
        cb.commit()
        cb.waitUntilCompleted()

        let prefillOut = Self.readHalves(outB, count: rows * cfg.valueDim)
        for row in 0..<rows {
            for i in 0..<cfg.valueDim {
                let got = prefillOut[row * cfg.valueDim + i]
                let want = decodeOutputs[row][i]
                let tolerance = max(2e-2, abs(want) * 4e-2)
                #expect(abs(got - want) <= tolerance,
                        "row \(row) element \(i): prefill \(got), decode \(want)")
            }
        }

        // Conv tails and states must agree between the two paths.
        let tailAH = Self.readHalves(tailA, count: (cfg.convKernelSize - 1) * cfg.qkvDim)
        let tailBH = Self.readHalves(tailB, count: (cfg.convKernelSize - 1) * cfg.qkvDim)
        for i in 0..<tailAH.count {
            #expect(abs(tailAH[i] - tailBH[i]) <= 1e-3,
                    "conv tail mismatch at \(i)")
        }
        let stateAPtr = stateA.contents().bindMemory(to: Float.self, capacity: stateCount)
        let stateBPtr = stateB.contents().bindMemory(to: Float.self, capacity: stateCount)
        var maxErr: Float = 0
        for i in 0..<stateCount {
            maxErr = max(maxErr, abs(stateAPtr[i] - stateBPtr[i]))
        }
        #expect(maxErr <= 5e-2, "state divergence \(maxErr)")
    }

    // MARK: - Fused input projection

    /// One INT4 projection packed as a single resident-style buffer holding
    /// `[pad | weights | scales | biases]`, mirroring the repacker's layout
    /// (2-byte but not 4-byte aligned sub-tensor offsets).
    private struct PackedProjection {
        let view: TensorView

        init(device: MTLDevice, rows: Int, n: Int, weightPad: Int,
             rng: inout SplitMix64) {
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
            bytes.replaceSubrange(weightsOffset..<(weightsOffset + weights.count),
                                  with: weights)
            scales.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(scaleOffset..<(scaleOffset + raw.count), with: raw)
            }
            biases.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(biasOffset..<(biasOffset + raw.count), with: raw)
            }
            let buffer = device.makeBuffer(bytes: bytes, length: total,
                                           options: .storageModeShared)!
            self.view = TensorView(buffer: buffer,
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
        let gdn = try GDN(context: ctx, config: cfg,
                          specializedHiddenSize: specialize ? hiddenSize : nil)
        let gemv = try DequantInt4GEMV(context: ctx)

        let qkv = PackedProjection(device: ctx.device, rows: cfg.qkvDim,
                                   n: hiddenSize, weightPad: weightPad, rng: &rng)
        let z = PackedProjection(device: ctx.device, rows: cfg.valueDim,
                                 n: hiddenSize, weightPad: weightPad, rng: &rng)
        let a = PackedProjection(device: ctx.device, rows: cfg.numVHeads,
                                 n: hiddenSize, weightPad: weightPad, rng: &rng)
        let b = PackedProjection(device: ctx.device, rows: cfg.numVHeads,
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
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }

        for (proj, out, rows) in [(qkv, qkvRef, cfg.qkvDim), (z, zRef, cfg.valueDim),
                                  (a, aRef, cfg.numVHeads), (b, bRef, cfg.numVHeads)] {
            let v = proj.view
            gemv.encode(commandBuffer: cb,
                        weights: v.buffer, weightsOffset: Int(v.offset),
                        scales: v.buffer, scalesOffset: Int(v.scaleOffset),
                        biases: v.buffer, biasesOffset: Int(v.biasOffset),
                        x: xBuf, y: out,
                        m: UInt32(rows), n: UInt32(hiddenSize))
        }
        gdn.encodeInputProjections(commandBuffer: cb, x: xBuf,
                                   qkv: qkv.view, qkvOut: qkvGot,
                                   z: z.view, zOut: zGot,
                                   a: a.view, aOut: aGot,
                                   b: b.view, bOut: bGot,
                                   hiddenSize: hiddenSize)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            Issue.record("Command buffer failed: \(error)"); return
        }

        func rawBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
            Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                count: count * MemoryLayout<Float16>.size))
        }
        #expect(rawBytes(qkvRef, count: cfg.qkvDim) == rawBytes(qkvGot, count: cfg.qkvDim),
                "qkv projection differs")
        #expect(rawBytes(zRef, count: cfg.valueDim) == rawBytes(zGot, count: cfg.valueDim),
                "z projection differs")
        #expect(rawBytes(aRef, count: cfg.numVHeads) == rawBytes(aGot, count: cfg.numVHeads),
                "a projection differs")
        #expect(rawBytes(bRef, count: cfg.numVHeads) == rawBytes(bGot, count: cfg.numVHeads),
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
        let ragged = LinearAttentionConfig(numKHeads: 1, numVHeads: 2,
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
            guard let tail = ctx.device.makeBuffer(length: tailBytes,
                                                   options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            memset(tail.contents(), 0, tailBytes)
            for chunk in chunkRows {
                let flat = chunk.flatMap { fixture.qkvRows[$0].map { Float16($0) } }
                guard let rowsBuf = Fp16Buffer.make(ctx.device, halves: flat),
                      let cb = ctx.queue.makeCommandBuffer() else {
                    throw MetalError.noDevice
                }
                gdn.encodeConvTailUpdate(commandBuffer: cb, tail: tail,
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
