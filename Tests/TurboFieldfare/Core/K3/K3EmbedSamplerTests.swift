import Testing
import Foundation
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
import TurboFieldfareValidationSupport

/// `K3Embed` (int8 gather + lm_head GEMV) and `K3Sampler` (fp32, no softcap)
/// against CPU references. Sampler assertions mirror the house order —
/// repetition penalty → softmax → top-p → top-k → temperature draw — with
/// `K3SamplerReference` providing the bit-comparable CPU pipeline.
@Suite struct K3EmbedSamplerTests {

    // MARK: helpers

    private static func f32Buffer(_ device: MTLDevice, _ values: [Float])
        -> MTLBuffer? {
        values.withUnsafeBufferPointer {
            device.makeBuffer(bytes: $0.baseAddress!,
                              length: values.count * MemoryLayout<Float>.stride,
                              options: .storageModeShared)
        }
    }

    private static func readF32(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    /// Deterministic toy weights, house-style formula.
    private static func toyValue(_ row: Int, _ col: Int) -> Float {
        Float((row % 7) - 3) * 0.01 + Float((col % 11) - 5) * 0.002
    }

    private struct Int8Table {
        var packed: [UInt8]
        var scales: [UInt16]
        var biases: [UInt16]
        var rows: [Quantization.Int8AffineRow]
    }

    private static func makeTable(rows: Int, columns: Int) -> Int8Table {
        let quantized = (0..<rows).map { row in
            Quantization.quantizeInt8Affine(
                (0..<columns).map { toyValue(row, $0) })
        }
        var table = Int8Table(packed: [], scales: [], biases: [], rows: quantized)
        for row in quantized {
            table.packed.append(contentsOf: row.packed)
            table.scales.append(contentsOf: row.scales)
            table.biases.append(contentsOf: row.biases)
        }
        return table
    }

    // MARK: - Embedding gather

    @Test func embedGatherMatchesCPUDequant() throws {
        let ctx = try MetalContext()
        let embed = try K3Embed(context: ctx)
        let vocab = 256
        let d = 64
        let table = Self.makeTable(rows: vocab, columns: d)

        let token = 137
        guard let w = ctx.device.makeBuffer(bytes: table.packed,
                                            length: table.packed.count,
                                            options: .storageModeShared),
              let s = table.scales.withUnsafeBufferPointer({
                  ctx.device.makeBuffer(bytes: $0.baseAddress!,
                                        length: table.scales.count * 2,
                                        options: .storageModeShared)
              }),
              let b = table.biases.withUnsafeBufferPointer({
                  ctx.device.makeBuffer(bytes: $0.baseAddress!,
                                        length: table.biases.count * 2,
                                        options: .storageModeShared)
              }),
              let out = Fp16Buffer.make(ctx.device, count: d),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        embed.encodeGather(commandBuffer: cmd,
                           table: w, scales: s, biases: b, out: out,
                           tokenID: UInt32(token), d: UInt32(d))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let gpu = Fp16Buffer.readHalf(out, count: d).map { $0.bitPattern }
        let cpu = Quantization.dequantizeInt8Affine(table.rows[token], n: d)
            .map { Float16($0).bitPattern }
        #expect(gpu == cpu)
    }

    // MARK: - lm_head GEMV

    @Test func lmHeadLogitsMatchReference() throws {
        let ctx = try MetalContext()
        let head = try K3LMHeadGEMV(context: ctx)
        let m = 256
        let n = 64
        let table = Self.makeTable(rows: m, columns: n)

        var rng = SeedTree(0xE1).key("lmhead-x")
        let xHalf = (0..<n).map { _ in Float16(rng.uniform(-0.75, 0.75)) }
        let xF32 = xHalf.map { Float($0) }

        guard let w = ctx.device.makeBuffer(bytes: table.packed,
                                            length: table.packed.count,
                                            options: .storageModeShared),
              let s = table.scales.withUnsafeBufferPointer({
                  ctx.device.makeBuffer(bytes: $0.baseAddress!,
                                        length: table.scales.count * 2,
                                        options: .storageModeShared)
              }),
              let b = table.biases.withUnsafeBufferPointer({
                  ctx.device.makeBuffer(bytes: $0.baseAddress!,
                                        length: table.biases.count * 2,
                                        options: .storageModeShared)
              }),
              let x = Fp16Buffer.make(ctx.device, halves: xHalf),
              let y = ctx.device.makeBuffer(length: m * MemoryLayout<Float>.stride,
                                            options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        head.encode(commandBuffer: cmd,
                    weights: w, scales: s, biases: b, x: x, y: y,
                    m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let gpu = Self.readF32(y, count: m)
        var cpu = [Float](repeating: 0, count: m)
        for row in 0..<m {
            let w = Quantization.dequantizeInt8Affine(table.rows[row], n: n)
            var acc: Float = 0
            for col in 0..<n { acc += w[col] * xF32[col] }
            cpu[row] = acc
        }
        let rel = RelError.compute(actual: gpu, reference: cpu)
        #expect(rel < Tolerance.identity, "lm_head logits rel=\(rel)")
    }

    // MARK: - Trunk GEMV adapter

    /// Emulates the resident layout: one buffer holding
    /// [weights | scales | biases] contiguously, addressed by offsets.
    private static func makeView(device: MTLDevice,
                                 payload: [UInt8],
                                 scales: [UInt8] = [],
                                 biases: [UInt8] = [],
                                 rows: Int, columns: Int,
                                 dtype: UInt8) throws -> TensorView {
        var combined = payload
        // Scales/biases are BF16: keep 2-byte alignment.
        if combined.count % 2 != 0 { combined.append(0) }
        let scaleOffset = combined.count
        combined.append(contentsOf: scales)
        if combined.count % 2 != 0 { combined.append(0) }
        let biasOffset = combined.count
        combined.append(contentsOf: biases)
        guard let buffer = combined.withUnsafeBufferPointer({
            device.makeBuffer(bytes: $0.baseAddress!, length: combined.count,
                              options: .storageModeShared)
        }) else {
            throw ModelError.residentBufferWrapFailed
        }
        return TensorView(
            buffer: buffer,
            offset: 0, length: UInt64(payload.count),
            scaleOffset: scales.isEmpty ? 0 : UInt64(scaleOffset),
            scaleLength: UInt64(scales.count),
            biasOffset: biases.isEmpty ? 0 : UInt64(biasOffset),
            biasLength: UInt64(biases.count),
            shape: (UInt32(rows), UInt32(columns), 0, 0),
            dtype: dtype)
    }

    @Test func trunkGEMVServesAllFourFormats() throws {
        let ctx = try MetalContext()
        let trunk = try K3TrunkGEMV(context: ctx)
        var rng = SeedTree(0x77).key("trunk-gemv")

        struct Case {
            let rows: Int
            let columns: Int
        }
        for (format, dims) in [
            (K3TrunkWeightFormat.int4G64, Case(rows: 64, columns: 128)),
            (.int8G64, Case(rows: 64, columns: 128)),
            (.bf16, Case(rows: 128, columns: 7_168)),
            (.fp32, Case(rows: 896, columns: 7_168)),  // canonical router shape
        ] as [(K3TrunkWeightFormat, Case)] {
            let rows = dims.rows
            let columns = dims.columns
            let weights = (0..<rows).map { r in
                (0..<columns).map { c in Self.toyValue(r, c) + rng.uniform(-0.01, 0.01) }
            }
            let xF32 = (0..<columns).map { _ in rng.uniform(-0.5, 0.5) }
            let xHalf = xF32.map { Float16($0) }

            var payload: [UInt8] = []
            var scales: [UInt8] = []
            var biases: [UInt8] = []
            var dtype = GTurboFormatV1.DType.u32.rawValue
            var referenceRows = weights
            switch format {
            case .int4G64, .int8G64:
                func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
                    for value in values {
                        bytes.append(UInt8(truncatingIfNeeded: value))
                        bytes.append(UInt8(truncatingIfNeeded: value >> 8))
                    }
                }
                if format == .int4G64 {
                    let quantized = weights.map { Quantization.quantizeInt4Affine($0) }
                    referenceRows = quantized.map {
                        Quantization.dequantizeInt4Affine($0, n: columns)
                    }
                    for row in quantized {
                        payload.append(contentsOf: row.packed)
                        appendU16(row.scales, to: &scales)
                        appendU16(row.biases, to: &biases)
                    }
                } else {
                    let quantized = weights.map { Quantization.quantizeInt8Affine($0) }
                    referenceRows = quantized.map {
                        Quantization.dequantizeInt8Affine($0, n: columns)
                    }
                    for row in quantized {
                        payload.append(contentsOf: row.packed)
                        appendU16(row.scales, to: &scales)
                        appendU16(row.biases, to: &biases)
                    }
                }
            case .bf16:
                dtype = GTurboFormatV1.DType.bf16.rawValue
                referenceRows = weights.map { row in
                    row.map { Quantization.bf16ToFloat(Quantization.bf16Bits($0)) }
                }
                for row in referenceRows {
                    for value in row {
                        let bits = Quantization.bf16Bits(value)
                        payload.append(UInt8(truncatingIfNeeded: bits))
                        payload.append(UInt8(truncatingIfNeeded: bits >> 8))
                    }
                }
            case .fp32:
                dtype = GTurboFormatV1.DType.fp32.rawValue
                for row in weights {
                    for var value in row {
                        withUnsafeBytes(of: &value) { payload.append(contentsOf: $0) }
                    }
                }
            }

            let view = try Self.makeView(
                device: ctx.device, payload: payload,
                scales: scales, biases: biases,
                rows: rows, columns: columns, dtype: dtype)
            #expect(try K3TrunkGEMV.format(of: view) == format)

            let fp32IO = format == .fp32
            guard let x = fp32IO
                    ? Self.f32Buffer(ctx.device, xF32)
                    : Fp16Buffer.make(ctx.device, halves: xHalf),
                  let y = ctx.device.makeBuffer(
                    length: rows * (fp32IO ? 4 : 2),
                    options: .storageModeShared),
                  let cmd = ctx.queue.makeCommandBuffer() else {
                Issue.record("buffer allocation failed")
                return
            }
            try trunk.encode(commandBuffer: cmd, view: view, x: x, y: y)
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)

            var cpu = [Float](repeating: 0, count: rows)
            for row in 0..<rows {
                var acc: Float = 0
                for col in 0..<columns { acc += referenceRows[row][col] * xF32[col] }
                cpu[row] = acc
            }
            let gpu: [Float]
            if fp32IO {
                gpu = Self.readF32(y, count: rows)
            } else {
                gpu = Fp16Buffer.read(y, count: rows)
            }
            let rel = RelError.compute(actual: gpu, reference: cpu)
            let bar: Float = fp32IO ? Tolerance.identity : Tolerance.fp16Reduction
            #expect(rel < bar, "trunk \(format) rel=\(rel)")
        }
    }

    @Test func trunkGEMVRejectsInconsistentViews() throws {
        let ctx = try MetalContext()
        _ = try K3TrunkGEMV(context: ctx)
        // int4 payload but no aux ranges.
        let view = try Self.makeView(device: ctx.device,
                                     payload: [UInt8](repeating: 0, count: 64 * 64 / 2),
                                     rows: 64, columns: 64,
                                     dtype: GTurboFormatV1.DType.u32.rawValue)
        #expect {
            try K3TrunkGEMV.format(of: view)
        } throws: { error in
            if case K3TrunkGEMVError.unsupportedView = error { return true }
            return false
        }
    }

    // MARK: - Sampler

    private static let samplerVocab = 4_096

    private static func samplerLogits(seed: UInt64) -> [Float] {
        var rng = SeedTree(seed).key("sampler-logits-\(samplerVocab)")
        return (0..<samplerVocab).map { _ in rng.uniform(-8, 8) }
    }

    /// Run the GPU sampler on a FRESH logits copy (the penalty path edits in
    /// place). Returns the token, the path taken, and the post-penalty
    /// logits as left in the shared buffer.
    private static func runSampler(ctx: MetalContext,
                                   sampler: K3Sampler,
                                   logits: [Float],
                                   config: GenerationConfig,
                                   position: Int,
                                   history: [Int32] = []) throws
        -> (token: UInt32, path: SamplePath, editedLogits: [Float]) {
        guard let logitBuf = Self.f32Buffer(ctx.device, logits),
              let probs = ctx.device.makeBuffer(
                length: logits.count * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let outToken = ctx.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return (0, .greedyGPU, [])
        }
        let path = sampler.sample(commandBuffer: cmd,
                                  logits: logitBuf, probs: probs,
                                  history: history, config: config,
                                  position: position, outToken: outToken)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        let token = outToken.contents().load(as: UInt32.self)
        return (token, path, Self.readF32(logitBuf, count: logits.count))
    }

    @Test func greedyDeterministicAndMatchesArgmax() throws {
        let ctx = try MetalContext()
        let sampler = try K3Sampler(context: ctx, vocab: Self.samplerVocab)
        let logits = Self.samplerLogits(seed: 0x51)
        let config = GenerationConfig(maxNewTokens: 1, temperature: 0, seed: 7)

        let first = try Self.runSampler(ctx: ctx, sampler: sampler,
                                        logits: logits, config: config, position: 0)
        let second = try Self.runSampler(ctx: ctx, sampler: sampler,
                                         logits: logits, config: config, position: 0)
        #expect(first.path == .greedyGPU)
        #expect(first.token == second.token)
        let cpu = K3SamplerReference.sample(logits: logits, temperature: 0,
                                            seed: 0, position: 0)
        #expect(first.token == cpu)
    }

    @Test func topK1EqualsGreedy() throws {
        let ctx = try MetalContext()
        let sampler = try K3Sampler(context: ctx, vocab: Self.samplerVocab)
        let logits = Self.samplerLogits(seed: 0x52)
        let greedy = try Self.runSampler(
            ctx: ctx, sampler: sampler, logits: logits,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0), position: 0)
        let topK1 = try Self.runSampler(
            ctx: ctx, sampler: sampler, logits: logits,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0.7,
                                     topK: 1, seed: 123),
            position: 0)
        #expect(topK1.path == .gpuSampled)
        #expect(topK1.token == greedy.token)
    }

    @Test func seededTopKTopPMatchesCPUReference() throws {
        let ctx = try MetalContext()
        let sampler = try K3Sampler(context: ctx, vocab: Self.samplerVocab)
        let logits = Self.samplerLogits(seed: 0x53)
        let config = GenerationConfig(maxNewTokens: 1, temperature: 0.8,
                                      topK: 32, topP: 0.9, seed: 0xDEAD)
        let position = 3
        let gpu = try Self.runSampler(ctx: ctx, sampler: sampler,
                                      logits: logits, config: config,
                                      position: position)
        let replay = try Self.runSampler(ctx: ctx, sampler: sampler,
                                         logits: logits, config: config,
                                         position: position)
        #expect(gpu.token == replay.token)

        let seed = Sampler.seedFor(config: config, position: position)
        let cpu = K3SamplerReference.sample(
            logits: logits, temperature: config.temperature,
            topK: config.topK ?? 0, topP: config.topP ?? 1.0,
            seed: seed, position: UInt32(position))
        #expect(gpu.token == cpu)
    }

    @Test func plainTemperatureFastPathMatchesReference() throws {
        let ctx = try MetalContext()
        let sampler = try K3Sampler(context: ctx, vocab: Self.samplerVocab)
        let logits = Self.samplerLogits(seed: 0x54)
        let config = GenerationConfig(maxNewTokens: 1, temperature: 0.9,
                                      seed: 0xBEEF)
        let position = 1
        let gpu = try Self.runSampler(ctx: ctx, sampler: sampler,
                                      logits: logits, config: config,
                                      position: position)
        #expect(gpu.path == .gpuSampled)
        let seed = Sampler.seedFor(config: config, position: position)
        let cpu = K3SamplerReference.sample(
            logits: logits, temperature: config.temperature,
            seed: seed, position: UInt32(position))
        #expect(gpu.token == cpu)
    }

    @Test func repetitionPenaltyMatchesReference() throws {
        let ctx = try MetalContext()
        let sampler = try K3Sampler(context: ctx, vocab: Self.samplerVocab)
        let logits = Self.samplerLogits(seed: 0x55)
        let greedyToken = Int32(K3SamplerReference.argmax(
            K3SamplerReference.softmax(logits)))
        let history: [Int32] = [greedyToken, 5, 99, greedyToken]  // dup on purpose
        let config = GenerationConfig(maxNewTokens: 1, temperature: 0.8,
                                      topK: 16, repetitionPenalty: 2.0,
                                      seed: 0xF00D)
        let position = 2
        let gpu = try Self.runSampler(ctx: ctx, sampler: sampler,
                                      logits: logits, config: config,
                                      position: position, history: history)
        #expect(gpu.path == .hostPenalty)

        // The host pass edits the shared logits buffer exactly like the
        // reference (unique ids, HF divide/multiply rule).
        var referenceLogits = logits
        K3SamplerReference.applyRepetitionPenalty(
            logits: &referenceLogits, history: history, penalty: 2.0)
        #expect(gpu.editedLogits == referenceLogits)

        let seed = Sampler.seedFor(config: config, position: position)
        let cpu = K3SamplerReference.sample(
            logits: logits, temperature: config.temperature,
            topK: config.topK ?? 0, topP: config.topP ?? 1.0,
            seed: seed, position: UInt32(position),
            repetitionPenalty: 2.0, history: history)
        #expect(gpu.token == cpu)
    }

    /// Greedy over the FULL canonical vocab: the max sits near the top row,
    /// catching any 16-bit indexing slip the small-vocab tests would miss.
    @Test func fullVocabGreedyBoundary() throws {
        let ctx = try MetalContext()
        let vocab = 163_840
        let sampler = try K3Sampler(context: ctx, vocab: vocab)
        var logits = [Float](repeating: -100, count: vocab)
        logits[163_000] = 50
        logits[12_345] = 49
        let config = GenerationConfig(maxNewTokens: 1, temperature: 0)
        guard let logitBuf = Self.f32Buffer(ctx.device, logits),
              let probs = ctx.device.makeBuffer(
                length: vocab * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let outToken = ctx.device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        sampler.sample(commandBuffer: cmd, logits: logitBuf, probs: probs,
                       history: [], config: config, position: 0,
                       outToken: outToken)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        #expect(outToken.contents().load(as: UInt32.self) == 163_000)
    }
}
