import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

private let mppTensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

@Suite struct MPPPrefillInt4QMMTests {
    private struct Inputs {
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
        let x: [Float16]
    }

    private static func makeInputs(m: Int,
                                   n: Int,
                                   k: Int,
                                   adversarialAffine: Bool = false) -> Inputs {
        let groups = k / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: n * k / 2)
        for index in packed.indices {
            packed[index] = UInt8(truncatingIfNeeded: index &* 37 &+ 0x29)
        }
        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        let adversarialScales: [UInt16] = [
            Quantization.bf16Bits(0.001),
            Quantization.bf16Bits(-0.0015),
            0x0001,
            0x8001,
        ]
        let adversarialBiases: [UInt16] = [
            Quantization.bf16Bits(-0.01),
            Quantization.bf16Bits(0.006),
            0x0001,
            0x8001,
        ]
        for row in 0..<n {
            for group in 0..<groups {
                let index = row * groups + group
                if adversarialAffine {
                    scales[index] = adversarialScales[(row + group) % adversarialScales.count]
                    biases[index] = adversarialBiases[(row * 3 + group) % adversarialBiases.count]
                } else {
                    scales[index] = Quantization.bf16Bits(
                        0.001 + Float((row + group) % 5) * 0.00025)
                    biases[index] = Quantization.bf16Bits(
                        -0.01 + Float((row * 3 + group) % 7) * 0.002)
                }
            }
        }
        var x = [Float16](repeating: 0, count: m * k)
        for index in x.indices {
            x[index] = Float16(Float((index * 11) % 29 - 14) / 64.0)
        }
        return Inputs(packed: packed, scales: scales, biases: biases, x: x)
    }

    private static func makeBuffer<T>(device: MTLDevice,
                                      values: [T],
                                      prefixBytes: Int = 0) -> MTLBuffer? {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: prefixBytes + byteCount,
                                             options: .storageModeShared) else {
            return nil
        }
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            buffer.contents().advanced(by: prefixBytes)
                .copyMemory(from: baseAddress, byteCount: byteCount)
        }
        return buffer
    }

    private static func cpuReference(_ inputs: Inputs,
                                     m: Int,
                                     n: Int,
                                     k: Int) -> [Float] {
        let groups = k / Quantization.groupSize
        let rowBytes = k / 2
        var output = [Float](repeating: 0, count: m * n)
        for token in 0..<m {
            for row in 0..<n {
                var accumulator: Float = 0
                for group in 0..<groups {
                    let scale = Quantization.bf16ToFloat(inputs.scales[row * groups + group])
                    let bias = Quantization.bf16ToFloat(inputs.biases[row * groups + group])
                    for localK in 0..<Quantization.groupSize {
                        let column = group * Quantization.groupSize + localK
                        let byte = inputs.packed[row * rowBytes + column / 2]
                        let q = column.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                        let weight = Float(q) * scale + bias
                        accumulator.addProduct(weight, Float(inputs.x[token * k + column]))
                    }
                }
                output[token * n + row] = Float(Float16(accumulator))
            }
        }
        return output
    }

    @discardableResult
    private static func runShape(context: MetalContext,
                                 candidate: MPPPrefillInt4QMM,
                                 baseline: PrefillInt4QMM,
                                 m: Int,
                                 n: Int,
                                 k: Int,
                                 adversarialAffine: Bool = false,
                                 weightOffset: Int = 0,
                                 scaleOffset: Int = 0,
                                 biasOffset: Int = 0,
                                 compareCPUReference: Bool = false,
                                 expectedPath: MPPPrefillInt4QMM.Path = .affineThreadgroupF16) throws
        -> MPPPrefillInt4QMM.PathMetadata {
        let inputs = makeInputs(m: m, n: n, k: k,
                                adversarialAffine: adversarialAffine)
        guard let weights = makeBuffer(device: context.device,
                                       values: inputs.packed,
                                       prefixBytes: weightOffset),
              let scaleBuffer = makeBuffer(device: context.device,
                                           values: inputs.scales,
                                           prefixBytes: scaleOffset),
              let biasBuffer = makeBuffer(device: context.device,
                                          values: inputs.biases,
                                          prefixBytes: biasOffset),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let expectedBuffer = Fp16Buffer.make(context.device, count: m * n),
              let actualBuffer = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }

        baseline.encode(commandBuffer: commandBuffer,
                        weights: weights,
                        weightsOffset: weightOffset,
                        scales: scaleBuffer,
                        scalesOffset: scaleOffset,
                        biases: biasBuffer,
                        biasesOffset: biasOffset,
                        x: input,
                        y: expectedBuffer,
                        t: m,
                        n: n,
                        k: k)
        let metadata = candidate.encode(commandBuffer: commandBuffer,
                                        weights: weights,
                                        weightsOffset: weightOffset,
                                        scales: scaleBuffer,
                                        scalesOffset: scaleOffset,
                                        biases: biasBuffer,
                                        biasesOffset: biasOffset,
                                        x: input,
                                        y: actualBuffer,
                                        m: m,
                                        n: n,
                                        k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(metadata.path == expectedPath)
        #expect(metadata.modelDerivedCPUHeapScratchBytes == 0)
        #expect(metadata.expandedWeightBufferBytes == 0)
        #expect(metadata.threadgroupBytes == (expectedPath == .affineThreadgroupF16Apple10V1
            ? 16_384 : 4096))
        #expect(metadata.cooperativeValueBytesEstimate ==
            (expectedPath == .affineThreadgroupF16Apple10V1 ? 32_768 : 16_384))
        #expect(metadata.simultaneousWeightTiles == 1)

        let baselineOutput = Fp16Buffer.read(expectedBuffer, count: m * n)
        let actual = Fp16Buffer.read(actualBuffer, count: m * n)
        let baselineMaxAbs = RelError.maxAbsDiff(actual, baselineOutput)
        let baselineRelative = RelError.compute(actual: actual, reference: baselineOutput)
        let byteExact = actual == baselineOutput
        #expect(baselineMaxAbs <= 0.03,
                "shape M=\(m) N=\(n) K=\(k) maxAbs=\(baselineMaxAbs) rel=\(baselineRelative) byteExact=\(byteExact)")
        #expect(baselineRelative <= 1e-3 || baselineMaxAbs <= 0.01,
                "shape M=\(m) N=\(n) K=\(k) maxAbs=\(baselineMaxAbs) rel=\(baselineRelative) byteExact=\(byteExact)")

        if compareCPUReference {
            let reference = cpuReference(inputs, m: m, n: n, k: k)
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let relative = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 0.03,
                    "CPU reference M=\(m) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(relative)")
            #expect(relative <= 1e-3,
                    "CPU reference M=\(m) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(relative)")
        }
        return metadata
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func affineThreadgroupCandidateMatchesFP32AffineReference() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        let baseline = try PrefillInt4QMM(context: context)
        #expect(candidate.isAvailable, "\(candidate.unavailableReason ?? "unknown error")")

        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 64, compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 128, adversarialAffine: true,
                          compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 17, n: 35, k: 64, compareCPUReference: true)
        let metadata = try Self.runShape(
            context: context, candidate: candidate, baseline: baseline,
            m: 17, n: 35, k: 128, adversarialAffine: true,
            weightOffset: 13, scaleOffset: 2, biasOffset: 6,
            compareCPUReference: true)
        #expect(metadata.weightBufferOffsetRemainder128 == 13)
    }

    @Test func apple10GeometryPreservesAffineReferenceAndFallsBackForShortM() throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let candidate = MPPPrefillInt4QMM(context: context, variant: .apple10V1)
        let baseline = try PrefillInt4QMM(context: context)
        #expect(candidate.isAvailable, "\(candidate.unavailableReason ?? "unknown error")")

        let selected = try Self.runShape(
            context: context, candidate: candidate, baseline: baseline,
            m: 64, n: 64, k: 128, adversarialAffine: true,
            compareCPUReference: true,
            expectedPath: .affineThreadgroupF16Apple10V1)
        #expect(selected.tileM == 64)
        #expect(selected.tileN == 64)
        #expect(selected.tileK == 128)
        #expect(selected.fallbackCount == 0)

        let fallback = try Self.runShape(
            context: context, candidate: candidate, baseline: baseline,
            m: 17, n: 35, k: 128, adversarialAffine: true,
            compareCPUReference: true)
        #expect(fallback.fallbackCount == 1)
    }

    @Test func apple10BF16ProjectorPreservesBF16AffineSemantics() throws {
        let context = try MetalContext()
        guard context.device.supportsFamily(.apple10) else { return }
        let m = 64, n = 64, k = 128
        let inputs = Self.makeInputs(m: m, n: n, k: k, adversarialAffine: true)
        let activations = inputs.x.map { Quantization.bf16Bits(Float($0)) }
        guard let weights = Self.makeBuffer(device: context.device, values: inputs.packed),
              let scales = Self.makeBuffer(device: context.device, values: inputs.scales),
              let biases = Self.makeBuffer(device: context.device, values: inputs.biases),
              let x = Self.makeBuffer(device: context.device, values: activations),
              let y = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let candidate = MPPPrefillInt4QMM(context: context, variant: .apple10BF16)
        #expect(candidate.isAvailable, "\(candidate.unavailableReason ?? "unknown error")")
        // A short output must still project. The BF16 variant is the vision
        // projector and has no control kernel, so falling back here meant not
        // projecting at all — a 3x3 patch grid pools to a single row and failed
        // outright. The Apple10 kernel bound-checks against M, so a partial tile
        // is correct.
        let shortOutput = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales, biases: biases,
            x: x, y: y, m: 17, n: 35, k: k)
        #expect(shortOutput.path == .affineThreadgroupBF16Apple10)
        #expect(shortOutput.fallbackCount == 0)
        let singleRow = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales, biases: biases,
            x: x, y: y, m: 1, n: 35, k: k)
        #expect(singleRow.path == .affineThreadgroupBF16Apple10)
        let metadata = candidate.encode(
            commandBuffer: commandBuffer,
            weights: weights, scales: scales, biases: biases,
            x: x, y: y, m: m, n: n, k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(metadata.path == .affineThreadgroupBF16Apple10)

        var reference = [Float](repeating: 0, count: m * n)
        let groups = k / Quantization.groupSize
        for token in 0..<m {
            for row in 0..<n {
                var accumulator: Float = 0
                for column in 0..<k {
                    let byte = inputs.packed[row * k / 2 + column / 2]
                    let q = column.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                    let group = row * groups + column / Quantization.groupSize
                    let weight = Quantization.bf16ToFloat(Quantization.bf16Bits(
                        Float(q) * Quantization.bf16ToFloat(inputs.scales[group])
                            + Quantization.bf16ToFloat(inputs.biases[group])))
                    accumulator.addProduct(
                        weight, Quantization.bf16ToFloat(activations[token * k + column]))
                }
                reference[token * n + row] = Float(Float16(accumulator))
            }
        }
        let actual = Fp16Buffer.read(y, count: m * n)
        #expect(RelError.maxAbsDiff(actual, reference) <= 0.03)
        #expect(RelError.compute(actual: actual, reference: reference) <= 1e-3)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func selectedProductionAttentionShapesMatchCurrentPolicy() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        let baseline = try PrefillInt4QMM(context: context)
        let shapes = [
            (name: "swa-q", n: 4096, k: 2816),
            (name: "swa-kv", n: 2048, k: 2816),
            (name: "swa-o", n: 2816, k: 4096),
            (name: "full-q", n: 8192, k: 2816),
            (name: "full-kv", n: 1024, k: 2816),
            (name: "full-o", n: 2816, k: 8192),
        ]
        for m in [32, 128] {
            for shape in shapes {
                let metadata = try Self.runShape(
                    context: context, candidate: candidate, baseline: baseline,
                    m: m, n: shape.n, k: shape.k)
                #expect(metadata.path == .affineThreadgroupF16,
                        "\(shape.name) M=\(m) unexpectedly fell back")
            }
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func fullProductionShapeIsByteStableAcross32Dispatches() throws {
        let m = 32
        let n = 2816
        let k = 8192
        let outputElements = m * n
        let outputBytes = outputElements * MemoryLayout<Float16>.stride
        let inputs = Self.makeInputs(m: m, n: n, k: k)
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        guard let weights = Self.makeBuffer(device: context.device, values: inputs.packed),
              let scales = Self.makeBuffer(device: context.device, values: inputs.scales),
              let biases = Self.makeBuffer(device: context.device, values: inputs.biases),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let outputs = context.device.makeBuffer(length: outputBytes * 32,
                                                      options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        for run in 0..<32 {
            let metadata = candidate.encode(commandBuffer: commandBuffer,
                                            weights: weights,
                                            scales: scales,
                                            biases: biases,
                                            x: input,
                                            y: outputs,
                                            yOffset: run * outputBytes,
                                            m: m,
                                            n: n,
                                            k: k)
            #expect(metadata.path == .affineThreadgroupF16)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let reference = outputs.contents().assumingMemoryBound(to: UInt16.self)
        for run in 1..<32 {
            let candidateOutput = outputs.contents()
                .advanced(by: run * outputBytes)
                .assumingMemoryBound(to: UInt16.self)
            var mismatch: Int?
            for index in 0..<outputElements where reference[index] != candidateOutput[index] {
                mismatch = index
                break
            }
            #expect(mismatch == nil, "dispatch \(run) first mismatch=\(mismatch ?? -1)")
        }
    }

    @Test func unsupportedOrUnalignedInputsReportFallback() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        guard let buffer = context.device.makeBuffer(length: 4096,
                                                     options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let unsupportedShape = candidate.encode(commandBuffer: commandBuffer,
                                                weights: buffer,
                                                scales: buffer,
                                                biases: buffer,
                                                x: buffer,
                                                y: buffer,
                                                m: 1,
                                                n: 1,
                                                k: 65)
        let unalignedScale = candidate.encode(commandBuffer: commandBuffer,
                                              weights: buffer,
                                              weightsOffset: 1,
                                              scales: buffer,
                                              scalesOffset: 1,
                                              biases: buffer,
                                              x: buffer,
                                              y: buffer,
                                              m: 1,
                                              n: 1,
                                              k: 64)
        #expect(unsupportedShape.path == .fallback)
        #expect(unsupportedShape.fallbackCount == 1)
        #expect(unalignedScale.path == .fallback)
        #expect(unalignedScale.fallbackCount == 1)
    }

    /// When a variant cannot come up it must name the function it actually
    /// needed. The BF16 variant — the vision projector — used to report the f16
    /// symbol as missing: a symbol that is present, on a device that may simply
    /// be too old. The failure held whichever way it went, so it sent whoever
    /// read it looking in the wrong place.
    ///
    /// Holds on both device classes: on an Apple10 GPU the variant comes up and
    /// there is no message to get wrong.
    @Test func anUnavailableVariantNamesItsOwnFunction() throws {
        let context = try MetalContext()
        let apple10 = context.device.supportsFamily(.apple10)

        for (variant, symbol) in [
            (MPPPrefillInt4QMM.Variant.apple10BF16,
             "mpp_prefill_affine_threadgroup_bf16_apple10_v1"),
            (MPPPrefillInt4QMM.Variant.apple10V1,
             "mpp_prefill_affine_threadgroup_f16_apple10_v1"),
        ] {
            let candidate = MPPPrefillInt4QMM(context: context, variant: variant)
            guard let reason = candidate.unavailableReason else {
                #expect(candidate.isAvailable,
                        "\(variant.rawValue) reported no reason and is not available")
                continue
            }
            #expect(reason.contains(symbol),
                    "\(variant.rawValue) failed without naming \(symbol): \(reason)")
            if variant == .apple10BF16 {
                #expect(!reason.contains("mpp_prefill_affine_threadgroup_f16 "),
                        "the BF16 variant blamed the f16 symbol: \(reason)")
            }
            if !apple10 {
                #expect(reason.contains("Apple10"),
                        "an old device was not named as the cause: \(reason)")
            }
        }
    }
}
