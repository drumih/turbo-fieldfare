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
                                 compareCPUReference: Bool = false) throws
        -> MPPPrefillInt4QMM.Path {
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
        let path = candidate.encode(commandBuffer: commandBuffer,
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
        #expect(path == .affineThreadgroupF16)

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
        return path
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func affineThreadgroupCandidateMatchesFP32AffineReference() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        let baseline = try PrefillInt4QMM(context: context)

        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 64, compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 128, adversarialAffine: true,
                          compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 17, n: 35, k: 64, compareCPUReference: true)
        try Self.runShape(
            context: context, candidate: candidate, baseline: baseline,
            m: 17, n: 35, k: 128, adversarialAffine: true,
            weightOffset: 13, scaleOffset: 2, biasOffset: 6,
            compareCPUReference: true)
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
                let path = try Self.runShape(
                    context: context, candidate: candidate, baseline: baseline,
                    m: m, n: shape.n, k: shape.k)
                #expect(path == .affineThreadgroupF16,
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
            let path = candidate.encode(commandBuffer: commandBuffer,
                                        weights: weights,
                                        scales: scales,
                                        biases: biases,
                                        x: input,
                                        y: outputs,
                                        yOffset: run * outputBytes,
                                        m: m,
                                        n: n,
                                        k: k)
            #expect(path == .affineThreadgroupF16)
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
        #expect(unsupportedShape == .unavailable)
        #expect(unalignedScale == .unavailable)
    }
}
