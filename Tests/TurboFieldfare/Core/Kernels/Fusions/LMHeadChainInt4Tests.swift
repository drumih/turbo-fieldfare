import Accelerate
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct LMHeadChainInt4Tests {
    private static let rmsEps: Float = 1e-6

    private static func cpuGreedy(hiddenFp16: [Float16],
                                  normWeightBF16: [UInt16],
                                  rows: [Quantization.Int4AffineRow],
                                  d: Int,
                                  v: Int) -> UInt32 {
        let hidden = hiddenFp16.map(Float.init)
        let normWeight = normWeightBF16.map(Quantization.bf16ToFloat)
        let normed = RmsNormRef.apply(x: hidden, weight: normWeight, eps: rmsEps)
            .map { Float(Float16($0)) }

        var bestValue: Float = -.infinity
        var bestIndex: UInt32 = 0
        for row in 0..<v {
            let dequantized = Quantization.dequantizeInt4Affine(rows[row], n: d)
            var dot: Float = 0
            dequantized.withUnsafeBufferPointer { weights in
                normed.withUnsafeBufferPointer { input in
                    vDSP_dotpr(weights.baseAddress!, 1,
                               input.baseAddress!, 1,
                               &dot,
                               vDSP_Length(d))
                }
            }
            if dot > bestValue {
                bestValue = dot
                bestIndex = UInt32(row)
            }
        }
        return bestIndex
    }

    private static func packRows(_ rows: [Quantization.Int4AffineRow])
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let packedRowCount = rows[0].packed.count
        let groupCount = rows[0].scales.count
        var packed = [UInt8](repeating: 0, count: rows.count * packedRowCount)
        var scales = [UInt16](repeating: 0, count: rows.count * groupCount)
        var biases = [UInt16](repeating: 0, count: rows.count * groupCount)
        for row in rows.indices {
            for index in 0..<packedRowCount {
                packed[row * packedRowCount + index] = rows[row].packed[index]
            }
            for index in 0..<groupCount {
                scales[row * groupCount + index] = rows[row].scales[index]
                biases[row * groupCount + index] = rows[row].biases[index]
            }
        }
        return (packed, scales, biases)
    }

    private static func gpuGreedy(hiddenFp16: [Float16],
                                  normBF16: [UInt16],
                                  rows: [Quantization.Int4AffineRow],
                                  d: Int,
                                  v: Int,
                                  hiddenOffset: Int = 0) throws -> UInt32 {
        let (packed, scales, biases) = Self.packRows(rows)
        let context = try MetalContext()
        let chain = try LMHeadChainInt4(context: context, maxD: d, maxVocab: v)

        guard let hidden = context.device.makeBuffer(
                  bytes: hiddenFp16,
                  length: hiddenFp16.count * MemoryLayout<Float16>.stride,
                  options: .storageModeShared),
              let norm = context.device.makeBuffer(
                  bytes: normBF16,
                  length: normBF16.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let weights = context.device.makeBuffer(
                  bytes: packed,
                  length: packed.count,
                  options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases,
                  length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let output = context.device.makeBuffer(
                  length: MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return 0
        }

        chain.encodeGreedyDecode(commandBuffer: commandBuffer,
                                 hidden: hidden,
                                 hiddenOffset: hiddenOffset,
                                 normWeight: norm,
                                 weights: weights,
                                 scales: scaleBuffer,
                                 biases: biasBuffer,
                                 outToken: output,
                                 d: UInt32(d),
                                 vocab: UInt32(v))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output.contents().load(as: UInt32.self)
    }

    @Test func rawGreedyMatchesCPUReference() throws {
        let d = 64
        let v = 1024
        var rng = SplitMix64(seed: 0x1A4E_4EAD_2026_0603)
        let hidden = (0..<d).map { _ in Float16(rng.uniform(-1, 1)) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.5, 1.5)) }
        let rows = (0..<v).map { _ in
            Quantization.quantizeInt4Affine((0..<d).map { _ in rng.uniform(-1, 1) })
        }

        let reference = Self.cpuGreedy(hiddenFp16: hidden,
                                       normWeightBF16: norm,
                                       rows: rows,
                                       d: d,
                                       v: v)
        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == reference)
    }

    @Test func allInvalidCandidatesFallBackToZero() throws {
        let d = 64
        let v = 1024
        let hidden = [Float16](repeating: .nan, count: d)
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { _ in
            Quantization.quantizeInt4Affine([Float](repeating: 0.25, count: d))
        }

        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == 0)
    }

    @Test(arguments: [1024, 2048, 4096])
    func greedyMatchesCPUReferenceAcrossVocabSizes(v: Int) throws {
        let d = 64
        let target = v - 3
        let hidden = (0..<d).map { Float16(0.25 + Float($0 % 7) * 0.01) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = row == target ? 1 : -0.5
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let reference = Self.cpuGreedy(hiddenFp16: hidden,
                                       normWeightBF16: norm,
                                       rows: rows,
                                       d: d,
                                       v: v)
        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == reference)
    }

    @Test func tiesResolveToLowestIndex() throws {
        let d = 64
        let v = 1024
        let hidden = (0..<d).map { Float16(0.2 + Float($0 % 5) * 0.02) }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = (row == 17 || row == 311) ? 1 : -0.25
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let result = try Self.gpuGreedy(hiddenFp16: hidden,
                                        normBF16: norm,
                                        rows: rows,
                                        d: d,
                                        v: v)
        #expect(result == 17)
    }

    @Test func readsSelectedHiddenRow() throws {
        let d = 64
        let v = 1024
        let rowStride = d + 16
        let selectedRow = 3
        let target = 777
        var hiddenRows = [Float16](repeating: 0, count: 5 * rowStride)
        for row in 0..<5 {
            for index in 0..<d {
                let value = row == selectedRow
                    ? Float(0.25 + Float(index % 7) * 0.01)
                    : Float(-0.4 - Float(index % 5) * 0.01)
                hiddenRows[row * rowStride + index] = Float16(value)
            }
        }
        let norm = (0..<d).map { _ in Quantization.bf16Bits(1) }
        let rows = (0..<v).map { row in
            let value: Float = row == target ? 1 : -0.25
            return Quantization.quantizeInt4Affine([Float](repeating: value, count: d))
        }

        let result = try Self.gpuGreedy(
            hiddenFp16: hiddenRows,
            normBF16: norm,
            rows: rows,
            d: d,
            v: v,
            hiddenOffset: selectedRow * rowStride * MemoryLayout<Float16>.stride)
        #expect(result == UInt32(target))
    }
}
