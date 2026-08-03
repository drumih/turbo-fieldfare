import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct Gemma4VisionKernelTests {
    @Test func denseBF16ProjectionMatchesCPU() throws {
        guard let context = try? MetalContext() else { return }
        let kernels = try Gemma4VisionKernels(context: context)
        let rows = 2, inputColumns = 64, outputColumns = 32
        let inputValues: [Float16] = (0..<(rows * inputColumns)).map {
            Float16(Float(($0 * 7) % 19 - 9) / 16)
        }
        let weightValues: [Float] = (0..<(outputColumns * inputColumns)).map {
            Float(($0 * 11) % 23 - 11) / 64
        }
        let weightBits = weightValues.map(Quantization.bf16Bits)
        let input = try #require(buffer(context.device, inputValues))
        let weights = try #require(buffer(context.device, weightBits))
        let converted = try #require(context.device.makeBuffer(
            length: weightBits.count * 2,
            options: .storageModePrivate))
        let output = try #require(context.device.makeBuffer(
            length: rows * outputColumns * 2,
            options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        kernels.encodeDenseBF16(
            commandBuffer: command,
            input: input,
            weight: weights,
            weightOffset: 0,
            convertedWeight: converted,
            output: output,
            rows: rows,
            outputColumns: outputColumns,
            inputColumns: inputColumns)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)

        let actual = output.contents().bindMemory(
            to: Float16.self,
            capacity: rows * outputColumns)
        for row in 0..<rows {
            for column in 0..<outputColumns {
                var expected: Float = 0
                for inner in 0..<inputColumns {
                    expected.addProduct(
                        Float(inputValues[row * inputColumns + inner]),
                        Quantization.bf16ToFloat(weightBits[column * inputColumns + inner]))
                }
                #expect(abs(Float(actual[row * outputColumns + column]) - expected) < 0.04)
            }
        }
    }

    @Test func axialRoPEUsesRotateHalfPairing() throws {
        guard let context = try? MetalContext() else { return }
        let kernels = try Gemma4VisionKernels(context: context)
        var values = [Float16](repeating: 0, count: 2 * 72)
        values[72] = 1
        values[72 + 18] = 2
        values[72 + 36] = 3
        values[72 + 54] = 4
        let query = try #require(buffer(context.device, values))
        let key = try #require(buffer(context.device, values))
        let command = try #require(context.queue.makeCommandBuffer())
        kernels.encodeRoPE(commandBuffer: command,
                           query: query,
                           key: key,
                           tokens: 2,
                           patchColumns: 2,
                           heads: 1,
                           headSize: 72)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)
        let result = query.contents().bindMemory(to: Float16.self, capacity: values.count)
        let cosine = cos(Float(1)), sine = sin(Float(1))
        #expect(abs(Float(result[72]) - (cosine - 2 * sine)) < 0.01)
        #expect(abs(Float(result[72 + 18]) - (2 * cosine + sine)) < 0.01)
        // token 1 has y == 0, so the y partition stays unchanged.
        #expect(result[72 + 36] == 3)
        #expect(result[72 + 54] == 4)
    }

    @Test func attentionIsNoncausalAndDoesNotAllocateScores() throws {
        guard let context = try? MetalContext() else { return }
        let kernels = try Gemma4VisionKernels(context: context)
        let zeros = [Float16](repeating: 0, count: 8)
        let query = try #require(buffer(context.device, zeros))
        let key = try #require(buffer(context.device, zeros))
        let value = try #require(buffer(context.device, [
            Float16(1), 2, 3, 4,
            3, 4, 5, 6,
        ]))
        let output = try #require(context.device.makeBuffer(
            length: 8 * 2,
            options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        kernels.encodeAttention(commandBuffer: command,
                                query: query,
                                key: key,
                                value: value,
                                output: output,
                                tokens: 2,
                                heads: 1,
                                headSize: 4)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)
        let result = output.contents().bindMemory(to: Float16.self, capacity: 8)
        for row in 0..<2 {
            for column in 0..<4 {
                #expect(abs(Float(result[row * 4 + column]) - Float(column + 2)) < 0.01)
            }
        }
    }

    private func buffer<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer? {
        values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base,
                                     length: bytes.count,
                                     options: .storageModeShared)
        }
    }
}
