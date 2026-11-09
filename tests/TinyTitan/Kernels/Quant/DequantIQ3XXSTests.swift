import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// IQ3_XXS on the GPU against the CPU decoder, then a rate on a real expert
/// matrix (640 x 2560 = 1,638,400 values in 6,400 blocks of 98 bytes).
@Suite struct DequantIQ3XXSTests {
    private static func blocks(count: Int, seed: UInt64) -> [UInt8] {
        var rng = SeedTree(seed).key("iq3xxs-blocks")
        var bytes: [UInt8] = []
        bytes.reserveCapacity(count * DequantIQ3XXS.blockBytes)
        for _ in 0..<count {
            let scale = Float16(rng.uniform(0.001, 0.02))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
            for _ in 0..<96 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
        }
        return bytes
    }

    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantIQ3XXS(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("IQ3_XXS test buffers")
        }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try kernel.encode(
                commandBuffer: commandBuffer, blocks: blockBuffer, count: count, out: outBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw MetalError.invalidState("\(error)")
            }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = outBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<count).map { pointer[$0] }, seconds)
    }

    @Test func decodesIQ3XXSBlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 16
        let bytes = Self.blocks(count: blockCount, seed: 0x33)
        let reference = try GGUFDequant.decode(type: .iq3XXS, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 256, iterations: 1)
        #expect(gpu.count == reference.count)
        var mismatches = 0
        for index in 0..<min(gpu.count, reference.count)
        where gpu[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
            if mismatches <= 5 {
                #expect(
                    gpu[index] == Quantization.bf16Bits(reference[index]),
                    "element \(index): gpu \(gpu[index]) cpu \(Quantization.bf16Bits(reference[index]))"
                )
            }
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) elements differ")
    }

    @Test func measuresIQ3XXSDecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = Self.blocks(count: count / 256, seed: 0x66)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print(
            "IQ3_XXS GPU dequant: \(String(format: "%.2f", rate)) G elements/s "
                + "(\(count) elements x \(iterations) in \(String(format: "%.3f", seconds)) s)")
        #expect(rate > 0)
    }
}
