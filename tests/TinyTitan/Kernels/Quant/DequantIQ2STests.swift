import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

@Suite struct DequantIQ2STests {
    private static func blocks(count: Int, seed: UInt64) -> [UInt8] {
        var rng = SeedTree(seed).key("iq2s-blocks")
        var bytes: [UInt8] = []
        bytes.reserveCapacity(count * DequantIQ2S.blockBytes)
        for _ in 0..<count {
            let scale = Float16(rng.uniform(0.001, 0.02))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
            for _ in 0..<80 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
        }
        return bytes
    }

    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantIQ2S(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("IQ2_S test buffers")
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

    @Test func decodesIQ2SBlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 16
        let bytes = Self.blocks(count: blockCount, seed: 0x2C)
        let reference = try GGUFDequant.decode(type: .iq2S, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 256, iterations: 1)
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

    @Test func measuresIQ2SDecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = Self.blocks(count: count / 256, seed: 0x5A)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print(
            "IQ2_S GPU dequant: \(String(format: "%.2f", rate)) G elements/s "
                + "(\(count) elements x \(iterations) in \(String(format: "%.3f", seconds)) s)")
        #expect(rate > 0)
    }
}
