import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Checks the GPU IQ4_NL decoder against the CPU one that is bit-exact against
/// ggml, then measures how fast the GPU decodes a real expert matrix.
///
/// The correctness assertion is what matters; the rate is printed because it is
/// the number the as-is GGUF path has to beat (the CPU decoder manages
/// 0.21-0.36 G elements/s).
@Suite struct DequantIQ4NLTests {
    private static func blocks(count: Int, seed: UInt64) -> [UInt8] {
        var rng = SeedTree(seed).key("iq4nl-blocks")
        var bytes: [UInt8] = []
        bytes.reserveCapacity(count * DequantIQ4NL.blockBytes)
        for _ in 0..<count {
            let scale = Float16(rng.uniform(0.001, 0.02))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
            for _ in 0..<16 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
        }
        return bytes
    }

    private static func decodeOnGPU(
        _ context: MetalContext, blocks: [UInt8], count: Int
    ) throws -> [UInt16] {
        let kernel = try DequantIQ4NL(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            throw MetalError.commandEncoderFailed
        }
        try kernel.encode(
            commandBuffer: commandBuffer, blocks: blockBuffer, count: count, out: outBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.error == nil else {
            throw MetalError.invalidState("\(String(describing: commandBuffer.error))")
        }
        let pointer = outBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { pointer[$0] }
    }

    @Test func decodesIQ4NLBlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 64
        let bytes = Self.blocks(count: blockCount, seed: 0x4E4C)
        let reference = try GGUFDequant.decode(type: .iq4NL, bytes: bytes)
        let gpu = try Self.decodeOnGPU(context, blocks: bytes, count: blockCount * 32)

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

    /// One real down-expert matrix: 640 x 2560 = 1,638,400 values in 51,200
    /// IQ4_NL blocks (921,600 bytes). 20 decodes, timed after one warm-up.
    @Test func measuresIQ4NLDecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = Self.blocks(count: count / 32, seed: 0x5153)
        _ = try Self.decodeOnGPU(context, blocks: bytes, count: count)  // warm-up

        let kernel = try DequantIQ4NL(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: bytes, length: bytes.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.commandEncoderFailed
        }
        let iterations = 20
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try kernel.encode(
                commandBuffer: commandBuffer, blocks: blockBuffer, count: count, out: outBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let rate = Double(count * iterations) / elapsed / 1e9
        print(
            "IQ4_NL GPU dequant: \(String(format: "%.2f", rate)) G elements/s "
                + "(\(count) elements x \(iterations) in \(String(format: "%.3f", elapsed)) s)")
        #expect(rate > 0)
    }
}
