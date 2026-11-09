import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

private func randomBlocks(
    count: Int, blockBytes: Int, key: String, seed: UInt64
) -> [UInt8] {
    var rng = SeedTree(seed).key(key)
    var bytes: [UInt8] = []
    bytes.reserveCapacity(count * blockBytes)
    for _ in 0..<count {
        let scale = Float16(rng.uniform(0.001, 0.02))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
        for _ in 0..<(blockBytes - 2) { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
    }
    return bytes
}

@Suite struct DequantIQ4XSTests {
    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantIQ4XS(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("IQ4_XS test buffers")
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
            if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = outBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<count).map { pointer[$0] }, seconds)
    }

    @Test func decodesIQ4XSBlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 16
        let bytes = randomBlocks(
            count: blockCount, blockBytes: 136, key: "iq4xs", seed: 0x4C)
        let reference = try GGUFDequant.decode(type: .iq4XS, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 256, iterations: 1)
        var mismatches = 0
        for index in 0..<min(gpu.count, reference.count)
        where gpu[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) elements differ")
    }

    @Test func measuresIQ4XSDecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = randomBlocks(count: count / 256, blockBytes: 136, key: "iq4xs", seed: 0x71)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print(
            "IQ4_XS GPU dequant: \(String(format: "%.2f", rate)) G elements/s")
        #expect(rate > 0)
    }
}

@Suite struct DequantQ2Tests {
    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantQ20(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("Q2_0 test buffers")
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
            if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = outBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<count).map { pointer[$0] }, seconds)
    }

    @Test func decodesQ2BlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 32
        let bytes = randomBlocks(count: blockCount, blockBytes: 18, key: "q20", seed: 0x20)
        let reference = try GGUFDequant.decode(type: .q2_0, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 64, iterations: 1)
        var mismatches = 0
        for index in 0..<min(gpu.count, reference.count)
        where gpu[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) elements differ")
    }

    @Test func measuresQ2DecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = randomBlocks(count: count / 64, blockBytes: 18, key: "q20", seed: 0x91)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print(
            "Q2_0 GPU dequant: \(String(format: "%.2f", rate)) G elements/s")
        #expect(rate > 0)
    }
}
