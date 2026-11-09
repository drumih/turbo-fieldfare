import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

private func q6kBlocks(_ count: Int, seed: UInt64) -> [UInt8] {
    var rng = SeedTree(seed).key("q6k-blocks")
    var bytes: [UInt8] = []
    bytes.reserveCapacity(count * 210)
    for _ in 0..<count {
        for _ in 0..<208 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
        let scale = Float16(rng.uniform(0.001, 0.01))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
    }
    return bytes
}

private func q8Blocks(_ count: Int, seed: UInt64) -> [UInt8] {
    var rng = SeedTree(seed).key("q8-blocks")
    var bytes: [UInt8] = []
    bytes.reserveCapacity(count * 34)
    for _ in 0..<count {
        let scale = Float16(rng.uniform(0.001, 0.02))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
        bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
        for _ in 0..<32 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
    }
    return bytes
}

@Suite struct DequantQ6KTests {
    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantQ6K(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("Q6_K test buffers")
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

    @Test func decodesQ6KBlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 16
        let bytes = q6kBlocks(blockCount, seed: 0x6B)
        let reference = try GGUFDequant.decode(type: .q6K, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 256, iterations: 1)
        var mismatches = 0
        for index in 0..<min(gpu.count, reference.count)
        where gpu[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) elements differ")
    }

    @Test func measuresQ6KDecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = q6kBlocks(count / 256, seed: 0x7B)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print("Q6_K GPU dequant: \(String(format: "%.2f", rate)) G elements/s")
        #expect(rate > 0)
    }
}

@Suite struct DequantQ8Tests {
    private static func run(
        _ context: MetalContext, blocks: [UInt8], count: Int, iterations: Int
    ) throws -> (values: [UInt16], seconds: Double) {
        let kernel = try DequantQ80(context: context)
        guard
            let blockBuffer = context.device.makeBuffer(
                bytes: blocks, length: blocks.count, options: .storageModeShared),
            let outBuffer = context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("Q8_0 test buffers")
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

    @Test func decodesQ8BlocksLikeTheCPU() throws {
        let context = try MetalContext()
        let blockCount = 64
        let bytes = q8Blocks(blockCount, seed: 0x08)
        let reference = try GGUFDequant.decode(type: .q8_0, bytes: bytes)
        let (gpu, _) = try Self.run(
            context, blocks: bytes, count: blockCount * 32, iterations: 1)
        var mismatches = 0
        for index in 0..<min(gpu.count, reference.count)
        where gpu[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
        }
        #expect(mismatches == 0, "\(mismatches) of \(gpu.count) elements differ")
    }

    @Test func measuresQ8DecoderThroughput() throws {
        let context = try MetalContext()
        let count = 1_638_400
        let bytes = q8Blocks(count / 32, seed: 0x18)
        _ = try Self.run(context, blocks: bytes, count: count, iterations: 1)
        let iterations = 20
        let (_, seconds) = try Self.run(
            context, blocks: bytes, count: count, iterations: iterations)
        let rate = Double(count * iterations) / seconds / 1e9
        print("Q8_0 GPU dequant: \(String(format: "%.2f", rate)) G elements/s")
        #expect(rate > 0)
    }
}
