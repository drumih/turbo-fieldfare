import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The batched IQ2_S GEMM: a prompt (several tokens) times one weight row per
/// threadgroup. Checked element for element against the CPU decoder and dots,
/// and measured so the prefill path has a rate rather than a hope.
@Suite struct IQ2SGemmTests {
    private static func blocks(rows: Int, columns: Int, seed: UInt64) -> [UInt8] {
        var rng = SeedTree(seed).key("iq2s-gemm")
        var bytes: [UInt8] = []
        bytes.reserveCapacity(rows * columns / 256 * 82)
        for _ in 0..<(rows * columns / 256) {
            let scale = Float16(rng.uniform(0.001, 0.01))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern))
            bytes.append(UInt8(truncatingIfNeeded: scale.bitPattern >> 8))
            for _ in 0..<80 { bytes.append(UInt8(truncatingIfNeeded: rng.next())) }
        }
        return bytes
    }

    private static func halves(_ values: [Float16]) -> [UInt8] {
        var bytes: [UInt8] = []
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value.bitPattern))
            bytes.append(UInt8(truncatingIfNeeded: value.bitPattern >> 8))
        }
        return bytes
    }

    @Test func matchesTheCPUDecoderAndDots() throws {
        let context = try MetalContext()
        let rows = 8
        let columns = 256
        let tokens = 4
        let weightBytes = Self.blocks(rows: rows, columns: columns, seed: 0xA2)
        var rng = SeedTree(0x2A).key("iq2s-gemm-a")
        let activations = (0..<(tokens * columns)).map { _ in Float16(rng.uniform(-1, 1)) }

        guard
            let weightBuffer = context.device.makeBuffer(
                bytes: weightBytes, length: weightBytes.count, options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: Self.halves(activations), length: activations.count * 2,
                options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: tokens * rows * 2, options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("IQ2_S GEMM buffers") }
        let gemm = try IQ2SGemm(context: context)
        try gemm.encode(
            commandBuffer: commandBuffer, weights: weightBuffer, x: xBuffer, out: yBuffer,
            rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }

        // CPU: decode the whole matrix, then dot each token with each row.
        let decoded = try GGUFDequant.decode(type: .iq2S, bytes: weightBytes)
        let pointer = yBuffer.contents().assumingMemoryBound(to: UInt16.self)
        var worst = 0.0
        for token in 0..<tokens {
            for row in 0..<rows {
                var reference = 0.0
                for k in 0..<columns {
                    reference +=
                        Double(activations[token * columns + k])
                        * Double(decoded[row * columns + k])
                }
                // The GEMM writes bf16: the top 16 bits of the fp32 pattern.
                let got = Double(Float(bitPattern: UInt32(pointer[token * rows + row]) << 16))
                worst = max(worst, abs(got - reference) / max(abs(reference), 1e-6))
            }
        }
        print("IQ2_S GEMM worst relative error: \(worst)")
        #expect(worst < 5e-3)
    }

    @Test func measuresPrefillStyleThroughput() throws {
        let context = try MetalContext()
        let rows = 2_560
        let columns = 2_560
        let tokens = 128
        let weightBytes = Self.blocks(rows: rows, columns: columns, seed: 0xB2)
        var rng = SeedTree(0x3A).key("iq2s-gemm-x")
        let activations = (0..<(tokens * columns)).map { _ in Float16(rng.uniform(-1, 1)) }
        guard
            let weightBuffer = context.device.makeBuffer(
                bytes: weightBytes, length: weightBytes.count, options: .storageModeShared),
            let xBuffer = context.device.makeBuffer(
                bytes: Self.halves(activations), length: activations.count * 2,
                options: .storageModeShared),
            let yBuffer = context.device.makeBuffer(
                length: tokens * rows * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("IQ2_S GEMM buffers") }
        let gemm = try IQ2SGemm(context: context)
        var best = Double.infinity
        for _ in 0..<3 {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try gemm.encode(
                commandBuffer: commandBuffer, weights: weightBuffer, x: xBuffer, out: yBuffer,
                rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens))
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            best = min(
                best,
                commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        }
        let flops = 2.0 * Double(rows) * Double(columns) * Double(tokens)
        print(
            "IQ2_S GEMM: \(String(format: "%.0f", flops / best / 1e9)) GFLOP/s, "
                + "\(String(format: "%.1f", best * 1000)) ms for 128 tokens x 2560 x 2560")
        #expect(best > 0)
    }
}
