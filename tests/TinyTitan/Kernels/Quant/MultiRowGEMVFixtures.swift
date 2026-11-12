import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Random weights, activations and byte comparison for the multi-row GEMV
/// identity tests. The weights are random bit patterns rather than quantized
/// from floats: the property under test is that the multi-row kernel performs
/// the decode kernel's arithmetic, which any packed value exercises.
enum MultiRowGEMVFixtures {
    struct Generator {
        private var state: UInt64
        init(seed: UInt64) { state = seed }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        mutating func uniform(_ low: Float, _ high: Float) -> Float {
            let unit = Float(next() >> 40) / Float(1 << 24)
            return low + (high - low) * unit
        }

        mutating func fill(_ pointer: UnsafeMutableRawPointer, bytes: Int) {
            var index = 0
            while index + 8 <= bytes {
                (pointer + index).storeBytes(of: next(), as: UInt64.self)
                index += 8
            }
            let tail = next()
            for offset in 0..<(bytes - index) {
                (pointer + index + offset).storeBytes(
                    of: UInt8(truncatingIfNeeded: tail >> (8 * offset)), as: UInt8.self)
            }
        }
    }

    static func bf16Bits(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    /// One weight matrix in a resident-style buffer: `[pad][weights][scales]
    /// [biases]`, so a view's offsets are what a model install hands the
    /// kernels. bf16 matrices carry no scales or biases.
    static func matrix(
        device: MTLDevice, format: MultiRowGEMV.Format,
        rows: Int, columns: Int, pad: Int,
        generator: inout Generator
    ) throws -> TensorView {
        let weightBytes: Int
        switch format {
        case .int4: weightBytes = rows * columns / 2
        case .affine(let bits): weightBytes = rows * columns * bits / 8
        case .bf16: weightBytes = rows * columns * 2
        }
        let groups = rows * columns / Quantization.groupSize
        let quantized = format != .bf16
        let scaleOffset = pad + weightBytes
        let biasOffset = scaleOffset + (quantized ? groups * 2 : 0)
        let total = biasOffset + (quantized ? groups * 2 : 0)
        let buffer = try #require(
            device.makeBuffer(length: max(total, 4), options: .storageModeShared))
        let base = buffer.contents()
        if quantized {
            generator.fill(base + pad, bytes: weightBytes)
            let scales = (base + scaleOffset).assumingMemoryBound(to: UInt16.self)
            let biases = (base + biasOffset).assumingMemoryBound(to: UInt16.self)
            for group in 0..<groups {
                scales[group] = bf16Bits(generator.uniform(-0.04, 0.04))
                biases[group] = bf16Bits(generator.uniform(-0.1, 0.1))
            }
        } else {
            let weights = (base + pad).assumingMemoryBound(to: UInt16.self)
            for index in 0..<(rows * columns) {
                weights[index] = bf16Bits(generator.uniform(-0.25, 0.25))
            }
        }
        return TensorView(
            buffer: buffer,
            offset: UInt64(pad), length: UInt64(weightBytes),
            scaleOffset: UInt64(scaleOffset), scaleLength: UInt64(quantized ? groups * 2 : 0),
            biasOffset: UInt64(biasOffset), biasLength: UInt64(quantized ? groups * 2 : 0),
            shape: (UInt32(rows), UInt32(columns), 1, 1),
            dtype: quantized ? 0 : 1)
    }

    /// `count` fp16 activations in [-1, 1].
    static func activations(
        device: MTLDevice, count: Int, generator: inout Generator
    ) throws -> MTLBuffer {
        let buffer = try #require(
            device.makeBuffer(length: count * 2, options: .storageModeShared))
        let halves = buffer.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<count {
            halves[index] = Float16(generator.uniform(-1, 1))
        }
        return buffer
    }

    static func zeroed(device: MTLDevice, halves: Int) throws -> MTLBuffer {
        let buffer = try #require(
            device.makeBuffer(length: halves * 2, options: .storageModeShared))
        memset(buffer.contents(), 0, halves * 2)
        return buffer
    }

    static func run(
        _ context: MetalContext, _ body: (MTLCommandBuffer) throws -> Void
    ) throws {
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try body(commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(
            commandBuffer.error == nil,
            "command buffer failed: \(String(describing: commandBuffer.error))")
    }

    /// Byte equality, with the first differing fp16 element named on failure.
    static func expectIdentical(
        _ expected: MTLBuffer, _ actual: MTLBuffer, halves: Int,
        actualOffsetHalves: Int = 0, _ label: String
    ) {
        let lhs = expected.contents().assumingMemoryBound(to: UInt16.self)
        let rhs = actual.contents().assumingMemoryBound(to: UInt16.self) + actualOffsetHalves
        let equal = memcmp(lhs, rhs, halves * 2) == 0
        #expect(equal, "\(label): multi-row output differs from decode")
        guard !equal else { return }
        var first = -1
        var count = 0
        for index in 0..<halves where lhs[index] != rhs[index] {
            if first < 0 { first = index }
            count += 1
        }
        let decodeBits = String(lhs[first], radix: 16)
        let multiRowBits = String(rhs[first], radix: 16)
        Issue.record(
            "\(label): \(count) of \(halves) differ, first at \(first) (\(decodeBits) vs \(multiRowBits))"
        )
    }
}
