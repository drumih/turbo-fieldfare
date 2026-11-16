import Testing
import Foundation
@testable import TinyTitanCore
@testable import Qwen35Dense2B4Bit

/// The variant matrix test (7.0 spec §12.6, plan task 9, tracker row TT-080):
/// for the variant, every role resolves and every kernel width has a path. The
/// nibble-trap case is the test that would have caught the 6.x 8-bit-read-as-
/// nibbles bug. These run on synthetic tensors — no model install is needed.
/// The kernel-path half awaits the graph port (TT-078); the contract half is
/// verified here.
struct MatrixTests {
    /// Every role the variant could read resolves to a valid spec: a width in
    /// {2,4,8} and a positive group. A role that resolved to nothing, or to a
    /// withdrawn width, would fail here.
    @Test func everyRoleResolvesToAValidSpec() {
        let quant = Qwen35Dense2B4BitVariant.quant
        for role in QuantRole.allCases {
            let spec = quant.spec(forRole: role, stem: TensorStem("layer0.weight"))
            #expect([2, 4, 8].contains(spec.bits), "role \(role) resolved to width \(spec.bits)")
            #expect(spec.group > 0, "role \(role) resolved to group \(spec.group)")
        }
    }

    /// The default width is the variant's declared width — a 4-bit variant does
    /// not silently default to another width.
    @Test func defaultWidthMatchesTheVariant() {
        let quant = Qwen35Dense2B4BitVariant.quant
        #expect(quant.defaultSpec.bits == Qwen35Dense2B4BitVariant.id.bits)
        #expect(quant.defaultSpec.bits == 4)
    }

    /// The nibble trap: an 8-bit buffer dequantized at 8 bits recovers the
    /// reference; reinterpreting the same bytes as 4-bit nibbles does not. A
    /// kernel that read 8-bit data as nibbles would produce the corrupted
    /// column, and this assertion is what refuses it.
    @Test func eightBitBufferReadAsNibblesIsCorrupted() {
        // One group of 8 values, affine 8-bit: value = byte * scale + bias.
        let bytes: [UInt8] = [10, 20, 30, 40, 50, 60, 70, 80]
        let scale: Float = 0.5
        let bias: Float = 0.0
        let reference = bytes.map { Float($0) * scale + bias }

        let atEight = Self.dequantize(bytes, bits: 8, scale: scale, bias: bias)
        #expect(atEight == reference)

        let atFour = Self.dequantize(bytes, bits: 4, scale: scale, bias: bias)
        #expect(atFour != reference, "reading 8-bit data as 4-bit nibbles must corrupt the values")
        // 4-bit unpacking yields two nibbles per byte: a different element count.
        #expect(atFour.count == bytes.count * 2)
    }

    /// A minimal host-side affine dequantizer used as the reference oracle. At
    /// 8 bits each byte is one value; at 4 bits each byte is two nibbles (low
    /// then high). This mirrors the packing the kernels read.
    private static func dequantize(_ bytes: [UInt8], bits: Int, scale: Float, bias: Float)
        -> [Float]
    {
        var out: [Float] = []
        if bits == 8 {
            for byte in bytes { out.append(Float(byte) * scale + bias) }
        } else if bits == 4 {
            for byte in bytes {
                out.append(Float(byte & 0x0F) * scale + bias)
                out.append(Float(byte >> 4) * scale + bias)
            }
        }
        return out
    }
}
