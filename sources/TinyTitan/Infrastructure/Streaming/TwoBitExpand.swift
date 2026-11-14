import Foundation
import TinyTitanCore

/// The cold tier's expansion step: 2-bit packed weights to 4-bit packed.
///
/// A tiered install stores cold experts at 2-bit and hot ones at 4-bit. The
/// existing 4-bit kernels run unchanged if a cold expert is expanded into the
/// slot on the way in, mapping `v` in 0..3 to `5v` in 0,5,10,15 and dividing the
/// group scale by 5 (`bias` unchanged), which makes
/// `q4 * scale/5 + bias == q2 * scale + bias` exactly.
///
/// The transform is a byte shuffle with no arithmetic dependency: 13.7 GB/s on
/// one thread and 28.1 GB/s on four on the 24 GiB M3, 0.27 ms per token for
/// ~3.2 expert misses (`benchmark/two_bit_expand_bench.swift`).
enum TwoBitExpand {
    /// Expands `byteCount` packed 2-bit source bytes into `2 * byteCount`
    /// packed 4-bit destination bytes. `src` and `dst` must not overlap.
    static func expand(
        source: UnsafePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        byteCount: Int
    ) {
        for index in 0..<byteCount {
            let byte = source[index]
            let v0 = UInt8((byte >> 6) & 0x3) * 5
            let v1 = UInt8((byte >> 4) & 0x3) * 5
            let v2 = UInt8((byte >> 2) & 0x3) * 5
            let v3 = UInt8(byte & 0x3) * 5
            // A 4-bit byte holds element 2k in the low nibble and 2k+1 in
            // the high one (`affine_quant_value` shifts by index*4), so the
            // first 2-bit value of the pair is the low nibble here.
            destination[2 * index] = (v1 << 4) | v0
            destination[2 * index + 1] = (v3 << 4) | v2
        }
    }

    /// The scale adjustment that pairs with `expand`: a 2-bit group scale must
    /// be divided by five so the expanded 4-bit values produce the same
    /// product. Bias is unchanged.
    static func adjustedScale(_ scale: Float) -> Float { scale / 5 }
}

/// One tensor inside an expert blob, before and after expansion.
///
/// A cold expert is written compact: its weight tensors are 2-bit and half the
/// size of the 4-bit slot they expand into. The scales and biases are already
/// the right size; only the scales change value (`/ 5`).
package struct ExpertBlobTensor {
    let sourceOffset: Int
    let sourceSize: Int
    let targetOffset: Int
    let targetSize: Int
    /// Stored width of a weight tensor: 2 or 3 for a cold tier, 0 otherwise.
    /// The expansion maps q2 -> 5q and q3 -> 2q, with the paired scale divisor.
    let bits: Int
    /// Weight tensors are expanded; scale and bias tensors are moved.
    let expands: Bool
    /// Scale tensors are divided by five, matching `adjustedScale`.
    let rescales: Bool
}

extension TwoBitExpand {
    /// The invariants `expandExpertInPlace` depends on, checked once where the
    /// maps are built at load. A violating layout would not crash — it would
    /// silently write over unread source bytes and decode wrong weights, so a
    /// bad map is refused before any expert is streamed.
    ///
    /// The invariants: every region sits inside the slot; a move or rescale
    /// keeps its size, a 2-bit expansion doubles it, a 3-bit one takes three
    /// bytes to four with no remainder; every target offset is at or beyond
    /// its source offset (the backwards loops rely on it); and processing
    /// from the highest source offset down, no write region reaches into a
    /// lower tensor's unread source.
    static func validateExpansionLayout(
        _ tensors: [ExpertBlobTensor], expertStride: Int
    ) throws {
        for tensor in tensors {
            guard tensor.sourceOffset >= 0, tensor.sourceSize >= 0,
                tensor.targetOffset >= tensor.sourceOffset,
                tensor.sourceOffset + tensor.sourceSize <= expertStride,
                tensor.targetOffset + tensor.targetSize <= expertStride
            else {
                throw ModelError.indexCorrupt(
                    detail: "expert tensor layout violates the slot bounds")
            }
            let expectedTarget: Int
            if tensor.expands, tensor.bits == 3 {
                guard tensor.sourceSize % 3 == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "3-bit tensor size \(tensor.sourceSize) is not "
                            + "a whole number of groups")
                }
                expectedTarget = tensor.sourceSize / 3 * 4
            } else if tensor.expands {
                expectedTarget = tensor.sourceSize * 2
            } else {
                if tensor.rescales {
                    guard tensor.sourceSize % 2 == 0 else {
                        throw ModelError.indexCorrupt(
                            detail: "rescaled tensor size \(tensor.sourceSize) "
                                + "is not a whole number of bf16 values")
                    }
                }
                expectedTarget = tensor.sourceSize
            }
            guard tensor.targetSize == expectedTarget else {
                throw ModelError.indexCorrupt(
                    detail: "expert tensor target size \(tensor.targetSize) "
                        + "does not match its source size \(tensor.sourceSize)")
            }
        }
        // Processing runs from the highest source offset down, so a write is
        // safe only if it misses every lower tensor's still-unread source and
        // every already-written target. Consecutive pairs are not enough (a
        // far-down write can reach past several tensors), so all pairs are
        // checked; the map is small and this runs once at load.
        for writer in tensors {
            let writeEnd = writer.targetOffset + writer.targetSize
            for reader in tensors where reader.sourceOffset != writer.sourceOffset {
                if writer.targetOffset < reader.sourceOffset + reader.sourceSize,
                    writeEnd > reader.sourceOffset,
                    reader.sourceOffset < writer.sourceOffset
                {
                    throw ModelError.indexCorrupt(
                        detail: "expert tensor at \(writer.sourceOffset) writes "
                            + "into the unread source of the tensor at "
                            + "\(reader.sourceOffset)")
                }
                if writer.targetOffset < reader.targetOffset + reader.targetSize,
                    writeEnd > reader.targetOffset
                {
                    throw ModelError.indexCorrupt(
                        detail: "expert tensors at \(writer.sourceOffset) and "
                            + "\(reader.sourceOffset) share a target region")
                }
            }
        }
    }

    /// Rewrites a compact 2-bit expert blob into the 4-bit slot layout, in place.
    ///
    /// Tensors are processed from the highest source offset down. Every target
    /// offset is at or beyond its source offset (weights double), so a write
    /// never lands on bytes that have not been read yet: the source of a tensor
    /// with a lower offset ends before this tensor's target begins. The slot
    /// must be at least as large as the 4-bit blob.
    static func expandExpertInPlace(
        slot: UnsafeMutableRawPointer,
        tensors: [ExpertBlobTensor]
    ) {
        let bytes = slot.assumingMemoryBound(to: UInt8.self)
        let checking =
            ProcessInfo.processInfo.environment["TINYTITAN_EXPAND_CHECK"] == "1"
        var corruption: String?
        for tensor in tensors.sorted(by: { $0.sourceOffset > $1.sourceOffset }) {
            let source = bytes + tensor.sourceOffset
            let destination = bytes + tensor.targetOffset
            if tensor.expands, tensor.bits == 3 {
                expand3BitReversed(
                    source: source, destination: destination,
                    groupCount: tensor.sourceSize / 3)
            } else if tensor.expands {
                // Backwards, because a doubling expansion overlaps itself --
                // including when the tensor stays at offset 0.
                expandReversed(
                    source: source, destination: destination,
                    byteCount: tensor.sourceSize)
            } else if tensor.rescales {
                if destination != source {
                    memmove(destination, source, tensor.sourceSize)
                }
                rescaleBF16InPlace(
                    destination, bytes: tensor.sourceSize,
                    divisor: tensor.bits == 3 ? 2 : 5)
            } else if destination != source {
                memmove(destination, source, tensor.sourceSize)
            }
            if checking, tensor.rescales {
                let halves = UnsafeMutableRawPointer(destination)
                    .assumingMemoryBound(to: UInt16.self)
                let count = tensor.sourceSize / 2
                var bad = 0
                for index in 0..<count {
                    let value = Float(bitPattern: UInt32(halves[index]) << 16)
                    if !value.isFinite { bad += 1 }
                }
                if bad > 0 {
                    corruption =
                        "role offset \(tensor.targetOffset): \(bad)/\(count) non-finite"
                }
            }
        }
        if let corruption {
            FileHandle.standardError.write(
                Data("[expand] \(corruption)\n".utf8))
        }
    }

    /// Expands a range whose destination overlaps and follows the source.
    private static func expandReversed(
        source: UnsafeMutablePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        byteCount: Int
    ) {
        var index = byteCount
        while index > 0 {
            index -= 1
            let byte = source[index]
            let v0 = UInt8((byte >> 6) & 0x3) * 5
            let v1 = UInt8((byte >> 4) & 0x3) * 5
            let v2 = UInt8((byte >> 2) & 0x3) * 5
            let v3 = UInt8(byte & 0x3) * 5
            destination[2 * index] = (v1 << 4) | v0
            destination[2 * index + 1] = (v3 << 4) | v2
        }
    }

    /// Expands a range of 3-bit groups (eight values per three bytes) to packed
    /// 4-bit, backwards so the doubling expansion cannot clobber unread input.
    private static func expand3BitReversed(
        source: UnsafeMutablePointer<UInt8>,
        destination: UnsafeMutablePointer<UInt8>,
        groupCount: Int
    ) {
        var group = groupCount
        while group > 0 {
            group -= 1
            let base = group * 3
            let word =
                UInt32(source[base])
                | (UInt32(source[base + 1]) << 8)
                | (UInt32(source[base + 2]) << 16)
            var packed: UInt32 = 0
            for slot in 0..<8 {
                let value = 2 * ((word >> (slot * 3)) & 0x7)
                packed |= value << (slot * 4)
            }
            let out = group * 4
            destination[out] = UInt8(truncatingIfNeeded: packed)
            destination[out + 1] = UInt8(truncatingIfNeeded: packed >> 8)
            destination[out + 2] = UInt8(truncatingIfNeeded: packed >> 16)
            destination[out + 3] = UInt8(truncatingIfNeeded: packed >> 24)
        }
    }

    /// Divides bf16 scale values by five in place.
    private static func rescaleBF16InPlace(
        _ pointer: UnsafeMutablePointer<UInt8>, bytes: Int, divisor: Float
    ) {
        let count = bytes / 2
        let halves = UnsafeMutableRawPointer(pointer)
            .assumingMemoryBound(to: UInt16.self)
        for index in 0..<count {
            let bits = UInt32(halves[index]) << 16
            let value = Float(bitPattern: bits) / divisor
            // Truncation costs at most one bf16 ulp on a scale the quantiser
            // already rounded; rounding here would only add code to audit.
            halves[index] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
    }
}
