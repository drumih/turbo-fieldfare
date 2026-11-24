import Foundation

/// Repacks one routed expert matrix from the checkpoint's planar layout into the
/// block layout the decode graph's kernels read (TT-125).
///
/// - NVFP4 (layers 0–19): planar values ([rows][cols/2], two E2M1 codes a byte,
///   even index in the low nibble) and E4M3 scales ([rows][cols/16]) become GGUF
///   NVFP4 blocks of 64 values: 4 scale bytes, then per 16-value group 8 bytes with
///   value j in the low nibble and value j+8 in the high one. Byte-identical with
///   the Q4 install's experts (measured on 9 experts in 3 layers).
/// - MXFP8 (layers 20–23): planar E4M3 values ([rows][cols]) and E8M0 scales
///   ([rows][cols/32]) become 34-byte blocks of 32 values: the scale byte, a zero
///   byte, then the 32 values -- the Q8_0 block's size, so strides match the Q4
///   install's.
///
/// `interleaved` is the checkpoint's `gate_up`: its rows alternate gate and up,
/// where the graph reads all gate rows, then all up rows.
enum InklingNVFP4Repack {
    /// Bytes of one GGUF NVFP4 block (64 values).
    static let nvfp4BlockBytes = 36
    /// Bytes of one MXFP8 block (32 values).
    static let mxfp8BlockBytes = 34

    static func nvfp4Bytes(rows: Int, columns: Int) -> Int { rows * columns / 64 * nvfp4BlockBytes }
    static func mxfp8Bytes(rows: Int, columns: Int) -> Int { rows * columns / 32 * mxfp8BlockBytes }

    /// The checkpoint row behind output row `row`.
    @inline(__always)
    static func sourceRow(_ row: Int, rows: Int, interleaved: Bool) -> Int {
        guard interleaved else { return row }
        let half = rows / 2
        return row < half ? 2 * row : 2 * (row - half) + 1
    }

    /// The repack the cache runs: one 64-bit word per 16-value group. The
    /// group's 8 source bytes hold value 2k in the low and value 2k+1 in the
    /// high nibble of byte k; output byte 2m is value 2m | value 2m+8 << 4 and
    /// byte 2m+1 is value 2m+1 | value 2m+9 << 4, which is the low nibbles of
    /// bytes m and m+4 (and the high ones) folded, then byte-interleaved.
    /// `nvfp4Reference` is the nibble-at-a-time definition it is tested against.
    static func nvfp4(
        values: UnsafeRawPointer, scales: UnsafeRawPointer, rows: Int, columns: Int,
        interleaved: Bool, into out: UnsafeMutableRawPointer
    ) {
        let valueRow = columns / 2
        let scaleRow = columns / 16
        let blocks = columns / 64
        let nibbles: UInt64 = 0x0F0F_0F0F_0F0F_0F0F
        @inline(__always) func spread(_ word: UInt64) -> UInt64 {
            var x = (word | word << 16) & 0x0000_FFFF_0000_FFFF
            x = (x | x << 8) & 0x00FF_00FF_00FF_00FF
            return x
        }
        for row in 0..<rows {
            let source = sourceRow(row, rows: rows, interleaved: interleaved)
            let rowValues = values + source * valueRow
            let rowScales = scales + source * scaleRow
            let rowOut = out + row * blocks * nvfp4BlockBytes
            for block in 0..<blocks {
                let blockOut = rowOut + block * nvfp4BlockBytes
                blockOut.storeBytes(
                    of: rowScales.loadUnaligned(fromByteOffset: block * 4, as: UInt32.self),
                    as: UInt32.self)
                for group in 0..<4 {
                    let word = rowValues.loadUnaligned(
                        fromByteOffset: block * 32 + group * 8, as: UInt64.self)
                    let low = word & nibbles
                    let high = (word >> 4) & nibbles
                    let a = (low & 0xFFFF_FFFF) | ((low >> 32) << 4)
                    let b = (high & 0xFFFF_FFFF) | ((high >> 32) << 4)
                    (blockOut + 4 + group * 8).storeBytes(
                        of: spread(a) | (spread(b) << 8), as: UInt64.self)
                }
            }
        }
    }

    static func nvfp4Reference(
        values: UnsafeRawPointer, scales: UnsafeRawPointer, rows: Int, columns: Int,
        interleaved: Bool, into out: UnsafeMutableRawPointer
    ) {
        let valueRow = columns / 2
        let scaleRow = columns / 16
        let blocks = columns / 64
        let v = values.assumingMemoryBound(to: UInt8.self)
        let s = scales.assumingMemoryBound(to: UInt8.self)
        let o = out.assumingMemoryBound(to: UInt8.self)
        for row in 0..<rows {
            let source = sourceRow(row, rows: rows, interleaved: interleaved)
            let rowValues = v + source * valueRow
            let rowScales = s + source * scaleRow
            let rowOut = o + row * blocks * nvfp4BlockBytes
            for block in 0..<blocks {
                let blockOut = rowOut + block * nvfp4BlockBytes
                for group in 0..<4 { blockOut[group] = rowScales[block * 4 + group] }
                for group in 0..<4 {
                    // 16 values = 8 source bytes, two codes each.
                    let src = rowValues + (block * 64 + group * 16) / 2
                    let dst = blockOut + 4 + group * 8
                    for j in 0..<8 {
                        let low = code(src, j)
                        let high = code(src, j + 8)
                        dst[j] = low | (high << 4)
                    }
                }
            }
        }
    }

    /// The E2M1 code of value `index` in pair-packed bytes.
    @inline(__always)
    private static func code(_ bytes: UnsafePointer<UInt8>, _ index: Int) -> UInt8 {
        let byte = bytes[index / 2]
        return index % 2 == 0 ? byte & 0x0F : byte >> 4
    }

    static func mxfp8(
        values: UnsafeRawPointer, scales: UnsafeRawPointer, rows: Int, columns: Int,
        interleaved: Bool, into out: UnsafeMutableRawPointer
    ) {
        let blocks = columns / 32
        let v = values.assumingMemoryBound(to: UInt8.self)
        let s = scales.assumingMemoryBound(to: UInt8.self)
        let o = out.assumingMemoryBound(to: UInt8.self)
        for row in 0..<rows {
            let source = sourceRow(row, rows: rows, interleaved: interleaved)
            let rowValues = v + source * columns
            let rowScales = s + source * blocks
            let rowOut = o + row * blocks * mxfp8BlockBytes
            for block in 0..<blocks {
                let blockOut = rowOut + block * mxfp8BlockBytes
                blockOut[0] = rowScales[block]
                blockOut[1] = 0
                (blockOut + 2).update(from: rowValues + block * 32, count: 32)
            }
        }
    }
}
