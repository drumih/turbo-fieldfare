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

    static func nvfp4(
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
