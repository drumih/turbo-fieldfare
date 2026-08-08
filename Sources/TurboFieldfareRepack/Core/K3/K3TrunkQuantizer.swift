import Accelerate
import Foundation

/// BF16 → affine-g64 (int4/int8) trunk quantizer for the K3 repack, plus the
/// FP32-widen and A_log-truncate companion transforms.
///
/// The numeric recipe mirrors `Quantization` (the house CPU reference the
/// runtime kernels are tested against) exactly: per group of 64 elements,
/// `scale = (max − min) / (2^bits − 1)`, `bias = min` (constant groups use
/// scale 1 / bias value), both companions rounded to BF16 first and the codes
/// quantized against the *rounded* values with round-to-nearest; packed
/// nibbles are low = even element index, matching the repo's int4 convention.
/// `TurboFieldfareTestsCore` asserts bit-identical output against
/// `Quantization.quantizeInt4Affine` / `quantizeInt8Affine`.
enum K3TrunkQuantizer {
    static let groupSize = 64

    // MARK: - BF16 helpers (same bit fiddles as `Quantization`)

    @inline(__always)
    static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - Row quantize / dequantize

    /// Quantize one row of `columns` floats (a multiple of 64) into packed
    /// codes + BF16 scale/bias companions. Bit-exact with the house reference.
    static func quantizeRowAffine(_ row: [Float], bits: Int)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        precondition(bits == 4 || bits == 8, "unsupported affine bit width \(bits)")
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")
        var packed = [UInt8](repeating: 0, count: row.count * bits / 8)
        var scales = [UInt16](repeating: 0, count: row.count / groupSize)
        var biases = [UInt16](repeating: 0, count: row.count / groupSize)
        row.withUnsafeBufferPointer { rowBuf in
            packed.withUnsafeMutableBufferPointer { packedBuf in
                scales.withUnsafeMutableBufferPointer { scalesBuf in
                    biases.withUnsafeMutableBufferPointer { biasesBuf in
                        quantizeRowAffine(row: rowBuf, bits: bits,
                                          packedOut: packedBuf,
                                          scalesOut: scalesBuf,
                                          biasesOut: biasesBuf)
                    }
                }
            }
        }
        return (packed, scales, biases)
    }

    /// Buffer-pointer core of `quantizeRowAffine`; the tile processors call
    /// this directly into preallocated scratch.
    static func quantizeRowAffine(row: UnsafeBufferPointer<Float>, bits: Int,
                                  packedOut: UnsafeMutableBufferPointer<UInt8>,
                                  scalesOut: UnsafeMutableBufferPointer<UInt16>,
                                  biasesOut: UnsafeMutableBufferPointer<UInt16>) {
        let columns = row.count
        let nGroups = columns / groupSize
        let maxCode = (1 << bits) - 1
        let norm = Float(maxCode)
        guard let rowBase = row.baseAddress,
              let packedBase = packedOut.baseAddress,
              let scalesBase = scalesOut.baseAddress,
              let biasesBase = biasesOut.baseAddress else { return }

        for g in 0..<nGroups {
            let group = rowBase.advanced(by: g * groupSize)
            var wmin: Float = 0
            var wmax: Float = 0
            vDSP_minv(group, 1, &wmin, vDSP_Length(groupSize))
            vDSP_maxv(group, 1, &wmax, vDSP_Length(groupSize))
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF: Float
            if wmax == wmin {
                scaleF = 1
                biasF = wmin
            } else {
                scaleF = (wmax - wmin) / norm
                biasF = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q this pass stores.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scalesBase[g] = sBits
            biasesBase[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            if bits == 4 {
                let packedGroup = packedBase.advanced(by: g * (groupSize / 2))
                for k in 0..<groupSize {
                    let w = group[k]
                    var q = Int(((w - bias) * invScale).rounded())
                    q = max(0, min(maxCode, q))
                    let nibble = UInt8(q) & 0x0F
                    let byteIdx = k / 2
                    if (k & 1) == 0 {
                        packedGroup[byteIdx] = (packedGroup[byteIdx] & 0xF0) | nibble
                    } else {
                        packedGroup[byteIdx] = (packedGroup[byteIdx] & 0x0F) | (nibble << 4)
                    }
                }
            } else {
                let packedGroup = packedBase.advanced(by: g * groupSize)
                for k in 0..<groupSize {
                    let w = group[k]
                    var q = Int(((w - bias) * invScale).rounded())
                    q = max(0, min(maxCode, q))
                    packedGroup[k] = UInt8(q)
                }
            }
        }
    }

    /// Reference dequantize (`w = q * scale + bias`) used by the repack tests
    /// to bound the quantization error against the BF16 source.
    static func dequantizeRowAffine(packed: [UInt8], scales: [UInt16], biases: [UInt16],
                                    count: Int, bits: Int) -> [Float] {
        precondition(bits == 4 || bits == 8, "unsupported affine bit width \(bits)")
        precondition(count % groupSize == 0)
        precondition(packed.count == count * bits / 8)
        precondition(scales.count == count / groupSize && biases.count == count / groupSize)
        var out = [Float](repeating: 0, count: count)
        let nGroups = count / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(scales[g])
            let bias = bf16ToFloat(biases[g])
            for k in 0..<groupSize {
                let q: Int
                if bits == 4 {
                    let b = packed[g * (groupSize / 2) + k / 2]
                    q = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                } else {
                    q = Int(packed[g * groupSize + k])
                }
                out[g * groupSize + k] = Float(q) * scale + bias
            }
        }
        return out
    }

    // MARK: - Phase-2 tile processors

    /// Read a BF16 matrix from the staging file, quantize every row, and
    /// write packed weights + scale/bias companions into the resident file.
    /// I/O is bounded to `WriterCore.tileBytes`-sized row tiles.
    static func quantizeStagedTensor(stagingFD: Int32, stagingPath: String,
                                     stagingOffset: UInt64,
                                     residentFD: Int32, residentPath: String,
                                     weightOffset: UInt64, scaleOffset: UInt64,
                                     biasOffset: UInt64,
                                     rows: Int, columns: Int, bits: Int,
                                     audit: RepackAudit) throws {
        precondition(columns > 0 && columns % groupSize == 0)
        let bytesPerRow = columns * 2
        let rowPackedBytes = columns * bits / 8
        let rowAuxBytes = (columns / groupSize) * 2
        let rowsPerTile = max(1, WriterCore.tileBytes / bytesPerRow)

        let readBuf = UnsafeMutableRawBufferPointer.allocate(
            byteCount: rowsPerTile * bytesPerRow, alignment: 16_384)
        defer { readBuf.deallocate() }
        var floatTile = [Float](repeating: 0, count: rowsPerTile * columns)
        var packedTile = [UInt8](repeating: 0, count: rowsPerTile * rowPackedBytes)
        var scalesTile = [UInt16](repeating: 0, count: rowsPerTile * (columns / groupSize))
        var biasesTile = [UInt16](repeating: 0, count: rowsPerTile * (columns / groupSize))
        audit.largestScratchBytes = max(audit.largestScratchBytes,
                                        readBuf.count + floatTile.count * 4
                                            + packedTile.count + 2 * scalesTile.count * 2)

        var row0 = 0
        while row0 < rows {
            try Task.checkCancellation()
            let tileRows = min(rowsPerTile, rows - row0)
            let readBytes = tileRows * bytesPerRow
            try Posix.preadAll(fd: stagingFD, path: stagingPath,
                               buf: readBuf.baseAddress!, count: readBytes,
                               offset: stagingOffset + UInt64(row0) * UInt64(bytesPerRow))
            audit.recordRead(bytes: readBytes)

            // BF16 -> FP32 (exact: top 16 bits of the fp32 pattern).
            readBuf.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: tileRows * columns) { src in
                floatTile.withUnsafeMutableBufferPointer { dst in
                    let n = tileRows * columns
                    var i = 0
                    // Widening shift; exact for every BF16 value.
                    while i < n {
                        dst[i] = Float(bitPattern: UInt32(src[i]) << 16)
                        i &+= 1
                    }
                }
            }

            floatTile.withUnsafeBufferPointer { floats in
                packedTile.withUnsafeMutableBufferPointer { packed in
                    scalesTile.withUnsafeMutableBufferPointer { scales in
                        biasesTile.withUnsafeMutableBufferPointer { biases in
                            for r in 0..<tileRows {
                                let rowSlice = UnsafeBufferPointer(
                                    start: floats.baseAddress!.advanced(by: r * columns),
                                    count: columns)
                                quantizeRowAffine(
                                    row: rowSlice, bits: bits,
                                    packedOut: UnsafeMutableBufferPointer(
                                        start: packed.baseAddress!.advanced(by: r * rowPackedBytes),
                                        count: rowPackedBytes),
                                    scalesOut: UnsafeMutableBufferPointer(
                                        start: scales.baseAddress!.advanced(by: r * (columns / groupSize)),
                                        count: columns / groupSize),
                                    biasesOut: UnsafeMutableBufferPointer(
                                        start: biases.baseAddress!.advanced(by: r * (columns / groupSize)),
                                        count: columns / groupSize))
                            }
                        }
                    }
                }
            }

            let base = UInt64(row0)
            try packedTile.withUnsafeBufferPointer { buf in
                try Posix.pwriteAll(fd: residentFD, path: residentPath,
                                    buf: buf.baseAddress!, count: tileRows * rowPackedBytes,
                                    offset: weightOffset + base * UInt64(rowPackedBytes))
            }
            try scalesTile.withUnsafeBufferPointer { buf in
                try Posix.pwriteAll(fd: residentFD, path: residentPath,
                                    buf: buf.baseAddress!, count: tileRows * rowAuxBytes,
                                    offset: scaleOffset + base * UInt64(rowAuxBytes))
            }
            try biasesTile.withUnsafeBufferPointer { buf in
                try Posix.pwriteAll(fd: residentFD, path: residentPath,
                                    buf: buf.baseAddress!, count: tileRows * rowAuxBytes,
                                    offset: biasOffset + base * UInt64(rowAuxBytes))
            }
            audit.recordWrite(bytes: tileRows * (rowPackedBytes + 2 * rowAuxBytes))
            Posix.adviseDontNeed(fd: stagingFD,
                                 offset: stagingOffset + base * UInt64(bytesPerRow),
                                 length: readBytes)
            row0 += tileRows
        }
    }

    /// Widen a staged BF16 tensor to FP32 in the resident file (router gate,
    /// q_a/kv_a layernorms). Exact widening, tile-bounded.
    static func widenStagedTensor(stagingFD: Int32, stagingPath: String,
                                  stagingOffset: UInt64,
                                  residentFD: Int32, residentPath: String,
                                  residentOffset: UInt64,
                                  elementCount: Int,
                                  audit: RepackAudit) throws {
        let elementsPerTile = max(1, WriterCore.tileBytes / 2)
        let readBuf = UnsafeMutableRawBufferPointer.allocate(
            byteCount: elementsPerTile * 2, alignment: 16_384)
        defer { readBuf.deallocate() }
        var floatTile = [Float](repeating: 0, count: elementsPerTile)
        audit.largestScratchBytes = max(audit.largestScratchBytes,
                                        readBuf.count + floatTile.count * 4)

        var done = 0
        while done < elementCount {
            try Task.checkCancellation()
            let count = min(elementsPerTile, elementCount - done)
            try Posix.preadAll(fd: stagingFD, path: stagingPath,
                               buf: readBuf.baseAddress!, count: count * 2,
                               offset: stagingOffset + UInt64(done) * 2)
            audit.recordRead(bytes: count * 2)
            readBuf.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: count) { src in
                for i in 0..<count {
                    floatTile[i] = Float(bitPattern: UInt32(src[i]) << 16)
                }
            }
            try floatTile.withUnsafeBufferPointer { buf in
                try Posix.pwriteAll(fd: residentFD, path: residentPath,
                                    buf: buf.baseAddress!, count: count * 4,
                                    offset: residentOffset + UInt64(done) * 4)
            }
            audit.recordWrite(bytes: count * 4)
            done += count
        }
    }

    /// A_log arrives head-padded in the source checkpoint (`[headDim]` floats
    /// with the trained prefix of `numHeads` values followed by a zero tail).
    /// Verify the zero tail and keep only the trained prefix.
    static func truncateStagedTensor(stagingFD: Int32, stagingPath: String,
                                     stagingOffset: UInt64,
                                     residentFD: Int32, residentPath: String,
                                     residentOffset: UInt64,
                                     sourceCount: Int, keepCount: Int,
                                     name: String,
                                     audit: RepackAudit) throws {
        precondition(sourceCount >= keepCount && keepCount > 0)
        let readBuf = UnsafeMutableRawBufferPointer.allocate(
            byteCount: sourceCount * 4, alignment: 16_384)
        defer { readBuf.deallocate() }
        audit.largestScratchBytes = max(audit.largestScratchBytes, readBuf.count)
        try Posix.preadAll(fd: stagingFD, path: stagingPath,
                           buf: readBuf.baseAddress!, count: sourceCount * 4,
                           offset: stagingOffset)
        audit.recordRead(bytes: sourceCount * 4)
        // The tail must be bit-zero: those slots belong to padding heads, and
        // silently keeping a trained tail would corrupt the delta-rule decay.
        for i in keepCount * 4..<sourceCount * 4 where readBuf[i] != 0 {
            throw RepackError.configurationInvalid(detail:
                "\(name) has a nonzero value past the \(keepCount)-head prefix")
        }
        try Posix.pwriteAll(fd: residentFD, path: residentPath,
                            buf: readBuf.baseAddress!, count: keepCount * 4,
                            offset: residentOffset)
        audit.recordWrite(bytes: keepCount * 4)
    }
}
