import Foundation

/// Little-endian binary encoders for the resident `IndexHeader` and 72-byte
/// `IndexEntry` records. Used by the resident writer and the
/// matching loader-side parsers; kept in one place so the on-disk layout
/// changes only here.
enum GTurboBinary {

    static let indexHeaderBytes: Int = 24
    static let indexEntryBytes: Int = 72

    /// Write `IndexHeader { indexSize, residentSize, entryCount }` (24 bytes, LE).
    static func writeIndexHeader(into buf: UnsafeMutableRawPointer,
                                        indexSize: UInt64,
                                        residentSize: UInt64,
                                        entryCount: UInt64) {
        var off = 0
        writeU64LE(buf, &off, indexSize)
        writeU64LE(buf, &off, residentSize)
        writeU64LE(buf, &off, entryCount)
    }

    /// Write one `IndexEntry` (72 bytes, LE) at `dst`. See gturbo-format.md.
    static func writeIndexEntry(into dst: UnsafeMutableRawPointer,
                                       entry: ResidentEntry,
                                       nameOffset: UInt32) {
        var off = 0
        writeU32LE(dst, &off, nameOffset)
        writeU16LE(dst, &off, UInt16(entry.name.utf8.count))
        writeU8(dst, &off, entry.dtype)
        writeU8(dst, &off, 0) // reserved
        writeU64LE(dst, &off, entry.fileOffset)
        writeU64LE(dst, &off, entry.sizeBytes)
        writeU32LE(dst, &off, entry.logicalShape4[0])
        writeU32LE(dst, &off, entry.logicalShape4[1])
        writeU32LE(dst, &off, entry.logicalShape4[2])
        writeU32LE(dst, &off, entry.logicalShape4[3])
        writeU64LE(dst, &off, entry.scaleOffset)
        writeU64LE(dst, &off, entry.scaleSize)
        writeU64LE(dst, &off, entry.biasOffset)
        writeU64LE(dst, &off, entry.biasSize)
        precondition(off == indexEntryBytes)
    }

    @inline(__always)
    private static func writeU64LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt64) {
        let p = buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            for i in 0..<8 { p[i] = src[i] }
        }
        off += 8
    }

    @inline(__always)
    private static func writeU32LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt32) {
        let p = buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            for i in 0..<4 { p[i] = src[i] }
        }
        off += 4
    }

    @inline(__always)
    private static func writeU16LE(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt16) {
        let p = buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { src in
            for i in 0..<2 { p[i] = src[i] }
        }
        off += 2
    }

    @inline(__always)
    private static func writeU8(_ buf: UnsafeMutableRawPointer, _ off: inout Int, _ v: UInt8) {
        buf.advanced(by: off).assumingMemoryBound(to: UInt8.self)[0] = v
        off += 1
    }

}
