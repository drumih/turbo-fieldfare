import Foundation
import Darwin

/// On-disk header. `indexSize` is the full byte size of the leading index
/// region — it INCLUDES the 24-byte header itself, the entry table, the
/// string table, and the writer's 16 KB page padding. The resident tensor
/// region starts at file byte `indexSize`.
struct ResidentIndexHeader: Sendable, Equatable {
    let indexSize: UInt64
    let residentSize: UInt64
    let entryCount: UInt64
}

public struct ResidentIndexEntry: Sendable, Equatable {
    public let name: String
    public let dtype: UInt8
    /// Absolute file offset of the packed weight bytes (≥ `indexSize`).
    public let fileOffset: UInt64
    public let sizeBytes: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    public let scaleOffset: UInt64
    public let scaleSize: UInt64
    public let biasOffset: UInt64
    public let biasSize: UInt64

    public static func == (a: ResidentIndexEntry, b: ResidentIndexEntry) -> Bool {
        a.name == b.name && a.dtype == b.dtype
            && a.fileOffset == b.fileOffset && a.sizeBytes == b.sizeBytes
            && a.shape.0 == b.shape.0 && a.shape.1 == b.shape.1
            && a.shape.2 == b.shape.2 && a.shape.3 == b.shape.3
            && a.scaleOffset == b.scaleOffset && a.scaleSize == b.scaleSize
            && a.biasOffset == b.biasOffset && a.biasSize == b.biasSize
    }
}

/// Parsed resident index: header + name-keyed entries.
struct ResidentIndex: Sendable {
    let header: ResidentIndexHeader
    let entries: [String: ResidentIndexEntry]
}

enum ResidentIndexReader {

    static let headerBytes = 24
    static let entryBytes = 72

    /// `pread` the header + index region out of `model_weights.bin`. The
    /// tensor data region (starting at byte `header.indexSize`) is **not**
    /// read here — that's the resident-buffer materialization job.
    static func load(fileURL: URL) throws -> ResidentIndex {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        // -- header
        var headerBuf = [UInt8](repeating: 0, count: headerBytes)
        var got = 0
        headerBuf.withUnsafeMutableBufferPointer { p in
            got = pread(fd, p.baseAddress!, headerBytes, 0)
        }
        guard got == headerBytes else {
            throw ModelError.indexCorrupt(detail: "short read for IndexHeader (\(got)/\(headerBytes))")
        }
        let header = headerBuf.withUnsafeBytes { raw -> ResidentIndexHeader in
            let base = raw.baseAddress!
            return ResidentIndexHeader(
                indexSize:    decodeU64LE(base, 0),
                residentSize: decodeU64LE(base, 8),
                entryCount:   decodeU64LE(base, 16))
        }

        guard header.indexSize >= UInt64(headerBytes) else {
            throw ModelError.indexCorrupt(
                detail: "indexSize \(header.indexSize) < header size \(headerBytes)")
        }
        let expectedEntryTableEnd = UInt64(headerBytes) + UInt64(header.entryCount) * UInt64(entryBytes)
        guard expectedEntryTableEnd <= header.indexSize else {
            throw ModelError.indexCorrupt(
                detail: "header+entries (\(expectedEntryTableEnd)) > indexSize \(header.indexSize)")
        }

        // -- entire index region (header + entries + string table + padding)
        let regionLen = Int(header.indexSize)
        var indexBuf = [UInt8](repeating: 0, count: regionLen)
        indexBuf.withUnsafeMutableBufferPointer { p in
            got = pread(fd, p.baseAddress!, regionLen, 0)
        }
        guard got == regionLen else {
            throw ModelError.indexCorrupt(detail: "short read for index region (\(got)/\(regionLen))")
        }

        // -- parse entries. Each IndexEntry stores `nameOffset` as an
        // absolute file offset; since we read the leading region starting
        // at file offset 0, that's also the buffer offset.
        var entries: [String: ResidentIndexEntry] = [:]
        entries.reserveCapacity(Int(header.entryCount))
        try indexBuf.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for i in 0..<Int(header.entryCount) {
                let p = base.advanced(by: headerBytes + i * entryBytes)
                let nameOffset = Int(decodeU32LE(p, 0))
                let nameLength = Int(decodeU16LE(p, 4))
                let dtype = decodeU8(p, 6)
                // byte 7 is reserved
                let fileOffset = decodeU64LE(p, 8)
                let sizeBytes  = decodeU64LE(p, 16)
                let s0 = decodeU32LE(p, 24)
                let s1 = decodeU32LE(p, 28)
                let s2 = decodeU32LE(p, 32)
                let s3 = decodeU32LE(p, 36)
                let scaleOffset = decodeU64LE(p, 40)
                let scaleSize   = decodeU64LE(p, 48)
                let biasOffset  = decodeU64LE(p, 56)
                let biasSize    = decodeU64LE(p, 64)

                guard nameOffset >= headerBytes,
                      nameOffset + nameLength <= regionLen else {
                    throw ModelError.indexCorrupt(
                        detail: "entry \(i) name range [\(nameOffset), \(nameOffset + nameLength)) " +
                                "out of index region [\(headerBytes), \(regionLen))")
                }
                let namePtr = base.advanced(by: nameOffset)
                let name = String(decoding: UnsafeRawBufferPointer(
                    start: namePtr, count: nameLength), as: UTF8.self)

                let entry = ResidentIndexEntry(
                    name: name, dtype: dtype,
                    fileOffset: fileOffset, sizeBytes: sizeBytes,
                    shape: (s0, s1, s2, s3),
                    scaleOffset: scaleOffset, scaleSize: scaleSize,
                    biasOffset: biasOffset, biasSize: biasSize)
                if entries[name] != nil {
                    throw ModelError.indexCorrupt(detail: "duplicate tensor name \(name)")
                }
                entries[name] = entry
            }
        }

        return ResidentIndex(header: header, entries: entries)
    }

    // MARK: - little-endian primitives

    @inline(__always)
    static func decodeU64LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt64 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var v: UInt64 = 0
        for i in (0..<8).reversed() { v = (v << 8) | UInt64(p[i]) }
        return v
    }

    @inline(__always)
    static func decodeU32LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt32 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        var v: UInt32 = 0
        for i in (0..<4).reversed() { v = (v << 8) | UInt32(p[i]) }
        return v
    }

    @inline(__always)
    static func decodeU16LE(_ base: UnsafeRawPointer, _ off: Int) -> UInt16 {
        let p = base.advanced(by: off).assumingMemoryBound(to: UInt8.self)
        return UInt16(p[0]) | (UInt16(p[1]) << 8)
    }

    @inline(__always)
    static func decodeU8(_ base: UnsafeRawPointer, _ off: Int) -> UInt8 {
        base.advanced(by: off).assumingMemoryBound(to: UInt8.self)[0]
    }
}
