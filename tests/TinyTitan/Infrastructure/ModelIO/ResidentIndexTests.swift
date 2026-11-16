import Foundation
import Testing
import TinyTitanFormat
@testable import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanRepackCore

@Suite struct ResidentIndexTests {

    static func dummySource(_ name: String) -> SourceTensor {
        SourceTensor(
            name: name, shardPath: "/dev/null", dtype: .u32,
            shape: [1024, 64], absoluteOffset: 0, sizeBytes: 0)
    }

    @Test func roundtripsWriterEncodedBytes() throws {
        // Two entries with distinct shapes / sizes / bias offsets so we
        // exercise every field of the 72-byte record. Encoding mirrors
        // ResidentWriter.write: header at byte 0, entries starting at byte 24,
        // string table after the entries, and nameOffset = absolute file
        // offset to the name inside the index region.
        let names = ["embedding.weight", "layer.0.q_proj.weight"]
        let stringTable = Data(names.joined().utf8)
        let headerBytes = SSDAIBinary.indexHeaderBytes
        let entryBytes = SSDAIBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let rawIndexBytes = stringTableBase + stringTable.count
        // Writer rounds the index region up to 16 KB; we keep the test snug
        // (no padding) since the parser only needs indexSize ≥ that minimum.
        // However, the validator now enforces 16KB alignment on the index size.
        let alignedIndexBytes = Int(
            ((UInt64(rawIndexBytes) + SSDAIFormatV1.alignmentBytes - 1)
                & ~(SSDAIFormatV1.alignmentBytes - 1)))
        let residentBytes = 96

        let entries: [ResidentEntry] = [
            ResidentEntry(
                name: names[0], dtype: 0,
                logicalShape4: [1024, 64, 0, 0],
                fileOffset: UInt64(alignedIndexBytes), sizeBytes: 32,
                scaleOffset: UInt64(alignedIndexBytes) + 32, scaleSize: 16,
                biasOffset: UInt64(alignedIndexBytes) + 48, biasSize: 16,
                quantSpec: nil,
                sourceWeight: Self.dummySource(names[0]),
                sourceScales: nil, sourceBiases: nil),
            ResidentEntry(
                name: names[1], dtype: 0,
                logicalShape4: [256, 64, 0, 0],
                fileOffset: UInt64(alignedIndexBytes) + 64, sizeBytes: 32,
                scaleOffset: 0, scaleSize: 0,
                biasOffset: 0, biasSize: 0,
                quantSpec: nil,
                sourceWeight: Self.dummySource(names[1]),
                sourceScales: nil, sourceBiases: nil),
        ]

        var fileBuf = [UInt8](repeating: 0, count: alignedIndexBytes + residentBytes)
        try fileBuf.withUnsafeMutableBytes { raw in
            let base = try #require(raw.baseAddress)
            SSDAIBinary.writeIndexHeader(
                into: base,
                indexSize: UInt64(alignedIndexBytes),
                residentSize: UInt64(residentBytes),
                entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                SSDAIBinary.writeIndexEntry(
                    into: dst, entry: e,
                    nameOffset: nameOffsets[i])
            }
            stringTable.withUnsafeBytes { sb in
                // An empty table has nothing to copy; the source pointer is only
                // meaningful when there is a byte to read.
                if let source = sb.baseAddress {
                    _ = memcpy(base.advanced(by: stringTableBase), source, stringTable.count)
                }
            }
            // Entry 1 payload (32 bytes at offset + 64).
            memset(base.advanced(by: Int(UInt64(alignedIndexBytes) + 64)), 0x22, 32)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-index-roundtrip-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(fileBuf).write(to: url)

        let parsed = try ResidentIndexReader.load(fileURL: url)
        #expect(parsed.header.entryCount == UInt64(entries.count))
        #expect(parsed.header.indexSize == UInt64(alignedIndexBytes))
        #expect(parsed.header.residentSize == UInt64(residentBytes))
        let e0 = try #require(parsed.entries[names[0]])
        #expect(e0.shape.0 == 1024 && e0.shape.1 == 64)
        #expect(e0.fileOffset == UInt64(alignedIndexBytes))
        #expect(e0.sizeBytes == 32)
        #expect(e0.scaleOffset == UInt64(alignedIndexBytes) + 32 && e0.scaleSize == 16)
        #expect(e0.biasOffset == UInt64(alignedIndexBytes) + 48 && e0.biasSize == 16)
        let e1 = try #require(parsed.entries[names[1]])
        #expect(e1.shape.0 == 256 && e1.shape.1 == 64)
        #expect(e1.fileOffset == UInt64(alignedIndexBytes) + 64)
        #expect(e1.sizeBytes == 32)
        #expect(e1.scaleSize == 0 && e1.biasSize == 0)
    }

    @Test func shortFileThrowsIndexCorrupt() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-short-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0, count: 8).write(to: url)
        #expect {
            _ = try ResidentIndexReader.load(fileURL: url)
        } throws: { error in
            if case ModelError.indexCorrupt = error { return true }
            return false
        }
    }

}

extension ResidentIndexTests {
    /// A header declaring a payload region past the end of the file is refused.
    ///
    /// Every entry bound the decoder enforces comes from that same header, so
    /// without a check against the real file an entry can point past the
    /// mapping and the reader builds a buffer over memory that was never
    /// mapped. The CPU dense loader hands those pointers out with no further
    /// size check, so the failure is a SIGBUS or silently wrong weights.
    @Test func aResidentRegionBeyondEOFIsRefused() throws {
        let headerBytes = SSDAIBinary.indexHeaderBytes
        let entryBytes = SSDAIBinary.indexEntryBytes
        let name = "layer.0.q_proj.weight"
        let stringTableBase = headerBytes + entryBytes
        let rawIndexBytes = stringTableBase + name.utf8.count
        let aligned = Int(
            ((UInt64(rawIndexBytes) + SSDAIFormatV1.alignmentBytes - 1)
                & ~(SSDAIFormatV1.alignmentBytes - 1)))
        // The file is exactly the index; the header claims a payload after it.
        var buf = [UInt8](repeating: 0, count: aligned)
        try buf.withUnsafeMutableBytes { raw in
            let base = try #require(raw.baseAddress)
            SSDAIBinary.writeIndexHeader(
                into: base,
                indexSize: UInt64(aligned),
                residentSize: 4096,
                entryCount: 1)
            SSDAIBinary.writeIndexEntry(
                into: base.advanced(by: headerBytes),
                entry: ResidentEntry(
                    name: name, dtype: 0, logicalShape4: [64, 64, 0, 0],
                    fileOffset: UInt64(aligned), sizeBytes: 4096,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: Self.dummySource(name),
                    sourceScales: nil, sourceBiases: nil),
                nameOffset: UInt32(stringTableBase))
            // The name is a non-empty literal; passing the array straight to
            // memcpy avoids a nil base address entirely.
            memcpy(
                base.advanced(by: stringTableBase), Array(name.utf8),
                name.utf8.count)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-overrun-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(buf).write(to: url)

        do {
            _ = try ResidentIndexReader.load(fileURL: url)
            Issue.record("a payload region past EOF was accepted")
        } catch let error as ModelError {
            #expect(
                "\(error)".contains("exceeds file size"),
                "refused for the wrong reason: \(error)")
        }
    }
}
