import Testing
import Foundation
import Darwin
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct ResidentIndexTests {

    static func dummySource(_ name: String) -> SourceTensor {
        SourceTensor(name: name, shardPath: "/dev/null", dtype: .u32,
                     shape: [1024, 64], absoluteOffset: 0, sizeBytes: 0)
    }

    @Test func roundtripsWriterEncodedBytes() throws {
        // Two entries with distinct shapes / sizes / bias offsets so we
        // exercise every field of the 72-byte record. Encoding mirrors
        // ResidentWriter.write: header at byte 0, entries starting at byte 24,
        // string table after the entries, and nameOffset = absolute file
        // offset to the name inside the index region.
        let names = ["embedding.weight", "layer.0.q_proj.weight"]
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes  = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = Int(GTurboFormatV1.alignmentBytes)
        let residentBytes = 96

        let entries: [ResidentEntry] = [
            ResidentEntry(
                name: names[0], dtype: 0,
                logicalShape4: [1024, 64, 0, 0],
                fileOffset: UInt64(indexBytes), sizeBytes: 32,
                scaleOffset: UInt64(indexBytes) + 32, scaleSize: 16,
                biasOffset:  UInt64(indexBytes) + 48, biasSize:  16,
                quantSpec: nil,
                sourceWeight: Self.dummySource(names[0]),
                sourceScales: nil, sourceBiases: nil),
            ResidentEntry(
                name: names[1], dtype: 0,
                logicalShape4: [256, 64, 0, 0],
                fileOffset: UInt64(indexBytes) + 64, sizeBytes: 32,
                scaleOffset: 0, scaleSize: 0,
                biasOffset:  0, biasSize:  0,
                quantSpec: nil,
                sourceWeight: Self.dummySource(names[1]),
                sourceScales: nil, sourceBiases: nil),
        ]

        var fileBuf = [UInt8](repeating: 0, count: indexBytes + residentBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: UInt64(indexBytes),
                                          residentSize: UInt64(residentBytes),
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameOffsets[i])
            }
            stringTable.withUnsafeBytes { sb in
                _ = memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-roundtrip-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(fileBuf).write(to: url)

        let parsed = try ResidentIndexReader.load(fileURL: url)
        #expect(parsed.header.entryCount == UInt64(entries.count))
        #expect(parsed.header.indexSize == UInt64(indexBytes))
        #expect(parsed.header.residentSize == UInt64(residentBytes))
        let e0 = try #require(parsed.entries[names[0]])
        #expect(e0.shape.0 == 1024 && e0.shape.1 == 64)
        #expect(e0.fileOffset == UInt64(indexBytes))
        #expect(e0.sizeBytes == 32)
        #expect(e0.scaleOffset == UInt64(indexBytes) + 32 && e0.scaleSize == 16)
        #expect(e0.biasOffset  == UInt64(indexBytes) + 48 && e0.biasSize  == 16)
        let e1 = try #require(parsed.entries[names[1]])
        #expect(e1.shape.0 == 256 && e1.shape.1 == 64)
        #expect(e1.fileOffset == UInt64(indexBytes) + 64)
        #expect(e1.sizeBytes == 32)
        #expect(e1.scaleSize == 0 && e1.biasSize == 0)
    }

    @Test func shortFileThrowsIndexCorrupt() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-short-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0, count: 8).write(to: url)
        #expect {
            _ = try ResidentIndexReader.load(fileURL: url)
        } throws: { error in
            if case ModelError.indexCorrupt = error { return true }
            return false
        }
    }

    @Test func acceptsIndexRegionExactlyAtMetadataCap() throws {
        let cap = ResidentIndexReader.defaultMaxBytes
        var data = Data(repeating: 0, count: Int(cap))
        data.withUnsafeMutableBytes {
            GTurboBinary.writeIndexHeader(
                into: $0.baseAddress!, indexSize: cap,
                residentSize: 0, entryCount: 0)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-cap-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        let parsed = try ResidentIndexReader.load(fileURL: url)
        #expect(parsed.header.indexSize == cap)
        #expect(parsed.entries.isEmpty)
    }

    @Test func rejectsIndexRegionAboveMetadataCapBeforeAllocation() throws {
        let claimed = ResidentIndexReader.defaultMaxBytes + 1
        var data = Data(repeating: 0, count: GTurboBinary.indexHeaderBytes)
        data.withUnsafeMutableBytes {
            GTurboBinary.writeIndexHeader(
                into: $0.baseAddress!, indexSize: claimed,
                residentSize: 0, entryCount: 0)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-over-cap-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        #expect {
            try ResidentIndexReader.load(fileURL: url)
        } throws: { error in
            guard case ModelError.indexCorrupt(let detail) = error else { return false }
            return detail.contains("metadata cap")
        }
    }

    @Test func rejectsIndexSizeSmallerThanHeader() throws {
        var data = Data(repeating: 0, count: GTurboBinary.indexHeaderBytes)
        data.withUnsafeMutableBytes {
            GTurboBinary.writeIndexHeader(
                into: $0.baseAddress!, indexSize: UInt64(GTurboBinary.indexHeaderBytes - 1),
                residentSize: 0, entryCount: 0)
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-small-header-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        #expect(throws: ModelError.self) {
            try ResidentIndexReader.load(fileURL: url)
        }
    }

    @Test func callerSelectedSymlinkToRegularIndexStillLoads() throws {
        let indexSize = Int(GTurboFormatV1.alignmentBytes)
        var data = Data(repeating: 0, count: indexSize)
        data.withUnsafeMutableBytes {
            GTurboBinary.writeIndexHeader(
                into: $0.baseAddress!, indexSize: UInt64(indexSize),
                residentSize: 0, entryCount: 0)
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-symlink-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("target.bin")
        let alias = directory.appendingPathComponent("alias.bin")
        try data.write(to: target)
        #expect(symlink(target.path, alias.path) == 0)
        let parsed = try ResidentIndexReader.load(fileURL: alias)
        #expect(parsed.header.entryCount == 0)
    }

    @Test func symlinkToFIFOIsRejectedWithoutReading() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-index-fifo-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fifo = directory.appendingPathComponent("index.fifo")
        let alias = directory.appendingPathComponent("alias.bin")
        #expect(mkfifo(fifo.path, 0o600) == 0)
        #expect(symlink(fifo.path, alias.path) == 0)
        #expect(throws: ModelError.self) {
            try ResidentIndexReader.load(fileURL: alias)
        }
    }

}
