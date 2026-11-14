import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The bound every metadata read in the load path now goes through.
///
/// The property under test is *when* the cap fires. `Data(contentsOf:)` reads the
/// whole document before anything can look at its size, and checking the byte
/// count afterwards is a correct TOCTOU fix that does not bound the allocation —
/// measured here on a 2 GiB sparse file: 0.350 s and +2,049 MB of process
/// footprint on a 24 GB machine. So one test below asserts the refusal and one
/// asserts the footprint, because the second is the defect.
@Suite struct BoundedMetadataReadTests {

    private static func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bounded-metadata-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A document whose *size* is `bytes` but whose allocated blocks are almost
    /// nothing: the tail is never written, so reading it is pure allocation. That
    /// is exactly the shape a planted file takes on a copied-off install.
    private static func sparseDocument(
        in directory: URL,
        named name: String,
        size: UInt64
    ) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(#"{"version":1}"#.utf8).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: size)
        try handle.close()
        return url
    }

    @Test("A document over the cap is refused before its bytes are allocated")
    func boundIsAppliedBeforeTheAllocation() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let documentSize: UInt64 = 256 * 1_024 * 1_024
        let url = try Self.sparseDocument(
            in: dir, named: "tokenizer.json", size: documentSize)

        let before = ProcessMemory.physFootprintMiB()
        #expect {
            _ = try BoundedMetadataRead.read(fileAt: url, maxBytes: 64)
        } throws: { error in
            guard
                case ModelError.metadataOverBound(
                    let document, let bytes, let cap) = error
            else { return false }
            return document == "tokenizer.json" && bytes == Int(documentSize) && cap == 64
        }
        let grew = ProcessMemory.physFootprintMiB() - before
        // Reading the same document into memory to discover it was too big would
        // move this number by the full 256 MiB. Allow generous slack so the
        // assertion survives whatever the rest of the suite has just allocated,
        // and still fails the read-then-check shape it was written for.
        #expect(
            grew < 32,
            "refusing a 256 MiB document grew the footprint by \(grew) MiB")
    }

    @Test("The refusal names the document, its size, and the cap")
    func refusalIsLegible() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = try Self.sparseDocument(
            in: dir, named: "config.json", size: 4 * 1_024 * 1_024)
        do {
            _ = try BoundedMetadataRead.read(fileAt: url, maxBytes: 1024)
            Issue.record("expected a refusal")
        } catch {
            let text = String(describing: error)
            #expect(
                text.contains("config.json") && text.contains("4194304")
                    && text.contains("1024-byte metadata bound"),
                "unhelpful refusal: \(text)")
        }
    }

    @Test("A document within the cap reads whole, and a short read is not padded")
    func readsWhatItPromises() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let payload = Data(#"{"hidden_size": 4096, "pad": "xxx"}"#.utf8)
        let url = dir.appendingPathComponent("config.json")
        try payload.write(to: url)
        #expect(try BoundedMetadataRead.read(fileAt: url, maxBytes: 1024) == payload)
        // The cap is inclusive at the boundary: the same file at its own size.
        #expect(
            try BoundedMetadataRead.read(fileAt: url, maxBytes: UInt64(payload.count))
                == payload)
    }

    @Test("A symlinked document is refused rather than followed")
    func symlinksAreNotFollowed() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let real = dir.appendingPathComponent("real.json")
        try Data(#"{"ok":true}"#.utf8).write(to: real)
        let link = dir.appendingPathComponent("tokenizer.json")
        try FileManager.default.createSymbolicLink(
            atPath: link.path, withDestinationPath: real.lastPathComponent)

        #expect(throws: (any Error).self) {
            _ = try BoundedMetadataRead.read(fileAt: link, maxBytes: 1024)
        }
        // The bytes come from the link's name or nowhere: reading the target
        // directly still works, so this is a refusal, not a broken directory.
        #expect(
            try BoundedMetadataRead.read(fileAt: real, maxBytes: 1024)
                == Data(#"{"ok":true}"#.utf8))
    }

    @Test("An absent document reports absent, not oversized")
    func absentIsAbsent() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect {
            _ = try BoundedMetadataRead.read(
                fileAt: dir.appendingPathComponent("metadata.json"), maxBytes: 1024)
        } throws: { error in
            if case ModelError.missingFile(let name) = error { return name == "metadata.json" }
            return false
        }
    }

    @Test("A directory in the document's place is refused, not read")
    func nonRegularFileIsRefused() throws {
        let dir = try Self.scratchDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let nested = dir.appendingPathComponent("config.json")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        #expect(throws: (any Error).self) {
            _ = try BoundedMetadataRead.read(fileAt: nested, maxBytes: 1024)
        }
    }
}
