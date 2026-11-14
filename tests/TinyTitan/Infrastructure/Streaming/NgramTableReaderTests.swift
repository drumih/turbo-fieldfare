import Foundation
import Testing

@testable import TinyTitan
import TinyTitanCore

/// Row gather over the n-gram table, against a synthetic file whose contents
/// encode their own row index so a misaddressed read is detectable rather than
/// merely wrong-looking.
@Suite("N-gram table reader")
struct NgramTableReaderTests {
    private static let rowDim = 8
    private static let rowCount: UInt64 = 64

    /// Row r is filled with the value r, so reading row r must yield r in
    /// every lane. An off-by-one or a wrong stride lands on a different row and
    /// fails loudly instead of returning plausible embedding data.
    private static func makeTable() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-\(UUID().uuidString).bin")
        var bytes = [Float16]()
        bytes.reserveCapacity(Int(rowCount) * rowDim)
        for r in 0..<Int(rowCount) {
            bytes += [Float16](repeating: Float16(r), count: rowDim)
        }
        try bytes.withUnsafeBufferPointer {
            try Data(buffer: $0).write(to: url)
        }
        return url
    }

    private static func reader(_ url: URL) throws -> NgramTableReader {
        // Page cache left on for the test: the synthetic file is tiny and
        // F_NOCACHE on a just-written temp file measures nothing useful.
        try NgramTableReader(
            path: url.path, rowDim: rowDim,
            rowCount: rowCount, bypassCache: false)
    }

    @Test("Gathers the requested rows in the requested order")
    func gathersInOrder() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        let want: [UInt32] = [7, 0, 63, 31, 7]
        var out = [Float16](repeating: 0, count: want.count * Self.rowDim)
        try out.withUnsafeMutableBytes { buffer in
            try r.gather(rows: want, into: try #require(buffer.baseAddress))
        }
        for (i, row) in want.enumerated() {
            for lane in 0..<Self.rowDim {
                #expect(
                    out[i * Self.rowDim + lane] == Float16(row),
                    "slot \(i) lane \(lane) should hold row \(row)")
            }
        }
    }

    @Test("A token's rows gathered concurrently match the serial gather")
    func concurrentRowsMatchSerial() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        let rows: [UInt32] = [7, 0, 63, 31, 7, 2, 40]
        var serial = [Float16](repeating: 0, count: rows.count * Self.rowDim)
        var concurrent = [Float16](repeating: 0, count: rows.count * Self.rowDim)
        try serial.withUnsafeMutableBytes { buffer in
            try r.gather(rows: rows, into: try #require(buffer.baseAddress))
        }
        try concurrent.withUnsafeMutableBytes { buffer in
            try r.gatherConcurrentlyRows(
                rows: rows, into: try #require(buffer.baseAddress), threads: 4)
        }
        #expect(serial == concurrent)
    }

    @Test("A row past the end is refused, not read out of bounds")
    func rejectsOutOfRange() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        var out = [Float16](repeating: 0, count: Self.rowDim)
        #expect(throws: NgramTableReader.Failure.self) {
            try out.withUnsafeMutableBytes { buffer in
                try r.gather(
                    rows: [UInt32(Self.rowCount)],
                    into: try #require(buffer.baseAddress))
            }
        }
    }

    @Test("A table whose size disagrees with the geometry is refused at open")
    func rejectsSizeMismatch() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        // Claiming more rows than the file holds would otherwise read
        // plausible values from wrong offsets rather than failing.
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: url.path, rowDim: Self.rowDim,
                rowCount: Self.rowCount + 1,
                bypassCache: false)
        }
        // A wrong row width is the same class of error.
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: url.path, rowDim: Self.rowDim * 2,
                rowCount: Self.rowCount,
                bypassCache: false)
        }
    }

    @Test("A missing table fails at open rather than at first gather")
    func rejectsMissingFile() {
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: "/nonexistent/ngram_table.bin",
                rowDim: 160, rowCount: 1, bypassCache: false)
        }
    }

    @Test("Production geometry moves ~5 KiB per token")
    func productionGatherBudget() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        // 160 fp16 x 16 heads at the real geometry; the fixture's rowDim is
        // smaller, so compute against the real numbers directly.
        let realRowBytes = 160 * MemoryLayout<Float16>.stride
        #expect(realRowBytes == 320)
        #expect(16 * realRowBytes == 5120)
        // And the reader reports its own geometry consistently.
        #expect(r.gatherBytes(headCount: 16) == 16 * r.rowBytes)
    }

    /// A concurrent gather must land the same bytes as the serial one: same
    /// rows, same order, one block per batch. The workers write disjoint
    /// regions, so the only way this fails is a wrong batch stride or a race,
    /// and a wrong stride would otherwise show up as a plausible embedding.
    @Test("Concurrent gather matches the serial one byte for byte")
    func concurrentMatchesSerial() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        let batches: [[UInt32]] = [
            [7, 0, 63], [1, 2, 3], [63, 62, 61],
            [31, 31, 31], [8, 16, 32], [0, 0, 63],
        ]
        let stride = batches[0].count * r.rowBytes
        let total = batches.count * stride
        var serial = [UInt8](repeating: 0xAA, count: total)
        var parallel = [UInt8](repeating: 0xAA, count: total)
        try serial.withUnsafeMutableBytes { buffer in
            let base = try #require(buffer.baseAddress)
            for (i, rows) in batches.enumerated() {
                try r.gather(
                    rows: rows, into: base.advanced(by: i * stride))
            }
        }
        try parallel.withUnsafeMutableBytes { buffer in
            try r.gatherConcurrently(
                batches: batches, chunkStride: stride,
                into: try #require(buffer.baseAddress), threads: 4)
        }
        #expect(serial == parallel)
        // Spot-check the addressing rather than trusting the bytewise match,
        // which two identically wrong loops would also satisfy.
        for (i, rows) in batches.enumerated() {
            for lane in 0..<Self.rowDim {
                let at = i * stride + lane * MemoryLayout<Float16>.stride
                let value = parallel.withUnsafeBytes {
                    $0.loadUnaligned(fromByteOffset: at, as: Float16.self)
                }
                #expect(
                    value == Float16(rows[0]),
                    "batch \(i) lane \(lane) should hold row \(rows[0])")
            }
        }
    }

    /// One batch degenerates to the serial path; an empty input is a no-op.
    @Test("Concurrent gather handles the degenerate sizes")
    func concurrentDegenerateSizes() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }
        let r = try Self.reader(url)
        var out = [Float16](repeating: 0, count: Self.rowDim)
        try out.withUnsafeMutableBytes { buffer in
            let base = try #require(buffer.baseAddress)
            try r.gatherConcurrently(
                batches: [], chunkStride: r.rowBytes, into: base)
            try r.gatherConcurrently(
                batches: [[9]], chunkStride: r.rowBytes, into: base)
        }
        #expect(out[0] == Float16(9))
    }
}

extension NgramTableReaderTests {
    /// Geometry comes from `ple_constants.json`, a file on disk, not a fact of
    /// the build — so an unusable one is reported, not trapped on.
    @Test("Unusable geometry is refused, not trapped on")
    func refusesUnusableGeometry() throws {
        let url = try Self.makeTable()
        defer { try? FileManager.default.removeItem(at: url) }

        for (rowDim, rowCount) in [(0, UInt64(64)), (8, UInt64(0))] {
            #expect(throws: NgramTableReader.Failure.self) {
                _ = try NgramTableReader(
                    path: url.path, rowDim: rowDim,
                    rowCount: rowCount, bypassCache: false)
            }
        }
        // The multiplication that sizes the table must not wrap: with a row
        // count near `UInt64.max` a wrapping product collapses `expected` to a
        // small number and the size guard below would accept a table built for
        // a different geometry.
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: url.path, rowDim: Int.max,
                rowCount: UInt64.max, bypassCache: false)
        }
    }
}
