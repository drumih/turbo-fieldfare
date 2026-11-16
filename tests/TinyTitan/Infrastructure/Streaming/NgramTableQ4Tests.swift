import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The 4-bit table form: header validation and the dequantising gather.
///
/// The synthetic rows are built so every expected value is exact in bf16:
/// scale 1.0, integer bias, and a nibble pattern; a wrong group, bias or nibble
/// order therefore lands on a value that is not in the set.
@Suite("N-gram table reader, 4-bit")
struct NgramTableQ4Tests {
    private static let rowDim = 32
    private static let rowCount: UInt64 = 4
    private static let rowBytes = 20  // 16 packed + 2 scale + 2 bias

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func bf16(_ value: Float) -> UInt16 {
        // Every value used here is exactly representable in bf16.
        UInt16((value.bitPattern >> 16) & 0xFFFF)
    }

    private static func makeQ4Table(magic: UInt32 = NgramTableReader.q4Magic) throws -> URL {
        var data = Data()
        append(magic, to: &data)
        append(NgramTableReader.q4Version, to: &data)
        append(UInt32(rowDim), to: &data)
        append(UInt32(4), to: &data)
        append(UInt32(32), to: &data)
        append(UInt32(rowBytes), to: &data)
        append(rowCount, to: &data)
        append(UInt32(40), to: &data)
        append(UInt32(0), to: &data)
        for row in 0..<Int(rowCount) {
            for group in 0..<1 {
                var packed = [UInt8](repeating: 0, count: rowDim / 2)
                for lane in 0..<rowDim {
                    let nibble = UInt8(lane % 16)
                    if lane % 2 == 0 {
                        packed[lane / 2] = nibble
                    } else {
                        packed[lane / 2] |= nibble << 4
                    }
                }
                data.append(contentsOf: packed)
                append(bf16(1.0), to: &data)
                append(bf16(Float(row * 10 + group)), to: &data)
            }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-q4-\(UUID().uuidString).bin")
        try data.write(to: url)
        return url
    }

    @Test func gathersAndDequantisesRowsInOrder() throws {
        let url = try Self.makeQ4Table()
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try NgramTableReader(
            path: url.path, rowDim: Self.rowDim, rowCount: Self.rowCount,
            bypassCache: false, format: .affine4Group32)
        let want: [UInt32] = [2, 0, 3]
        var out = [Float16](repeating: 0, count: want.count * Self.rowDim)
        try out.withUnsafeMutableBytes { buffer in
            try reader.gather(rows: want, into: try #require(buffer.baseAddress))
        }
        for (i, row) in want.enumerated() {
            for lane in 0..<Self.rowDim {
                let expected = Float(row) * 10 + Float(lane % 16)
                #expect(
                    out[i * Self.rowDim + lane] == Float16(expected),
                    "row \(row) lane \(lane)")
            }
        }
    }

    @Test func concurrentRowsMatchSerialForTheDequantisingGather() throws {
        let url = try Self.makeQ4Table()
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try NgramTableReader(
            path: url.path, rowDim: Self.rowDim, rowCount: Self.rowCount,
            bypassCache: false, format: .affine4Group32)
        let rows: [UInt32] = [2, 0, 3, 1, 2]
        var serial = [Float16](repeating: 0, count: rows.count * Self.rowDim)
        var concurrent = [Float16](repeating: 0, count: rows.count * Self.rowDim)
        try serial.withUnsafeMutableBytes { buffer in
            try reader.gather(rows: rows, into: try #require(buffer.baseAddress))
        }
        try concurrent.withUnsafeMutableBytes { buffer in
            try reader.gatherConcurrentlyRows(
                rows: rows, into: try #require(buffer.baseAddress), threads: 3)
        }
        #expect(serial == concurrent)
    }

    @Test func foreignMagicIsRefused() throws {
        let url = try Self.makeQ4Table(magic: 0xDEAD_BEEF)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: url.path, rowDim: Self.rowDim, rowCount: Self.rowCount,
                bypassCache: false, format: .affine4Group32)
        }
    }

    @Test func geometryThatDisagreesWithTheHeaderIsRefused() throws {
        let url = try Self.makeQ4Table()
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: NgramTableReader.Failure.self) {
            _ = try NgramTableReader(
                path: url.path, rowDim: Self.rowDim, rowCount: Self.rowCount + 1,
                bypassCache: false, format: .affine4Group32)
        }
    }
}
