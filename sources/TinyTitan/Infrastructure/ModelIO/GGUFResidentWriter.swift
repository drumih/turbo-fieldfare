import Foundation
import TinyTitanFormat

package enum GGUFResidentWidth: Sendable, Equatable {
    case affine4
    case affine8
    case bf16

    var dtype: UInt8 {
        switch self {
        case .affine4, .affine8: return 0
        case .bf16: return 1
        }
    }

    var bits: Int {
        switch self {
        case .affine4: return 4
        case .affine8: return 8
        case .bf16: return 16
        }
    }
}

package enum GGUFResidentWriterError: Error, Equatable, CustomStringConvertible {
    case missingWidth(String)
    case unknownTensor(String)
    case rowNotGroupAligned(name: String, columns: UInt64, group: Int)
    case writeFailed(path: String, errno: Int32)
    case fileSizeMismatch(path: String, expected: UInt64, actual: UInt64)
    case emptyTensorList

    package var description: String {
        switch self {
        case .missingWidth(let name):
            return "no resident width was given for \(name)"
        case .unknownTensor(let name):
            return "the GGUF has no tensor named \(name)"
        case .rowNotGroupAligned(let name, let columns, let group):
            return "\(name) rows are \(columns) wide, not a multiple of \(group)"
        case .writeFailed(let path, let code):
            return "cannot write \(path): errno \(code)"
        case .fileSizeMismatch(let path, let expected, let actual):
            return "\(path) is \(actual) bytes, expected \(expected)"
        case .emptyTensorList:
            return "the resident writer was given no tensors"
        }
    }
}

package struct GGUFResidentWriteResult: Sendable {
    package let url: URL
    package let indexSize: UInt64
    package let residentSize: UInt64
    package let entries: [ResidentIndexEntry]
}

/// Writes `model_weights.bin`: a binary index (header, 72-byte entries, string
/// table, padded to 16 KiB) followed by the resident tensor payload.
///
/// The width per tensor is the caller's, not this type's: which role gets 4
/// bits, 8 bits or the checkpoint's bf16 is the conversion policy, and it has to
/// match the manifest's quant slots. This writer only has to make the bytes and
/// the index agree with what `ResidentIndexReader` and the kernels expect:
/// dtype 0 for affine weights with BF16 scale/bias companions, dtype 1 for bf16
/// with no companions, every tensor 16 KiB aligned.
package enum GGUFResidentWriter {
    package static let alignment: UInt64 = SSDAIFormatV1.alignmentBytes
    private static let headerBytes: UInt64 = 24
    private static let entryBytes: UInt64 = 72

    package static func write(
        names: [String],
        from model: GGUFModel,
        widths: [String: GGUFResidentWidth],
        to url: URL
    ) throws -> GGUFResidentWriteResult {
        guard !names.isEmpty else { throw GGUFResidentWriterError.emptyTensorList }
        let payloads = try encodePayloads(names: names, from: model, widths: widths)
        let layout = try assembleIndex(payloads: payloads, model: model)
        try writeFile(url: url, layout: layout, payloads: payloads)
        return GGUFResidentWriteResult(
            url: url, indexSize: layout.indexSize,
            residentSize: layout.residentSize, entries: layout.entries)
    }

    private struct Payload {
        let name: String
        let data: [UInt8]
        let width: GGUFResidentWidth
    }

    private struct IndexLayout {
        let index: [UInt8]
        let entries: [ResidentIndexEntry]
        let indexSize: UInt64
        let residentSize: UInt64
        let payloadOffsets: [UInt64]
    }

    private static func encodePayloads(
        names: [String], from model: GGUFModel, widths: [String: GGUFResidentWidth]
    ) throws -> [Payload] {
        var payloads: [Payload] = []
        payloads.reserveCapacity(names.count)
        for name in names {
            guard let width = widths[name] else {
                throw GGUFResidentWriterError.missingWidth(name)
            }
            guard let location = model.tensor(named: name) else {
                throw GGUFResidentWriterError.unknownTensor(name)
            }
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let raw = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
            guard raw.count == Int(location.tensor.byteCount) else {
                throw GGUFResidentWriterError.writeFailed(path: shard.url.path, errno: EIO)
            }
            let values = try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](raw))
            payloads.append(
                Payload(
                    name: name,
                    data: try encode(
                        values, dimensions: location.tensor.dimensions, width: width),
                    width: width))
        }
        return payloads
    }

    private static func assembleIndex(
        payloads: [Payload], model: GGUFModel
    ) throws -> IndexLayout {
        let stringTable = payloads.map { Array($0.name.utf8) }
        let tableBytes = stringTable.reduce(0) { $0 + $1.count }
        let rawIndexSize = headerBytes + UInt64(payloads.count) * entryBytes + UInt64(tableBytes)
        let indexSize = align(rawIndexSize)
        var payloadOffsets: [UInt64] = []
        var cursor: UInt64 = 0
        for payload in payloads {
            payloadOffsets.append(cursor)
            cursor = align(cursor + UInt64(payload.data.count))
        }

        var index = [UInt8](repeating: 0, count: Int(indexSize))
        writeU64(&index, 0, indexSize)
        writeU64(&index, 8, cursor)
        writeU64(&index, 16, UInt64(payloads.count))
        var entryList: [ResidentIndexEntry] = []
        var nameOffsets: [UInt32] = []
        var nameCursor = UInt32(headerBytes + UInt64(payloads.count) * entryBytes)
        for (position, payload) in payloads.enumerated() {
            nameOffsets.append(nameCursor)
            nameCursor += UInt32(payload.name.utf8.count)
            guard let location = model.tensor(named: payload.name) else {
                throw GGUFResidentWriterError.unknownTensor(payload.name)
            }
            let dimensions = location.tensor.dimensions
            let rows = dimensions.count > 1 ? dimensions[1] : 1
            let columns = dimensions[0]
            let groups = columns / UInt64(Quantization.groupSize)
            let weightsSize: UInt64
            let scaleSize: UInt64
            switch payload.width {
            case .affine4, .affine8:
                weightsSize = rows * columns * UInt64(payload.width.bits) / 8
                scaleSize = rows * groups * 2
            case .bf16:
                weightsSize = rows * columns * 2
                scaleSize = 0
            }
            let fileOffset = indexSize + payloadOffsets[position]
            entryList.append(
                ResidentIndexEntry(
                    name: payload.name,
                    dtype: payload.width.dtype,
                    fileOffset: fileOffset,
                    sizeBytes: weightsSize,
                    shape: (UInt32(rows), UInt32(columns), 1, 1),
                    scaleOffset: scaleSize > 0 ? fileOffset + weightsSize : 0,
                    scaleSize: scaleSize,
                    biasOffset: scaleSize > 0 ? fileOffset + weightsSize + scaleSize : 0,
                    biasSize: scaleSize))
        }
        writeEntries(&index, entries: entryList, nameOffsets: nameOffsets)
        writeStringTable(&index, payloads: payloads, stringTable: stringTable)
        return IndexLayout(
            index: index, entries: entryList, indexSize: indexSize,
            residentSize: cursor, payloadOffsets: payloadOffsets)
    }

    private static func writeEntries(
        _ index: inout [UInt8], entries: [ResidentIndexEntry], nameOffsets: [UInt32]
    ) {
        for (position, entry) in entries.enumerated() {
            var start = Int(headerBytes) + position * Int(entryBytes)
            writeU32(&index, start, nameOffsets[position])
            start += 4
            let utf8 = Array(entry.name.utf8)
            writeU16(&index, start, UInt16(utf8.count))
            start += 2
            index[start] = entry.dtype
            index[start + 1] = 0
            start += 2
            writeU64(&index, start, entry.fileOffset)
            start += 8
            writeU64(&index, start, entry.sizeBytes)
            start += 8
            for dimension in [entry.shape.0, entry.shape.1, entry.shape.2, entry.shape.3] {
                writeU32(&index, start, dimension)
                start += 4
            }
            writeU64(&index, start, entry.scaleOffset)
            start += 8
            writeU64(&index, start, entry.scaleSize)
            start += 8
            writeU64(&index, start, entry.biasOffset)
            start += 8
            writeU64(&index, start, entry.biasSize)
        }
    }

    private static func writeStringTable(
        _ index: inout [UInt8], payloads: [Payload], stringTable: [[UInt8]]
    ) {
        var cursor = Int(headerBytes + UInt64(payloads.count) * entryBytes)
        for bytes in stringTable {
            index.replaceSubrange(cursor..<(cursor + bytes.count), with: bytes)
            cursor += bytes.count
        }
    }

    private static func writeFile(
        url: URL, layout: IndexLayout, payloads: [Payload]
    ) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw GGUFResidentWriterError.writeFailed(path: url.path, errno: errno)
        }
        guard let handle = FileHandle(forWritingAtPath: url.path) else {
            throw GGUFResidentWriterError.writeFailed(path: url.path, errno: errno)
        }
        defer { try? handle.close() }
        do {
            try handle.write(contentsOf: Data(layout.index))
            for (position, payload) in payloads.enumerated() {
                try handle.write(contentsOf: Data(payload.data))
                let written =
                    layout.indexSize + layout.payloadOffsets[position]
                    + UInt64(payload.data.count)
                let next = align(written)
                if next > written {
                    try handle.write(contentsOf: Data(count: Int(next - written)))
                }
            }
        } catch {
            throw GGUFResidentWriterError.writeFailed(path: url.path, errno: errno)
        }
        let actual = try handle.seekToEnd()
        guard actual == layout.indexSize + layout.residentSize else {
            throw GGUFResidentWriterError.fileSizeMismatch(
                path: url.path, expected: layout.indexSize + layout.residentSize, actual: actual)
        }
    }

    /// Encode one tensor at the requested width. Rows are quantised in groups of
    /// 64 (the engine's affine format); bf16 is written plane.
    static func encode(
        _ values: [Float], dimensions: [UInt64], width: GGUFResidentWidth
    ) throws -> [UInt8] {
        let columns = Int(dimensions[0])
        let rows = dimensions.count > 1 ? Int(dimensions[1]) : 1
        // Grouping is an affine constraint; a bf16 tensor keeps its own width.
        if width != .bf16, columns % Quantization.groupSize != 0 {
            throw GGUFResidentWriterError.rowNotGroupAligned(
                name: "tensor", columns: UInt64(columns), group: Quantization.groupSize)
        }
        var packed: [UInt8] = []
        var scales: [UInt8] = []
        var biases: [UInt8] = []
        switch width {
        case .affine4:
            for row in 0..<rows {
                let start = row * columns
                let rowValues = Array(values[start..<(start + columns)])
                let quantized = Quantization.quantizeInt4Affine(rowValues)
                packed.append(contentsOf: quantized.packed)
                append(quantized.scales, to: &scales)
                append(quantized.biases, to: &biases)
            }
            return packed + scales + biases
        case .affine8:
            for row in 0..<rows {
                let start = row * columns
                let rowValues = Array(values[start..<(start + columns)])
                let quantized = Quantization.quantizeInt8Affine(rowValues)
                packed.append(contentsOf: quantized.packed)
                append(quantized.scales, to: &scales)
                append(quantized.biases, to: &biases)
            }
            return packed + scales + biases
        case .bf16:
            var bytes: [UInt8] = []
            bytes.reserveCapacity(values.count * 2)
            for value in values {
                let bits = Quantization.bf16Bits(value)
                bytes.append(UInt8(truncatingIfNeeded: bits))
                bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
            }
            return bytes
        }
    }

    private static func append(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    private static func align(_ value: UInt64) -> UInt64 {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }

    private static func writeU16(_ bytes: inout [UInt8], _ offset: Int, _ value: UInt16) {
        bytes[offset] = UInt8(truncatingIfNeeded: value)
        bytes[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
    }

    private static func writeU32(_ bytes: inout [UInt8], _ offset: Int, _ value: UInt32) {
        for index in 0..<4 {
            bytes[offset + index] = UInt8(truncatingIfNeeded: value >> (8 * index))
        }
    }

    private static func writeU64(_ bytes: inout [UInt8], _ offset: Int, _ value: UInt64) {
        for index in 0..<8 {
            bytes[offset + index] = UInt8(truncatingIfNeeded: value >> (8 * index))
        }
    }
}
