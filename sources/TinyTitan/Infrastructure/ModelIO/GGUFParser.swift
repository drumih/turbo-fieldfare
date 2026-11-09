import Foundation

struct GGUFParser {
    let url: URL
    let data: Data
    let fileSize: UInt64
    var offset: Int = 0

    init(url: URL, data: Data, fileSize: UInt64) {
        self.url = url
        self.data = data
        self.fileSize = fileSize
    }

    mutating func parse() throws -> GGUFShard {
        let magicBytes = try raw(4)
        // The magic is four bytes of ASCII in every valid file, but a hostile
        // one can put anything there; the refusal must still be printable.
        let magic = String(bytes: magicBytes, encoding: .utf8) ?? "<non-UTF-8>"
        guard magic == "GGUF" else { throw GGUFError.badMagic(magic) }
        let version = try uint32()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }
        let tensorCount = try uint64()
        let metadataCount = try uint64()
        guard metadataCount <= 1_000_000 else {
            throw GGUFError.metadataCountTooLarge(metadataCount)
        }
        guard tensorCount <= 1_000_000 else {
            throw GGUFError.tensorCountTooLarge(tensorCount)
        }

        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let key = try string(field: "metadata key")
            guard metadata[key] == nil else {
                throw GGUFError.duplicateMetadataKey(key)
            }
            let type = try uint32()
            metadata[key] = try value(type: type, depth: 0)
        }

        var tensors: [GGUFTensor] = []
        tensors.reserveCapacity(Int(tensorCount))
        var names = Set<String>()
        for _ in 0..<tensorCount {
            let name = try string(field: "tensor name")
            guard names.insert(name).inserted else {
                throw GGUFError.duplicateTensorName(name)
            }
            let dimensions = try tensorDimensions(name: name)
            let rawType = try uint32()
            guard let type = GGUFType(rawValue: rawType) else {
                throw GGUFError.unknownTensorType(name: name, type: rawType)
            }
            let tensorOffset = try uint64()
            let (elementCount, byteCount) = try geometry(
                name: name, dimensions: dimensions, type: type)
            tensors.append(
                GGUFTensor(
                    name: name,
                    dimensions: dimensions,
                    type: type,
                    offset: tensorOffset,
                    elementCount: elementCount,
                    byteCount: byteCount))
        }

        let headerEnd = UInt64(offset)
        let alignment = try alignment(from: metadata)
        let dataOffset = align(headerEnd, to: alignment)
        // A shard with no tensors is a metadata shard and may end exactly at the
        // directory: the Unsloth Qwen3.8 split puts all 67 architecture and
        // tokenizer keys in shard 1, 10,946,618 bytes that are not a multiple of
        // the 32-byte alignment and are followed by no data section at all.
        if !tensors.isEmpty {
            guard dataOffset <= fileSize else {
                throw GGUFError.missingDataSection(
                    headerEnd: headerEnd, alignment: alignment, fileSize: fileSize)
            }
        }
        for tensor in tensors {
            guard tensor.byteCount == 0 || tensor.offset % alignment == 0 else {
                throw GGUFError.tensorOffsetMisaligned(
                    name: tensor.name, offset: tensor.offset, alignment: alignment)
            }
            let start = dataOffset.addingReportingOverflow(tensor.offset)
            let end = start.partialValue.addingReportingOverflow(tensor.byteCount)
            guard !start.overflow, !end.overflow, end.partialValue <= fileSize else {
                throw GGUFError.tensorOutOfRange(
                    name: tensor.name,
                    end: end.overflow ? UInt64.max : end.partialValue,
                    fileSize: fileSize)
            }
        }
        return GGUFShard(
            url: url,
            version: version,
            alignment: alignment,
            metadata: metadata,
            tensors: tensors,
            dataOffset: dataOffset)
    }

    private func align(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }

    private func alignment(from metadata: [String: GGUFValue]) throws -> UInt64 {
        guard let value = metadata["general.alignment"]?.integer else { return 32 }
        guard value >= 1, value <= 1 << 24, value & (value - 1) == 0 else {
            throw GGUFError.invalidAlignment(UInt64(bitPattern: value))
        }
        return UInt64(value)
    }

    private mutating func tensorDimensions(name: String) throws -> [UInt64] {
        let count = try uint32()
        guard count >= 1, count <= 8 else { throw GGUFError.invalidDimensions(name: name) }
        var dimensions: [UInt64] = []
        dimensions.reserveCapacity(Int(count))
        for _ in 0..<count {
            let extent = try uint64()
            guard extent > 0 else { throw GGUFError.invalidDimensions(name: name) }
            dimensions.append(extent)
        }
        return dimensions
    }

    private func geometry(
        name: String, dimensions: [UInt64], type: GGUFType
    ) throws -> (elements: UInt64, bytes: UInt64) {
        var elements: UInt64 = 1
        for extent in dimensions {
            let (product, overflow) = elements.multipliedReportingOverflow(by: extent)
            guard !overflow else { throw GGUFError.invalidDimensions(name: name) }
            elements = product
        }
        guard elements % type.blockSize == 0 else {
            throw GGUFError.tensorNotBlockAligned(
                name: name, elements: elements, block: type.blockSize)
        }
        let blocks = elements / type.blockSize
        let (bytes, overflow) = blocks.multipliedReportingOverflow(by: type.blockBytes)
        guard !overflow else { throw GGUFError.invalidDimensions(name: name) }
        return (elements, bytes)
    }

    private mutating func value(type: UInt32, depth: Int) throws -> GGUFValue {
        switch type {
        case 0: return .uint8(try rawByte())
        case 1: return .int8(Int8(bitPattern: try rawByte()))
        case 2: return .uint16(try uint16())
        case 3: return .int16(Int16(bitPattern: try uint16()))
        case 4: return .uint32(try uint32())
        case 5: return .int32(Int32(bitPattern: try uint32()))
        case 6: return .float32(Float(bitPattern: try uint32()))
        case 7:
            let byte = try rawByte()
            switch byte {
            case 0: return .boolean(false)
            case 1: return .boolean(true)
            default: throw GGUFError.invalidBoolean(byte)
            }
        case 8: return .string(try string(field: "metadata string"))
        case 9:
            guard depth < 8 else { throw GGUFError.arrayNestingTooDeep }
            let elementType = try uint32()
            let count = try uint64()
            guard count <= 10_000_000 else { throw GGUFError.arrayCountTooLarge(count) }
            var values: [GGUFValue] = []
            values.reserveCapacity(min(Int(count), 4096))
            for _ in 0..<count {
                values.append(try value(type: elementType, depth: depth + 1))
            }
            return .array(values)
        case 10: return .uint64(try uint64())
        case 11: return .int64(Int64(bitPattern: try uint64()))
        case 12: return .float64(Double(bitPattern: try uint64()))
        default: throw GGUFError.unknownValueType(type)
        }
    }

    private func remaining() -> Int { data.count - offset }

    private mutating func raw(_ count: Int) throws -> Data.SubSequence {
        guard count >= 0, count <= remaining() else {
            throw GGUFError.truncated(offset: UInt64(offset))
        }
        let range = offset..<(offset + count)
        offset += count
        return data[range]
    }

    private mutating func rawByte() throws -> UInt8 {
        guard offset < data.count else {
            throw GGUFError.truncated(offset: UInt64(offset))
        }
        let byte = data[offset]
        offset += 1
        return byte
    }

    private mutating func uint16() throws -> UInt16 {
        let bytes = try raw(2)
        return UInt16(bytes[bytes.startIndex])
            | (UInt16(bytes[bytes.startIndex + 1]) << 8)
    }

    private mutating func uint32() throws -> UInt32 {
        let bytes = try raw(4)
        var value: UInt32 = 0
        for index in 0..<4 {
            value |= UInt32(bytes[bytes.startIndex + index]) << (8 * index)
        }
        return value
    }

    private mutating func uint64() throws -> UInt64 {
        let bytes = try raw(8)
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(bytes[bytes.startIndex + index]) << (8 * index)
        }
        return value
    }

    private mutating func string(field: String) throws -> String {
        let length = try uint64()
        guard length <= UInt64(remaining()) else {
            throw GGUFError.truncated(offset: UInt64(offset))
        }
        let bytes = try raw(Int(length))
        guard let text = String(bytes: bytes, encoding: .utf8) else {
            throw GGUFError.invalidUTF8(field: field)
        }
        return text
    }
}
