import Foundation

package struct GTurboResidentIndexHeaderV1: Equatable, Sendable {
    package let indexSize: UInt64
    package let residentSize: UInt64
    package let entryCount: UInt64

    package init(indexSize: UInt64, residentSize: UInt64, entryCount: UInt64) {
        self.indexSize = indexSize
        self.residentSize = residentSize
        self.entryCount = entryCount
    }
}

package struct GTurboResidentIndexEntryV1: Equatable, Sendable {
    package let name: String
    package let dtype: UInt8
    package let fileOffset: UInt64
    package let sizeBytes: UInt64
    package let shape: [UInt32]
    package let scaleOffset: UInt64
    package let scaleSize: UInt64
    package let biasOffset: UInt64
    package let biasSize: UInt64

    package init(name: String, dtype: UInt8, fileOffset: UInt64, sizeBytes: UInt64,
                 shape: [UInt32], scaleOffset: UInt64, scaleSize: UInt64,
                 biasOffset: UInt64, biasSize: UInt64) {
        self.name = name
        self.dtype = dtype
        self.fileOffset = fileOffset
        self.sizeBytes = sizeBytes
        self.shape = shape
        self.scaleOffset = scaleOffset
        self.scaleSize = scaleSize
        self.biasOffset = biasOffset
        self.biasSize = biasSize
    }
}

package enum GTurboResidentIndexCodec {
    package static func decodeHeader(_ bytes: UnsafeRawBufferPointer) throws -> GTurboResidentIndexHeaderV1 {
        guard bytes.count >= GTurboFormatV1.residentHeaderBytes else {
            throw GTurboFormatError.truncated(field: "resident.header")
        }
        let base = bytes.baseAddress!
        return GTurboResidentIndexHeaderV1(
            indexSize: readU64(base, 0),
            residentSize: readU64(base, 8),
            entryCount: readU64(base, 16))
    }

    package static func decodeRegion(_ bytes: UnsafeRawBufferPointer,
                                     header: GTurboResidentIndexHeaderV1) throws -> [GTurboResidentIndexEntryV1] {
        guard header.indexSize <= GTurboFormatV1.residentIndexMaxBytes else {
            throw GTurboFormatError.invalid(
                field: "resident.indexSize", reason: "exceeds v1 metadata cap")
        }
        guard header.indexSize <= UInt64(bytes.count),
              header.indexSize >= UInt64(GTurboFormatV1.residentHeaderBytes),
              header.indexSize % GTurboFormatV1.alignmentBytes == 0 else {
            throw GTurboFormatError.truncated(field: "resident.index")
        }
        let tableBytes = try gturboCheckedMultiply(header.entryCount,
                                                   UInt64(GTurboFormatV1.residentEntryBytes),
                                                   field: "resident.entryTable")
        let tableEnd = try gturboCheckedAdd(UInt64(GTurboFormatV1.residentHeaderBytes),
                                            tableBytes, field: "resident.entryTable")
        guard tableEnd <= header.indexSize, header.entryCount <= UInt64(Int.max) else {
            throw GTurboFormatError.invalid(field: "resident.entryTable", reason: "outside index")
        }
        let residentEnd = try gturboCheckedAdd(header.indexSize, header.residentSize,
                                               field: "resident.payload")
        let base = bytes.baseAddress!
        var result: [GTurboResidentIndexEntryV1] = []
        result.reserveCapacity(Int(header.entryCount))
        var names = Set<String>()
        var payloadRanges: [(start: UInt64, end: UInt64, field: String)] = []
        payloadRanges.reserveCapacity(Int(header.entryCount) * 3)
        for index in 0..<Int(header.entryCount) {
            let offset = GTurboFormatV1.residentHeaderBytes + index * GTurboFormatV1.residentEntryBytes
            let entry = base.advanced(by: offset)
            let nameOffset = UInt64(readU32(entry, 0))
            let nameLength = UInt64(readU16(entry, 4))
            guard readU8(entry, 7) == 0 else {
                throw GTurboFormatError.invalid(field: "resident.entries[\(index)].reserved",
                                                reason: "must be zero")
            }
            let nameEnd = try gturboCheckedAdd(nameOffset, nameLength,
                                               field: "resident.entries[\(index)].name")
            guard nameOffset >= tableEnd, nameEnd <= header.indexSize,
                  nameLength <= UInt64(Int.max) else {
                throw GTurboFormatError.invalid(field: "resident.entries[\(index)].name",
                                                reason: "range outside string table")
            }
            let nameBytes = UnsafeRawBufferPointer(start: base.advanced(by: Int(nameOffset)),
                                                   count: Int(nameLength))
            guard let name = String(bytes: nameBytes, encoding: .utf8),
                  !name.isEmpty, names.insert(name).inserted else {
                throw GTurboFormatError.invalid(field: "resident.entries[\(index)].name",
                                                reason: "invalid UTF-8 or duplicate")
            }
            let dtype = readU8(entry, 6)
            guard GTurboFormatV1.DType(rawValue: dtype) != nil else {
                throw GTurboFormatError.invalid(field: "resident.entries[\(index)].dtype",
                                                reason: "unknown dtype")
            }
            let fileOffset = readU64(entry, 8)
            let sizeBytes = readU64(entry, 16)
            let scaleOffset = readU64(entry, 40)
            let scaleSize = readU64(entry, 48)
            let biasOffset = readU64(entry, 56)
            let biasSize = readU64(entry, 64)
            let shape = [readU32(entry, 24), readU32(entry, 28),
                         readU32(entry, 32), readU32(entry, 36)]
            try validateShape(shape, index: index)
            try validatePrimaryPayloadRange(offset: fileOffset, size: sizeBytes,
                                            header: header, residentEnd: residentEnd,
                                            field: "resident.entries[\(index)].weights")
            try validateOptionalPayloadRange(offset: scaleOffset, size: scaleSize,
                                             header: header, residentEnd: residentEnd,
                                             field: "resident.entries[\(index)].scales")
            try validateOptionalPayloadRange(offset: biasOffset, size: biasSize,
                                             header: header, residentEnd: residentEnd,
                                             field: "resident.entries[\(index)].biases")
            payloadRanges.append((fileOffset, fileOffset + sizeBytes,
                                  "resident.entries[\(index)].weights"))
            if scaleSize > 0 {
                payloadRanges.append((scaleOffset, scaleOffset + scaleSize,
                                      "resident.entries[\(index)].scales"))
            }
            if biasSize > 0 {
                payloadRanges.append((biasOffset, biasOffset + biasSize,
                                      "resident.entries[\(index)].biases"))
            }
            result.append(GTurboResidentIndexEntryV1(
                name: name, dtype: dtype, fileOffset: fileOffset, sizeBytes: sizeBytes,
                shape: shape,
                scaleOffset: scaleOffset, scaleSize: scaleSize,
                biasOffset: biasOffset, biasSize: biasSize))
        }
        payloadRanges.sort {
            $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
        }
        if payloadRanges.count > 1 {
            for index in 1..<payloadRanges.count
                where payloadRanges[index].start < payloadRanges[index - 1].end {
                throw GTurboFormatError.invalid(
                    field: payloadRanges[index].field,
                    reason: "overlaps \(payloadRanges[index - 1].field)")
            }
        }
        return result
    }

    package static func writeHeader(into buffer: UnsafeMutableRawPointer,
                                    header: GTurboResidentIndexHeaderV1) {
        writeU64(buffer, 0, header.indexSize)
        writeU64(buffer, 8, header.residentSize)
        writeU64(buffer, 16, header.entryCount)
    }

    package static func writeEntry(into buffer: UnsafeMutableRawPointer,
                                   entry: GTurboResidentIndexEntryV1,
                                   nameOffset: UInt32) {
        precondition(entry.shape.count == 4)
        writeU32(buffer, 0, nameOffset)
        writeU16(buffer, 4, UInt16(entry.name.utf8.count))
        writeU8(buffer, 6, entry.dtype)
        writeU8(buffer, 7, 0)
        writeU64(buffer, 8, entry.fileOffset)
        writeU64(buffer, 16, entry.sizeBytes)
        for i in 0..<4 { writeU32(buffer, 24 + i * 4, entry.shape[i]) }
        writeU64(buffer, 40, entry.scaleOffset)
        writeU64(buffer, 48, entry.scaleSize)
        writeU64(buffer, 56, entry.biasOffset)
        writeU64(buffer, 64, entry.biasSize)
    }

    private static func validatePrimaryPayloadRange(offset: UInt64, size: UInt64,
                                                    header: GTurboResidentIndexHeaderV1,
                                                    residentEnd: UInt64,
                                                    field: String) throws {
        guard size > 0 else {
            throw GTurboFormatError.invalid(field: field, reason: "primary payload is empty")
        }
        try validateContainedPayloadRange(offset: offset, size: size,
                                          header: header, residentEnd: residentEnd,
                                          field: field)
    }

    private static func validateOptionalPayloadRange(offset: UInt64, size: UInt64,
                                                     header: GTurboResidentIndexHeaderV1,
                                                     residentEnd: UInt64,
                                                     field: String) throws {
        if size == 0 {
            guard offset == 0 else {
                throw GTurboFormatError.invalid(field: field, reason: "absent payload offset must be zero")
            }
            return
        }
        try validateContainedPayloadRange(offset: offset, size: size,
                                          header: header, residentEnd: residentEnd,
                                          field: field)
    }

    private static func validateContainedPayloadRange(offset: UInt64, size: UInt64,
                                                      header: GTurboResidentIndexHeaderV1,
                                                      residentEnd: UInt64,
                                                      field: String) throws {
        let end = try gturboCheckedAdd(offset, size, field: field)
        guard offset >= header.indexSize, end <= residentEnd else {
            throw GTurboFormatError.invalid(field: field, reason: "range outside resident payload")
        }
    }

    private static func validateShape(_ shape: [UInt32], index: Int) throws {
        guard shape.count == 4, shape[0] > 0 else {
            throw GTurboFormatError.invalid(
                field: "resident.entries[\(index)].shape", reason: "first dimension must be positive")
        }
        var sawZero = false
        for dimension in shape {
            if dimension == 0 {
                sawZero = true
            } else if sawZero {
                throw GTurboFormatError.invalid(
                    field: "resident.entries[\(index)].shape",
                    reason: "zero dimensions must be trailing")
            }
        }
    }

    @inline(__always) private static func readU8(_ p: UnsafeRawPointer, _ o: Int) -> UInt8 {
        p.advanced(by: o).load(as: UInt8.self)
    }
    @inline(__always) private static func readU16(_ p: UnsafeRawPointer, _ o: Int) -> UInt16 {
        let q = p.advanced(by: o).assumingMemoryBound(to: UInt8.self)
        return UInt16(q[0]) | UInt16(q[1]) << 8
    }
    @inline(__always) private static func readU32(_ p: UnsafeRawPointer, _ o: Int) -> UInt32 {
        let q = p.advanced(by: o).assumingMemoryBound(to: UInt8.self)
        var value: UInt32 = 0
        for i in (0..<4).reversed() { value = value << 8 | UInt32(q[i]) }
        return value
    }
    @inline(__always) private static func readU64(_ p: UnsafeRawPointer, _ o: Int) -> UInt64 {
        let q = p.advanced(by: o).assumingMemoryBound(to: UInt8.self)
        var value: UInt64 = 0
        for i in (0..<8).reversed() { value = value << 8 | UInt64(q[i]) }
        return value
    }
    @inline(__always) private static func writeU8(_ p: UnsafeMutableRawPointer, _ o: Int, _ v: UInt8) {
        p.advanced(by: o).storeBytes(of: v, as: UInt8.self)
    }
    @inline(__always) private static func writeU16(_ p: UnsafeMutableRawPointer, _ o: Int, _ v: UInt16) {
        var x = v.littleEndian
        _ = withUnsafeBytes(of: &x) { memcpy(p.advanced(by: o), $0.baseAddress!, 2) }
    }
    @inline(__always) private static func writeU32(_ p: UnsafeMutableRawPointer, _ o: Int, _ v: UInt32) {
        var x = v.littleEndian
        _ = withUnsafeBytes(of: &x) { memcpy(p.advanced(by: o), $0.baseAddress!, 4) }
    }
    @inline(__always) private static func writeU64(_ p: UnsafeMutableRawPointer, _ o: Int, _ v: UInt64) {
        var x = v.littleEndian
        _ = withUnsafeBytes(of: &x) { memcpy(p.advanced(by: o), $0.baseAddress!, 8) }
    }
}
