import Foundation

/// The header of a GGUF file — metadata and tensor table — read from its first
/// bytes, so an installer can plan a conversion before fetching the data
/// section (TT-121).
///
/// It keeps what a planner needs: every scalar and string, numeric arrays up
/// to `maximumKeptArrayCount` elements, and only the length of longer arrays
/// (a 200K-entry tokenizer vocabulary is carried into an install by copying the
/// header bytes, not by parsing it here). Tensor offsets are absolute, already
/// including the data section's start. A header longer than the bytes given
/// throws `.needMoreBytes` with a lower bound to fetch, so a remote reader can
/// widen its range and retry.
package struct GGUFHeaderV3: Sendable {
    package enum Value: Sendable, Equatable {
        case integer(Int64)
        case unsigned(UInt64)
        case real(Double)
        case boolean(Bool)
        case string(String)
        case numbers([Double])
        /// An array kept only as its length.
        case longArray(count: UInt64)
    }

    package struct Tensor: Sendable, Equatable {
        package let name: String
        /// ggml order: `dimensions[0]` is the row length (fastest varying).
        package let dimensions: [UInt64]
        package let ggmlType: UInt32
        /// Absolute file offset of the tensor's first byte.
        package let absoluteOffset: UInt64
    }

    package enum ReadError: Error, Equatable, CustomStringConvertible {
        case needMoreBytes(atLeast: Int)
        case invalid(String)

        package var description: String {
            switch self {
            case .needMoreBytes(let count):
                "GGUF header continues past the bytes read; read at least \(count)"
            case .invalid(let reason): "GGUF header: \(reason)"
            }
        }
    }

    package static let maximumKeptArrayCount: UInt64 = 65_536

    package let version: UInt32
    package let metadata: [String: Value]
    package let tensors: [Tensor]
    package let alignment: UInt64
    /// Absolute offset of the data section: the header's end, aligned.
    package let dataStart: UInt64

    package init(bytes: [UInt8]) throws {
        var cursor = Cursor(bytes: bytes)
        guard Array(try cursor.raw(4)) == Array("GGUF".utf8) else {
            throw ReadError.invalid("bad magic")
        }
        version = try cursor.u32()
        guard version == 2 || version == 3 else { throw ReadError.invalid("version \(version)") }
        let tensorCount = try cursor.u64()
        let keyCount = try cursor.u64()
        guard tensorCount < 1 << 20, keyCount < 1 << 20 else {
            throw ReadError.invalid("implausible counts")
        }
        var metadata: [String: Value] = [:]
        for _ in 0..<keyCount {
            let key = try cursor.string()
            metadata[key] = try cursor.value(type: try cursor.u32())
        }
        var pending: [(String, [UInt64], UInt32, UInt64)] = []
        for _ in 0..<tensorCount {
            let name = try cursor.string()
            let rank = try cursor.u32()
            guard rank >= 1, rank <= 4 else { throw ReadError.invalid("\(name): rank \(rank)") }
            var dimensions: [UInt64] = []
            for _ in 0..<rank { dimensions.append(try cursor.u64()) }
            pending.append((name, dimensions, try cursor.u32(), try cursor.u64()))
        }
        var alignment: UInt64 = 32
        switch metadata["general.alignment"] {
        case .unsigned(let value)?: alignment = value
        case .integer(let value)? where value > 0: alignment = UInt64(value)
        default: break
        }
        guard alignment > 0, alignment & (alignment - 1) == 0 else {
            throw ReadError.invalid("alignment \(alignment)")
        }
        let headerEnd = UInt64(cursor.position)
        let start = (headerEnd + alignment - 1) / alignment * alignment
        dataStart = start
        self.alignment = alignment
        self.metadata = metadata
        tensors = try pending.map { name, dimensions, type, offset in
            guard offset % alignment == 0 else {
                throw ReadError.invalid("\(name): unaligned offset")
            }
            let (absolute, overflow) = offset.addingReportingOverflow(start)
            guard !overflow else { throw ReadError.invalid("\(name): offset overflows") }
            return Tensor(
                name: name, dimensions: dimensions, ggmlType: type, absoluteOffset: absolute)
        }
    }

    /// A metadata number, whatever integer or float type the file used.
    package func number(_ key: String) -> Double? {
        switch metadata[key] {
        case .integer(let value)?: Double(value)
        case .unsigned(let value)?: Double(value)
        case .real(let value)?: value
        case .boolean(let value)?: value ? 1 : 0
        default: nil
        }
    }

    package func string(_ key: String) -> String? {
        if case .string(let value)? = metadata[key] { return value }
        return nil
    }

    package func numbers(_ key: String) -> [Double]? {
        if case .numbers(let value)? = metadata[key] { return value }
        return nil
    }
}

private struct Cursor {
    let bytes: [UInt8]
    var position = 0

    mutating func raw(_ count: Int) throws -> ArraySlice<UInt8> {
        guard count >= 0, position + count <= bytes.count else {
            throw GGUFHeaderV3.ReadError.needMoreBytes(
                atLeast: max(bytes.count * 2, position + count))
        }
        defer { position += count }
        return bytes[position..<position + count]
    }

    mutating func u32() throws -> UInt32 {
        try raw(4).reversed().reduce(0) { $0 << 8 | UInt32($1) }
    }

    mutating func u64() throws -> UInt64 {
        try raw(8).reversed().reduce(0) { $0 << 8 | UInt64($1) }
    }

    mutating func string() throws -> String {
        let length = try u64()
        guard length < 1 << 30 else {
            throw GGUFHeaderV3.ReadError.invalid("string length \(length)")
        }
        guard let text = String(bytes: try raw(Int(length)), encoding: .utf8) else {
            throw GGUFHeaderV3.ReadError.invalid("string is not UTF-8")
        }
        return text
    }

    /// Reads one value of ggml metadata `type`; arrays are kept as numbers or
    /// as a length, never recursed into storage.
    mutating func value(type: UInt32) throws -> GGUFHeaderV3.Value {
        switch type {
        case 0: return .unsigned(UInt64((try raw(1).first ?? 0)))
        case 1: return .integer(Int64(Int8(bitPattern: (try raw(1).first ?? 0))))
        case 2:
            return .unsigned(
                UInt64(try raw(2).reversed().reduce(UInt16(0)) { $0 << 8 | UInt16($1) }))
        case 3:
            let raw16 = try raw(2).reversed().reduce(UInt16(0)) { $0 << 8 | UInt16($1) }
            return .integer(Int64(Int16(bitPattern: raw16)))
        case 4: return .unsigned(UInt64(try u32()))
        case 5: return .integer(Int64(Int32(bitPattern: try u32())))
        case 6: return .real(Double(Float(bitPattern: try u32())))
        case 7: return .boolean((try raw(1).first ?? 0) != 0)
        case 8: return .string(try string())
        case 9: return try array()
        case 10: return .unsigned(try u64())
        case 11: return .integer(Int64(bitPattern: try u64()))
        case 12: return .real(Double(bitPattern: try u64()))
        default: throw GGUFHeaderV3.ReadError.invalid("metadata type \(type)")
        }
    }

    private mutating func array() throws -> GGUFHeaderV3.Value {
        let elementType = try u32()
        let count = try u64()
        guard elementType != 9 else { throw GGUFHeaderV3.ReadError.invalid("nested array") }
        let keep = count <= GGUFHeaderV3.maximumKeptArrayCount && elementType != 8
        var numbers: [Double] = []
        for _ in 0..<count {
            let element = try value(type: elementType)
            guard keep else { continue }
            switch element {
            case .integer(let v): numbers.append(Double(v))
            case .unsigned(let v): numbers.append(Double(v))
            case .real(let v): numbers.append(v)
            case .boolean(let v): numbers.append(v ? 1 : 0)
            default: break
            }
        }
        return keep ? .numbers(numbers) : .longArray(count: count)
    }
}
