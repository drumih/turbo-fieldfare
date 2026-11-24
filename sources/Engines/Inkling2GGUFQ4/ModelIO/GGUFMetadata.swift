import Foundation
import TinyTitanCore

// The GGUF header's metadata, for the Inkling 2 Q4 plugin's tokenizer (TT-130).
// `GGUFValue` is copied from sources/TinyTitan/Infrastructure/ModelIO/GGUFReader.swift;
// the reader is metadata-only: the install keeps the GGUF header the tokenizer is
// rebuilt from, and no tensor directory is needed for that, so none is parsed.

package enum GGUFValue: Sendable, Equatable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case boolean(Bool)
    case string(String)
    case array([GGUFValue])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    /// Integer value with the width ignored, which is how architecture keys are
    /// read: the checkpoint's `u32` and a hand-written `u64` mean the same thing.
    package var integer: Int64? {
        switch self {
        case .uint8(let v): return Int64(v)
        case .int8(let v): return Int64(v)
        case .uint16(let v): return Int64(v)
        case .int16(let v): return Int64(v)
        case .uint32(let v): return Int64(v)
        case .int32(let v): return Int64(v)
        case .uint64(let v): return v <= UInt64(Int64.max) ? Int64(v) : nil
        case .int64(let v): return v
        default: return nil
        }
    }

    package var double: Double? {
        switch self {
        case .float32(let v): return Double(v)
        case .float64(let v): return v
        default: return integer.map(Double.init)
        }
    }

    package var string: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    package var boolean: Bool? {
        if case .boolean(let value) = self { return value }
        return nil
    }

    package var array: [GGUFValue]? {
        if case .array(let values) = self { return values }
        return nil
    }

    package var strings: [String]? {
        guard let values = array else { return nil }
        var result: [String] = []
        result.reserveCapacity(values.count)
        for value in values {
            guard let text = value.string else { return nil }
            result.append(text)
        }
        return result
    }
}

/// The metadata of a GGUF header: every key/value pair before the tensor
/// directory.
package struct GGUFModel: Sendable {
    package let metadata: [String: GGUFValue]

    /// `<general.architecture>.<key>`, the way architecture keys are spelled.
    package func architectureValue(_ key: String) -> GGUFValue? {
        guard let architecture = metadata["general.architecture"]?.string else { return nil }
        return metadata["\(architecture).\(key)"]
    }
}

package enum GGUFMetadataError: Error, Equatable, CustomStringConvertible {
    case notGGUF
    case truncated
    case unsupportedValueType(UInt32)

    package var description: String {
        switch self {
        case .notGGUF: "not a GGUF header"
        case .truncated: "the GGUF header ends inside its metadata"
        case .unsupportedValueType(let type): "GGUF metadata value type \(type) is unknown"
        }
    }
}

/// Reads a GGUF header's metadata (version 2 or 3, little endian).
package enum GGUFModelReader {
    /// The header file is the install's own `inkling2_gguf_header.bin`;
    /// 256 MiB bounds the read far above any tokenizer's metadata.
    package static func readHeader(at url: URL) throws -> GGUFModel {
        let data = try BoundedMetadataRead.read(fileAt: url, maxBytes: 256 << 20)
        var reader = Cursor(data: data)
        guard try reader.u32() == 0x4655_4747 else { throw GGUFMetadataError.notGGUF }
        _ = try reader.u32()  // version
        _ = try reader.u64()  // tensor count
        let count = try reader.u64()
        var metadata: [String: GGUFValue] = [:]
        for _ in 0..<count {
            let key = try reader.string()
            metadata[key] = try reader.value(type: try reader.u32())
        }
        return GGUFModel(metadata: metadata)
    }

    struct Cursor {
        let data: Data
        var offset = 0

        mutating func bytes(_ count: Int) throws -> Data {
            guard count >= 0, offset + count <= data.count else {
                throw GGUFMetadataError.truncated
            }
            defer { offset += count }
            return data.subdata(in: (data.startIndex + offset)..<(data.startIndex + offset + count))
        }

        mutating func integer<T: FixedWidthInteger>(_: T.Type) throws -> T {
            let raw = try bytes(MemoryLayout<T>.size)
            var value: T = 0
            for (shift, byte) in raw.enumerated() { value |= T(byte) << (8 * shift) }
            return value
        }

        mutating func u32() throws -> UInt32 { try integer(UInt32.self) }
        mutating func u64() throws -> UInt64 { try integer(UInt64.self) }

        mutating func string() throws -> String {
            let length = try u64()
            guard length <= UInt64(data.count) else { throw GGUFMetadataError.truncated }
            guard let text = String(bytes: try bytes(Int(length)), encoding: .utf8) else {
                throw GGUFMetadataError.notGGUF
            }
            return text
        }

        mutating func value(type: UInt32) throws -> GGUFValue {
            switch type {
            case 0: return .uint8(try integer(UInt8.self))
            case 1: return .int8(try integer(Int8.self))
            case 2: return .uint16(try integer(UInt16.self))
            case 3: return .int16(try integer(Int16.self))
            case 4: return .uint32(try u32())
            case 5: return .int32(try integer(Int32.self))
            case 6: return .float32(Float(bitPattern: try u32()))
            case 7: return .boolean(try integer(UInt8.self) != 0)
            case 8: return .string(try string())
            case 9:
                let element = try u32()
                let count = try u64()
                guard count <= UInt64(data.count) else { throw GGUFMetadataError.truncated }
                return .array(try (0..<count).map { _ in try value(type: element) })
            case 10: return .uint64(try u64())
            case 11: return .int64(try integer(Int64.self))
            case 12: return .float64(Double(bitPattern: try u64()))
            default: throw GGUFMetadataError.unsupportedValueType(type)
            }
        }
    }
}
