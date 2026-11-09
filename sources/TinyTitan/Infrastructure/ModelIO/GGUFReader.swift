import Darwin
import Foundation

/// Failure modes of the GGUF reader.
///
/// One namespace for the whole format, for the same reason `ModelError` is one:
/// a caller deciding whether a file is a usable model should not have to match
/// three unrelated error types to say "no". Every case names the tensor, key or
/// offset it refused, because the reader's job is to make a malformed file
/// diagnosable rather than to produce a plausible-looking directory.
package enum GGUFError: Error, Equatable, CustomStringConvertible {
    case unreadable(path: String, errno: Int32)
    case statFailed(path: String, errno: Int32)
    case notRegularFile(path: String)
    case tooSmall(path: String, bytes: UInt64)
    case badMagic(String)
    case unsupportedVersion(UInt32)
    case truncated(offset: UInt64)
    case directoryOverBound(cap: UInt64)
    case metadataCountTooLarge(UInt64)
    case tensorCountTooLarge(UInt64)
    case duplicateMetadataKey(String)
    case duplicateTensorName(String)
    case invalidUTF8(field: String)
    case unknownValueType(UInt32)
    case arrayNestingTooDeep
    case arrayCountTooLarge(UInt64)
    case invalidBoolean(UInt8)
    case invalidDimensions(name: String)
    case unknownTensorType(name: String, type: UInt32)
    case tensorNotBlockAligned(name: String, elements: UInt64, block: UInt64)
    case tensorOffsetMisaligned(name: String, offset: UInt64, alignment: UInt64)
    case tensorOutOfRange(name: String, end: UInt64, fileSize: UInt64)
    case invalidAlignment(UInt64)
    case missingDataSection(headerEnd: UInt64, alignment: UInt64, fileSize: UInt64)
    case ambiguousModelDirectory(String)
    case noGGUFFiles(String)
    case splitMismatch(detail: String)

    package var description: String {
        switch self {
        case .unreadable(let path, let code):
            return "cannot open GGUF file \(path): errno \(code)"
        case .statFailed(let path, let code):
            return "cannot stat GGUF file \(path): errno \(code)"
        case .notRegularFile(let path):
            return "\(path) is not a regular file"
        case .tooSmall(let path, let bytes):
            return "\(path) is \(bytes) bytes, shorter than a GGUF header"
        case .badMagic(let magic):
            return "not a GGUF file: magic is \(magic), expected GGUF"
        case .unsupportedVersion(let version):
            return "GGUF version \(version) is not supported (this reader takes 2 and 3)"
        case .truncated(let offset):
            return "GGUF directory is truncated at byte \(offset)"
        case .directoryOverBound(let cap):
            return "GGUF directory exceeds the \(cap)-byte bound"
        case .metadataCountTooLarge(let count):
            return "GGUF metadata count \(count) is implausible"
        case .tensorCountTooLarge(let count):
            return "GGUF tensor count \(count) is implausible"
        case .duplicateMetadataKey(let key):
            return "GGUF metadata repeats key \(key)"
        case .duplicateTensorName(let name):
            return "GGUF tensor \(name) appears twice"
        case .invalidUTF8(let field):
            return "GGUF \(field) is not valid UTF-8"
        case .unknownValueType(let type):
            return "GGUF metadata value type \(type) is unknown"
        case .arrayNestingTooDeep:
            return "GGUF metadata arrays nest deeper than 8 levels"
        case .arrayCountTooLarge(let count):
            return "GGUF metadata array of \(count) elements is implausible"
        case .invalidBoolean(let value):
            return "GGUF boolean is \(value), expected 0 or 1"
        case .invalidDimensions(let name):
            return "GGUF tensor \(name) has a zero or overflowing dimension"
        case .unknownTensorType(let name, let type):
            return "GGUF tensor \(name) uses unknown GGML type \(type)"
        case .tensorNotBlockAligned(let name, let elements, let block):
            return
                "GGUF tensor \(name) has \(elements) elements, not a multiple of its block \(block)"
        case .tensorOffsetMisaligned(let name, let offset, let alignment):
            return "GGUF tensor \(name) starts at \(offset), not aligned to \(alignment)"
        case .tensorOutOfRange(let name, let end, let fileSize):
            return "GGUF tensor \(name) ends at byte \(end), past the \(fileSize)-byte file"
        case .invalidAlignment(let value):
            return "GGUF general.alignment = \(value) is not a power of two"
        case .missingDataSection(let headerEnd, let alignment, let fileSize):
            return
                "GGUF data section starts at \(headerEnd) padded to \(alignment), past the \(fileSize)-byte file"
        case .ambiguousModelDirectory(let path):
            return "\(path) holds no unambiguous GGUF model"
        case .noGGUFFiles(let path):
            return "\(path) contains no .gguf files"
        case .splitMismatch(let detail):
            return "GGUF split set is inconsistent: \(detail)"
        }
    }
}

/// A GGML tensor type, with the block geometry its byte count needs.
///
/// These ids and block sizes are GGML's, not ours; the table was checked against
/// `ggml.h` / `ggml-common.h` and every type this tree's Unsloth checkpoint
/// actually uses was confirmed against the file's own tensor offsets (a derived
/// bytes-per-element of 0.28125 identified `q2_0` before the header was read).
/// A type missing from this enum is **refused**, never assumed: a guessed block
/// size yields offsets that look sane and are wrong, which is the failure the
/// directory exists to catch.
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

/// One tensor's place in the shard it was declared by.
package struct GGUFTensor: Sendable, Equatable {
    package let name: String
    /// GGML extents, fastest-varying first. `dimensions[0]` is the row width.
    package let dimensions: [UInt64]
    package let type: GGUFType
    /// Byte offset from the shard's data section, which starts at `dataOffset`.
    package let offset: UInt64
    package let elementCount: UInt64
    package let byteCount: UInt64
}

/// The parsed header and directory of one GGUF shard.
package struct GGUFShard: Sendable {
    package let url: URL
    package let version: UInt32
    package let alignment: UInt64
    package let metadata: [String: GGUFValue]
    package let tensors: [GGUFTensor]
    /// Absolute file offset of tensor 0 relative to the data section.
    package let dataOffset: UInt64

    package func tensor(named name: String) -> GGUFTensor? {
        tensors.first { $0.name == name }
    }
}

/// Bounded, validated GGUF v2/v3 directory reader.
package enum GGUFReader {
    /// Cap on the header + metadata + tensor directory. The tokenizer usually
    /// dominates it: the Unsloth Qwen3.8 shard holding the vocabulary needs
    /// 10.9 MB for 248,320 tokens and 247,587 merges, and every split shard
    /// after it needs less than 77 KB. 64 MiB is the ceiling the manifest
    /// reader uses, and it is a *parse* bound, not an allocation of the whole
    /// file: at most `cap` bytes are read, never the 83 GB behind them.
    package static let defaultMaxDirectoryBytes: UInt64 = 64 * 1024 * 1024

    private static let initialWindow: UInt64 = 1 << 20

    /// Read one shard, growing the read window from 1 MiB by doubling so a
    /// typical directory costs one small page-cache read rather than the cap.
    package static func read(
        shardAt url: URL,
        maxDirectoryBytes: UInt64 = defaultMaxDirectoryBytes
    ) throws -> GGUFShard {
        let path = url.path
        let fd = open(path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw GGUFError.unreadable(path: path, errno: errno) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw GGUFError.statFailed(path: path, errno: errno)
        }
        guard (st.st_mode & S_IFMT) == S_IFREG else {
            throw GGUFError.notRegularFile(path: path)
        }
        let fileSize = UInt64(st.st_size)
        guard fileSize >= 24 else { throw GGUFError.tooSmall(path: path, bytes: fileSize) }

        var window = min(fileSize, initialWindow, maxDirectoryBytes)
        guard window >= 24 else { throw GGUFError.directoryOverBound(cap: maxDirectoryBytes) }
        while true {
            let data = try readExact(fd: fd, path: path, count: Int(window))
            do {
                var parser = GGUFParser(url: url, data: data, fileSize: fileSize)
                return try parser.parse()
            } catch GGUFError.truncated(let offset) {
                if window >= fileSize { throw GGUFError.truncated(offset: offset) }
                guard window < maxDirectoryBytes else {
                    throw GGUFError.directoryOverBound(cap: maxDirectoryBytes)
                }
                window = min(fileSize, window * 2, maxDirectoryBytes)
            }
        }
    }

    private static func readExact(fd: Int32, path: String, count: Int) throws -> Data {
        var data = Data(count: count)
        var total = 0
        try data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            while total < count {
                let n = pread(fd, base.advanced(by: total), count - total, off_t(total))
                if n < 0 {
                    throw GGUFError.unreadable(path: path, errno: errno)
                }
                if n == 0 { throw GGUFError.truncated(offset: UInt64(total)) }
                total += n
            }
        }
        return data
    }
}

/// Cursor over the bounded window. Every read is bounds-checked; the parser
/// never indexes past the bytes it actually obtained.
