import Foundation

/// A range reader for the checkpoint's safetensors file.
///
/// The 20 GB checkpoint is never loaded whole. The header is a small JSON
/// document after an 8-byte length; each tensor then names a byte range in the
/// data section. This type validates those ranges and reads one tensor at a
/// time, which is the foundation for preserving the NVFP4/MXFP8 payloads.
public struct Inkling2Safetensors: Sendable {
    public enum DType: String, Sendable, Equatable {
        case bf16 = "BF16"
        case f32 = "F32"
        case f8e4m3 = "F8_E4M3"
        case u8 = "U8"

        public var byteWidth: Int {
            switch self {
            case .bf16: 2
            case .f32: 4
            case .f8e4m3, .u8: 1
            }
        }
    }

    public struct Tensor: Sendable, Equatable {
        public let name: String
        public let dtype: DType
        public let shape: [Int]
        public let offset: Int
        public let byteCount: Int
    }

    public enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case invalid(String)
        case missing(String)
        case shortRead(String, expected: Int, got: Int)

        public var description: String {
            switch self {
            case .invalid(let detail): "invalid safetensors: \(detail)"
            case .missing(let name): "safetensors has no tensor \(name)"
            case .shortRead(let name, let expected, let got):
                "short read for \(name): expected \(expected), got \(got)"
            }
        }
    }

    public let tensors: [String: Tensor]
    private let fileURL: URL
    private let dataStart: Int

    public init(url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let lengthData = try handle.read(upToCount: 8) ?? Data()
        guard lengthData.count == 8 else {
            throw Error.invalid("file shorter than the 8-byte header length")
        }
        let headerLength = lengthData.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerLength > 0, headerLength <= 16 * 1024 * 1024 else {
            throw Error.invalid("implausible header length \(headerLength)")
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw Error.invalid("short header read")
        }
        guard let root = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw Error.invalid("header is not a JSON object")
        }
        var tensors: [String: Tensor] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                let dtypeName = entry["dtype"] as? String,
                let dtype = DType(rawValue: dtypeName),
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw Error.invalid("malformed tensor entry \(name)")
            }
            let start = offsets[0]
            let end = offsets[1]
            guard start >= 0, end >= start else {
                throw Error.invalid("bad offsets for \(name)")
            }
            let elements = shape.reduce(1, *)
            let expected = elements * dtype.byteWidth
            guard end - start == expected else {
                throw Error.invalid(
                    "\(name) range \(end - start) does not match \(elements) x \(dtype.byteWidth)")
            }
            tensors[name] = Tensor(
                name: name, dtype: dtype, shape: shape,
                offset: start, byteCount: end - start)
        }
        guard !tensors.isEmpty else { throw Error.invalid("no tensors") }
        self.tensors = tensors
        self.fileURL = url
        self.dataStart = 8 + Int(headerLength)
    }

    public func data(for name: String) throws -> Data {
        guard let tensor = tensors[name] else { throw Error.missing(name) }
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(dataStart + tensor.offset))
        let bytes = try handle.read(upToCount: tensor.byteCount) ?? Data()
        guard bytes.count == tensor.byteCount else {
            throw Error.shortRead(
                name, expected: tensor.byteCount, got: bytes.count)
        }
        return bytes
    }

    public func data(for name: String, offset: Int, count: Int) throws -> Data {
        guard let tensor = tensors[name] else { throw Error.missing(name) }
        guard offset >= 0, count >= 0, offset + count <= tensor.byteCount else {
            throw Error.invalid(
                "\(name) slice [\(offset), \(offset + count)) outside \(tensor.byteCount)")
        }
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(dataStart + tensor.offset + offset))
        let bytes = try handle.read(upToCount: count) ?? Data()
        guard bytes.count == count else {
            throw Error.shortRead(name, expected: count, got: bytes.count)
        }
        return bytes
    }

    public func tensor(_ name: String) -> Tensor? { tensors[name] }
}
