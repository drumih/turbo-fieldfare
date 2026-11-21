import Darwin
import Foundation

/// A range reader for the checkpoint's safetensors file.
///
/// The 20 GB checkpoint is never loaded whole. The header is a small JSON
/// document after an 8-byte length; each tensor then names a byte range in the
/// data section. This type validates those ranges and reads one tensor at a
/// time, which is the foundation for preserving the NVFP4/MXFP8 payloads.
/// unchecked-invariant: `fd` is opened once in `init` and only read through
/// `pread`; `deinit` closes it after the last reference is gone.
public final class Inkling2Safetensors: @unchecked Sendable {
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
    private let dataStart: Int
    private let fd: Int32

    public init(url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw Error.invalid("cannot open \(url.path)")
        }
        self.fd = descriptor
        let lengthData = try Self.pread(descriptor, offset: 0, count: 8)
        guard lengthData.count == 8 else {
            throw Error.invalid("file shorter than the 8-byte header length")
        }
        let headerLength = lengthData.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerLength > 0, headerLength <= 16 * 1024 * 1024 else {
            throw Error.invalid("implausible header length \(headerLength)")
        }
        let headerData = try Self.pread(
            descriptor, offset: 8, count: Int(headerLength))
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
        self.dataStart = 8 + Int(headerLength)
    }

    deinit { close(fd) }

    private static func pread(_ fd: Int32, offset: Int, count: Int) throws -> Data {
        var data = Data(count: count)
        let read = data.withUnsafeMutableBytes { raw -> Int in
            guard count > 0, let base = raw.baseAddress else { return 0 }
            var done = 0
            while done < count {
                let result = Darwin.pread(
                    fd, base.advanced(by: done), count - done,
                    off_t(offset + done))
                if result < 0 {
                    if errno == EINTR { continue }
                    return -1
                }
                if result == 0 { break }
                done += result
            }
            return done
        }
        guard read == count else {
            throw Error.shortRead("<range>", expected: count, got: max(read, 0))
        }
        return data
    }

    public func data(for name: String) throws -> Data {
        guard let tensor = tensors[name] else { throw Error.missing(name) }
        let bytes = try Self.pread(
            fd, offset: dataStart + tensor.offset, count: tensor.byteCount)
        return bytes
    }

    public func data(for name: String, offset: Int, count: Int) throws -> Data {
        guard let tensor = tensors[name] else { throw Error.missing(name) }
        guard offset >= 0, count >= 0, offset + count <= tensor.byteCount else {
            throw Error.invalid(
                "\(name) slice [\(offset), \(offset + count)) outside \(tensor.byteCount)")
        }
        return try Self.pread(
            fd, offset: dataStart + tensor.offset + offset, count: count)
    }

    /// Reads a dense BF16/F32 tensor as floats. Expert tensors use the range
    /// reader instead; this is for the attention, norms, router and embedding.
    public func floatValues(_ name: String) throws -> [Float] {
        guard let tensor = tensors[name] else { throw Error.missing(name) }
        guard tensor.dtype == .bf16 || tensor.dtype == .f32 else {
            throw Error.invalid("\(name) is \(tensor.dtype.rawValue), not BF16/F32")
        }
        let bytes = try data(for: name)
        let count = bytes.count / tensor.dtype.byteWidth
        var values = [Float](repeating: 0, count: count)
        bytes.withUnsafeBytes { raw in
            if tensor.dtype == .f32 {
                for index in 0..<count {
                    values[index] = Float(
                        bitPattern: raw.loadUnaligned(
                            fromByteOffset: index * 4, as: UInt32.self))
                }
            } else {
                for index in 0..<count {
                    let bits = raw.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self)
                    values[index] = Float(bitPattern: UInt32(bits) << 16)
                }
            }
        }
        return values
    }

    public func tensor(_ name: String) -> Tensor? { tensors[name] }
}
