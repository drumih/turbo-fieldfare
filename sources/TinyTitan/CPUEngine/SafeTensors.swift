import Foundation
import TinyTitanCore

/// A safetensors file, memory-mapped.
///
/// TinyTitan serves its own models from the SSDAI format, which is built for
/// streaming experts off SSD. The side-engine's model is a different
/// problem: two gigabytes that stay resident, produced by this project's own
/// converter, and read start to finish for every token. Mapping the
/// safetensors file the converter already writes is the whole loader, and it
/// avoids a second conversion step whose only job would be to restate the
/// same bytes.
///
/// Mapping rather than reading matters here. The file is larger than the
/// side-engine's whole memory budget would be if it were copied, and the
/// kernel's page cache is exactly the right owner of pages that are read
/// sequentially and never written.
public struct SafeTensorsFile: Sendable {

    public struct Entry: Sendable, Equatable {
        public let dtype: String
        public let shape: [Int]
        /// Offsets relative to the start of the payload, not the file.
        public let start: Int
        public let end: Int
        /// The product of `shape`, computed once while the header is validated.
        ///
        /// This used to be `shape.reduce(1, *)` on every access: a trapping
        /// multiplication over values taken straight out of the file, so a
        /// header declaring an absurd shape parsed cleanly and aborted the
        /// process the first time anything asked -- which makes it look like a
        /// crash in whatever happened to ask first.
        public let count: Int
    }

    public enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case malformed(String)
        case missing(String)
        case unsupported(dtype: String, name: String)

        public var description: String {
            switch self {
            case .unreadable(let path): "cannot open \(path)"
            case .malformed(let detail): "malformed safetensors: \(detail)"
            case .missing(let name): "no tensor named \(name)"
            case .unsupported(let dtype, let name): "unsupported dtype \(dtype) for \(name)"
            }
        }
    }

    public let url: URL
    public let entries: [String: Entry]
    private let mapping: Mapping
    private let payload: Int

    /// The mapping's lifetime, so the pointer stays valid for as long as any
    /// tensor view taken from it.
    ///
    /// unchecked-invariant: `base` and `length` are `let`, set once before
    /// the instance escapes, and describe a read-only `MAP_PRIVATE` mapping
    /// that nothing ever writes to. The only mutation in the type's life is
    /// `munmap` in `deinit`, which by definition runs after the last
    /// reference is gone.
    private final class Mapping: @unchecked Sendable {
        let base: UnsafeRawPointer
        let length: Int
        init(base: UnsafeRawPointer, length: Int) {
            self.base = base
            self.length = length
        }
        deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }
    }

    public init(url: URL) throws {
        // O_NOFOLLOW because a model install is copied in from another machine,
        // which is the attacker model `SSDAIModelDirectory` exists for. AUD-110
        // and AUD-144 closed this same opener in the installer and the journals;
        // this is the sibling those fixes did not reach, where a link planted at
        // `model.safetensors` mapped a file outside the install as its weights.
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.unreadable(url.path) }
        try self.init(url: url, descriptor: descriptor)
    }

    /// Map a shard the caller has already opened through a validated directory
    /// descriptor, so the bytes mapped are the bytes that were checked -- an
    /// open-by-path after the check would leave the window open again.
    /// Takes ownership of `descriptor` and closes it, including on failure.
    init(url: URL, descriptor: Int32) throws {
        self.url = url
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 8 else {
            throw Failure.unreadable(url.path)
        }
        let length = Int(status.st_size)
        guard let raw = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, 0),
            raw != MAP_FAILED
        else {
            throw Failure.unreadable(url.path)
        }
        // WILLNEED, emphatically not SEQUENTIAL. The access pattern *is*
        // sequential, but SEQUENTIAL also tells the kernel it may free pages
        // once they are behind the read point -- and this file is read from
        // end to end again for the very next token, milliseconds later. With
        // SEQUENTIAL the engine re-faulted the whole model every token and
        // ran at a third of its speed; the right hint is that all of it will
        // be wanted, which on a machine with room keeps it resident.
        madvise(raw, length, MADV_WILLNEED)
        let base = UnsafeRawPointer(raw)
        mapping = Mapping(base: base, length: length)

        let headerLength = Int(base.loadUnaligned(as: UInt64.self))
        guard headerLength > 0, 8 + headerLength <= length else {
            throw Failure.malformed("header length \(headerLength) does not fit \(length)")
        }
        payload = 8 + headerLength
        let json = Data(bytes: base.advanced(by: 8), count: headerLength)
        guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.malformed("header is not an object")
        }
        var parsed: [String: Entry] = [:]
        parsed.reserveCapacity(object.count)
        for (name, value) in object where name != "__metadata__" {
            guard let fields = value as? [String: Any],
                let dtype = fields["dtype"] as? String,
                let shape = fields["shape"] as? [Int],
                let offsets = fields["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw Failure.malformed("entry \(name)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0],
                payload + offsets[1] <= length
            else {
                throw Failure.malformed("entry \(name) runs past the file")
            }
            // The shape is validated with reporting arithmetic before anything
            // can multiply it: a dimension from the file must not be negative
            // and the product must not overflow `Int`, or `count` would trap.
            // Checked here so the failure names the tensor at load.
            var count = 1
            for dimension in shape {
                guard dimension >= 0 else {
                    throw Failure.malformed("entry \(name) has a negative shape \(shape)")
                }
                let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
                guard !overflow else {
                    throw Failure.malformed("entry \(name) has an unusable shape \(shape)")
                }
                count = product
            }
            parsed[name] = Entry(
                dtype: dtype, shape: shape,
                start: offsets[0], end: offsets[1], count: count)
        }
        entries = parsed
    }

    /// Pull the whole file into memory and keep it there.
    ///
    /// Mapping alone leaves residency to the page cache, which is usually
    /// right — this file is read end to end every token, so it stays warm on
    /// its own. It is not right when something else needs the memory: a 35B
    /// streaming experts off SSD will evict a 2 GB side model between
    /// requests, and the first token after that pays for the whole thing
    /// again.
    ///
    /// So a model served CPU-only is faulted in once, up front, and the cost
    /// is paid where it is visible. `MADV_WILLNEED` asks the kernel to read
    /// ahead; the touch loop is what actually makes the pages resident,
    /// because the advice is a hint and this is not.
    ///
    /// Returns the bytes touched, so a caller can say what it did.
    @discardableResult
    public func makeResident() -> Int {
        madvise(
            UnsafeMutableRawPointer(mutating: mapping.base),
            mapping.length, MADV_WILLNEED)
        let pageSize = Int(getpagesize())
        var checksum: UInt64 = 0
        var offset = 0
        while offset < mapping.length {
            checksum &+= UInt64(mapping.base.load(fromByteOffset: offset, as: UInt8.self))
            offset += pageSize
        }
        // The sum is never used; it exists so the reads cannot be optimized
        // away, which would leave the pages exactly as cold as before.
        Self.residencyChecksum = checksum
        return mapping.length
    }

    /// Written only by `makeResident`, and never read for meaning.
    nonisolated(unsafe) private static var residencyChecksum: UInt64 = 0

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw Failure.missing(name) }
        return entry
    }

    /// A raw view of one tensor. Valid for the life of this file.
    public func bytes(_ name: String) throws -> UnsafeRawBufferPointer {
        let entry = try entry(name)
        return UnsafeRawBufferPointer(
            start: mapping.base.advanced(by: payload + entry.start),
            count: entry.end - entry.start)
    }

    /// A tensor as `Float`, whatever it is stored as.
    ///
    /// The small tensors -- every norm, `A_log`, `dt_bias`, the convolution
    /// taps -- are F32 or BF16 and are read once at load, so a copy is the
    /// right shape here. The large ones are quantized and are never read
    /// through this path.
    public func floats(_ name: String) throws -> [Float] {
        let entry = try entry(name)
        let raw = try bytes(name)
        switch entry.dtype {
        case "F32":
            return Array(raw.bindMemory(to: Float.self))
        case "BF16":
            // Widened by bit pattern: the top sixteen bits of a float32 are
            // exactly a bfloat16, which is the whole point of the format.
            return raw.bindMemory(to: UInt16.self).map {
                Float(bitPattern: UInt32($0) << 16)
            }
        case "F16":
            return raw.bindMemory(to: Float16.self).map(Float.init)
        default:
            throw Failure.unsupported(dtype: entry.dtype, name: name)
        }
    }
}
