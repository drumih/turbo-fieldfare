import Darwin
import Foundation
import Metal

package enum GGUFMappedShardsError: Error, Equatable, CustomStringConvertible {
    case openFailed(String, Int32)
    case mapFailed(String)

    package var description: String {
        switch self {
        case .openFailed(let path, let code):
            return "could not open \(path) (errno \(code))"
        case .mapFailed(let name):
            return "could not map \(name)"
        }
    }
}

/// The checkpoint's tensors mapped read-only, one page-aligned window per
/// tensor, wrapped in no-copy Metal buffers, so the fused GEMVs read the block
/// formats straight out of the file's pages.
///
/// This replaces materialising the weights: the bf16 arena was 2-4x the bytes
/// the file stores and made ~7 GB of a 24 GB machine unavailable, which is what
/// put the end-to-end run into swap. A mapped window costs page cache, which is
/// evictable. Windows are mapped per tensor because a whole shard is 50 GB and
/// Metal refuses a single buffer that large.
package final class GGUFMappedShards {
    package struct Weight {
        package let buffer: MTLBuffer
        package let offset: Int
        package let type: GGUFType
        package let rows: Int
        package let columns: Int
    }

    private struct Location {
        let url: URL
        let fileOffset: UInt64
        let byteCount: UInt64
        let type: GGUFType
        let dimensions: [UInt64]
    }

    private let device: MTLDevice
    private let locations: [String: Location]
    private let lock = NSLock()
    private var mapped: [String: Weight] = [:]
    private var regions: [(base: UnsafeMutableRawPointer, length: Int)] = []

    package init(model: GGUFModel, device: MTLDevice) throws {
        self.device = device
        var table: [String: Location] = [:]
        for (name, location) in model.tensors {
            let shard = model.shards[location.shardIndex]
            table[name] = Location(
                url: shard.url,
                fileOffset: shard.dataOffset + location.tensor.offset,
                byteCount: location.tensor.byteCount,
                type: location.tensor.type,
                dimensions: location.tensor.dimensions)
        }
        self.locations = table
    }

    deinit {
        for (base, length) in regions { munmap(base, length) }
    }

    /// A tensor's mapped bytes and geometry, mapped on first use. `columns` is
    /// the GGUF fast axis (ne0) and `rows` the product of the rest, which is the
    /// shape a GEMV reads.
    package func weight(named name: String) throws -> Weight? {
        lock.lock()
        defer { lock.unlock() }
        if let existing = mapped[name] { return existing }
        guard let location = locations[name], location.dimensions.count >= 2 else {
            return nil
        }
        let pageSize = UInt64(getpagesize())
        let pageStart = location.fileOffset / pageSize * pageSize
        let end = location.fileOffset + location.byteCount
        let windowLength = Int((end - pageStart + pageSize - 1) / pageSize * pageSize)
        let fd = open(location.url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw GGUFMappedShardsError.openFailed(location.url.path, errno)
        }
        defer { close(fd) }
        guard
            let base = mmap(nil, windowLength, PROT_READ, MAP_PRIVATE, fd, off_t(pageStart)),
            base != MAP_FAILED
        else {
            throw GGUFMappedShardsError.mapFailed(name)
        }
        let pointer = base.assumingMemoryBound(to: UInt8.self)
        _ = posix_madvise(base, windowLength, POSIX_MADV_RANDOM)
        guard
            let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: windowLength, options: .storageModeShared,
                deallocator: nil)
        else {
            munmap(base, windowLength)
            throw MetalError.bufferAllocationFailed("mapped \(name)")
        }
        regions.append((base, windowLength))
        let columns = Int(location.dimensions[0])
        let rows = location.dimensions.dropFirst().reduce(1) { $0 * Int($1) }
        let weight = Weight(
            buffer: buffer, offset: Int(location.fileOffset - pageStart),
            type: location.type, rows: rows, columns: columns)
        mapped[name] = weight
        return weight
    }

    /// Maps an arbitrary byte range of a shard, keyed by `key`, and caches it.
    /// This is how the routed experts are read: their ranges belong to one
    /// 50 GB tensor, which cannot be one Metal buffer, but each expert's slice
    /// maps cleanly and the page cache does the caching.
    package func weight(
        key: String, url: URL, fileOffset: UInt64, byteCount: UInt64,
        type: GGUFType, rows: Int, columns: Int
    ) throws -> Weight {
        lock.lock()
        defer { lock.unlock() }
        if let existing = mapped[key] { return existing }
        let pageSize = UInt64(getpagesize())
        let pageStart = fileOffset / pageSize * pageSize
        let windowLength = Int(
            (fileOffset + byteCount - pageStart + pageSize - 1)
                / pageSize * pageSize)
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw GGUFMappedShardsError.openFailed(url.path, errno) }
        defer { close(fd) }
        guard
            let base = mmap(nil, windowLength, PROT_READ, MAP_PRIVATE, fd, off_t(pageStart)),
            base != MAP_FAILED
        else {
            throw GGUFMappedShardsError.mapFailed(key)
        }
        let pointer = base.assumingMemoryBound(to: UInt8.self)
        _ = posix_madvise(base, windowLength, POSIX_MADV_RANDOM)
        guard
            let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: windowLength, options: .storageModeShared,
                deallocator: nil)
        else {
            munmap(base, windowLength)
            throw MetalError.bufferAllocationFailed(key)
        }
        regions.append((base, windowLength))
        let weight = Weight(
            buffer: buffer, offset: Int(fileOffset - pageStart),
            type: type, rows: rows, columns: columns)
        mapped[key] = weight
        return weight
    }

    /// Bytes mapped so far (page-cache backed, not resident).
    package var mappedBytes: Int { regions.reduce(0) { $0 + $1.length } }
}
