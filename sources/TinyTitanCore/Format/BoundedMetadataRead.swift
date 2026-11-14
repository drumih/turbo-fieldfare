import Darwin
import Foundation

/// Read one metadata document from a directory the engine was handed, with the
/// allocation capped **before** it happens.
///
/// `Data(contentsOf:)` materializes the whole file first. Measured on this host
/// against a 2 GiB sparse file (0 B allocated on disk, so the read is pure
/// allocation): 0.350 s and **+2,049 MB of process footprint** on a 24 GB
/// machine. A bound applied to the bytes *after* such a read therefore refuses a
/// hostile document only after it has already cost its full size in memory,
/// which is the thing the bound exists to prevent — and it is why the sites this
/// replaced were a finding rather than a style preference.
///
/// One file descriptor is held from `open` through `fstat` through the read, so
/// both properties hold at once: the cap fires before the allocation, and the
/// size checked belongs to the file actually read. A `stat` by *path* followed by
/// a re-read by path does not give the second property — the path can name a
/// different inode in between (K17).
///
/// Reads go through `SSDAIModelDirectory`, so they inherit its rules as well: the
/// path is validated relative to the root, every directory component is opened
/// `O_NOFOLLOW`, the document itself must be a regular file and not a link, and a
/// short read surfaces as an error instead of zero padding. A symlinked
/// `tokenizer.json` inside a model directory is therefore refused now, exactly as
/// a symlinked `manifest.json` has been since Task 8 of the AUD-113 shape.
package enum BoundedMetadataRead {
    package static func read(fileAt url: URL, maxBytes: UInt64) throws -> Data {
        let name = url.lastPathComponent
        let directory = try SSDAIModelDirectory(rootURL: url.deletingLastPathComponent())
        let fd = try directory.openFile(name)
        defer { close(fd) }
        let size = try directory.fileSize(fileDescriptor: fd, relativePath: name)
        guard size <= maxBytes else {
            throw ModelError.metadataOverBound(document: name, bytes: Int(size), cap: maxBytes)
        }
        return try directory.readMetadata(
            fileDescriptor: fd, relativePath: name, maxBytes: maxBytes)
    }
}
