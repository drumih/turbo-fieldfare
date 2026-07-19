import Foundation
import CryptoKit

/// Streaming SHA-256 hasher. Wraps CryptoKit's incremental API so we can hash
/// a file as we walk it (mmap'd source pages + zero-filled gaps) without
/// allocating a Swift heap buffer for the whole file.
struct Sha256Stream {
    private var hasher: SHA256

    init() { self.hasher = SHA256() }

    mutating func update(_ ptr: UnsafeRawBufferPointer) {
        hasher.update(bufferPointer: ptr)
    }

    func finalizeHexString() -> String {
        let digest = self.hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// One-shot helper to hash a file from disk in tile-bounded chunks. Used
    /// for fingerprinting `model.safetensors.index.json`.
    static func hashFile(path: String,
                                tileBytes: Int = 65_536,
                                noCache: Bool = false) throws -> String {
        let fd = open(path, O_RDONLY)
        if fd < 0 { throw RepackError.fileOpenFailed(path: path, errno: errno) }
        defer { close(fd) }
        if noCache {
            _ = fcntl(fd, F_NOCACHE, 1)
        }
        var hasher = Sha256Stream()
        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: tileBytes, alignment: 16_384)
        defer { buf.deallocate() }
        var off: off_t = 0
        while true {
            let got = pread(fd, buf.baseAddress, tileBytes, off)
            if got < 0 { throw RepackError.preadShort(path: path, expected: tileBytes, got: 0, errno: errno) }
            if got == 0 { break }
            hasher.update(UnsafeRawBufferPointer(start: buf.baseAddress, count: got))
            off += off_t(got)
        }
        return hasher.finalizeHexString()
    }
}
