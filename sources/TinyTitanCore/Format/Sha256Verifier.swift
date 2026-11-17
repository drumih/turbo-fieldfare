import CryptoKit
import Darwin
import Foundation

public enum Sha256Verifier {

    /// Compute the lowercase-hex SHA-256 of the entire file at `fileURL` by
    /// streaming through a fixed-size scratch read. Does not allocate the
    /// whole file.
    public static func hashFile(
        at fileURL: URL,
        chunkBytes: Int = 1 << 20
    ) throws -> String {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        return try hashFile(
            fileDescriptor: fd, displayName: fileURL.path,
            chunkBytes: chunkBytes)
    }

    package static func hashFile(
        fileDescriptor fd: Int32,
        displayName: String,
        chunkBytes: Int = 1 << 20
    ) throws -> String {
        guard chunkBytes > 0 else {
            throw ModelError.indexCorrupt(detail: "SHA-256 chunk size must be positive")
        }
        var hasher = SHA256()
        var buf = [UInt8](repeating: 0, count: chunkBytes)
        while true {
            let got: Int = buf.withUnsafeMutableBytes { raw -> Int in
                // `buf` is `chunkBytes > 0` long, so this is unreachable; an
                // empty buffer reads nothing rather than trapping.
                guard let base = raw.baseAddress else { return 0 }
                return read(fd, base, chunkBytes)
            }
            if got == 0 { break }
            if got < 0, errno == EINTR { continue }
            if got < 0 {
                throw ModelError.posixFailed(call: "read(\(displayName))", errno: errno)
            }
            // `got` is in 1...chunkBytes by here: 0 broke out of the loop and a
            // negative read threw, so the slice is never empty and never longer
            // than the buffer.
            buf[0..<got].withUnsafeBytes {
                hasher.update(bufferPointer: $0)
            }
        }
        return hex(of: hasher.finalize())
    }

    /// SHA-256 of an in-memory buffer. Cannot fail: CryptoKit's hasher has no
    /// error state to discard, which is why this is `hashData` rather than a
    /// throwing function wrapping a CommonCrypto context.
    public static func hashData(_ data: Data) -> String {
        hex(of: SHA256.hash(data: data))
    }

    /// Throw `ModelError.checksumMismatch(file)` if the on-disk file's
    /// SHA-256 does not match `expectedHex`. Hex comparison is
    /// case-insensitive on the expected side (writer outputs lowercase).
    public static func verifyFile(
        at fileURL: URL,
        named name: String,
        expectedHex: String
    ) throws {
        let actual = try hashFile(at: fileURL)
        if actual.lowercased() != expectedHex.lowercased() {
            throw ModelError.checksumMismatch(file: name)
        }
    }

    package static func verifyFile(
        fileDescriptor fd: Int32,
        named name: String,
        expectedHex: String
    ) throws {
        let actual = try hashFile(fileDescriptor: fd, displayName: name)
        if actual.lowercased() != expectedHex.lowercased() {
            throw ModelError.checksumMismatch(file: name)
        }
    }

    private static func hex(of digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
