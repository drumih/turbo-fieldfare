import Foundation
import CommonCrypto
import Darwin

public enum Sha256Verifier {

    /// Compute the lowercase-hex SHA-256 of the entire file at `fileURL` by
    /// streaming through a fixed-size scratch read. Does not allocate the
    /// whole file.
    public static func hashFile(at fileURL: URL,
                                chunkBytes: Int = 1 << 20) throws -> String {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        var ctx = CC_SHA256_CTX()
        CC_SHA256_Init(&ctx)
        var buf = [UInt8](repeating: 0, count: chunkBytes)
        while true {
            let got: Int = buf.withUnsafeMutableBytes { raw -> Int in
                return read(fd, raw.baseAddress!, chunkBytes)
            }
            if got == 0 { break }
            if got < 0 {
                throw ModelError.posixFailed(call: "read", errno: errno)
            }
            buf.withUnsafeBytes { raw in
                _ = CC_SHA256_Update(&ctx, raw.baseAddress!, CC_LONG(got))
            }
        }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        digest.withUnsafeMutableBytes { raw in
            _ = CC_SHA256_Final(raw.baseAddress!.assumingMemoryBound(to: UInt8.self), &ctx)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public static func hashData(_ data: Data) -> String {
        var ctx = CC_SHA256_CTX()
        CC_SHA256_Init(&ctx)
        data.withUnsafeBytes { raw in
            if let base = raw.baseAddress, raw.count > 0 {
                _ = CC_SHA256_Update(&ctx, base, CC_LONG(raw.count))
            }
        }
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        digest.withUnsafeMutableBytes { raw in
            _ = CC_SHA256_Final(raw.baseAddress!.assumingMemoryBound(to: UInt8.self), &ctx)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Throw `ModelError.checksumMismatch(file)` if the on-disk file's
    /// SHA-256 does not match `expectedHex`. Hex comparison is
    /// case-insensitive on the expected side (writer outputs lowercase).
    public static func verifyFile(at fileURL: URL,
                                  named name: String,
                                  expectedHex: String) throws {
        let actual = try hashFile(at: fileURL)
        if actual.lowercased() != expectedHex.lowercased() {
            throw ModelError.checksumMismatch(file: name)
        }
    }
}
