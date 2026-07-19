import Testing
import Foundation
@testable import TurboFieldfare

@Suite struct Sha256VerifierTests {

    /// SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    @Test func hashesEmptyFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-empty-\(UUID().uuidString).bin")
        try Data().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let h = try Sha256Verifier.hashFile(at: url)
        #expect(h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test func chunkSizeDoesNotAffectDigest() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-2m-\(UUID().uuidString).bin")
        // Two chunks at the default 1 MB chunkBytes — exercises the loop.
        var data = Data(count: 2 << 20)
        for i in 0..<data.count { data[i] = UInt8(i & 0xFF) }
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let small = try Sha256Verifier.hashFile(at: url, chunkBytes: 64 << 10)
        let big   = try Sha256Verifier.hashFile(at: url)
        #expect(small == big, "chunk size must not affect digest")
    }

    @Test func verifyMatches() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-match-\(UUID().uuidString).bin")
        let payload = Data("hello world".utf8)
        try payload.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let hex = try Sha256Verifier.hashFile(at: url)
        try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: hex)
    }

    @Test func verifyMismatchThrowsChecksumMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-sha-bad-\(UUID().uuidString).bin")
        try Data("hello world".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let wrong = String(repeating: "0", count: 64)
        #expect(throws: ModelError.checksumMismatch(file: "hello")) {
            try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: wrong)
        }
    }
}
