import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

@Suite struct Sha256VerifierTests {

    /// SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    @Test func hashesEmptyFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-empty-\(UUID().uuidString).bin")
        try Data().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let h = try Sha256Verifier.hashFile(at: url)
        #expect(h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test func chunkSizeDoesNotAffectDigest() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-2m-\(UUID().uuidString).bin")
        // Two chunks at the default 1 MB chunkBytes — exercises the loop.
        var data = Data(count: 2 << 20)
        for i in 0..<data.count { data[i] = UInt8(i & 0xFF) }
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let small = try Sha256Verifier.hashFile(at: url, chunkBytes: 64 << 10)
        let big = try Sha256Verifier.hashFile(at: url)
        #expect(small == big, "chunk size must not affect digest")
    }

    /// The expected digest is pinned from outside the implementation: feeding
    /// `verifyFile` whatever `hashFile` just returned would pass for any
    /// deterministic wrong digest, because both sides would agree with each
    /// other and with nothing else.
    @Test func verifyMatches() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-match-\(UUID().uuidString).bin")
        let payload = Data("hello world".utf8)
        try payload.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let published = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        #expect(try Sha256Verifier.hashFile(at: url) == published)
        try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: published)
    }

    @Test func verifyMismatchThrowsChecksumMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-bad-\(UUID().uuidString).bin")
        try Data("hello world".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let wrong = String(repeating: "0", count: 64)
        #expect(throws: ModelError.checksumMismatch(file: "hello")) {
            try Sha256Verifier.verifyFile(at: url, named: "hello", expectedHex: wrong)
        }
    }

    // MARK: - The digests are the published ones, on both paths

    /// SHA-256("abc") from FIPS 180-4's own worked example. The verifier used to
    /// wrap a CommonCrypto context whose `Init`/`Update`/`Final` statuses it
    /// threw away, so a failed call would have surfaced here as a digest of
    /// whatever the context held rather than as an error -- which is also why
    /// the vector is pinned from the standard and not from the implementation.
    @Test func hashDataMatchesThePublishedVector() {
        #expect(
            Sha256Verifier.hashData(Data("abc".utf8))
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    /// The empty buffer is the case the discarded statuses could reach most
    /// easily -- `Update` is never called, so only `Init` and `Final` decide the
    /// answer -- and `hashData` is what `Model+Loading` runs over
    /// `manifest.json` to bind the receipt to it.
    @Test func hashDataOfNothingIsTheEmptyDigest() {
        #expect(
            Sha256Verifier.hashData(Data())
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    /// One million `a`s, the other worked example in the standard: 1,000,000
    /// bytes is bigger than the default 1 MB chunk, so the streaming loop runs
    /// at least twice and a digest that only covered the first chunk cannot pass.
    @Test func oneMillionBytesStreamToThePublishedDigest() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-1m-\(UUID().uuidString).bin")
        try Data(repeating: UInt8(ascii: "a"), count: 1_000_000).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let streamed = try Sha256Verifier.hashFile(at: url, chunkBytes: 64 << 10)
        #expect(
            streamed
                == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            Comment(rawValue: "streamed \(streamed) is not the published digest"))
    }

    /// The two paths must not be allowed to drift: `Model+Loading` hashes the
    /// manifest through `hashData` and the weights through `hashFile`, and the
    /// receipt compares the two against one manifest.
    @Test func bothPathsAgreeOnTheSameBytes() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-agree-\(UUID().uuidString).bin")
        var payload = Data(count: 3 << 20)
        for i in 0..<payload.count { payload[i] = UInt8((i * 7) & 0xFF) }
        try payload.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let inMemory = Sha256Verifier.hashData(payload)
        for chunk in [1, 4096, 64 << 10, 1 << 20, 8 << 20] {
            #expect(
                try Sha256Verifier.hashFile(at: url, chunkBytes: chunk) == inMemory,
                Comment(rawValue: "chunkBytes \(chunk) disagreed with hashData"))
        }
    }

    /// A file that cannot be opened is an error, not a digest of nothing.
    @Test func anUnopenablePathThrowsRatherThanHashingEmpty() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssdai-sha-missing-\(UUID().uuidString).bin")
        #expect(throws: (any Error).self) {
            try Sha256Verifier.hashFile(at: missing)
        }
    }
}
