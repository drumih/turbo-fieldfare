import Foundation
import Testing

@testable import TinyTitan
import TinyTitanFormat
@testable import TinyTitanCore

/// The compressed-expert format's codec, pinned.
///
/// The engine has no compression dependency, so the codec is LZFSE from the
/// system framework. These cases fix the contract the streamer will rely on: a
/// blob decodes to exactly the byte count the install's geometry says, and a
/// blob that does not is refused rather than leaving a slot half-overwritten.
@Suite("Expert blob codec")
struct ExpertBlobCodecTests {
    /// Quantized weights are close to uniform, so a compressible payload here
    /// would test the codec against data the real format never stores.
    private static func syntheticBlob(_ count: Int) -> [UInt8] {
        var generator = SystemRandomNumberGenerator()
        var bytes = [UInt8](repeating: 0, count: count)
        for index in bytes.indices {
            bytes[index] = UInt8.random(in: 0...255, using: &generator)
        }
        return bytes
    }

    @Test("A blob round-trips to exactly its own bytes")
    func roundTrips() throws {
        let original = Self.syntheticBlob(64 * 1024)
        let packed = try original.withUnsafeBytes { raw in
            try ExpertBlobCodec.compress(
                source: try #require(raw.baseAddress), count: raw.count)
        }
        var restored = [UInt8](repeating: 0, count: original.count)
        try restored.withUnsafeMutableBytes { destination in
            try ExpertBlobCodec.decompress(
                packed, into: try #require(destination.baseAddress),
                expected: original.count)
        }
        #expect(restored == original)
    }

    /// Random bytes are incompressible, so the encoder must be allowed to emit
    /// slightly more than it read. A fixed-size destination would truncate here
    /// and the failure would surface at decode, far from the cause.
    @Test("An incompressible blob is stored, not truncated")
    func storesIncompressibleBytes() throws {
        let original = Self.syntheticBlob(32 * 1024)
        let packed = try original.withUnsafeBytes { raw in
            try ExpertBlobCodec.compress(
                source: try #require(raw.baseAddress), count: raw.count)
        }
        #expect(packed.count > 0)
        var restored = [UInt8](repeating: 0, count: original.count)
        try restored.withUnsafeMutableBytes { destination in
            try ExpertBlobCodec.decompress(
                packed, into: try #require(destination.baseAddress),
                expected: original.count)
        }
        #expect(restored == original)
    }

    /// The geometry says how long the blob must be; a blob that decodes to
    /// anything else is refused, so a swapped or truncated file cannot leave a
    /// slot holding the previous expert's weights.
    @Test("A blob that does not fill the geometry is refused")
    func refusesWrongLength() throws {
        let original = Self.syntheticBlob(16 * 1024)
        let packed = try original.withUnsafeBytes { raw in
            try ExpertBlobCodec.compress(
                source: try #require(raw.baseAddress), count: raw.count)
        }
        var destination = [UInt8](repeating: 0, count: original.count * 2)
        #expect(throws: ExpertBlobCodec.Failure.self) {
            try destination.withUnsafeMutableBytes { buffer in
                try ExpertBlobCodec.decompress(
                    packed, into: try #require(buffer.baseAddress),
                    expected: original.count * 2)
            }
        }
    }

    @Test("An empty blob is refused rather than decoding to nothing")
    func refusesEmpty() {
        var destination = [UInt8](repeating: 0, count: 64)
        #expect(throws: ExpertBlobCodec.Failure.self) {
            try destination.withUnsafeMutableBytes { buffer in
                try ExpertBlobCodec.decompress(
                    Data(), into: try #require(buffer.baseAddress), expected: 64)
            }
        }
    }
}
