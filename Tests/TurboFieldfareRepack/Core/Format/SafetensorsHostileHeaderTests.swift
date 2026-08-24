import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// Shard headers are attacker-shaped input: only `model.safetensors.index.json`
/// is fingerprinted, so the per-shard JSON that drives every offset computation
/// arrives unpinned. Each case here used to trap — an abort, not a failed
/// download — because the arithmetic was unchecked.
@Suite struct SafetensorsHostileHeaderTests {
    private static func header(_ entries: [String: Any]) -> Data {
        try! JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
    }

    private static func parse(
        _ entries: [String: Any],
        fileSize: UInt64 = 1_024
    ) throws -> Safetensors.Header {
        try Safetensors.parseHeaderBytes(
            path: "hostile.safetensors", fileSize: fileSize,
            headerBytes: header(entries))
    }

    private static func entry(
        dtype: String = "BF16",
        shape: [Any],
        offsets: [Any]
    ) -> [String: Any] {
        ["weight": ["dtype": dtype, "shape": shape, "data_offsets": offsets]]
    }

    @Test func dataOffsetsEndBeforeBeginIsRejected() throws {
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [512, 8]))
        }
    }

    @Test func negativeOffsetsAreRejected() throws {
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [-1, 8]))
        }
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [0, -8]))
        }
    }

    @Test func negativeShapeExtentIsRejected() throws {
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [-2, 2], offsets: [0, 8]))
        }
    }

    /// `payloadBase + begin` and `abs + size` both overflow here.
    @Test func offsetsNearTheTopOfTheAddressSpaceAreRejected() throws {
        let top = UInt64.max
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [top - 4, top]))
        }
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [top, top]))
        }
    }

    /// The shape product is folded before it is compared against the size, so a
    /// header can make it overflow without any tensor being large.
    @Test func shapeProductOverflowIsRejected() throws {
        let big = UInt64(1) << 62
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [big, big, big], offsets: [0, 8]))
        }
        // Product fits, but multiplying by the element size does not.
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [UInt64.max], offsets: [0, 8]))
        }
    }

    /// A file shorter than its own 8-byte length prefix underflowed the header
    /// bound into ~2^64, which admitted any header size at all.
    /// JSON has one number type, so a fractional or huge literal reaches the
    /// parser as an ordinary number. Reading it as an integer truncates `1.5`
    /// to 1 and saturates `1e30` at `UInt64.max`, either of which lets a header
    /// describe a tensor other than the one it declares.
    @Test func nonIntegerShapeOrOffsetIsRejected() throws {
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [1.5, 2], offsets: [0, 8]))
        }
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [1e30, 2], offsets: [0, 8]))
        }
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [0.5, 8]))
        }
        #expect(throws: RepackError.self) {
            _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [0, 1e30]))
        }
    }

    @Test func fileShorterThanTheLengthPrefixIsRejected() throws {
        for size: UInt64 in [0, 1, 7] {
            #expect(throws: RepackError.self) {
                _ = try Self.parse(Self.entry(shape: [2, 2], offsets: [0, 8]),
                                   fileSize: size)
            }
        }
    }

    /// The checks must not have cost the ordinary case its acceptance.
    @Test func wellFormedHeaderStillParses() throws {
        let parsed = try Self.parse(Self.entry(shape: [2, 2], offsets: [0, 8]))
        #expect(parsed.tensors.count == 1)
        let tensor = try #require(parsed.tensors.first)
        #expect(tensor.sizeBytes == 8)
        #expect(tensor.shape == [2, 2])
        #expect(tensor.absoluteOffset == parsed.payloadBaseOffset)
    }
}
