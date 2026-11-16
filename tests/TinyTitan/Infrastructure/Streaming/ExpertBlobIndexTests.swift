import Foundation
import Testing

@testable import TinyTitan
import TinyTitanFormat
@testable import TinyTitanCore

/// The seek index a compressed expert layer cannot do without: an uncompressed
/// layer is `n * stride`, a compressed one is wherever its blobs landed, and a
/// split layer has two blobs per expert.
///
/// Every case here is a failure the streamer would otherwise turn into silent
/// weight corruption -- the wrong expert's, or the wrong half's, bytes decoded
/// into a slot -- rather than a read error.
@Suite("Expert blob index")
struct ExpertBlobIndexTests {
    private static func sample(_ count: Int, frames: Int) -> [ExpertBlobIndex.Entry] {
        var entries: [ExpertBlobIndex.Entry] = []
        var offset: UInt64 = 0
        for expert in 0..<count {
            var blobs: [ExpertBlobIndex.Blob] = []
            for frame in 0..<frames {
                let length = UInt32(1_000 + expert * 10 + frame)
                blobs.append(.init(offset: offset, length: length))
                offset += UInt64(length)
            }
            entries.append(.init(frames: blobs))
        }
        return entries
    }

    @Test("A two-frame index round-trips")
    func roundTripsSplit() throws {
        let entries = Self.sample(8, frames: 2)
        let decoded = try ExpertBlobIndex.decode(
            ExpertBlobIndex.encode(entries, framesPerExpert: 2),
            expectedCount: 8, expectedFrames: 2)
        #expect(decoded == entries)
    }

    @Test("A single-frame index round-trips")
    func roundTripsWhole() throws {
        let entries = Self.sample(4, frames: 1)
        let decoded = try ExpertBlobIndex.decode(
            ExpertBlobIndex.encode(entries, framesPerExpert: 1),
            expectedCount: 4, expectedFrames: 1)
        #expect(decoded == entries)
    }

    /// The reader decides its decode sizes from the layer's split, so an index
    /// built for the other shape would hand it blobs of the wrong length.
    @Test("An index written for the other split is refused")
    func refusesWrongSplit() {
        let data = ExpertBlobIndex.encode(Self.sample(4, frames: 2), framesPerExpert: 2)
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 4, expectedFrames: 1)
        }
    }

    @Test("A truncated index is refused, not read past its end")
    func refusesTruncation() {
        let data = ExpertBlobIndex.encode(Self.sample(8, frames: 2), framesPerExpert: 2)
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(
                data.prefix(data.count - 4), expectedCount: 8, expectedFrames: 2)
        }
    }

    @Test("An index written for another layer is refused")
    func refusesWrongExpertCount() {
        let data = ExpertBlobIndex.encode(Self.sample(8, frames: 2), framesPerExpert: 2)
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 9, expectedFrames: 2)
        }
    }

    @Test("A file that is not an index is refused at the magic")
    func refusesForeignFile() {
        var data = ExpertBlobIndex.encode(Self.sample(2, frames: 2), framesPerExpert: 2)
        data[0] = UInt8(ascii: "X")
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 2, expectedFrames: 2)
        }
    }

    /// Overlapping blobs would decode one frame's bytes into another's slot; the
    /// index cannot say so at read time, so it says so at decode.
    @Test("Overlapping frames are refused")
    func refusesOverlap() {
        let entries = [
            ExpertBlobIndex.Entry(frames: [
                .init(offset: 0, length: 100), .init(offset: 100, length: 100),
            ]),
            ExpertBlobIndex.Entry(frames: [
                .init(offset: 150, length: 100), .init(offset: 250, length: 100),
            ]),
        ]
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(
                ExpertBlobIndex.encode(entries, framesPerExpert: 2),
                expectedCount: 2, expectedFrames: 2)
        }
    }

    /// The streamer sizes a compressed layer's file check from this, so it has
    /// to be the last byte any frame addresses -- not the sum of the lengths,
    /// which would disagree as soon as two experts' frames are adjacent.
    @Test("The highest frame end is the layer's stream size")
    func maximumEndIsTheHighestFrame() {
        let entries = [
            ExpertBlobIndex.Entry(frames: [
                .init(offset: 0, length: 100), .init(offset: 100, length: 50),
            ]),
            ExpertBlobIndex.Entry(frames: [
                .init(offset: 150, length: 25), .init(offset: 175, length: 75),
            ]),
        ]
        #expect(ExpertBlobIndex.maximumEnd(entries) == 250)
        #expect(ExpertBlobIndex.maximumEnd(entries.map(\.frames)) == 250)
    }

    /// A wrapping `offset + length` would read as "inside the file" to every
    /// later `end <= streamSize` guard, so it is refused at decode.
    @Test("A frame that wraps the address space is refused")
    func refusesWrappingFrame() {
        let entries = [ExpertBlobIndex.Entry(frames: [.init(offset: UInt64.max - 4, length: 16)])]
        let data = ExpertBlobIndex.encode(entries, framesPerExpert: 1)
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(data, expectedCount: 1, expectedFrames: 1)
        }
    }

    @Test("A zero-length frame is refused")
    func refusesEmptyFrame() {
        let entries = [ExpertBlobIndex.Entry(frames: [.init(offset: 0, length: 0)])]
        #expect(throws: ExpertBlobIndex.Failure.self) {
            _ = try ExpertBlobIndex.decode(
                ExpertBlobIndex.encode(entries, framesPerExpert: 1),
                expectedCount: 1, expectedFrames: 1)
        }
    }
}

/// The whole scheme on a synthetic layer: experts split into gate|up and tail,
/// each half compressed as its own frame, laid out sequentially with an index,
/// then read back the way the streamer would.
///
/// This is the test the repacker and the streamer will both have to satisfy, and
/// it runs without a model: if the layout, the index and the codec disagree,
/// this fails before either side is written.
@Suite("Compressed expert layer")
struct CompressedExpertLayerTests {
    private static let expertStride = 1_000
    private static let split = 670

    /// Deterministic but incompressible-ish, so the frames are real frames.
    private static func expertBytes(_ expert: Int) -> [UInt8] {
        var state = UInt64(expert) &* 6_364_136_223_846_793_005 &+ 1
        return (0..<expertStride).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return UInt8(truncatingIfNeeded: state >> 33)
        }
    }

    @Test("A layer round-trips through two frames per expert and the index")
    func layerRoundTrips() throws {
        let experts = 4
        var file = Data()
        var entries: [ExpertBlobIndex.Entry] = []
        var originals: [[UInt8]] = []
        for expert in 0..<experts {
            let raw = Self.expertBytes(expert)
            originals.append(raw)
            var blobs: [ExpertBlobIndex.Blob] = []
            for range in [0..<Self.split, Self.split..<Self.expertStride] {
                let slice = Array(raw[range])
                let frame = try slice.withUnsafeBytes { buffer in
                    try ExpertBlobCodec.compress(
                        source: try #require(buffer.baseAddress), count: buffer.count)
                }
                blobs.append(.init(offset: UInt64(file.count), length: UInt32(frame.count)))
                file.append(frame)
            }
            entries.append(.init(frames: blobs))
        }
        let index = ExpertBlobIndex.encode(entries, framesPerExpert: 2)
        let decoded = try ExpertBlobIndex.decode(
            index, expectedCount: experts, expectedFrames: 2)

        for expert in 0..<experts {
            var restored = [UInt8](repeating: 0, count: Self.expertStride)
            let frames = decoded[expert].frames
            #expect(frames.count == 2)
            let halfSizes = [Self.split, Self.expertStride - Self.split]
            var cursor = 0
            for (frame, size) in zip(frames, halfSizes) {
                let packed = file.subdata(
                    in: Int(frame.offset)..<Int(frame.offset) + Int(frame.length))
                try restored.withUnsafeMutableBytes { buffer in
                    try ExpertBlobCodec.decompress(
                        packed,
                        into: try #require(buffer.baseAddress).advanced(by: cursor),
                        expected: size)
                }
                cursor += size
            }
            #expect(restored == originals[expert])
        }
    }
}
