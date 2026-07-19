import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension PreadExpertStreamerTests {
  @Test func preadRoundTrip_matchesTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    for e in 0..<Self.numExperts {
      let r = try streamer.loadExpert(layer: 0, expert: e)
      #expect(r.offset == 0)
      #expect(r.size == UInt64(Self.expertStride))
      let got = Self.bytes(of: r.buffer, offset: 0, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(e) },
        "expert \(e) slot not uniformly tagged")
    }
  }

  @Test func shortRead_throwsSizeMismatch() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 1)

    // Truncate the file on disk to just past expert 0; the already-open fd
    // now hits EOF mid-read for any later expert.
    let truncatedLen = off_t(Self.streamOffset) + off_t(Self.expertStride)
    #expect(truncate(url.path, truncatedLen) == 0)

    #expect(throws: StreamerError.self) {
      _ = try streamer.loadExpert(layer: 0, expert: Self.numExperts - 1)
    }
  }

  @Test func slotReuse_roundRobinOverwrites() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    // With slotCount=2, experts 0,1,2,3 land in slots 0,1,0,1. Capture the
    // first buffer (slot 0, expert 0), then load expert 2 which reuses
    // slot 0 — the same MTLBuffer must now hold expert 2's tag.
    let r0 = try streamer.loadExpert(layer: 0, expert: 0)
    let r1 = try streamer.loadExpert(layer: 0, expert: 1)
    let r2 = try streamer.loadExpert(layer: 0, expert: 2)
    let r3 = try streamer.loadExpert(layer: 0, expert: 3)

    #expect(r0.buffer === r2.buffer, "expert 0 and 2 should share slot 0's buffer")
    #expect(r1.buffer === r3.buffer, "expert 1 and 3 should share slot 1's buffer")
    #expect(r0.buffer !== r1.buffer, "slots 0 and 1 must be distinct buffers")

    // r0 was overwritten by r2; reading slot 0 now yields expert 2's tag.
    let slot0 = Self.bytes(of: r2.buffer, offset: 0, count: Self.expertStride)
    #expect(slot0.allSatisfy { $0 == Self.tagByte(2) })
    let slot1 = Self.bytes(of: r3.buffer, offset: 0, count: Self.expertStride)
    #expect(slot1.allSatisfy { $0 == Self.tagByte(3) })
  }

}
