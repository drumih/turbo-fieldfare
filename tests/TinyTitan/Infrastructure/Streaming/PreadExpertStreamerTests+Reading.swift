import Darwin
import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

extension PreadExpertStreamerTests {
    /// A stride that cannot be represented as an `Int` used to trap in the slot
    /// allocation (`Int(layout.expertStride)`), which is a crash on a value that
    /// arrived in a layout file. The stream range is small and fits the synthetic
    /// file, so the open-time check passes and the allocation is what is asked
    /// about.
    @Test func aStrideTooLargeToAllocateIsRefused() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let huge = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: Self.streamOffset + Self.streamSize,
            expertsPerLayer: 1,
            expertStride: UInt64.max)

        #expect(throws: StreamerError.self) {
            _ = try PreadExpertStreamer(layout: huge, device: device, slotCount: 1)
        }
    }

    /// A layout whose stream range wraps must be refused, not wrapped. `required`
    /// came out of an unchecked `streamOffset + streamSize`, so an install layout
    /// with a corrupt offset could produce a small `required`, pass the file-size
    /// check, and leave every later offset -- all of them `streamOffset +
    /// regionOffset` -- pointing outside the file.
    @Test func aWrappingStreamRangeIsRefused() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let wrapping = StreamLayout(
            path: url.path,
            streamOffset: UInt64.max - 8,
            streamSize: 64,
            expertsPerLayer: Self.numExperts,
            expertStride: UInt64(Self.expertStride))

        #expect(throws: StreamerError.self) {
            _ = try PreadExpertStreamer(layout: wrapping, device: device, slotCount: 2)
        }

        // The valid layout still opens, so this is not a guard that refuses
        // everything.
        _ = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path),
            device: device, slotCount: 2)
    }

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

    // MARK: - Split blob reads (plan §5 / B3)

    /// A layout with a phase-1 boundary reads each miss as gate+up then tail;
    /// the assembled slot must be byte-identical to a whole-region load, and
    /// the expert must still end `.resident` (a re-plan is a hit, not a miss).
    @Test func splitMissesResidentWithIdenticalBytes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let phase1 = 1024
        let splitLayout = StreamLayout(
            path: url.path,
            streamOffset: Self.streamOffset,
            streamSize: Self.streamSize,
            expertsPerLayer: Self.numExperts,
            expertStride: UInt64(Self.expertStride),
            expertOffsets: nil,
            phase1Bytes: UInt64(phase1))
        let whole = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)
        let split = try PreadExpertStreamer(
            layout: splitLayout, device: device, slotCount: 2)

        let wanted = [3, 1]
        _ = try whole.loadExpertsCached(experts: wanted)
        _ = try split.loadExpertsCached(experts: wanted)
        // The two streamers allocate their own slots; compare against the file
        // tags instead of each other's buffers.
        for e in wanted {
            let plan = try split.planExpertsCached(experts: [e])
            #expect(plan.misses.isEmpty, "expert \(e) did not end resident after a split load")
            #expect(plan.hits == 1, "expert \(e) should be a hit on re-plan")
        }
        let r = try split.loadExpertsCached(experts: wanted)
        for (i, e) in wanted.enumerated() {
            let got = Self.bytes(of: r[i].buffer, offset: 0, count: Self.expertStride)
            #expect(
                got.allSatisfy { $0 == Self.tagByte(e) },
                "expert \(e) slot not uniformly tagged after split reads")
        }
    }

    /// AUD2-045: the tiered read branch skips the phase-1 split, because a
    /// tiered layer's experts are shorter than the slot stride. The operation
    /// still declared `publishesFirstStage` from `phase1Bytes` alone, so
    /// `encodeDecodeEarlyMissFixup` awaited a stage this branch never publishes
    /// -- no hang (completion resumes the waiters) but no overlap either.
    @Test func aTieredLayerDoesNotDeclareAFirstStageTheReadNeverPublishes() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device

        let tiered = StreamLayout(
            path: url.path,
            streamOffset: Self.streamOffset,
            streamSize: Self.streamSize,
            expertsPerLayer: Self.numExperts,
            expertStride: UInt64(Self.expertStride),
            expertSizes: [
                UInt64(Self.expertStride),
                UInt64(Self.expertStride / 2),
                UInt64(Self.expertStride),
                UInt64(Self.expertStride),
            ],
            phase1Bytes: 1024)
        let tieredStreamer = try PreadExpertStreamer(
            layout: tiered, device: device, slotCount: 1)
        let tieredOperation = try tieredStreamer.beginExpertCachePlan(
            tieredStreamer.planExpertsCached(experts: [0]))
        #expect(
            !tieredOperation.publishesFirstStage,
            "a tiered layer's read branch never publishes a first stage")
        try tieredOperation.wait()

        // Control: the same layer with uniform sizes does take the split, so the
        // guard must not simply switch the stage off for every layer.
        let split = StreamLayout(
            path: url.path,
            streamOffset: Self.streamOffset,
            streamSize: Self.streamSize,
            expertsPerLayer: Self.numExperts,
            expertStride: UInt64(Self.expertStride),
            phase1Bytes: 1024)
        let splitStreamer = try PreadExpertStreamer(
            layout: split, device: device, slotCount: 1)
        let splitOperation = try splitStreamer.beginExpertCachePlan(
            splitStreamer.planExpertsCached(experts: [0]))
        #expect(splitOperation.publishesFirstStage, "a uniform split still publishes")
        try splitOperation.wait()
    }

    /// The split boundary must sit strictly inside the stride: a 0 or
    /// stride-sized boundary would make one of the two ranges empty and the
    /// reader refuses zero-length ranges.
    @Test func aNonInteriorPhase1BoundaryIsRefused() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        for bad in [UInt64(0), UInt64(Self.expertStride)] {
            let layout = StreamLayout(
                path: url.path,
                streamOffset: Self.streamOffset,
                streamSize: Self.streamSize,
                expertsPerLayer: Self.numExperts,
                expertStride: UInt64(Self.expertStride),
                expertOffsets: nil,
                phase1Bytes: bad)
            #expect(throws: StreamerError.self) {
                _ = try PreadExpertStreamer(layout: layout, device: device, slotCount: 2)
            }
        }
    }

}
