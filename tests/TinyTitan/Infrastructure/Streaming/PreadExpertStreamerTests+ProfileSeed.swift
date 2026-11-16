import Darwin
import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

// A layer opened with a persisted routing profile restores that prior at the
// prefill-to-decode reset instead of zeroing the LFU score. Without a profile
// the reset keeps its original behaviour.

extension PreadExpertStreamerTests {
    @Test func seededUseCountsAreTheStartingScoreAndSurviveReset() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let seed = [0, 7, 0, 3]
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device,
            slotCount: 4, seedUseCounts: seed)
        #expect(streamer.expertUseCountsSnapshot() == seed)

        _ = try streamer.loadExpertsCached(experts: [1, 1, 3])
        #expect(streamer.expertUseCountsSnapshot()[1] > seed[1])

        streamer.resetExpertUseCounts()
        #expect(streamer.expertUseCountsSnapshot() == seed)
    }

    @Test func aSeedForAnotherGeometryIsIgnored() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device,
            slotCount: 4, seedUseCounts: [1, 2, 3])
        #expect(
            streamer.expertUseCountsSnapshot()
                == [Int](repeating: 0, count: Self.numExperts))
    }

    @Test func noSeedStillZeroesAtReset() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
        _ = try streamer.loadExpertsCached(experts: [2, 2])
        #expect(streamer.expertUseCountsSnapshot()[2] > 0)
        streamer.resetExpertUseCounts()
        #expect(
            streamer.expertUseCountsSnapshot()
                == [Int](repeating: 0, count: Self.numExperts))
    }
}

extension PreadExpertStreamerTests {
    @Test func adoptedCountsBecomeTheNextPriorCapped() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let device = try MetalContext().device
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path), device: device,
            slotCount: 4, seedUseCounts: [0, 1, 0, 0])
        streamer.adoptExpertUseCounts([0, 900, 0, 4])
        streamer.resetExpertUseCounts()
        #expect(
            streamer.expertUseCountsSnapshot()
                == [0, ExpertUsageProfile.seedCap, 0, 4])
    }
}
