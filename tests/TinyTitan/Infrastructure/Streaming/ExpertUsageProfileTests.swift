import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

// The persisted routing profile: format round trip, dimension checks, and the
// decay/cap rule the streamer's LFU score is seeded with.

@Suite struct ExpertUsageProfileTests {
    @Test func encodedProfileRoundTrips() {
        var profile = ExpertUsageProfile(layers: 3, expertsPerLayer: 5)
        profile.setCount(layer: 0, expert: 2, to: 7)
        profile.setCount(layer: 2, expert: 4, to: 4095)
        let decoded = ExpertUsageProfile.decode(
            profile.encoded(), layers: 3, expertsPerLayer: 5)
        #expect(decoded == profile)
        #expect(decoded?.count(layer: 0, expert: 2) == 7)
        #expect(decoded?.count(layer: 2, expert: 4) == 4095)
    }

    @Test func profileForAnotherGeometryIsRefused() {
        let profile = ExpertUsageProfile(layers: 2, expertsPerLayer: 4)
        let data = profile.encoded()
        #expect(
            ExpertUsageProfile.decode(data, layers: 2, expertsPerLayer: 5) == nil)
        #expect(
            ExpertUsageProfile.decode(data, layers: 3, expertsPerLayer: 4) == nil)
    }

    @Test func truncatedOrForeignBytesAreRefused() {
        let profile = ExpertUsageProfile(layers: 2, expertsPerLayer: 4)
        var data = profile.encoded()
        #expect(ExpertUsageProfile.decode(Data(), layers: 2, expertsPerLayer: 4) == nil)
        data.removeLast()
        #expect(ExpertUsageProfile.decode(data, layers: 2, expertsPerLayer: 4) == nil)
        data = Data("not a profile".utf8)
        #expect(ExpertUsageProfile.decode(data, layers: 2, expertsPerLayer: 4) == nil)
    }

    @Test func seedHalvesAndCapsTheStoredCount() {
        var profile = ExpertUsageProfile(layers: 1, expertsPerLayer: 3)
        profile.setCount(layer: 0, expert: 0, to: 2000)
        profile.setCount(layer: 0, expert: 1, to: 300)
        profile.setCount(layer: 0, expert: 2, to: 1)
        let seed = profile.seedCounts(layer: 0)
        #expect(seed == [ExpertUsageProfile.seedCap, 150, 0])
    }

    @Test func storedCountsSaturate() {
        var profile = ExpertUsageProfile(layers: 1, expertsPerLayer: 1)
        profile.setCount(layer: 0, expert: 0, to: 99_999)
        #expect(profile.counts[0] == ExpertUsageProfile.storedCap)
        profile.setCounts(layer: 0, [1_000_000])
        #expect(profile.counts[0] == ExpertUsageProfile.storedCap)
    }

    @Test func allZeroIsEmpty() {
        var profile = ExpertUsageProfile(layers: 2, expertsPerLayer: 3)
        #expect(profile.isEmpty)
        profile.setCount(layer: 1, expert: 2, to: 1)
        #expect(!profile.isEmpty)
        #expect(profile.seedCounts(layer: 1) == [0, 0, 0])
        #expect(profile.seedCounts(layer: 9) == nil)
    }

    @Test func writeAndLoadFromDisk() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-profile-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        var profile = ExpertUsageProfile(layers: 2, expertsPerLayer: 3)
        profile.setCount(layer: 1, expert: 1, to: 42)
        try profile.write(to: url)
        let loaded = ExpertUsageProfile.load(
            url: url, layers: 2, expertsPerLayer: 3)
        #expect(loaded == profile)
    }
}
