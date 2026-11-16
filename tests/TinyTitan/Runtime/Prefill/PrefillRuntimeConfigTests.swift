import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct PrefillRuntimeConfigTests {
    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096])
    func productionUsesCompleteChunkedPath(_ chunkTokens: Int) throws {
        let config = PrefillRuntimeConfig.production(chunkTokens: chunkTokens)
        #expect(config.mode == .chunked)
        #expect(config.chunkTokens == chunkTokens)
    }

    @Test func offDisablesChunkedPrefill() {
        let config = PrefillRuntimeConfig.off
        #expect(config.mode == .off)
        #expect(!config.enabled)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }

    @Test func plannerSupportsLongQwenChunks() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 8_194,
            startPosition: 0,
            config: .production(chunkTokens: 4_096))
        #expect(spans.map(\.tokenCount) == [4_096, 4_096, 2])
        #expect(spans.map(\.startPosition) == [0, 4_096, 8_192])
    }
}
