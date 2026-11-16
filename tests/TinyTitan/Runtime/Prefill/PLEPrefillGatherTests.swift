import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// AUD2-036: the PLE prefill gather's destination is sized by the runner's own
/// `prefillChunkTokens`, but `PrefillRuntimeConfig.production(chunkTokens:)`
/// accepts any allowed chunk. The scratch guard bounds the caller's chunk, not
/// that allocation, so the gather has to refuse a row count larger than
/// `PLEBlock.maxRows` before it writes.
@Suite struct PLEPrefillGatherTests {
    @Test func aGatherLargerThanTheBlockIsRefused() {
        #expect(throws: PrefillError.self) {
            try PLEPrefillGather.validate(rows: 5, maxRows: 4)
        }
    }

    @Test func aGatherAtTheBlockCapacityIsAccepted() {
        #expect(throws: Never.self) {
            try PLEPrefillGather.validate(rows: 4, maxRows: 4)
        }
    }

    @Test func aSmallerGatherIsAccepted() {
        #expect(throws: Never.self) {
            try PLEPrefillGather.validate(rows: 3, maxRows: 4)
        }
    }
}
