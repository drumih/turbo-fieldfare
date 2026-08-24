import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        // 29 MiB above the older figures at every size, because the sliding
        // ring is sized for the widest prefill chunk the runtime may see - the
        // pooled image-token count - and the estimate had used the smaller text
        // chunk. The menu deltas are unchanged: every option grew equally.
        #expect(mebibytes == [334, 414, 574, 894, 1_534])
        // Deltas are relative to the default, which is 8K as of 2026-08-17 so
        // that an image and its prompt fit without the user changing anything.
        // 334 - 414 = -80 MiB, 574 - 414 = +160 MiB, and so on.
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, -85 MB",
            "8K, Default",
            "16K, +170 MB",
            "32K, +500 MB",
            "64K, +1.17 GB",
        ])
    }
}
