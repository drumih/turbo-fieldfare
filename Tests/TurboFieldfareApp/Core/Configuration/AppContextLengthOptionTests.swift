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
        #expect(mebibytes == [305, 385, 545, 865, 1_505])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, Default",
            "8K, +85 MB",
            "16K, +250 MB",
            "32K, +590 MB",
            "64K, +1.26 GB",
        ])
    }
}
