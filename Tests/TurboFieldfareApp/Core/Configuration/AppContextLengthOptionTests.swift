import Testing
import TurboFieldfare

@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func candidatesUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.candidateTokens
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = [4_096, 8_192, 16_384, 32_768, 65_536].map {
            AppContextLengthOption.fp16KVBytes(architecture: .gemma4_26B_A4B, tokens: $0)
                / 1_048_576
        }
        #expect(mebibytes == [305, 385, 545, 865, 1_505])
    }

    @Test func labelsCarryTheContextLengthAndItsCost() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: 0,
            installedRAMBytes: 32 * 1_073_741_824)
        #expect(options.first { $0.tokens == 4_096 }?.label == "4K, 305 MB")
        #expect(options.first { $0.tokens == 65_536 }?.label == "64K, 1.47 GB")
    }
}
