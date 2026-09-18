import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppContextLengthOptionTests {
    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
    }

    /// The ceiling is the checkpoint's `max_position_embeddings`, so the menu
    /// cannot offer a position the model was never trained to address.
    @Test func thelargestOptionIsTheModelCeiling() {
        #expect(AppContextLengthOption.allCases.last?.tokens
            == ArchConfig.gemma4_26B_A4B.maxPositionEmbeddings)
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        // 29 MiB above the older figures at every size, because the sliding
        // ring is sized for the widest prefill chunk the runtime may see — the
        // pooled image-token count — and the estimate had used the smaller text
        // chunk. The menu deltas are unchanged: every option grew equally.
        #expect(mebibytes == [334, 414, 574, 894, 1_534, 2_814, 5_374])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, -85 MB",
            "8K, Default",
            "16K, +170 MB",
            "32K, +505 MB",
            "64K, +1.17 GB",
            "128K, +2.52 GB",
            "256K, +5.20 GB",
        ])
    }

    /// The option is only a face on `ContextAdmission`; if it ever computed its
    /// own figures the menu label and the loader's refusal would drift apart.
    @Test func everyFigureComesFromTheSharedAdmissionRule() {
        for option in AppContextLengthOption.allCases {
            let config = ArchConfig.gemma4_26B_A4B
            #expect(option.fp16KVBytes
                == ContextAdmission.fp16KVBytes(config: config, maxContext: option.tokens))
            #expect(option.projectedFootprintBytes
                == ContextAdmission.projectedFootprintBytes(config: config,
                                                            maxContext: option.tokens))
            #expect(option.minimumHostMemoryBytes
                == ContextAdmission.minimumHostMemoryBytes(config: config,
                                                           maxContext: option.tokens))
        }
    }

    @Test func aneightGigabyteMacIsOfferedEverythingUpTo128K() {
        #expect(AppContextLengthOption.available(on: eightGigabyteHost).map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072])
        #expect(AppContextLengthOption.twoFiftySixK.availability(
            hostMemoryBytes: eightGigabyteHost)
            == .needsMemory(minimumHostBytes: 11_004_477_440))
        #expect(AppContextLengthOption.largestAvailable(on: eightGigabyteHost)
            == .oneTwentyEightK)
    }

    @Test func asixteenGigabyteMacIsOfferedEverySize() {
        #expect(AppContextLengthOption.available(on: sixteenGigabyteHost)
            == AppContextLengthOption.allCases)
        #expect(AppContextLengthOption.largestAvailable(on: sixteenGigabyteHost)
            == .twoFiftySixK)
    }

    /// The rule admits at exactly the minimum. One byte below it 128K has to
    /// disappear from the menu, or an 8 GB Mac whose margin the plan puts at
    /// 270 MB would be offered a context the loader then refuses.
    @Test func themenuBoundaryIsTheAdmissionBoundary() {
        let minimum = AppContextLengthOption.oneTwentyEightK.minimumHostMemoryBytes
        #expect(AppContextLengthOption.available(on: minimum - 1).map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536])
        #expect(AppContextLengthOption.available(on: minimum).map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072])
        #expect(AppContextLengthOption.largestAvailable(on: minimum - 1) == .sixtyFourK)
    }

    @Test func theneedDescriptionNamesTheRequirementAndTheHost() {
        let message = AppContextLengthOption.twoFiftySixK
            .needDescription(hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("262,144"))
        #expect(message.contains("16 GB"))
        #expect(message.contains("8 GB"))
    }

    @Test func defaultOptionMatchesRuntimeDefault() {
        #expect(AppContextLengthOption.defaultOption.tokens
            == 8_192)
        #expect(MacAppSettings().contextTokens
            == AppContextLengthOption.defaultOption.tokens)
        #expect(AppContextLengthOption.defaultOption == .eightK)
        // Every Mac this app runs on has at least 8 GB, so the shipped default
        // is never one of the rows admission hides.
        for host in [eightGigabyteHost, sixteenGigabyteHost] {
            #expect(AppContextLengthOption.available(on: host)
                .contains(AppContextLengthOption.defaultOption),
                "the default context was hidden on a \(host)-byte host")
        }
    }
}
