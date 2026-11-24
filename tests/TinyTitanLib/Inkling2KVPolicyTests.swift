import Testing
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanVariants

/// TT-123: Inkling 2 keeps its K/V cache at 16 bits. Quantized K/V was built and
/// measured (8 bits 95/96, 4 bits 91/96 top-1 against llama.cpp) and rejected,
/// so an explicit request is refused by name and a default one is announced.
@Suite struct Inkling2KVPolicyTests {
    @Test func sixteenBitsIsAccepted() {
        #expect(EngineFactories.inkling2RefusesKVBits(.fp16) == nil)
    }

    @Test func eightAndFourBitsAreRefusedWithTheMeasurement() throws {
        for precision in [KVCachePrecision.int8, .int4] {
            let refusal = try #require(EngineFactories.inkling2RefusesKVBits(precision))
            #expect(refusal.contains("--kv-bits \(precision.rawValue)"))
            #expect(refusal.contains("95/96"))
        }
    }

    @Test func theDefaultIsAnnouncedNotSilentlyIgnored() {
        let notice = EngineFactories.inkling2KVNotice(.int8)
        #expect(notice.contains("does not apply"))
        #expect(notice.contains("16-bit"))
    }
}
