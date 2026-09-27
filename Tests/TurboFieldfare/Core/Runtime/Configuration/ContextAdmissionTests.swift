import Testing
import Foundation
@testable import TurboFieldfare

/// Pins the admission arithmetic every context-picking surface shares. The
/// byte figures are the FP16 KV allocation the runtime actually makes — 25
/// ring-capped sliding layers plus 5 linear full layers — so a change to the
/// ring geometry or the layer mask fails here before it silently moves a menu
/// label, a server refusal and a CLI warning apart.
@Suite struct ContextAdmissionTests {

    @Test(arguments: [24, 32])
    func extraCacheSlotsAreChargedAtTheAdmissionBoundary(slots: Int) {
        let config = ArchConfig.gemma4_26B_A4B
        let baseline = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072)
        let page = Int(getpagesize())
        let slotBytes = ((3_358_720 + page - 1) / page) * page
        let expected = baseline + UInt64((slots - 16) * 30 * slotBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 131_072,
                                                        expertCacheSlots: slots) == expected)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected - 1, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: expected, expertCacheSlots: slots) == .available)
        #expect(ContextAdmission.availability(config: config, maxContext: 131_072,
                                              hostMemoryBytes: 8 << 30, expertCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
    }

    @Test func smallerCacheDoesNotDiscountTheMeasuredRuntimeAllowance() {
        #expect(ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072,
                                                         expertCacheSlots: 8)
                == ContextAdmission.projectedFootprintBytes(config: config, maxContext: 131_072))
    }

    /// One retained slot is the lineage `fp16KVBytes` already charges, so the
    /// default must leave every figure in this file exactly where it was
    /// before retention existed. An accidental charge here would move a server
    /// refusal and a menu row for operators who never touched the flag.
    @Test(arguments: [4_096, 16_384, 65_536, 262_144])
    func oneRetainedSlotLeavesAdmissionUnchanged(context: Int) {
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: context,
                                                        promptCacheSlots: 1)
                == ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: context))
    }

    /// Each slot past the first is a whole KV lineage, so it is charged at the
    /// same rate as the first one.
    @Test(arguments: [2, 3, 4])
    func retainedSlotsAreChargedAtTheAdmissionBoundary(slots: Int) {
        let baseline = ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 16_384)
        let expected = baseline
            + UInt64(slots - 1) * ContextAdmission.fp16KVBytes(config: config, maxContext: 16_384)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 16_384,
                                                        promptCacheSlots: slots) == expected)
        #expect(ContextAdmission.availability(config: config, maxContext: 16_384,
                                              hostMemoryBytes: expected - 1, promptCacheSlots: slots)
                == .needsMemory(minimumHostBytes: expected))
        #expect(ContextAdmission.availability(config: config, maxContext: 16_384,
                                              hostMemoryBytes: expected, promptCacheSlots: slots) == .available)
    }

    /// The combination the charge exists to catch: every rung is available on
    /// a 16 GB Mac at one slot, but four slots at the top rung is four times
    /// the KV and has to be refused rather than failing in the allocator.
    @Test func fourSlotsAtTheTopRungIsRefusedOnASixteenGigabyteHost() {
        #expect(ContextAdmission.availability(config: config, maxContext: 262_144,
                                              hostMemoryBytes: sixteenGigabyteHost,
                                              promptCacheSlots: 1) == .available)
        #expect(ContextAdmission.availability(config: config, maxContext: 262_144,
                                              hostMemoryBytes: sixteenGigabyteHost,
                                              promptCacheSlots: 4)
                == .needsMemory(minimumHostBytes: 27_911_782_400))
        // Four slots at the default context still fits an 8 GB Mac: the charge
        // scales with the context, so it does not foreclose the common case.
        #expect(ContextAdmission.availability(config: config, maxContext: 16_384,
                                              hostMemoryBytes: eightGigabyteHost,
                                              promptCacheSlots: 4) == .available)
    }


    private let config = ArchConfig.gemma4_26B_A4B
    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let sixteenGigabyteHost: UInt64 = 17_179_869_184

    @Test func fp16KVBytesMatchesTheRuntimeAllocationAtEverySize() {
        let expected: [(context: Int, bytes: UInt64)] = [
            (4_096, 350_945_280),
            (8_192, 434_831_360),
            (16_384, 602_603_520),
            (32_768, 938_147_840),
            (65_536, 1_609_236_480),
            (131_072, 2_951_413_760),
            (262_144, 5_635_768_320),
        ]
        for row in expected {
            #expect(ContextAdmission.fp16KVBytes(config: config, maxContext: row.context)
                    == row.bytes,
                    "context \(row.context)")
        }
    }

    /// Only the five full-attention layers grow with the context; the sliding
    /// ring is fixed. 20,480 B per token across those five is the whole slope.
    @Test func onlyFullAttentionLayersGrowWithContext() {
        let small = ContextAdmission.fp16KVBytes(config: config, maxContext: 8_192)
        let large = ContextAdmission.fp16KVBytes(config: config, maxContext: 16_384)
        #expect(large - small == UInt64((16_384 - 8_192) * 20_480))
    }

    @Test func productionSlidingRingRowsMatchTheRunnersChunkFloor() {
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 8_192)
                == 1_304)
        // A context below the ring size caps the ring at the context.
        #expect(ContextAdmission.productionSlidingRingRows(config: config, maxContext: 1_024)
                == 1_024)
    }

    @Test func eightGigabyteHostBacks128KButNot256K() {
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: eightGigabyteHost)
                == .available)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 262_144,
                                              hostMemoryBytes: eightGigabyteHost)
                == .needsMemory(minimumHostBytes: 11_004_477_440))
    }

    @Test func sixteenGigabyteHostBacksEverySize() {
        for context in [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144] {
            #expect(ContextAdmission.availability(config: config,
                                                  maxContext: context,
                                                  hostMemoryBytes: sixteenGigabyteHost)
                    == .available,
                    "context \(context)")
        }
    }

    /// The rule admits at exactly the minimum, not one byte above it: an
    /// off-by-one here would hide 128K from every 8 GB Mac, whose 270 MB of
    /// margin is the whole question step 9 of the plan measures.
    @Test func admissionBoundaryIsInclusive() {
        let minimum = ContextAdmission.minimumHostMemoryBytes(config: config,
                                                              maxContext: 131_072)
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum - 1)
                == .needsMemory(minimumHostBytes: minimum))
        #expect(ContextAdmission.availability(config: config,
                                              maxContext: 131_072,
                                              hostMemoryBytes: minimum)
                == .available)
    }

    @Test func projectedFootprintIsKVPlusTheMeasuredNonKVRuntime() {
        let projected = ContextAdmission.projectedFootprintBytes(config: config,
                                                                 maxContext: 262_144)
        #expect(projected == 5_635_768_320 + ContextAdmission.nonKVRuntimeFootprintBytes)
        #expect(ContextAdmission.minimumHostMemoryBytes(config: config, maxContext: 262_144)
                == projected + ContextAdmission.hostReserveBytes)
    }

    @Test func needDescriptionNamesBothTheRequirementAndTheHost() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 262_144,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("16 GB"))
        #expect(message.contains("8 GB"))
        #expect(message.contains("262,144"))
    }

    /// 128K needs 8,320,122,880 bytes, which an 8 GB Mac has. Reporting that
    /// as "needs 16 GB" while `availability` calls it available would put the
    /// caption and the menu in direct contradiction.
    @Test func needDescriptionRoundsToASizeThatActuallyBacksTheContext() {
        let message = ContextAdmission.needDescription(config: config,
                                                       maxContext: 131_072,
                                                       hostMemoryBytes: eightGigabyteHost)
        #expect(message.contains("needs 8 GB"))
    }

    @Test func contextCeilingComesFromTheCheckpoint() {
        #expect(config.maxPositionEmbeddings == 262_144)
    }
}
