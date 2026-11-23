import Testing

@testable import TinyTitan

/// The window an Inkling2 GGUF session allocates (TT-120). Model-free: the
/// device and the file enter as numbers, so the arithmetic is pinned here and
/// the measured envelope (24 GB M3, 17.76 GiB working set) is a fixture.
@Suite struct Inkling2GGUFContextBudgetTests {
    /// 8 global visits x 8 KV heads x 128 dims x K+V x fp16.
    static let perPosition = 8 * 8 * 128 * 2 * 2
    /// `recommendedMaxWorkingSetSize` of the 24 GB M3 the cap was measured on.
    static let m3WorkingSet = 19_069_665_280
    static let q4File = 17_629_759_456
    static let q2File = 9_138_720_000

    @Test func perPositionCostIsThirtyTwoKilobytes() {
        #expect(Self.perPosition == 32_768)
    }

    @Test func q4OnA24GBMacGetsWhatTheWorkingSetLeaves() {
        let window = Inkling2GGUFContextBudget.affordable(
            requested: 262_144, model: 1_048_576, workingSet: Self.m3WorkingSet,
            weightBytes: Self.q4File, bytesPerPosition: Self.perPosition)
        // (19,069,665,280 - 17,629,759,456 - 536,870,912) / 32,768 = 27,558,
        // rounded down to a multiple of 1,024.
        #expect(window == 26_624)
    }

    @Test func q2OnA24GBMacGetsTheWholeRequest() {
        let window = Inkling2GGUFContextBudget.affordable(
            requested: 262_144, model: 1_048_576, workingSet: Self.m3WorkingSet,
            weightBytes: Self.q2File, bytesPerPosition: Self.perPosition)
        // 8.5 GiB of weights leave room for 286,684 positions.
        #expect(window == 262_144)
    }

    @Test func aSmallerRequestIsKept() {
        let window = Inkling2GGUFContextBudget.affordable(
            requested: 8_192, model: 1_048_576, workingSet: Self.m3WorkingSet,
            weightBytes: Self.q2File, bytesPerPosition: Self.perPosition)
        #expect(window == 8_192)
    }

    @Test func aDeviceTooSmallStillGetsTheMinimum() {
        let window = Inkling2GGUFContextBudget.affordable(
            requested: 262_144, model: 1_048_576, workingSet: Self.q4File,
            weightBytes: Self.q4File, bytesPerPosition: Self.perPosition)
        #expect(window == Inkling2GGUFContextBudget.minimumContext)
    }
}
