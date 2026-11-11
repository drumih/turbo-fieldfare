import Testing

@testable import TinyTitan

/// The PLE n-gram window: the token, then up to `ngram_size - 1` predecessors,
/// newest first. Qwen3.8-Flash-Next declares `ple.ngram_size = 3`; a window one
/// token short hashes a different 3-gram row from position 2 on, which is the
/// defect this pins (the decoder used to pass only one predecessor, and none at
/// all for the first generated token).
@Suite struct GGUFPLEWindowTests {
    private let sequence: [Int32] = [760, 6511, 314, 9338, 369, 11751]

    @Test func windowReachesTwoPredecessorsForATrigram() {
        #expect(GGUFPLEStage.window(of: sequence, at: 4, ngramSize: 3) == [369, 9338, 314])
        #expect(GGUFPLEStage.window(of: sequence, at: 5, ngramSize: 3) == [11751, 369, 9338])
    }

    @Test func windowIsTruncatedAtTheSequenceStart() {
        #expect(GGUFPLEStage.window(of: sequence, at: 0, ngramSize: 3) == [760])
        #expect(GGUFPLEStage.window(of: sequence, at: 1, ngramSize: 3) == [6511, 760])
    }

    @Test func windowFollowsTheDeclaredSize() {
        #expect(GGUFPLEStage.window(of: sequence, at: 4, ngramSize: 2) == [369, 9338])
        #expect(GGUFPLEStage.window(of: sequence, at: 4, ngramSize: 1) == [369])
    }

    /// The window feeds `PLEHash.rows`, which reads slot 2 for the 3-gram heads:
    /// dropping the second predecessor changes exactly those rows.
    @Test func aShortWindowChangesTheTrigramRows() {
        let hash = PLEHash(
            multipliers: [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            offsets: (0..<16).map { UInt64($0) * 20_000_003 },
            vocabSizes: Array(repeating: 20_000_003, count: 16),
            ngramSize: 3, headsPerNgram: 8, eosTokenID: 248_044)
        let full = hash.rows(context: GGUFPLEStage.window(of: sequence, at: 4, ngramSize: 3))
        let short = hash.rows(context: [369, 9338])
        #expect(full.count == 16)
        #expect(Array(full[0..<8]) == Array(short[0..<8]))
        #expect(Array(full[8..<16]) != Array(short[8..<16]))
    }
}
