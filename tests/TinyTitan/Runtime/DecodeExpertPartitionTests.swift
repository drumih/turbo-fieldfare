import Testing

@testable import TinyTitan
import TinyTitanCore

@Suite struct DecodeExpertPartitionTests {
    @Test func mixedResidencyPreservesTopKPositions() {
        var hits: [UInt32] = []
        var misses: [UInt32] = []

        DecodeExpertPartition.populate(
            topK: 8,
            missIndices: [1, 4, 7],
            hits: &hits,
            misses: &misses)

        #expect(hits == [0, 2, 3, 5, 6])
        #expect(misses == [1, 4, 7])
    }

    @Test func reusedScratchDoesNotRetainPriorPartition() {
        var hits: [UInt32] = [99]
        var misses: [UInt32] = [98]

        DecodeExpertPartition.populate(
            topK: 4,
            missIndices: [],
            hits: &hits,
            misses: &misses)
        #expect(hits == [0, 1, 2, 3])
        #expect(misses.isEmpty)

        DecodeExpertPartition.populate(
            topK: 4,
            missIndices: [0, 1, 2, 3],
            hits: &hits,
            misses: &misses)
        #expect(hits.isEmpty)
        #expect(misses == [0, 1, 2, 3])
    }
}
