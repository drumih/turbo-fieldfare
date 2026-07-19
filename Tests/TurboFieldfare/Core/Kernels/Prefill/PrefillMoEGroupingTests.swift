import Testing
@testable import TurboFieldfare

@Suite struct PrefillMoEGroupingTests {
    @Test func groupingSortsPairsAndBuildsOffsetsCountsAndTiles() throws {
        let pairs = [
            Self.pair(token: 0, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 0, expert: 1, rank: 1, weightBits: 11),
            Self.pair(token: 0, expert: 4, rank: 2, weightBits: 12),
            Self.pair(token: 1, expert: 1, rank: 0, weightBits: 13),
            Self.pair(token: 1, expert: 3, rank: 1, weightBits: 14),
            Self.pair(token: 1, expert: 2, rank: 2, weightBits: 15),
            Self.pair(token: 2, expert: 4, rank: 0, weightBits: 16),
            Self.pair(token: 2, expert: 3, rank: 1, weightBits: 17),
            Self.pair(token: 2, expert: 1, rank: 2, weightBits: 18),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 19),
            Self.pair(token: 3, expert: 1, rank: 1, weightBits: 20),
            Self.pair(token: 3, expert: 4, rank: 2, weightBits: 21),
        ].reversed()

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            Array(pairs),
            queryCount: 4,
            topK: 3,
            numExperts: 6,
            tileExpertCount: 2)

        #expect(grouped.sortedPairs.map { $0.expert } == [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5])
        #expect(grouped.sortedPairs.map { [$0.token, $0.rank] } == [
            [0, 1], [1, 0], [2, 2], [3, 1],
            [1, 2],
            [0, 0], [1, 1], [2, 1],
            [0, 2], [2, 0], [3, 2],
            [3, 0],
        ])
        #expect(grouped.perExpertOffsets == [UInt32.max, 0, 4, 5, 8, 11])
        #expect(grouped.perExpertCounts == [0, 4, 1, 3, 3, 1])
        #expect(grouped.groups == [
            PrefillMoEGroup(expert: 1, pairStart: 0, pairCount: 4),
            PrefillMoEGroup(expert: 2, pairStart: 4, pairCount: 1),
            PrefillMoEGroup(expert: 3, pairStart: 5, pairCount: 3),
            PrefillMoEGroup(expert: 4, pairStart: 8, pairCount: 3),
            PrefillMoEGroup(expert: 5, pairStart: 11, pairCount: 1),
        ])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 2, pairStart: 0, pairCount: 5),
            PrefillMoETile(groupStart: 2, groupCount: 2, pairStart: 5, pairCount: 6),
            PrefillMoETile(groupStart: 4, groupCount: 1, pairStart: 11, pairCount: 1),
        ])
        #expect(grouped.maxPairsPerExpert == 4)
        #expect(grouped.maxPairsPerTile == 6)
        #expect(grouped.maxLiveExpertsPerTile == 2)
    }

    @Test func groupingIsDeterministicAcrossInputOrderAndPreservesWeightBits() throws {
        let tokenMajor = [
            Self.pair(token: 0, expert: 2, rank: 0, weightBits: 101),
            Self.pair(token: 0, expert: 1, rank: 1, weightBits: 102),
            Self.pair(token: 1, expert: 2, rank: 0, weightBits: 103),
            Self.pair(token: 1, expert: 0, rank: 1, weightBits: 104),
            Self.pair(token: 2, expert: 1, rank: 0, weightBits: 105),
            Self.pair(token: 2, expert: 2, rank: 1, weightBits: 106),
        ]
        let shuffled = [tokenMajor[5], tokenMajor[1], tokenMajor[3],
                        tokenMajor[0], tokenMajor[4], tokenMajor[2]]

        let a = try PrefillMoEGrouping.groupTokenExpertPairs(
            tokenMajor,
            queryCount: 3,
            topK: 2,
            numExperts: 3,
            tileExpertCount: 2)
        let b = try PrefillMoEGrouping.groupTokenExpertPairs(
            shuffled,
            queryCount: 3,
            topK: 2,
            numExperts: 3,
            tileExpertCount: 2)

        #expect(a == b)
        #expect(a.sortedPairs.map { $0.weightBitsAndReserved } == [104, 102, 105, 101, 103, 106])
    }

    @Test func groupingCanOrderTilesByExpertSortKeysWhileKeepingPairRangesContiguous() throws {
        let pairs = [
            Self.pair(token: 0, expert: 1, rank: 0, weightBits: 10),
            Self.pair(token: 0, expert: 2, rank: 1, weightBits: 20),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 30),
            Self.pair(token: 1, expert: 1, rank: 1, weightBits: 11),
            Self.pair(token: 2, expert: 2, rank: 0, weightBits: 21),
            Self.pair(token: 2, expert: 3, rank: 1, weightBits: 31),
        ]

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 4,
            tileExpertCount: 2,
            expertSortKeys: [0, 30, 10, 20])

        #expect(grouped.groups.map(\.expert) == [2, 3, 1])
        #expect(grouped.sortedPairs.map(\.expert) == [2, 2, 3, 3, 1, 1])
        #expect(grouped.perExpertOffsets == [UInt32.max, 4, 0, 2])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 2, pairStart: 0, pairCount: 4),
            PrefillMoETile(groupStart: 2, groupCount: 1, pairStart: 4, pairCount: 2),
        ])
        for tile in grouped.tiles {
            let slice = grouped.sortedPairs[Int(tile.pairStart)..<Int(tile.pairStart + tile.pairCount)]
            let tileExperts = grouped.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                .map(\.expert)
            #expect(Set(slice.map(\.expert)) == Set(tileExperts))
        }
    }

    @Test func groupingRejectsInvalidMetadataBeforeKernelUse() throws {
        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 1,
                tileExpertCount: 17)
        } throws: { error in
            if case PrefillMoEGroupingError.invalidTileExpertCount(17) = error { return true }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 2, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 2,
                tileExpertCount: 1)
        } throws: { error in
            if case PrefillMoEGroupingError.expertOutOfRange(2) = error { return true }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [
                    Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1),
                    Self.pair(token: 0, expert: 1, rank: 0, weightBits: 2),
                ],
                queryCount: 1,
                topK: 2,
                numExperts: 2,
                tileExpertCount: 1)
        } throws: { error in
            if case PrefillMoEGroupingError.duplicateTokenRank(token: 0, rank: 0) = error {
                return true
            }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 2,
                tileExpertCount: 1,
                expertSortKeys: [0])
        } throws: { error in
            if case PrefillMoEGroupingError.expertSortKeyCountMismatch(expected: 2, actual: 1) = error {
                return true
            }
            return false
        }
    }

    @Test func groupingToleratesRealTracePairPressureAtT32() throws {
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(256)
        for token in 0..<32 {
            for rank in 0..<8 {
                let expert = UInt32((token + rank * 7) % 74)
                pairs.append(Self.pair(token: UInt32(token),
                                       expert: expert,
                                       rank: UInt32(rank),
                                       weightBits: UInt32(token * 8 + rank)))
            }
        }

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 32,
            topK: 8,
            numExperts: 128,
            tileExpertCount: 16)

        #expect(grouped.sortedPairs.count == 256)
        #expect(grouped.groups.count <= 74)
        #expect(grouped.maxLiveExpertsPerTile <= 16)
        #expect(grouped.maxPairsPerTile <= 256)
        for tile in grouped.tiles {
            #expect(tile.groupCount <= 16)
            #expect(tile.pairCount > 0)
        }
    }

    private static func pair(token: UInt32,
                             expert: UInt32,
                             rank: UInt32,
                             weightBits: UInt32) -> PrefillTokenExpertPair {
        PrefillTokenExpertPair(token: token,
                               expert: expert,
                               rank: rank,
                               weightBitsAndReserved: weightBits)
    }
}
