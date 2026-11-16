import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

/// The indexer's selection: which keys a query is allowed to attend to.
///
/// The kernels around it (pool, norm, rope, score) are checked against the
/// reference implementation numerically; this pins the selection rule, which
/// is where the reference's semantics are easiest to get subtly wrong and
/// where being wrong costs no error and no crash.
@Suite("QSA key selection")
struct QSAIndexerTests {
    private static let config = SparseIndexerConfig(
        numHeads: 4, numKVHeads: 1, headDim: 128,
        budget: 64, compressRatio: 4)

    private static func makeIndexer(capacity: Int = 512) throws -> QSAIndexer {
        try QSAIndexer(
            context: try MetalContext(), config: config,
            ropeTheta: 10_000_000, capacity: capacity)
    }

    /// Writes block scores straight into the indexer's scratch, standing in
    /// for the scoring kernel so the selection can be driven with known
    /// rankings.
    private static func setScores(_ indexer: QSAIndexer, _ scores: [Float]) {
        let ptr = indexer.scoresForTesting.contents()
            .bindMemory(to: Float.self, capacity: scores.count)
        for (i, value) in scores.enumerated() { ptr[i] = value }
    }

    private static func keptCells(_ buffer: MTLBuffer, count: Int) -> [Int] {
        let ptr = buffer.contents().bindMemory(to: UInt8.self, capacity: count)
        return (0..<count).filter { ptr[$0] != 0 }
    }

    @Test("Inside the window nothing is selected")
    func denseWindowSkipsSelection() throws {
        let indexer = try Self.makeIndexer()
        // budget 64 + ratio 4 - 1 = 67 cells, so 67 visible keys are all kept
        // and the caller should get no mask at all rather than an all-ones one.
        #expect(indexer.selectionWidth == 67)
        #expect(indexer.selectKeys(visibleKeys: 67) == nil)
        #expect(indexer.selectKeys(visibleKeys: 68) != nil)
    }

    @Test("The query's ragged tail is kept regardless of score")
    func tailAlwaysSurvives() throws {
        let indexer = try Self.makeIndexer()
        let visible = 103  // 25 complete blocks + 3 tail
        let blocks = visible / 4
        // Every complete block scores far above zero; the tail cells belong to
        // no complete block and must still be kept.
        Self.setScores(indexer, (0..<blocks + 1).map { _ in 1000 })
        guard let mask = indexer.selectKeys(visibleKeys: visible) else {
            Issue.record("expected a selection past the window")
            return
        }
        let kept = Self.keptCells(mask, count: visible)
        #expect(kept.contains(100) && kept.contains(101) && kept.contains(102))
        #expect(kept.count == indexer.selectionWidth)
    }

    @Test("Blocks are taken by score, and the budget can cut one in half")
    func budgetCutsTheLastBlock() throws {
        let indexer = try Self.makeIndexer()
        // 68 visible keys: 17 complete blocks, no tail (68 % 4 == 0). The
        // budget is 67 cells, so 16 blocks go in whole and the 17th
        // contributes 3 of its 4 -- lowest index first.
        let visible = 68
        var scores = [Float](repeating: 0, count: 18)
        for b in 0..<17 { scores[b] = Float(17 - b) }  // block 0 best
        Self.setScores(indexer, scores)
        guard let mask = indexer.selectKeys(visibleKeys: visible) else {
            Issue.record("expected a selection past the window")
            return
        }
        let kept = Self.keptCells(mask, count: visible)
        #expect(kept.count == 67)
        // Block 16 is the worst, so it is the one cut: cells 64, 65, 66 in,
        // cell 67 out.
        #expect(kept.contains(64) && kept.contains(66))
        #expect(!kept.contains(67))
    }

    @Test("Equal scores break toward the lower block index")
    func tiesFavourLowerIndex() throws {
        let indexer = try Self.makeIndexer()
        let visible = 72  // 18 complete blocks, no tail
        // All equal: the reference's stable cell ordering reduces to taking
        // the lowest block indices, because every cell in a block shares its
        // block's score.
        Self.setScores(indexer, [Float](repeating: 5, count: 19))
        guard let mask = indexer.selectKeys(visibleKeys: visible) else {
            Issue.record("expected a selection past the window")
            return
        }
        let kept = Self.keptCells(mask, count: visible)
        #expect(kept.count == 67)
        #expect(kept.first == 0)
        #expect(kept.last == 66)
    }
}
