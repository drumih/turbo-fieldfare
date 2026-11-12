import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

@Suite struct DraftAcceptanceTests {
    /// A stand-in target: the token after a context is a fixed hash of it, and
    /// "sampling" with a per-position seed mixes the position in -- the shape
    /// of the real sampler's determinism, with no model.
    static func target(_ context: [Int32], position: Int, vocab: Int32 = 5) -> Int32 {
        var hash: UInt64 = 1469598103934665603
        for token in context {
            hash = (hash ^ UInt64(UInt32(bitPattern: token))) &* 1099511628211
        }
        hash = (hash ^ UInt64(position)) &* 1099511628211
        return Int32(hash % UInt64(vocab))
    }

    /// What scalar decode emits for `count` tokens after `history`.
    static func scalar(_ history: [Int32], count: Int) -> [Int32] {
        var context = history
        var out: [Int32] = []
        for _ in 0..<count {
            let token = target(context, position: context.count)
            out.append(token)
            context.append(token)
        }
        return out
    }

    static func verify(_ history: [Int32], drafts: [Int32]) -> DraftAcceptance.Outcome {
        DraftAcceptance.resolve(drafts: drafts[...]) { row, accepted in
            let context = history + accepted
            return target(context, position: history.count + row)
        }
    }

    @Test func allAcceptedEmitsEveryDraftPlusTheBonusToken() {
        let history: [Int32] = [1, 2, 3]
        let truth = Self.scalar(history, count: 6)
        let outcome = Self.verify(history, drafts: Array(truth.prefix(5)))
        #expect(outcome.acceptedDrafts == 5)
        #expect(outcome.emitted == truth)
    }

    @Test func aWrongDraftIsReplacedByTheTargetsToken() {
        let history: [Int32] = [4, 4, 0]
        let truth = Self.scalar(history, count: 3)
        var drafts = Array(truth.prefix(3))
        drafts[1] = (drafts[1] + 1) % 5
        let outcome = Self.verify(history, drafts: drafts)
        #expect(outcome.acceptedDrafts == 1)
        #expect(outcome.emitted == Array(truth.prefix(2)))
    }

    @Test func noDraftsIsAScalarStep() {
        let history: [Int32] = [2]
        let outcome = Self.verify(history, drafts: [])
        #expect(outcome.acceptedDrafts == 0)
        #expect(outcome.emitted == Self.scalar(history, count: 1))
    }

    @Test func rowsAfterTheFirstRejectionAreNeverSampled() {
        var sampledRows: [Int] = []
        let outcome = DraftAcceptance.resolve(drafts: [7, 7, 7][...]) { row, _ in
            sampledRows.append(row)
            return row == 0 ? 7 : 3
        }
        #expect(sampledRows == [0, 1])
        #expect(outcome.emitted == [7, 3])
    }

    /// The property the plan rests on: speculation emits exactly the scalar
    /// stream, whatever the drafts, when sampling is a function of context and
    /// position.
    @Test func speculativeDecodeEqualsScalarDecodeForAnyDrafts() {
        var generator = SplitMix64(seed: 42)
        for _ in 0..<200 {
            var history: [Int32] = [Int32(generator.next() % 5)]
            let truth = Self.scalar(history, count: 40)
            var produced: [Int32] = []
            while produced.count < 40 {
                let width = Int(generator.next() % 8)
                let drafts = (0..<width).map { offset in
                    // Half the drafts copy the truth, half guess.
                    generator.next() % 2 == 0 && produced.count + offset < truth.count
                        ? truth[produced.count + offset] : Int32(generator.next() % 5)
                }
                let outcome = Self.verify(history, drafts: drafts)
                produced += outcome.emitted
                history += outcome.emitted
            }
            #expect(Array(produced.prefix(40)) == truth)
        }
    }
}
