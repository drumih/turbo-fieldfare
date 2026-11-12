/// What a verify pass costs, against a scalar token. Linear in the pass's row
/// count, which is the shape gate 2 of docs/plan-wide-verify.md fits
/// (`TinyTitanBench passcost`); the defaults are the plan's projection and are
/// replaced by the measured fit before anything ships.
public struct WideVerifyCostModel: Sendable, Equatable {
    public var scalarMilliseconds: Double
    public var passBaseMilliseconds: Double
    public var passRowMilliseconds: Double

    public static let planProjection = WideVerifyCostModel(
        scalarMilliseconds: 211, passBaseMilliseconds: 370, passRowMilliseconds: 55)

    public init(
        scalarMilliseconds: Double, passBaseMilliseconds: Double, passRowMilliseconds: Double
    ) {
        precondition(
            scalarMilliseconds > 0 && passBaseMilliseconds >= 0 && passRowMilliseconds >= 0,
            "cost model needs a positive scalar token and non-negative pass terms")
        self.scalarMilliseconds = scalarMilliseconds
        self.passBaseMilliseconds = passBaseMilliseconds
        self.passRowMilliseconds = passRowMilliseconds
    }

    public func passMilliseconds(rows: Int) -> Double {
        passBaseMilliseconds + passRowMilliseconds * Double(rows)
    }
}

/// Decides, per step, whether to verify a draft and how much of it.
///
/// Acceptance is modelled per drafted position as geometric: given the tokens
/// before it were accepted, a drafted token is accepted with probability `p`,
/// estimated separately for each matched n-gram length (a long match copies
/// more reliably than a two-token one). A pass of `d` drafted tokens then
/// emits `1 + p + ... + p^d` tokens in expectation -- row 0 always yields the
/// target's own token -- at `passMilliseconds(rows: d + 1)`. The controller
/// takes the `d` with the best expected tokens per millisecond, and drafts only
/// when that beats a scalar token's `1 / scalarMilliseconds`.
///
/// The estimates forget at `decay` per recorded pass, so a session that moves
/// from an edit (copy-heavy) to prose (not) stops paying for passes quickly.
public struct WideVerifyController: Sendable {
    public let costModel: WideVerifyCostModel
    /// Rows per pass, the boundary token included: W in the plan.
    public let maxRows: Int
    public let decay: Double
    /// Accepted and rejected drafted positions, per matched n (index = n).
    private var accepted: [Double]
    private var rejected: [Double]

    /// Pseudo-counts behind each estimate before any pass is seen: a prior of
    /// `priorAcceptance` worth `priorWeight` observations.
    public init(
        costModel: WideVerifyCostModel = .planProjection,
        maxRows: Int = 8,
        decay: Double = 0.98,
        priorAcceptance: Double = 0.6,
        priorWeight: Double = 4
    ) {
        precondition(maxRows >= 2, "a verify pass has the boundary row and at least one draft")
        precondition(decay > 0 && decay <= 1, "decay is in (0, 1]")
        precondition(priorAcceptance > 0 && priorAcceptance < 1 && priorWeight > 0)
        self.costModel = costModel
        self.maxRows = maxRows
        self.decay = decay
        let slots = PromptLookupDrafter.maximumN + 1
        accepted = Array(repeating: priorAcceptance * priorWeight, count: slots)
        rejected = Array(repeating: (1 - priorAcceptance) * priorWeight, count: slots)
    }

    /// The per-position acceptance estimate for a match of length `matchedN`.
    public func acceptance(matchedN: Int) -> Double {
        let n = bucket(matchedN)
        return accepted[n] / (accepted[n] + rejected[n])
    }

    /// Expected tokens one pass emits with `drafted` tokens at acceptance `p`.
    public static func expectedEmitted(drafted: Int, acceptance p: Double) -> Double {
        // 1 + p + ... + p^drafted, written so p == 1 is exact.
        var total = 1.0
        var term = 1.0
        for _ in 0..<max(0, drafted) {
            term *= p
            total += term
        }
        return total
    }

    /// How many of `available` drafted tokens to verify; 0 means take a scalar
    /// step instead.
    public func draftLength(available: Int, matchedN: Int) -> Int {
        guard matchedN > 0 else { return 0 }
        let p = acceptance(matchedN: matchedN)
        let scalarRate = 1 / costModel.scalarMilliseconds
        var best = 0
        var bestRate = scalarRate
        for drafted in stride(from: 1, through: min(available, maxRows - 1), by: 1) {
            let rate =
                Self.expectedEmitted(drafted: drafted, acceptance: p)
                / costModel.passMilliseconds(rows: drafted + 1)
            if rate > bestRate {
                best = drafted
                bestRate = rate
            }
        }
        return best
    }

    /// Record a verified pass: `drafted` tokens were proposed for a match of
    /// length `matchedN` and the first `acceptedCount` of them were accepted.
    public mutating func record(matchedN: Int, drafted: Int, acceptedCount: Int) {
        precondition(drafted > 0 && (0...drafted).contains(acceptedCount))
        for index in accepted.indices {
            accepted[index] *= decay
            rejected[index] *= decay
        }
        let n = bucket(matchedN)
        accepted[n] += Double(acceptedCount)
        // The first rejection ends the run; positions after it were never
        // judged, so they are not evidence either way.
        if acceptedCount < drafted { rejected[n] += 1 }
    }

    private func bucket(_ matchedN: Int) -> Int {
        min(max(matchedN, 0), PromptLookupDrafter.maximumN)
    }
}
