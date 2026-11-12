import Foundation
import Metal

/// Opt-in wide verify for `runRawCompletion` (docs/plan-wide-verify.md,
/// task 11): drafts copied from the prompt and the output so far are verified
/// several rows per pass, and the loop emits exactly the tokens scalar decode
/// would have. Off unless a caller passes options; the server reads
/// `TINYTITAN_WIDE_VERIFY=1`.
public struct WideVerifyDecodeOptions: Sendable, Equatable {
    /// Rows per pass, the boundary token included (W). Clamped to the
    /// runner's `wideVerifyMaximumRows`.
    public var maxRows: Int
    public var costModel: WideVerifyCostModel
    public var minN: Int
    public var maxN: Int

    public init(
        maxRows: Int = 9,
        costModel: WideVerifyCostModel = .measuredDedicatedVerify,
        minN: Int = 2,
        maxN: Int = PromptLookupDrafter.maximumN
    ) {
        self.maxRows = maxRows
        self.costModel = costModel
        self.minN = minN
        self.maxN = maxN
    }

    /// `TINYTITAN_WIDE_VERIFY=1` turns the mode on; `TINYTITAN_WIDE_VERIFY_ROWS`
    /// optionally narrows W (2...9). Nil when the mode is off.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> WideVerifyDecodeOptions? {
        guard environment["TINYTITAN_WIDE_VERIFY"] == "1" else { return nil }
        var options = WideVerifyDecodeOptions()
        if let raw = environment["TINYTITAN_WIDE_VERIFY_ROWS"], let rows = Int(raw),
            rows >= 2
        {
            options.maxRows = rows
        }
        return options
    }
}

/// What wide verify did during one generation.
public struct WideVerifyDecodeStats: Sendable, Equatable {
    /// The W the generation ran at (0 when the producer could not verify).
    public var maxRows = 0
    public var passes = 0
    /// Steps where the controller declined (no match, or a pass would not pay).
    public var scalarSteps = 0
    public var draftedTokens = 0
    public var acceptedDrafts = 0
    /// Passes that rejected at least one draft.
    public var rejectingPasses = 0
    /// Tokens the passes emitted that reached the output, after stop truncation.
    public var emittedTokens = 0
    /// Passes whose emitted tokens a stop cut short (rolled back past rows a
    /// pass had already fed).
    public var passesCutAtStop = 0
    public var passNanos: UInt64 = 0

    public var acceptanceRate: Double {
        draftedTokens > 0 ? Double(acceptedDrafts) / Double(draftedTokens) : 0
    }

    /// One log line for a real run.
    public var summary: String {
        String(
            format:
                "TinyTitan wide-verify rows=%d passes=%d drafted=%d accepted=%d "
                + "acceptance=%.1f%% rejecting_passes=%d emitted=%d emitted_per_pass=%.3f "
                + "scalar_steps=%d pass_ms=%.1f",
            maxRows, passes, draftedTokens, acceptedDrafts, acceptanceRate * 100,
            rejectingPasses, emittedTokens,
            passes > 0 ? Double(emittedTokens) / Double(passes) : 0,
            scalarSteps, passes > 0 ? Double(passNanos) / Double(passes) / 1_000_000 : 0)
    }
}

/// The decode loop's side of wide verify: drafter, controller, the pass, and
/// the rollback owed after it.
///
/// Row i of a pass feeds `x_i` at `P + i`; `x_0` is the token scalar decode
/// would feed next. Accepting k drafts emits k + 1 tokens: `x_1..x_k` (already
/// fed by rows 1...k) and the target's own token for row k (not fed). The loop
/// pops the emitted tokens one at a time through its ordinary stop logic, and
/// the rollback is settled when that is done:
///
/// - a token popped at index e < k with no stop was fed by row e + 1, so the
///   loop only advances its position;
/// - the last token (e == k) is pending like any sampled token: the pass is
///   rolled back to k + 1 rows (x_0..x_k), and the loop advances as usual;
/// - a stop at index e (stop token, stop string, `maxNewTokens`, external):
///   scalar decode would have fed x_0..x_e and then sampled the stopping
///   token without feeding it, so the pass is rolled back to e + 1 rows. For
///   e < k that discards fed rows past the stop, which the returned
///   `kvPosition` / `kvBackedTokenIDs` (and any continuation built on them)
///   would otherwise disagree with.
///
/// Between the pass and the settle only host work runs (detokenizing, stop
/// matching, callbacks), as the rollback window's contract requires. An error
/// thrown in between leaves the window open; the next generation's
/// `resetSequence` / `prepareForContinuation` closes it, as for the benches.
final class WideVerifyDecodeDriver {
    private struct Pass {
        let tokens: [Int32]
        let startPosition: Int
        let acceptedDrafts: Int
        let emitted: [Int32]
        var popped = 0
    }

    private let runner: RealForwardRunner
    private let maxRows: Int
    private let greedyRows: Bool
    private var drafter: PromptLookupDrafter
    private var controller: WideVerifyController
    private let sampler: WideVerifySampler?
    private let rowLogits: MTLBuffer?
    private var pass: Pass?
    private(set) var stats = WideVerifyDecodeStats()

    /// Nil when the request must stay on scalar decode: a producer other than
    /// the plain runner (MTP decoder, batched multi-slot producer), a slot
    /// other than 0, a grammar, or a runner with no room for a two-row pass.
    init?(
        options: WideVerifyDecodeOptions?,
        producer: any LogitProducer,
        slot: Int,
        config: GenerationConfig,
        scratch: RawCompletionScratch,
        context: MetalContext,
        fusedGreedy: Bool
    ) throws {
        guard let options, slot == 0, config.constraint == nil,
            let runner = producer as? RealForwardRunner
        else { return nil }
        let rows = min(options.maxRows, runner.wideVerifyMaximumRows)
        guard rows >= 2 else { return nil }
        self.runner = runner
        self.maxRows = rows
        // The fused head's GPU argmax is what scalar decode emits there, so the
        // pass's argmax stands in for it. With a logits head scalar decode runs
        // the sampler even for greedy configs, so the rows run it too.
        self.greedyRows = fusedGreedy
        self.drafter = PromptLookupDrafter(minN: options.minN, maxN: options.maxN)
        self.controller = WideVerifyController(costModel: options.costModel, maxRows: rows)
        if fusedGreedy {
            sampler = nil
            rowLogits = nil
        } else {
            sampler = try WideVerifySampler(scratch: scratch, context: context, config: config)
            rowLogits = try runner.makeWideVerifyRowLogitsBuffer(rows: rows)
        }
        stats.maxRows = rows
    }

    /// Feeds the drafter a committed token (prompt or output).
    func observe<S: Sequence>(_ tokens: S) where S.Element == Int32 {
        drafter.append(contentsOf: tokens)
    }

    /// The next token the open pass emitted, if any.
    func popEmitted() -> Int32? {
        guard var current = pass, current.popped < current.emitted.count else { return nil }
        let token = current.emitted[current.popped]
        current.popped += 1
        pass = current
        return token
    }

    /// After a popped token passed the stop checks: true when a pass row
    /// already fed it (the loop advances its position and skips the feed).
    /// Otherwise any open pass is settled and the loop feeds the token.
    func consumeFed() throws -> Bool {
        guard let current = pass else { return false }
        if current.popped <= current.acceptedDrafts { return true }
        try settle(rows: current.popped)
        return false
    }

    /// The loop stopped on the last popped token: keep only the rows scalar
    /// decode would have fed.
    func settleAtStop() throws {
        guard let current = pass else { return }
        if current.popped < current.emitted.count { stats.passesCutAtStop += 1 }
        try settle(rows: current.popped)
    }

    private func settle(rows: Int) throws {
        guard let current = pass else { return }
        pass = nil
        stats.emittedTokens += current.popped
        if rows == current.tokens.count {
            runner.endWideVerifyRollbackWindow()
        } else {
            try runner.rollbackWideVerify(
                acceptedRows: rows, tokens: current.tokens,
                startPosition: current.startPosition)
        }
    }

    /// Runs one pass for `boundary` (the last entry of `history`, not yet fed)
    /// at `position` when the controller wants one. False: take a scalar step.
    func tryPass(
        boundary: Int32, history: [Int32], position: Int, generated: Int, remaining: Int
    ) async throws -> Bool {
        // Emitting at most `remaining` tokens keeps the pass inside the
        // context bound the loop already checked.
        let limit = min(maxRows - 1, remaining - 1)
        let proposal = drafter.propose(limit: limit)
        let drafted = controller.draftLength(
            available: proposal.tokens.count, matchedN: proposal.matchedN)
        guard drafted > 0 else {
            stats.scalarSteps += 1
            return false
        }
        let tokens = [boundary] + proposal.tokens.prefix(drafted)
        let drafts = tokens[1...]
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let outcome: DraftAcceptance.Outcome
        if greedyRows {
            let verified = try await runner.verifyGreedyRows(tokens, startPosition: position)
            outcome = DraftAcceptance.resolve(drafts: drafts) { row, _ in
                verified.predictions[row]
            }
        } else if let sampler, let rowLogits {
            _ = try await runner.verifyRowLogits(tokens, startPosition: position, into: rowLogits)
            outcome = try sampler.resolve(
                rowLogits: rowLogits, drafts: drafts, history: history,
                firstSampleIndex: generated, timing: runner)
        } else {
            throw ModelError.internalInconsistency(detail: "wide verify sampler missing")
        }
        stats.passNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- start
        controller.record(
            matchedN: proposal.matchedN, drafted: drafted,
            acceptedCount: outcome.acceptedDrafts)
        stats.passes += 1
        stats.draftedTokens += drafted
        stats.acceptedDrafts += outcome.acceptedDrafts
        if outcome.acceptedDrafts < drafted { stats.rejectingPasses += 1 }
        pass = Pass(
            tokens: tokens, startPosition: position,
            acceptedDrafts: outcome.acceptedDrafts, emitted: outcome.emitted)
        return true
    }
}
