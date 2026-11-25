import Foundation
import TinyTitanCore

/// Greedy speculative decode loop using Inkling 2's bundled DSpark MTP draft
/// model (TT-129): proposes 7 draft tokens per verify block of up to 8 tokens,
/// verifies in one target chunk pass, and rolls back conv states on rejection.
/// lint:allow-long speculative loop tracks prefill, draft, verify, rollback,
/// stop matching, and acceptance counters in one tight state machine.
func runInklingQ4MTPCompletion(
    _ request: RawCompletionRequest, runner: InklingQ4Runner, stops: Set<Int32>
) async throws -> RawDecodeResult {
    let config = request.configuration
    let prompt = request.promptIDs
    let tokenizer = request.tokenizer

    let prefillStart = Date()
    var history: [Int32] = []
    history.reserveCapacity(prompt.count)
    var segmentStart = 0
    for point in request.checkpointPositions.sorted() where point > 0 && point < prompt.count {
        _ = try runner.prefillMTP(prompt[segmentStart..<point].map(Int.init), at: segmentStart)
        history.append(contentsOf: prompt[segmentStart..<point])
        request.onProgress(.prefill(done: point, total: prompt.count))
        segmentStart = point
        request.onPrefillCheckpoint?(point)
    }
    let prefill = try runner.prefillMTP(prompt[segmentStart...].map(Int.init), at: segmentStart)
    history.append(contentsOf: prompt[segmentStart...])
    request.onProgress(.prefill(done: prompt.count, total: prompt.count))
    let prefillSeconds = Date().timeIntervalSince(prefillStart)

    let decodeStart = Date()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var text = InklingQ4TextStream(tokenizer: tokenizer)
    var generated = 0
    var reason: StopReason = .maxTokens
    var boundary: [Int32] = []

    var anchorToken = prefill.next
    var lastTapRow = prefill.lastTapRow
    var totalDrafted = 0
    var totalAccepted = 0
    var verifyPasses = 0
    var draftSec = 0.0
    var verifySec = 0.0
    var commitSec = 0.0

    func emit(_ token: Int32) -> Bool {
        generated += 1
        boundary = [token]
        if stops.contains(token) {
            reason = token == tokenizer.endOfTurnID ? .endOfTurn : .eos
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            return true
        }
        let visible = stopMatcher.push(text.push(token))
        request.onProgress(.token(index: generated - 1, id: token, delta: visible))
        if stopMatcher.isStopped || request.shouldStop() || generated >= config.maxNewTokens {
            let tail = stopMatcher.push(text.flush()) + stopMatcher.finish()
            request.onProgress(.tail(tail))
            reason =
                stopMatcher.isStopped
                ? .stopString
                : (generated >= config.maxNewTokens ? .maxTokens : .external)
            return true
        }
        return false
    }

    while true {
        try Task.checkCancellation()
        let anchorID = Int32(anchorToken)
        if emit(anchorID) { break }

        let pos = runner.position(forSlot: 0)
        let remainingGen = config.maxNewTokens - generated
        let remainingCtx = runner.maxContext - pos - 1
        let maxDraft = min(remainingGen, remainingCtx)

        if maxDraft <= 0 {
            history.append(anchorID)
            let next = try runner.feed([anchorToken], at: pos, slot: 0, logits: false).next
            boundary = []
            anchorToken = next
            continue
        }

        let t0 = Date()
        let fullDraft = try runner.draftMTP(
            anchorToken: anchorToken, position: pos, lastTapRow: lastTapRow)
        let t1 = Date()
        let draft = Array(fullDraft.prefix(maxDraft))
        totalDrafted += draft.count
        verifyPasses += 1

        let block = [anchorToken] + draft
        let verifiedNext = try runner.verifyMTP(block, at: pos)
        let t2 = Date()

        var accepted = 0
        while accepted < draft.count && verifiedNext[accepted] == draft[accepted] {
            accepted += 1
        }
        totalAccepted += accepted

        // Commit anchor + accepted draft tokens into KV/conv and draft KV.
        let committed = 1 + accepted
        try runner.commitMTP(at: pos, acceptedCount: committed)
        let t3 = Date()
        draftSec += t1.timeIntervalSince(t0)
        verifySec += t2.timeIntervalSince(t1)
        commitSec += t3.timeIntervalSince(t2)
        history.append(anchorID)
        boundary = []

        var stopped = false
        for i in 0..<accepted {
            let draftID = Int32(draft[i])
            history.append(draftID)
            if emit(draftID) {
                stopped = true
                break
            }
            boundary = []
        }
        if stopped { break }

        anchorToken = verifiedNext[accepted]
        lastTapRow = accepted
    }

    if verifyPasses > 0 {
        let rate = totalDrafted > 0 ? Double(totalAccepted) / Double(totalDrafted) * 100.0 : 0
        let avgTok = Double(generated) / Double(verifyPasses)
        let msg = String(
            format: "Inkling MTP: %d/%d drafts accepted (%.1f%%), %d passes (%.2f tok/pass) "
                + "[draft=%.2fs verify=%.2fs commit=%.2fs]\n",
            totalAccepted, totalDrafted, rate, verifyPasses, avgTok, draftSec, verifySec, commitSec)
        FileHandle.standardError.write(Data(msg.utf8))
    }

    return RawDecodeResult(
        prefillTokens: prompt.count, cachedPromptTokens: 0,
        computedPrefillTokens: prompt.count, prefillSeconds: prefillSeconds,
        newTokens: generated, decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason, kvPosition: runner.position(forSlot: 0), kvBackedTokenIDs: history,
        uncommittedBoundaryTokenIDs: boundary)
}
