import Foundation
import Metal
import TinyTitanCore

extension InklingQ4Decoder {
    /// Prefills `prompt` starting at `position` while capturing MTP target
    /// features and populating the draft model's context KV caches.
    /// Returns the greedy next token and the row index of the last committed
    /// token inside `mtpDraft.state.targetTaps`.
    func prefillWithMTP(_ prompt: [Int], position: Int) throws -> (next: Int, lastTapRow: Int) {
        guard let mtp = mtpDraft else {
            throw MetalError.invalidState("MTP draft model is not enabled")
        }
        guard !prompt.isEmpty, position + prompt.count <= state.maxContext else {
            throw MetalError.invalidState(
                "\(prompt.count) tokens at \(position) for context \(state.maxContext)")
        }
        captureMTPTaps = true
        captureConvSnapshots = false
        defer { captureMTPTaps = false }

        var offset = 0
        var lastNext = 0
        var lastCount = 0
        while offset < prompt.count {
            let chunk = Array(prompt[offset..<min(prompt.count, offset + state.maxChunk)])
            let result = try run(chunk, position: position + offset, slot: 0, returnLogits: false)
            try mtp.storeContextKV(startPosition: position + offset, count: chunk.count)
            lastNext = result.next
            lastCount = chunk.count
            offset += chunk.count
        }
        return (next: lastNext, lastTapRow: lastCount - 1)
    }

    /// Runs the target model on `tokens` (`<= blockSize`) at `position`,
    /// capturing target taps, per-token conv snapshots, and per-token greedy
    /// next tokens for speculative verification.
    func verifyBlock(_ tokens: [Int], position: Int) throws -> [Int] {
        guard let mtp = mtpDraft else {
            throw MetalError.invalidState("MTP draft model is not enabled")
        }
        guard !tokens.isEmpty, tokens.count <= mtp.state.maxVerifyTokens,
            position + tokens.count <= state.maxContext
        else {
            throw MetalError.invalidState(
                "verify block \(tokens.count) at \(position) out of bounds")
        }
        captureMTPTaps = true
        captureConvSnapshots = true
        defer {
            captureMTPTaps = false
            captureConvSnapshots = false
        }
        _ = try run(tokens, position: position, slot: 0, returnLogits: false)
        let ptr = mtp.state.verifyNextTokens.contents()
            .bindMemory(to: Int32.self, capacity: tokens.count)
        return (0..<tokens.count).map { Int(ptr[$0]) }
    }

    /// Commits the first `acceptedCount` tokens of the most recent `verifyBlock`
    /// call at `position`: rolls back every visit's 3-tap conv state to the
    /// snapshot after token `acceptedCount - 1` and stores those accepted
    /// tokens' target features into the draft KV caches.
    func commitVerified(position: Int, acceptedCount: Int) throws {
        guard let mtp = mtpDraft else {
            throw MetalError.invalidState("MTP draft model is not enabled")
        }
        guard acceptedCount >= 1, acceptedCount <= mtp.state.maxVerifyTokens else {
            throw MetalError.invalidState("invalid acceptedCount \(acceptedCount)")
        }
        let strideBytes = mtp.state.convSnapshotStrideFloats * 4
        let snapOffset = (acceptedCount - 1) * strideBytes
        let slotConv = state.slotStates[0].convStates
        for visit in 0..<manifest.unrolledLayers {
            let src = mtp.state.convSnapshots[visit].contents() + snapOffset
            memcpy(slotConv[visit].contents(), src, strideBytes)
        }
        try mtp.storeContextKV(startPosition: position, count: acceptedCount)
    }
}
