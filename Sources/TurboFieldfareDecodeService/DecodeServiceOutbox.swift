import Foundation
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

final class DecodeServiceOutbox: @unchecked Sendable {
    private struct PrefillProgress {
        var done: Int
        var total: Int
    }

    private struct State {
        var pendingText = ""
        var latestPrefill: PrefillProgress?
        var latestToken: AppTokenEvent?
        var terminal: DecodeServiceEvent?
        var terminalCommitted = false
        var finished = false
        var sequence: UInt64 = 0
    }

    private let condition = NSCondition()
    private var state = State()
    private let generationID: UUID
    private let memorySampler = AppMemorySampler()

    /// Bytes of image tower currently held mapped, or nil when there is no
    /// vision runtime. Sampled per event so Keep Ready is visible while a run
    /// is happening, not only in its final diagnostics.
    private let towerBytes: @Sendable () -> UInt64?

    init(generationID: UUID,
         towerBytes: @escaping @Sendable () -> UInt64? = { nil }) {
        self.generationID = generationID
        self.towerBytes = towerBytes
        memorySampler.resetPeak()
    }

    func publish(_ event: AppInferenceEvent) {
        condition.lock()
        switch event {
        case .memorySample:
            // The writer samples on its own schedule; an inbound reading is the
            // runtime's, and there is nothing to queue.
            break
        case .prefillProgress(let done, let total):
            state.latestPrefill = PrefillProgress(done: done, total: total)
            condition.signal()
        case .token(let token):
            state.pendingText += token.textDelta
            state.latestToken = token
        case .finished(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.finished, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .cancelled(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.cancelled, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .failed(let error, let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(
                    .failed, diagnostics: diagnostics, error: error.userMessage)
                state.terminalCommitted = true
            }
        }
        if state.terminal != nil { condition.signal() }
        condition.unlock()
    }

    func finish(error: Error? = nil) {
        condition.lock()
        if !state.terminalCommitted, let error {
            state.terminal = DecodeServiceEvent(
                kind: .failed, generationID: generationID, error: "\(error)")
            state.terminalCommitted = true
        }
        state.finished = true
        condition.broadcast()
        condition.unlock()
    }

    func runWriter(to handle: FileHandle) throws {
        while true {
            condition.lock()
            if state.terminal == nil, !state.finished {
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            let prefill = state.latestPrefill
            let text = state.pendingText
            let token = state.latestToken
            let terminal = state.terminal
            let done = state.finished
            state.latestPrefill = nil
            state.pendingText = ""
            state.latestToken = nil
            state.terminal = nil
            var prefillSequence: UInt64?
            if prefill != nil {
                state.sequence &+= 1
                prefillSequence = state.sequence
            }
            var tokenSequence: UInt64?
            if !text.isEmpty || token != nil {
                state.sequence &+= 1
                tokenSequence = state.sequence
            }
            condition.unlock()

            _ = memorySampler.sample()

            if prefill == nil, text.isEmpty, token == nil, terminal == nil, !done {
                let snapshot = DecodeServiceEvent(
                    kind: .memory, generationID: generationID,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
                continue
            }
            if let prefill, let prefillSequence {
                let snapshot = DecodeServiceEvent(
                    kind: .prefill, generationID: generationID,
                    sequence: prefillSequence,
                    prefillDone: prefill.done, prefillTotal: prefill.total,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            if !text.isEmpty || token != nil {
                let elapsed = token?.elapsedDecodeSeconds ?? 0
                let count = (token?.index ?? -1) + 1
                let snapshot = DecodeServiceEvent(
                    kind: .snapshot, generationID: generationID,
                    sequence: tokenSequence ?? 0, textDelta: text, tokenCount: count,
                    decodeSeconds: elapsed,
                    tokensPerSecond: elapsed > 0 ? Double(count) / elapsed : 0,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            if let terminal {
                try handle.write(contentsOf: DecodeFrameCodec.encode(terminal))
            }
            if terminal != nil { return }
            if done { return }
        }
    }

    private func terminal(_ kind: DecodeServiceEventKind,
                          diagnostics: AppDiagnostics?,
                          error: String? = nil) -> DecodeServiceEvent {
        DecodeServiceEvent(
            kind: kind, generationID: generationID,
            tokenCount: diagnostics?.generatedTokens ?? 0,
            promptTokenCount: diagnostics?.promptTokenCount,
            prefillSeconds: diagnostics?.prefillSeconds,
            timeToFirstTokenSeconds: diagnostics?.timeToFirstTokenSeconds,
            decodeSeconds: diagnostics?.decodeSeconds ?? 0,
            tokensPerSecond: diagnostics?.tokensPerSecond ?? 0,
            stopReason: diagnostics?.stopReason.rawValue,
            error: error,
            currentMemoryBytes: memorySampler.sample(),
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: diagnostics?.visionTowerMappedBytes,
            prefill: diagnostics?.prefill.map(Self.prefillDiagnostics),
            runner: diagnostics?.runner.map(Self.runnerDiagnostics))
    }

    private static func prefillDiagnostics(_ value: PrefillExecutionDiagnostics)
        -> DecodePrefillDiagnostics {
        DecodePrefillDiagnostics(
            requestedMode: value.requestedMode.rawValue,
            executedMode: value.executedMode.rawValue,
            kvStorageMode: value.kvStorageMode?.rawValue,
            chunkCompleteness: value.chunkCompleteness.rawValue,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: AppRunnerDiagnostics)
        -> DecodeRunnerDiagnostics {
        DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }
}
