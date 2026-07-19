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
        var finished = false
        var sequence: UInt64 = 0
    }

    private let condition = NSCondition()
    private var state = State()
    private let generationID: UUID
    private let memorySampler = AppMemorySampler()

    init(generationID: UUID) {
        self.generationID = generationID
        memorySampler.resetPeak()
    }

    func publish(_ event: AppInferenceEvent) {
        condition.lock()
        switch event {
        case .prefillProgress(let done, let total):
            state.latestPrefill = PrefillProgress(done: done, total: total)
            condition.signal()
        case .token(let token):
            state.pendingText += token.textDelta
            state.latestToken = token
        case .finished(let diagnostics):
            state.terminal = terminal(.finished, diagnostics: diagnostics)
        case .cancelled(let diagnostics):
            state.terminal = terminal(.cancelled, diagnostics: diagnostics)
        case .failed(let error, let diagnostics):
            state.terminal = terminal(
                .failed, diagnostics: diagnostics, error: error.userMessage)
        }
        if state.terminal != nil { condition.signal() }
        condition.unlock()
    }

    func finish(error: Error? = nil) {
        condition.lock()
        if state.terminal == nil, let error {
            state.terminal = DecodeServiceEvent(
                kind: .failed, generationID: generationID, error: "\(error)")
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

            if let prefill, let prefillSequence {
                let snapshot = DecodeServiceEvent(
                    kind: .prefill, generationID: generationID,
                    sequence: prefillSequence,
                    prefillDone: prefill.done, prefillTotal: prefill.total)
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
                    peakMemoryBytes: memorySampler.peakBytes)
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            if let terminal {
                try handle.write(contentsOf: DecodeFrameCodec.encode(terminal))
            }
            if done, terminal == nil { return }
            if terminal != nil && done { return }
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
