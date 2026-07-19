import Foundation
@testable import TurboFieldfareAppCore

/// Test double: streams canned word tokens behind the `AppInferenceClient`
/// protocol so AppModel control flow stays testable without a model load.
final class MockInferenceClient: AppInferenceClient, @unchecked Sendable {
    var response: String
    var tokenDelayNanos: UInt64
    var failureMessage: String?
    var prefillSteps: Int = 3

    private let lock = NSLock()
    private var activeTask: Task<Void, Never>?
    private var activeGenerationID: UUID?
    private let memorySampler: AppMemorySampler

    init(response: String = "This is a lightweight mock response streaming through the TurboFieldfare Mac shell.",
         tokenDelayNanos: UInt64 = 35_000_000,
         memorySampler: AppMemorySampler = AppMemorySampler(),
         failureMessage: String? = nil) {
        self.response = response
        self.tokenDelayNanos = tokenDelayNanos
        self.memorySampler = memorySampler
        self.failureMessage = failureMessage
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            do {
                try request.validate()
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }

            lock.lock()
            if activeTask != nil {
                lock.unlock()
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            let generationID = UUID()
            let task = Task { [self] in
                await run(request: request, generationID: generationID, continuation: continuation)
            }
            activeTask = task
            activeGenerationID = generationID
            lock.unlock()

            continuation.onTermination = { [weak self] _ in
                self?.cancel()
            }
        }
    }

    func cancel() {
        lock.lock()
        let task = activeTask
        activeTask = nil
        activeGenerationID = nil
        lock.unlock()
        task?.cancel()
    }

    private func clearActiveTask(generationID: UUID) {
        lock.lock()
        if activeGenerationID == generationID {
            activeTask = nil
            activeGenerationID = nil
        }
        lock.unlock()
    }

    private func run(request: AppGenerationRequest,
                     generationID: UUID,
                     continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        defer { clearActiveTask(generationID: generationID) }

        if let failureMessage {
            let error = AppInferenceError.unknown(failureMessage)
            let diagnostics = makeDiagnostics(request: request, generated: 0, start: Date(),
                                              firstToken: nil, stopReason: .failed)
            continuation.yield(.failed(error, partial: diagnostics))
            continuation.finish(throwing: error)
            return
        }

        let start = Date()
        var firstTokenDate: Date?
        var prefillEndDate: Date?
        let pieces = Array(responsePieces(for: request)).prefix(request.maxNewTokens)
        var generated = 0

        for step in 0..<max(prefillSteps, 0) {
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: 0,
                                                  start: start, firstToken: nil,
                                                  prefillEnd: Date(),
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            try? await Task.sleep(nanoseconds: tokenDelayNanos)
            continuation.yield(.prefillProgress(done: step + 1, total: prefillSteps))
        }
        prefillEndDate = Date()

        for (index, piece) in pieces.enumerated() {
            let decodeStart = prefillEndDate ?? start
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: generated,
                                                  start: start, firstToken: firstTokenDate,
                                                  prefillEnd: prefillEndDate,
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            try? await Task.sleep(nanoseconds: tokenDelayNanos)
            if Task.isCancelled {
                let diagnostics = makeDiagnostics(request: request, generated: generated,
                                                  start: start, firstToken: firstTokenDate,
                                                  prefillEnd: prefillEndDate,
                                                  stopReason: .cancelled)
                continuation.yield(.cancelled(diagnostics))
                continuation.finish(throwing: AppInferenceError.cancelled)
                return
            }
            if firstTokenDate == nil { firstTokenDate = Date() }
            generated += 1
            _ = memorySampler.sample()
            continuation.yield(.token(AppTokenEvent(
                index: index,
                textDelta: piece,
                elapsedDecodeSeconds: max(Date().timeIntervalSince(decodeStart), 0))))
        }

        let diagnostics = makeDiagnostics(request: request, generated: generated,
                                          start: start, firstToken: firstTokenDate,
                                          prefillEnd: prefillEndDate,
                                          stopReason: generated >= request.maxNewTokens ? .maxTokens : .eos)
        continuation.yield(.finished(diagnostics))
        continuation.finish()
    }

    private func responsePieces(for request: AppGenerationRequest) -> [String] {
        let trimmed = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = trimmed.isEmpty ? response : "\(response) Prompt received: \(trimmed)"
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        guard !words.isEmpty else { return [""] }
        return words.enumerated().map { index, word in
            index == 0 ? String(word) : " " + word
        }
    }

    private func makeDiagnostics(request: AppGenerationRequest,
                                 generated: Int,
                                 start: Date,
                                 firstToken: Date?,
                                 prefillEnd: Date? = nil,
                                 stopReason: AppStopReason) -> AppDiagnostics {
        _ = memorySampler.sample()
        let decodeStart = prefillEnd ?? start
        let decodeElapsed = max(Date().timeIntervalSince(decodeStart), 0)
        return AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: mockPromptTokenCount(request.prompt),
            prefillSeconds: prefillEnd.map { max($0.timeIntervalSince(start), 0) },
            timeToFirstTokenSeconds: firstToken.map { max($0.timeIntervalSince(decodeStart), 0) },
            decodeSeconds: decodeElapsed,
            tokensPerSecond: decodeElapsed > 0 ? Double(generated) / decodeElapsed : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            runtimeOptions: request.runtimeOptions,
            runner: nil)
    }

    private func mockPromptTokenCount(_ prompt: String) -> Int {
        max(1, prompt.split(whereSeparator: \.isWhitespace).count)
    }
}
