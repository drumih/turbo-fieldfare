import Foundation
import Synchronization

@testable import TurboFieldfareAppCore

final class FakeInferenceClient: AppModelLifecycleClient, Sendable {
    private struct State: Sendable {
        var loadedKey: AppLoadedRuntimeKey?
    }

    private final class StateBox: Sendable {
        let value = Mutex(State())
    }

    private let state = StateBox()
    private let generationTasks = GenerationTaskRegistry()
    private let eventDelay: Duration

    init(eventDelay: Duration = .milliseconds(80)) {
        self.eventDelay = eventDelay
    }

    func ensureLoaded(modelDirectory: URL,
                      maxContextTokens: Int,
                      options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        let start = Date()
        onState(.loading(.validatingDirectory))
        try await Task.sleep(for: eventDelay)
        onState(.loading(.preparingRunner))
        try await Task.sleep(for: eventDelay)
        try Task.checkCancellation()
        state.value.withLock {
            $0.loadedKey = AppLoadedRuntimeKey(modelDirectory: modelDirectory,
                                               maxContextTokens: maxContextTokens,
                                               options: options,
                                               forceLogitsHead: forceLogitsHead)
        }
        onState(.ready(
            modelDirectory: modelDirectory.standardizedFileURL,
            loadSeconds: Date().timeIntervalSince(start)))
    }

    func unload() async {
        cancel()
        state.value.withLock { $0.loadedKey = nil }
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            do {
                try request.validate(requireModelDirectory: false)
                let expected = AppLoadedRuntimeKey(
                    modelDirectory: request.modelDirectory,
                    maxContextTokens: request.maxContextTokens,
                    options: request.runtimeOptions,
                    forceLogitsHead: !request.isPureGreedy)
                guard let loaded = state.value.withLock({ $0.loadedKey }) else {
                    throw AppInferenceError.modelNotLoaded
                }
                guard loaded == expected else { throw AppInferenceError.reloadRequired }
            } catch {
                let appError = error as? AppInferenceError ?? .unknown("\(error)")
                continuation.yield(.failed(appError, partial: nil))
                continuation.finish(throwing: appError)
                return
            }

            let id = UUID()
            guard generationTasks.reserve(id) else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                await streamResponse(request: request,
                                     continuation: continuation,
                                     generationID: id)
            }
            generationTasks.attach(task, to: id)

            continuation.onTermination = { [generationTasks] _ in
                generationTasks.take(id)?.cancel()
            }
        }
    }

    func cancel() {
        generationTasks.takeCurrent()?.cancel()
    }

    private func streamResponse(
        request: AppGenerationRequest,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        generationID: UUID
    ) async {
        let start = Date()
        var firstTokenDate: Date?
        var prefillEnd = start
        var generated = 0

        do {
            for step in 1...3 {
                try await Task.sleep(for: eventDelay)
                try Task.checkCancellation()
                continuation.yield(.prefillProgress(done: step, total: 3))
            }
            prefillEnd = Date()

            let response = "Simulated response. Your prompt was: \(request.prompt)"
            let words = response.split(whereSeparator: \.isWhitespace)
            let pieces = words.enumerated().map { index, word in
                index == 0 ? String(word) : " " + word
            }.prefix(request.maxNewTokens)

            for (index, piece) in pieces.enumerated() {
                try await Task.sleep(for: eventDelay)
                try Task.checkCancellation()
                if firstTokenDate == nil { firstTokenDate = Date() }
                generated += 1
                continuation.yield(.token(AppTokenEvent(
                    index: index,
                    textDelta: piece,
                    elapsedDecodeSeconds: Date().timeIntervalSince(prefillEnd))))
            }

            let end = Date()
            let decodeSeconds = max(end.timeIntervalSince(prefillEnd), 0)
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: generated,
                stopReason: generated >= request.maxNewTokens ? .maxTokens : .eos,
                promptTokenCount: max(1, request.prompt.split(whereSeparator: \.isWhitespace).count),
                prefillSeconds: prefillEnd.timeIntervalSince(start),
                timeToFirstTokenSeconds: firstTokenDate.map { $0.timeIntervalSince(prefillEnd) },
                decodeSeconds: decodeSeconds,
                tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish()
        } catch is CancellationError {
            let end = Date()
            let decodeSeconds = max(end.timeIntervalSince(prefillEnd), 0)
            continuation.yield(.cancelled(AppDiagnostics(
                generatedTokens: generated,
                stopReason: .cancelled,
                promptTokenCount: max(1, request.prompt.split(whereSeparator: \.isWhitespace).count),
                prefillSeconds: prefillEnd.timeIntervalSince(start),
                timeToFirstTokenSeconds: firstTokenDate.map { $0.timeIntervalSince(prefillEnd) },
                decodeSeconds: decodeSeconds,
                tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch {
            let appError = error as? AppInferenceError ?? .unknown("\(error)")
            continuation.yield(.failed(appError, partial: nil))
            continuation.finish(throwing: appError)
        }

        generationTasks.clear(generationID)
    }
}
