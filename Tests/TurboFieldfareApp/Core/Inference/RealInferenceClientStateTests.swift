import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Model-free state coverage for the real client: load failure surfaces
/// before any network or Metal work, idle cancel is a no-op, and a bad
/// request fails the stream with a typed error.
@Suite struct RealInferenceClientStateTests {
    @Test func generationRegistryScopesTerminationToOwningID() async {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        registry.clear(first)
        #expect(registry.reserve(second))
        let secondTask = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(secondTask, to: second)

        #expect(registry.take(first) == nil)
        #expect(!secondTask.isCancelled)
        registry.take(second)?.cancel()
        #expect(secondTask.isCancelled)
    }

    @Test func generationRegistryRejectsConcurrentReservationAndClearsByOwner() {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        #expect(!registry.reserve(second))
        registry.clear(second)
        #expect(!registry.reserve(second))
        registry.clear(first)
        #expect(registry.reserve(second))
        registry.clear(second)
    }

    @Test func generationRegistryCancelsTaskAttachedAfterReservationEnded() async {
        let registry = GenerationTaskRegistry()
        let id = UUID()
        #expect(registry.reserve(id))
        registry.clear(id)
        let task = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(task, to: id)
        #expect(task.isCancelled)
        let next = UUID()
        #expect(registry.reserve(next))
        registry.clear(next)
    }

    @Test func generationRunnerPolicyKeepsFusionHeadForPureGreedyChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello",
            temperature: 0,
            repetitionPenalty: 1)

        #expect(!RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationRunnerPolicyForcesLogitsForSamplingChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello",
            temperature: 0.7,
            repetitionPenalty: 1)

        #expect(RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationConfigCarriesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello")

        let config = RealInferenceSession.generationConfig(for: request)
        #expect(config.temperature == 1.0)
        #expect(config.topK == 64)
        #expect(config.topP == 0.95)
        #expect(config.repetitionPenalty == 1)
    }

    @Test func tokenizerDirectoryCacheReloadsOnlyWhenModelDirectoryChanges() {
        var cache = TokenizerDirectoryCache()
        let first = URL(fileURLWithPath: "/tmp/first.gturbo")
        let second = URL(fileURLWithPath: "/tmp/second.gturbo")

        #expect(cache.shouldReload(for: first))
        cache.markLoaded(for: first)
        #expect(!cache.shouldReload(for: first))
        #expect(cache.shouldReload(for: second))
        cache.clear()
        #expect(cache.shouldReload(for: first))
    }

    @Test func generateWithoutLoadedModelFailsWithoutPartialDiagnostics() async throws {
        let client = RealInferenceClient()
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-prefill-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "hello",
            runtimeOptions: AppRuntimeOptions(prefillEnabled: false,
                                              turboQuantKVEnabled: true))

        var failure: AppInferenceError?
        var partial: AppDiagnostics?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, let diagnostics) = event {
                    failure = error
                    partial = diagnostics
                }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        #expect(failure != nil)
        #expect(partial == nil)
    }

    @Test func ensureLoadedFailsFastForMissingDirectory() async {
        let client = RealInferenceClient()
        var states: [AppModelLoadState] = []
        let recorder = StateRecorder()

        await #expect(throws: AppInferenceError.self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
                maxContextTokens: 1024,
                options: AppRuntimeOptions(),
                forceLogitsHead: false,
                onState: { recorder.append($0) })
        }
        states = recorder.snapshot()
        #expect(states.first == .loading(.validatingDirectory))
        #expect(states.last?.isFailed == true)
        #expect(!states.contains(.loading(.tokenizer)))
    }

    @Test func generateWithMissingDirectoryFailsStream() async {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            prompt: "hello")

        var failure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, _) = event { failure = error }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(failure == .modelNotFound("/nonexistent/model.gturbo"))
    }

    @Test func prefillFailureDiagnosticsMarksUnsupportedModeAndReason() {
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)

        let diagnostics = RealInferenceSession.prefillFailureDiagnostics(
            config: config,
            kvStorageMode: .fp16,
            reason: "chunked prefill synthetic unsupported diagnostic")

        #expect(diagnostics.requestedMode == .chunked)
        #expect(diagnostics.executedMode == .unsupported)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.kvStorageMode == .fp16)
        #expect(diagnostics.unsupportedReason?.contains("synthetic unsupported") == true)
    }

    @Test func cancelWhenIdleIsNoOp() {
        let client = RealInferenceClient()
        client.cancel()
        client.cancel()
    }

    @Test func unloadWhenIdleIsSafe() async {
        let client = RealInferenceClient()
        await client.unload()
    }
}

private final class StateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppModelLoadState] = []

    func append(_ state: AppModelLoadState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func snapshot() -> [AppModelLoadState] {
        lock.lock()
        defer { lock.unlock() }
        return states
    }
}
