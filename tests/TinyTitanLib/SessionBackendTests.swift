import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitanLib

/// The session is backend-agnostic: the GPU engine and the CPU engine differ
/// only in the `ServerInferenceBackend` behind it. These tests use a recording
/// backend so the raw-prompt contract (the backend tokenizes, `generate` sees
/// `renderedPromptIDs`) is pinned without a model.
private struct Stop: Error {}

private actor RecordingBackend: ServerInferenceBackend {
    nonisolated let maximumContext = 100
    nonisolated let samplingDefaults: SamplingProfile = .house
    private(set) var captured: ValidatedChatRequest?

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        captured = request
        throw Stop()
    }

    func tokenizeRawPrompt(_ text: String) async throws -> [Int32] {
        [7, 8]
    }
}

private actor UntokenizingBackend: ServerInferenceBackend {
    nonisolated let maximumContext = 100
    nonisolated let samplingDefaults: SamplingProfile = .house

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        throw Stop()
    }
}

@Suite struct SessionBackendTests {
    private func makeSession(
        _ backend: any ServerInferenceBackend
    ) -> (Session, BackendBox) {
        let box = BackendBox(backend)
        return (
            Session(
                system: nil,
                box: box,
                descriptor: ModelDescriptor(
                    id: "test", family: "qwen35_dense", contextWindow: 100,
                    weightBytes: 0, expertCacheBytes: 0),
                reasoningProfile: ServerReasoningProfile(
                    family: "qwen35_dense", thinkingMode: .off,
                    reasoningEffort: nil, reasoningControl: .binaryThinking)),
            box
        )
    }

    @Test func aRawPromptReachesTheBackendAsRenderedIDs() async throws {
        let backend = RecordingBackend()
        let (session, box) = makeSession(backend)
        defer { withExtendedLifetime(box) {} }
        do {
            _ = try await session.respond(to: .raw("hello")) { _ in }
            Issue.record("the recording backend throws Stop")
        } catch is Stop {
            // Expected: the request was recorded before the throw.
        }
        let captured = await backend.captured
        #expect(captured?.renderedPromptIDs == [7, 8])
    }

    @Test func aBackendWithoutATokenizerRefusesRawPrompts() async {
        let (session, box) = makeSession(UntokenizingBackend())
        defer { withExtendedLifetime(box) {} }
        do {
            _ = try await session.respond(to: .raw("hello")) { _ in }
            Issue.record("raw prompts need a tokenizer")
        } catch is RawPromptUnsupported {
            // Expected.
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
