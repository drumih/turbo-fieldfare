import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct MockInferenceClientTests {
    @Test func streamsTokensAndFinishesWithDiagnostics() async throws {
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 1_000_000)
        let request = AppGenerationRequest(modelDirectory: FileManager.default.temporaryDirectory, prompt: "go", maxNewTokens: 8)
        var text = ""
        var diagnostics: AppDiagnostics?
        var prefillEvents: [(Int, Int)] = []
        var sawTokenBeforePrefillEnded = false
        var firstTokenElapsed: Double?

        for try await event in client.generate(request) {
            switch event {
            case .prefillProgress(let done, let total):
                prefillEvents.append((done, total))
            case .token(let token):
                if prefillEvents.count < client.prefillSteps { sawTokenBeforePrefillEnded = true }
                if firstTokenElapsed == nil { firstTokenElapsed = token.elapsedDecodeSeconds }
                text += token.textDelta
            case .finished(let d):
                diagnostics = d
            default:
                break
            }
        }

        #expect(text.contains("one two three"))
        #expect(prefillEvents.count == client.prefillSteps)
        #expect(prefillEvents.last?.0 == client.prefillSteps)
        #expect(!sawTokenBeforePrefillEnded)
        let d = try #require(diagnostics)
        #expect(d.generatedTokens > 0)
        #expect(d.promptTokenCount == 1)
        let ttft = try #require(d.timeToFirstTokenSeconds)
        let prefillSeconds = try #require(d.prefillSeconds)
        let firstElapsed = try #require(firstTokenElapsed)
        #expect(prefillSeconds >= 0)
        #expect(ttft >= 0)
        #expect(firstElapsed >= ttft)
        #expect(firstElapsed <= d.decodeSeconds)
        #expect(d.tokensPerSecond >= 0)
    }

    @Test func cancellationEmitsCancelledDiagnostics() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 5_000_000)
        let request = AppGenerationRequest(modelDirectory: FileManager.default.temporaryDirectory, prompt: "go", maxNewTokens: 10)
        var sawToken = false
        var sawCancel = false
        var cancelledDiagnostics: AppDiagnostics?

        do {
            for try await event in client.generate(request) {
                switch event {
                case .token:
                    sawToken = true
                    client.cancel()
                case .cancelled(let diagnostics):
                    sawCancel = diagnostics.stopReason == .cancelled
                    cancelledDiagnostics = diagnostics
                default:
                    break
                }
            }
        } catch AppInferenceError.cancelled {
        }

        #expect(sawToken)
        #expect(sawCancel)
        #expect(cancelledDiagnostics?.promptTokenCount == 1)
        #expect(cancelledDiagnostics?.prefillSeconds != nil)
    }

    @Test func concurrentGenerationRejected() async throws {
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 20_000_000)
        let request = AppGenerationRequest(modelDirectory: FileManager.default.temporaryDirectory, prompt: "go", maxNewTokens: 3)
        let first = client.generate(request)
        let second = client.generate(request)

        var firstIterator = first.makeAsyncIterator()
        _ = try await firstIterator.next()

        await #expect(throws: AppInferenceError.self) {
            for try await _ in second {}
        }
        client.cancel()
    }
}
