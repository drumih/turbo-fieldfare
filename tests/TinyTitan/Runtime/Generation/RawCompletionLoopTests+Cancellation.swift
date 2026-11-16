import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

extension RawCompletionLoopTests {
    @Test func cancellationPropagatesMidDecode() async throws {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let idA = try #require(tok.encode("a", addBOS: false).first)
        let producer = ScriptedLogitProducer(
            vocabSize: tok.vocabSize,
            step: automaton([idA, idA], end: idA))
        let promptIds = tok.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)

        let task = Task {
            try await runRawCompletion(
                producer: producer, tokenizer: tok,
                promptIds: promptIds,
                config: GenerationConfig(maxNewTokens: 100_000, temperature: 0),
                context: ctx, scratch: scratch,
                prefillConfig: .off
            ) { progress in
                if case .token(let index, _, _) = progress, index == 2 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
    }
}
