import Foundation
import Metal
import Testing
import TurboFieldfareValidationSupport

@testable import TurboFieldfare

extension RawCompletionLoopTests {
  @Test func cancellationPropagatesMidDecode() async throws {
    let ctx = try MetalContext()
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
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

  /// A caller that asks the loop to stop is not a stop-string match. Reporting
  /// it as one made a user pressing Stop indistinguishable from a configured
  /// stop string, and it disagreed with `stopStringFiltered`, which the server
  /// computes from the matcher rather than from the reason.
  @Test func askingTheLoopToStopIsReportedAsCancelledNotAsAStopString() async throws {
    let ctx = try MetalContext()
    let tok = try await GFTokenizer.load()
    let idA = tok.encode("a", addBOS: false).first!
    let scratch = try RawCompletionScratch(context: ctx, vocab: tok.vocabSize)
    let promptIds = tok.encode("go", addBOS: true)

    var produced = 0
    let cancelled = try await runRawCompletion(
      producer: ScriptedLogitProducer(
        vocabSize: tok.vocabSize, step: automaton([idA, idA, idA], end: idA)),
      tokenizer: tok,
      promptIds: promptIds,
      config: GenerationConfig(maxNewTokens: 100_000, temperature: 0),
      context: ctx, scratch: scratch, prefillConfig: .off,
      shouldStop: { produced >= 2 }
    ) { progress in
      if case .token = progress { produced += 1 }
    }
    #expect(cancelled.reason == .cancelled)
    #expect(cancelled.newTokens >= 2)

    // A real stop string still reports itself, so the two stay distinguishable.
    var config = GenerationConfig(maxNewTokens: 100_000, temperature: 0)
    config = GenerationConfig(
      maxNewTokens: 100_000, temperature: 0, stopStrings: ["a"])
    let stopped = try await runRawCompletion(
      producer: ScriptedLogitProducer(
        vocabSize: tok.vocabSize, step: automaton([idA, idA, idA], end: idA)),
      tokenizer: tok,
      promptIds: promptIds,
      config: config,
      context: ctx, scratch: try RawCompletionScratch(context: ctx, vocab: tok.vocabSize),
      prefillConfig: .off
    ) { _ in }
    #expect(stopped.reason == .stopString)
  }
}
