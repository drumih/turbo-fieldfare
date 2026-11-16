import Foundation
import Metal
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

/// `runRawCompletion(slot:)` must carry the slot to the producer on both the
/// chunked prefill and the decode step. Non-zero slots used to fall back to
/// sequential prefill, which does not match the chunked path numerically.
extension RawCompletionLoopTests {

    /// Records which prefill path and which produce overload the loop reached,
    /// and with which slot.
    final class SlotRecordingProducer: LogitProducer, ChunkedPrefillRunner,
        @unchecked Sendable
    {
        let vocabSize: Int
        private let nextToken: Int32
        private(set) var slotCalls: [Int] = []
        private(set) var legacyProduceCalls = 0
        private(set) var chunkedCalls = 0
        private(set) var chunkedSlots: [Int] = []

        init(vocabSize: Int, nextToken: Int32) {
            self.vocabSize = vocabSize
            self.nextToken = nextToken
        }

        func reset() {
            slotCalls.removeAll()
            legacyProduceCalls = 0
            chunkedCalls = 0
            chunkedSlots.removeAll()
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            legacyProduceCalls += 1
            write(into: logits)
        }

        func produce(
            token: Int32, position: Int, slot: Int,
            into logits: MTLBuffer
        ) async throws {
            slotCalls.append(slot)
            write(into: logits)
        }

        func prefillChunked(
            tokens: ArraySlice<Int32>,
            startPosition: Int,
            outputMode: PrefillOutputMode,
            config: PrefillRuntimeConfig,
            into logits: MTLBuffer,
            onProgress: (Int) -> Void
        ) async throws -> PrefillResult {
            try await prefillChunked(
                tokens: tokens, startPosition: startPosition,
                slot: 0, outputMode: outputMode, config: config,
                into: logits, onProgress: onProgress)
        }

        /// The slot-aware overload is the one `runRawCompletion` calls, so this
        /// is what records the slot a batched prefill actually landed in.
        func prefillChunked(
            tokens: ArraySlice<Int32>,
            startPosition: Int,
            slot: Int,
            outputMode: PrefillOutputMode,
            config: PrefillRuntimeConfig,
            into logits: MTLBuffer,
            onProgress: (Int) -> Void
        ) async throws -> PrefillResult {
            chunkedCalls += 1
            chunkedSlots.append(slot)
            write(into: logits)
            onProgress(tokens.count)
            return PrefillResult(
                newPosition: startPosition + tokens.count,
                seed: .logitsWritten)
        }

        private func write(into logits: MTLBuffer) {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { ptr[i] = Float16(-30) }
            ptr[Int(nextToken)] = Float16(30)
        }
    }

    /// A non-zero slot takes the same chunked path as slot 0, and the slot
    /// reaches both the prefill and the decode step.
    @Test func nonZeroSlotUsesSlotAwareChunkedPrefill() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let tokenA = try #require(tokenizer.encode("a", addBOS: false).first)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let producer = SlotRecordingProducer(
            vocabSize: tokenizer.vocabSize,
            nextToken: tokenA)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 2, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32),
            slot: 2
        ) { _ in }

        #expect(result.newTokens == 2)
        #expect(
            producer.chunkedCalls == 1,
            "slot 2 must take the chunked prefill path")
        #expect(
            producer.chunkedSlots == [2],
            "the prefill must land in slot 2, got \(producer.chunkedSlots)")
        #expect(
            producer.legacyProduceCalls == 0,
            "the slot-aware overload must be the one reached")
        // Chunked prefill leaves the first row's logits, so only the second
        // generated token costs a decode produce step.
        #expect(
            producer.slotCalls == [2],
            "the decoded step must name slot 2, got \(producer.slotCalls)")
    }

    @Test func slotZeroKeepsTheChunkedPrefillPath() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
        let tokenA = try #require(tokenizer.encode("a", addBOS: false).first)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let producer = SlotRecordingProducer(
            vocabSize: tokenizer.vocabSize,
            nextToken: tokenA)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 2, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32),
            slot: 0
        ) { _ in }

        #expect(result.newTokens == 2)
        #expect(
            producer.chunkedCalls == 1,
            "slot 0 must keep the chunked prefill fast path")
        #expect(
            producer.chunkedSlots == [0],
            "the prefill must land in slot 0, got \(producer.chunkedSlots)")
        // Chunked prefill leaves the first row's logits, so only the second
        // generated token costs a produce step.
        #expect(
            producer.slotCalls == [0],
            "only the decoded step goes through produce, in slot 0")
    }
}
