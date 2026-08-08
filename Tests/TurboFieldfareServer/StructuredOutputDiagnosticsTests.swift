import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Structured output diagnostics")
struct StructuredOutputDiagnosticsTests {
    private struct UnexpectedError: Error, CustomStringConvertible {
        let description = "private-error-description"
    }

    @Test func orphanResponseIncludesBoundaryEvidenceWithoutContent() {
        let diagnostics = makeDiagnostics()
        let error = StructuredOutputFailure(
            kind: .orphanToolResponse,
            cause: .none,
            diagnostics: diagnostics)
        let reflected = String(reflecting: error)

        #expect(diagnostics.completionTokens == 3)
        #expect(diagnostics.toolStartCount == 0)
        #expect(diagnostics.toolEndCount == 0)
        #expect(diagnostics.toolResponseCount == 1)
        #expect(diagnostics.toolResponseEndCount == 0)
        #expect(diagnostics.lastToolResponseOffset == 2)
        #expect(diagnostics.effectiveCountMatchesResult)
        #expect(diagnostics.effectivePrefixMatchesKV)
        #expect(diagnostics.kvPositionMatchesHistory)
        #expect(diagnostics.completionCountMatchesHistory)
        #expect(diagnostics.prefillAccountingMatches)
        #expect(isLowercaseSHA256(diagnostics.renderedPromptHash))
        #expect(isLowercaseSHA256(diagnostics.effectivePromptHash))
        #expect(isLowercaseSHA256(diagnostics.generatedHash))
        let generated: [Int32] = [20, 21, 102]
        #expect(diagnostics.generatedHash
            == StructuredOutputFailureDiagnostics.i32leSHA256([generated[...]]))
        #expect(reflected.hasPrefix(
            "structured_output_failure kind=orphan_tool_response cause=none "))
        #expect(("error=" + reflected).hasPrefix(
            "error=structured_output_failure kind=orphan_tool_response cause=none "))
        #expect(reflected.contains("completion_tokens=3"))
        #expect(reflected.contains("tool_response_count=1"))
        #expect(reflected.contains("last_tool_response_offset=2"))
        #expect(!reflected.contains("\n"))
        #expect(!reflected.contains("["))
        #expect(!reflected.contains("]"))
    }

    @Test func parserCausesAreFixedAndUnknownToolNameIsDiscarded() {
        #expect(StructuredOutputFailureCause.classify(
            GemmaToolCallParserError.malformed) == .malformed)
        #expect(StructuredOutputFailureCause.classify(
            GemmaToolCallParserError.oversized) == .oversized)
        #expect(StructuredOutputFailureCause.classify(
            UnexpectedError()) == .unexpected)

        let cause = StructuredOutputFailureCause.classify(
            GemmaToolCallParserError.unknownTool("private-name"))
        let reflected = String(reflecting: StructuredOutputFailure(
            kind: .decoderConsume,
            cause: cause,
            diagnostics: makeDiagnostics()))
        #expect(cause == .unknownTool)
        #expect(reflected.contains("cause=unknown_tool"))
        #expect(!reflected.contains("private-name"))

        let unexpected = String(reflecting: StructuredOutputFailure(
            kind: .decoderFinish,
            cause: .classify(UnexpectedError()),
            diagnostics: makeDiagnostics()))
        #expect(unexpected.contains("cause=unexpected"))
        #expect(!unexpected.contains("private-error-description"))
    }

    @Test func allFailureKindsShareTheSameDiagnosticSuffix() {
        let diagnostics = makeDiagnostics()
        for kind in [
            StructuredOutputFailureKind.decoderConsume,
            .decoderFinish,
            .orphanToolResponse,
        ] {
            let reflected = String(reflecting: StructuredOutputFailure(
                kind: kind,
                cause: .none,
                diagnostics: diagnostics))
            #expect(reflected.hasPrefix(
                "structured_output_failure kind=\(kind.rawValue) cause=none "))
            #expect(reflected.hasSuffix(diagnostics.logDescription))
        }
    }

    /// The grammar's veto counters ride along with the existing evidence: a
    /// failure that still happens with the constraint on means the grammar has
    /// a hole, and the counters say how hard it was fighting the model.
    @Test func vetoCountersAreReportedInTheLogLine() {
        var result = makeResult()
        result.toolGrammarVetoes = 7
        result.toolNameVetoes = 3
        let diagnostics = StructuredOutputFailureDiagnostics(
            renderedPromptIDs: [8, 10, 11, 12],
            effectivePromptIDs: [10, 11, 12],
            result: result,
            maxCompletionTokens: 8,
            decodedCalls: 0,
            visibleBytes: 0,
            stopStringMatched: false,
            toolStartID: 48,
            toolEndID: 49,
            toolResponseID: 102,
            toolResponseEndID: 103)
        #expect(diagnostics.toolGrammarVetoes == 7)
        #expect(diagnostics.toolNameVetoes == 3)
        #expect(diagnostics.logDescription.contains("tool_grammar_vetoes=7"))
        #expect(diagnostics.logDescription.contains("tool_name_vetoes=3"))
        #expect(makeDiagnostics().logDescription.contains("tool_grammar_vetoes=0"))
    }

    @Test func tokenHashUsesUInt32LittleEndianBytes() {
        let tokens: [Int32] = [1, -1]
        #expect(StructuredOutputFailureDiagnostics.i32leSHA256([tokens[...]])
            == "b15348c8f462384c01e83b6d499c6faf3f96808f5aa07c6bab4b65b36b4445d4")
    }

    @Test func invalidPrefillCountCannotCrashDiagnostics() {
        let diagnostics = makeDiagnostics(prefillTokens: 99)
        #expect(!diagnostics.effectiveCountMatchesResult)
        #expect(!diagnostics.completionCountMatchesHistory)
        #expect(!diagnostics.prefillAccountingMatches)
    }

    private func makeResult(prefillTokens: Int = 3) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prefillTokens,
            cachedPromptTokens: 1,
            computedPrefillTokens: 2,
            prefillSeconds: 0,
            newTokens: 3,
            decodeSeconds: 0,
            reason: .toolCalls,
            kvPosition: 5,
            kvBackedTokenIDs: [10, 11, 12, 20, 21],
            uncommittedBoundaryTokenIDs: [102])
    }

    private func makeDiagnostics(prefillTokens: Int = 3)
        -> StructuredOutputFailureDiagnostics {
        let renderedPrompt: [Int32] = [8, 10, 11, 12]
        let effectivePrompt: [Int32] = [10, 11, 12]
        let result = makeResult(prefillTokens: prefillTokens)
        return StructuredOutputFailureDiagnostics(
            renderedPromptIDs: renderedPrompt,
            effectivePromptIDs: effectivePrompt,
            result: result,
            maxCompletionTokens: 8,
            decodedCalls: 0,
            visibleBytes: 0,
            stopStringMatched: false,
            toolStartID: 100,
            toolEndID: 101,
            toolResponseID: 102,
            toolResponseEndID: 103)
    }

    private func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64
            && value.allSatisfy { $0.isNumber || ("a"..."f").contains(String($0)) }
    }
}
