import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server image token budget")
struct ServerImageTokenBudgetTests {
    @Test func eachImageCostsItsProjectedTokensPlusTwoMarkers() {
        #expect(ServerImageTokenBudget.imageTokens(softTokenCounts: []) == 0)
        #expect(ServerImageTokenBudget.imageTokens(softTokenCounts: [280]) == 282)
        #expect(ServerImageTokenBudget.imageTokens(
            softTokenCounts: [280, 280, 280, 280]) == 1_128)
        #expect(ServerImageTokenBudget.imageTokens(
            softTokenCounts: [1, 1, 1]) == 9)
    }

    @Test func budgetAdmitsExactlyUpToTheContextAndRejectsOneOver() {
        // 100 text + (280 + 2) image = 382; must be strictly under the context.
        #expect(ServerImageTokenBudget.fits(
            softTokenCounts: [280], textTokens: 100, maxContext: 383))
        #expect(!ServerImageTokenBudget.fits(
            softTokenCounts: [280], textTokens: 100, maxContext: 382))
    }

    /// Clients routinely request the whole context as reply; pinned Pi 0.82.1
    /// sends max_tokens 16384 — the entire default context — on every request.
    /// The text path clamps rather than refusing, so the image budget must not
    /// count the requested reply or it rejects requests the server would
    /// otherwise serve, on any configured context.
    @Test func aHugeRequestedReplyDoesNotRejectAPromptThatFits() {
        #expect(ServerImageTokenBudget.fits(
            softTokenCounts: [258], textTokens: 1_627, maxContext: 8_192))
    }

    /// The point of the change: how many images fit depends on the context, not
    /// on a fixed number. A small context admits fewer than the old cap of four
    /// and a large one admits far more.
    @Test func admissibleImageCountScalesWithContext() {
        func maximumFullSizeImages(context: Int) -> Int {
            var count = 0
            while ServerImageTokenBudget.fits(
                softTokenCounts: Array(repeating: 280, count: count + 1),
                textTokens: 200,
                maxContext: context) {
                count += 1
            }
            return count
        }
        #expect(maximumFullSizeImages(context: 4_096) == 13)
        #expect(maximumFullSizeImages(context: 8_192) == 28)
        #expect(maximumFullSizeImages(context: 65_536) == 231)
        // A context too small for even one full-size image admits none.
        #expect(maximumFullSizeImages(context: 400) == 0)
    }

    /// Small images cost proportionally less, so many of them fit where a few
    /// full-size ones would not.
    @Test func smallImagesCostLessThanFullSizeOnes() throws {
        let config = VisionConfig()
        // A 4:3 photo does not reach the theoretical maximum: the processed side
        // rounds down to a multiple of patch x pooling, so it lands at 266 of
        // the 280 ceiling. Only aspect ratios that tile exactly, such as the
        // 42x60 patch grid, reach it.
        let full = try Gemma4ImageGeometry(
            sourceWidth: 4_032, sourceHeight: 3_024, config: config)
        #expect(full.softTokenCount == 266)
        #expect(full.softTokenCount <= config.maximumPooledTokens)

        let small = try Gemma4ImageGeometry(
            sourceWidth: 96, sourceHeight: 96, config: config)
        #expect(small.softTokenCount < full.softTokenCount)
        #expect(small.softTokenCount > 0)

        let manySmall = Array(repeating: small.softTokenCount, count: 20)
        let fewFull = Array(repeating: full.softTokenCount, count: 20)
        #expect(ServerImageTokenBudget.imageTokens(softTokenCounts: manySmall)
                < ServerImageTokenBudget.imageTokens(softTokenCounts: fewFull))
    }

    /// Only the prompt has to fit: the reply budget is deliberately not
    /// counted, so what rejects a request is its images and text alone.
    @Test func manySmallImagesStillHaveToFitTheContext() {
        #expect(ServerImageTokenBudget.fits(
            softTokenCounts: Array(repeating: 20, count: 50),
            textTokens: 100, maxContext: 4_096))
        #expect(!ServerImageTokenBudget.fits(
            softTokenCounts: Array(repeating: 280, count: 20),
            textTokens: 100, maxContext: 4_096))
    }

    @Test func rejectionNamesTheRealCostRatherThanACount() {
        let error = ServerImageTokenBudget.rejection(
            imageCount: 3, softTokenCounts: [280, 280, 280],
            textTokens: 40, maxContext: 800)
        guard case .invalid(let message, let param, let code) = error else {
            Issue.record("expected an invalid-request rejection")
            return
        }
        #expect(param == "messages")
        #expect(code == "context_length_exceeded")
        #expect(message.contains("846"))
        #expect(message.contains("800"))
        #expect(!message.contains("four"))
    }
}
