import Testing
import TinyTitanCore

/// Model-free (TT-122): where the prompt cache's interior checkpoints find
/// message starts, for ChatML (one start token, the last start always the
/// generation header) and for Inkling (role tokens, a header only when the
/// prompt opened the answer's text message itself).
@Suite struct PromptMessageMarkersTests {
    // ChatML: 1 = <|im_start|>.
    let chatML = PromptMessageMarkers.chatML(imStartID: 1)
    // Inkling: 10 user, 11 system, 12 model, 13 tool; 20 = <|content_text|>.
    let inkling = PromptMessageMarkers(startIDs: [10, 11, 12, 13], headerSuffix: [12, 20])

    @Test func chatMLDropsTheGenerationHeader() {
        // system, user, generation header.
        let ids: [Int32] = [1, 5, 5, 1, 6, 6, 1, 7]
        #expect(chatML.messageStarts(ids) == [0, 3])
    }

    @Test func chatMLWithOneMarkerHasNoMessageStart() {
        #expect(chatML.messageStarts([1, 5, 5]).isEmpty)
    }

    @Test func inklingKeepsTheLastTurnWhenTheModelOpensItsOwn() {
        // system, effort system, user: the model opens its own turn.
        let ids: [Int32] = [11, 20, 5, 11, 20, 6, 10, 20, 7]
        #expect(inkling.messageStarts(ids) == [0, 3, 6])
    }

    @Test func inklingDropsAnOpenedTextMessage() {
        // user, then the opened answer message <|message_model|><|content_text|>.
        let ids: [Int32] = [11, 20, 6, 10, 20, 7, 12, 20]
        #expect(inkling.messageStarts(ids) == [0, 3])
    }

    @Test func aModelTurnInHistoryIsAMessageStart() {
        // user, a replayed model turn, user again; no opened header.
        let ids: [Int32] = [10, 20, 5, 12, 20, 6, 10, 20, 7]
        #expect(inkling.messageStarts(ids) == [0, 3, 6])
    }
}
