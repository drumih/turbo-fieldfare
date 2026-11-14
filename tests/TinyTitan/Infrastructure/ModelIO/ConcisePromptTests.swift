import Foundation
import Testing

@testable import TinyTitan
import TinyTitanCore

@Suite struct ConcisePromptTests {
    /// The shipped prompt is the only prompt, and it is not the unshipped one.
    ///
    /// This replaced `promptSelectionIsConsistentAcrossQuantizations`, which
    /// asserted that `prompt(forRoutedExpertBits:)` returned `standard` for 4, 6
    /// and 8 bits. It did — for every value, because no per-width variant ever
    /// existed — while both callers read the manifest to compute the width it
    /// ignored. The function is gone rather than left selecting nothing, so the
    /// assertion worth keeping is that the shipped text is the standard one.
    @Test func theShippedPromptIsTheStandardOne() {
        #expect(ConcisePrompt.standard != ConcisePrompt.strengthened)
        #expect(ConcisePrompt.standard.contains("Never:"))
    }

    @Test func appendsAfterExistingSystemGuidance() {
        let messages = [
            GFTokenizer.Message(role: .user, content: "first"),
            GFTokenizer.Message(role: .system, content: "user system"),
            GFTokenizer.Message(role: .user, content: "second"),
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 3)
        #expect(injected[1].role == .system)
        #expect(injected[1].content == "user system\n\nCONCISE")
    }

    @Test func mergesIntoDeveloperGuidanceToo() {
        let messages = [
            GFTokenizer.Message(role: .developer, content: "dev"),
            GFTokenizer.Message(role: .user, content: "question"),
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 2)
        #expect(injected[0].role == .system)
        #expect(injected[0].content == "dev\n\nCONCISE")
    }

    @Test func opensTheMessagesWhenNoSystemGuidanceExists() {
        let messages = [
            GFTokenizer.Message(role: .user, content: "question")
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 2)
        #expect(injected[0].role == .system)
        #expect(injected[0].content == "CONCISE")
        #expect(injected[1].role == .user)
    }

    @Test func standardAndStrengthenedPromptsAreNonEmpty() {
        #expect(!ConcisePrompt.standard.isEmpty)
        #expect(!ConcisePrompt.strengthened.isEmpty)
        #expect(ConcisePrompt.standard.contains("Never:"))
        #expect(ConcisePrompt.strengthened.contains("Never:"))
    }
}
