import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@Suite struct LatestPromptEditPolicyTests {
    @Test func exposesOnlyTheNewestUserMessage() {
        let messages = [
            AppChatMessage(role: .user, content: "first"),
            AppChatMessage(role: .assistant, content: "first answer"),
            AppChatMessage(role: .user, content: "second"),
        ]

        #expect(LatestPromptEditPolicy.editableUserMessageIndex(
            in: messages,
            canEditLastPrompt: true) == 2)
    }

    @Test func hidesEditingWhenTheModelCannotReplaceTheLastTurn() {
        let messages = [AppChatMessage(role: .user, content: "question")]

        #expect(LatestPromptEditPolicy.editableUserMessageIndex(
            in: messages,
            canEditLastPrompt: false) == nil)
    }

    @Test func ignoresTranscriptsWithoutAUserMessage() {
        let messages = [AppChatMessage(role: .assistant, content: "answer")]

        #expect(LatestPromptEditPolicy.editableUserMessageIndex(
            in: messages,
            canEditLastPrompt: true) == nil)
    }
}
