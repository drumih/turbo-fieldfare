import Foundation
import Testing

@testable import TinyTitan

/// Model-free: Inkling's message format as the CLI and the server render it.
@Suite struct Inkling2GGUFSessionTests {
    @Test func rendersEveryRoleAndOpensAModelTurn() {
        let text = Inkling2GGUFSession.render([
            (.system, "Be brief."), (.user, "Hi"), (.assistant, "Hello."), (.tool, "{}"),
        ])
        #expect(
            text
                == "<|message_system|><|content_text|>Be brief.<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
                + "<|message_model|><|content_text|>Hello.<|end_message|>"
                + "<|message_tool|><|content_text|>{}<|end_message|>"
                + "<|message_model|><|content_text|>")
    }

    @Test func emptyConversationStillOpensAModelTurn() {
        #expect(Inkling2GGUFSession.render([]) == "<|message_model|><|content_text|>")
    }

    @Test func modelFileFindsOnlyInklingGGUFs() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "inkling2-gguf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        #expect(Inkling2GGUFSession.modelFile(at: folder) == nil)
        let other = folder.appendingPathComponent("config.json")
        try Data("{}".utf8).write(to: other)
        #expect(Inkling2GGUFSession.modelFile(at: other) == nil)
    }
}
