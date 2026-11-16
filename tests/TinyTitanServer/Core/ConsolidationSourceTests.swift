import Foundation
import Testing
import TinyTitan

@testable import TinyTitanMemory
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// Consolidation is where the authority comes from, so the flag has to
/// survive the extraction's own output shapes.
@Suite struct ConsolidationSourceTests {

    private func records(_ text: String) -> [MemoryRecord] {
        ServerMemory.consolidationRecords(from: text)
    }

    @Test func sourceUserMarksTheRecord() {
        let parsed = records(
            """
            ```json
            [{"key": "setting/town", "value": "Ashgrove", "source": "user"},
             {"key": "state/draft", "value": "chapter 12 written", "source": "assistant"}]
            ```
            """)
        #expect(parsed.count == 2)
        #expect(parsed[0].isUserAsserted)
        #expect(!parsed[1].isUserAsserted)
    }

    /// The safe direction. An extraction that omits the field, or answers
    /// something unexpected, must degrade to today's behaviour -- a
    /// mislabelled invention would be *protected*, which is worse than
    /// merely being wrong.
    @Test func anythingButUserIsTheModelsOwn() {
        for entry in [
            #"{"key": "a/b", "value": "v"}"#,
            #"{"key": "a/b", "value": "v", "source": "USER "}"#,
            #"{"key": "a/b", "value": "v", "source": "person"}"#,
            #"{"key": "a/b", "value": "v", "source": 1}"#,
            #"{"key": "a/b", "value": "v", "source": null}"#,
        ] {
            let parsed = records("[\(entry)]")
            #expect(parsed.count == 1, "\(entry)")
            #expect(parsed[0].isUserAsserted == false, "\(entry)")
        }
    }

    /// "user" survives the three shapes the parser already recovers: a
    /// fenced block, a bare object, and an array the output cap truncated.
    @Test func flagSurvivesEveryRecoveredShape() {
        #expect(
            records(#"{"key": "a/b", "value": "v", "source": "user"}"#)
                .first?.isUserAsserted == true)
        #expect(
            records(
                """
                [{"key": "a/b", "value": "v", "source": "user"},
                 {"key": "c/d", "value": "w", "sour
                """
            ).first?.isUserAsserted == true)
        #expect(
            records(#"prose ```json [{"key":"a/b","value":"v","source":"user"}] ``` more"#)
                .first?.isUserAsserted == true)
    }

    /// The prompt has to ask for it, or the model has no way to answer.
    @Test func thePromptAsksForTheSource() {
        let turn = JournalTurn(
            session: "s", workspace: "w", index: 0,
            prompt: "the town is Ashgrove", reply: "Noted.")
        let request = ServerMemory.consolidationRequest(
            turns: [turn], existing: [], workspace: "w")
        let system = request.messages.first { $0.role == .system }?.content ?? ""
        #expect(system.contains("\"source\""))
        #expect(system.contains("USER"))
    }
}
