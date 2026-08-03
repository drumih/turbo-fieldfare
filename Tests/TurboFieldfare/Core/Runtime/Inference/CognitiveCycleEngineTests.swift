import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct CognitiveCycleEngineTests {
    @Test func nextStepReturnsPlanThenDraftThenCritiqueThenFinal() {
        var engine = CognitiveCycleEngine(userPrompt: "Explain Metal")

        #expect(engine.nextStep()?.kind == .plan)
        engine.record(output: "plan")
        #expect(engine.nextStep()?.kind == .draft)
        engine.record(output: "draft")
        #expect(engine.nextStep()?.kind == .critique)
        engine.record(output: "critique")
        #expect(engine.nextStep()?.kind == .final)
        engine.record(output: "final")
        #expect(engine.nextStep() == nil)
    }

    @Test func nextStepIsIdempotentUntilRecord() {
        // Calling nextStep twice without recording returns the same step.
        var engine = CognitiveCycleEngine(userPrompt: "p")
        #expect(engine.nextStep()?.kind == .plan)
        #expect(engine.nextStep()?.kind == .plan)
    }

    @Test func planPromptContainsUserPrompt() {
        var engine = CognitiveCycleEngine(userPrompt: "Explain Metal")
        let plan = engine.nextStep()

        #expect(plan?.kind == .plan)
        #expect(plan?.prompt.contains("Explain Metal") == true)
        #expect(plan?.prompt.contains("Plan your answer") == true)
    }

    @Test func draftPromptEmbedsPlanOutput() {
        var engine = CognitiveCycleEngine(userPrompt: "Explain Metal")
        _ = engine.nextStep()  // plan
        engine.record(output: "1. History\n2. Architecture")

        let draft = engine.nextStep()

        #expect(draft?.kind == .draft)
        #expect(draft?.prompt.contains("1. History") == true)
        #expect(draft?.prompt.contains("2. Architecture") == true)
    }

    @Test func critiquePromptEmbedsDraftOutput() {
        var engine = CognitiveCycleEngine(userPrompt: "Explain Metal")
        _ = engine.nextStep()  // plan
        engine.record(output: "plan text")
        _ = engine.nextStep()  // draft
        engine.record(output: "draft text")

        let critique = engine.nextStep()

        #expect(critique?.kind == .critique)
        #expect(critique?.prompt.contains("draft text") == true)
        #expect(critique?.prompt.contains("Critique") == true)
    }

    @Test func finalPromptEmbedsDraftAndCritique() {
        var engine = CognitiveCycleEngine(userPrompt: "Explain Metal")
        _ = engine.nextStep()  // plan
        engine.record(output: "plan text")
        _ = engine.nextStep()  // draft
        engine.record(output: "draft text")
        _ = engine.nextStep()  // critique
        engine.record(output: "critique text")

        let final = engine.nextStep()

        #expect(final?.kind == .final)
        #expect(final?.prompt.contains("draft text") == true)
        #expect(final?.prompt.contains("critique text") == true)
        #expect(final?.prompt.contains("Output only the final") == true)
    }

    @Test func engineFinishesAfterAllFourOutputs() {
        var engine = CognitiveCycleEngine(userPrompt: "p")
        for _ in 0..<4 {
            guard let step = engine.nextStep() else {
                Issue.record("Expected four steps")
                return
            }
            engine.record(output: "output for \(step.kind.rawValue)")
        }

        #expect(engine.isFinished)
        #expect(engine.nextStep() == nil)
        #expect(engine.outputs.count == 4)
    }

    @Test func recordFillsTheNextExpectedPass() {
        // record assigns to the next step the engine would emit, so an
        // orchestrator can record even before the first nextStep call.
        var engine = CognitiveCycleEngine(userPrompt: "p")
        engine.record(output: "orphan")

        #expect(engine.outputs[.plan] == "orphan")
        #expect(!engine.isFinished)
        #expect(engine.nextStep()?.kind == .draft)
    }

    @Test func headersMatchPassKind() {
        #expect(CognitiveStepKind.plan.header == "Plan")
        #expect(CognitiveStepKind.draft.header == "Draft")
        #expect(CognitiveStepKind.critique.header == "Critique")
        #expect(CognitiveStepKind.final.header == "Final answer")
    }

    @Test func cycleIteratesWithoutDroppingOutputs() {
        var engine = CognitiveCycleEngine(userPrompt: "p")
        var seen: [CognitiveStepKind] = []

        while let step = engine.nextStep() {
            seen.append(step.kind)
            engine.record(output: "text \(step.kind.rawValue)")
        }

        #expect(seen == [.plan, .draft, .critique, .final])
    }
}
