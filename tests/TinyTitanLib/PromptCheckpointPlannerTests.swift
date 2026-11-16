import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

/// T22: the interior boundaries are located on the rendered ids, not on a
/// re-render of truncated messages. The ChatML template renders the last
/// assistant turn differently depending on whether a user turn follows it, so
/// the second form would not be a prefix of the first.
@Suite struct PromptCheckpointPlannerTests {
    private let imStart: Int32 = 9

    private func message(_ role: GFTokenizer.Role, _ content: String)
        -> GFTokenizer.Message
    {
        GFTokenizer.Message(role: role, content: content)
    }

    @Test func findsSystemEndAndLastTurnStart() {
        // system | user | assistant | user | generation prompt
        let ids: [Int32] = [
            9, 1, 2,
            9, 3, 4,
            9, 5, 6,
            9, 7, 8,
            9,
        ]
        let messages = [
            message(.system, "system"),
            message(.user, "first"),
            message(.assistant, "answer"),
            message(.user, "edited"),
        ]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart)
        #expect(boundaries.count == 2)
        #expect(
            boundaries.contains(
                PromptCheckpointBoundary(kind: .systemEnd, position: 3)))
        #expect(
            boundaries.contains(
                PromptCheckpointBoundary(kind: .lastTurnStart, position: 9)))
    }

    @Test func aSingleUserTurnUsesOnePosition() {
        // user | generation prompt; there is no system block and the only
        // user turn is both the first message and the edited turn.
        let ids: [Int32] = [9, 1, 2, 9]
        let messages = [message(.user, "only")]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart)
        // Position 0 is not a checkpoint: there is nothing before it to
        // restore. The planner must not invent one.
        #expect(boundaries.isEmpty)
    }

    @Test func aChatEndingOnAnAssistantTurnHasNoEditedTurnPoint() {
        let ids: [Int32] = [9, 1, 2, 9, 3, 4, 9]
        let messages = [
            message(.user, "question"),
            message(.assistant, "answer"),
        ]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart)
        #expect(!boundaries.contains { $0.kind == .lastTurnStart })
    }

    @Test func aMarkerInsideContentCannotMoveTheLastTurnPastTheRender() {
        // A literal <|im_start|> inside the last user message is another
        // occurrence of the id. The planner treats the last message start as
        // the last occurrence before the generation prompt; a content marker
        // is inside that message and is still a valid prefix boundary (the
        // edited-turn key misses when the edit crosses it), never a wrong
        // restore.
        // message starts are 0 and 2; the 9 at index 5 is literal content and
        // index 7 is the generation prompt.
        let ids: [Int32] = [9, 1, 9, 2, 3, 9, 4, 9]
        let messages = [
            message(.user, "first"),
            message(.user, "second"),
        ]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart)
        #expect(
            boundaries
                == [PromptCheckpointBoundary(kind: .lastTurnStart, position: 5)])
    }

    @Test func conversationKeyIsStableAcrossGrowthAndEdits() {
        let domain = ServerPromptCacheDomain(
            modelID: "ornith",
            sourceSnapshotHash: "source",
            runtimeProfileHash: "profile",
            maximumContext: 8192,
            kvStorage: "int8",
            fp16RingEnabled: false,
            templateSHA256: "template")
        let before = [
            message(.system, "system"),
            message(.user, "first"),
        ]
        let edited = [
            message(.system, "system"),
            message(.user, "first"),
            message(.assistant, "answer"),
            message(.user, "edited last turn"),
        ]
        let other = [
            message(.system, "system"),
            message(.user, "a different first turn"),
        ]
        #expect(
            PromptCheckpointPlanner.conversationKey(
                domain: domain, messages: before)
                == PromptCheckpointPlanner.conversationKey(
                    domain: domain, messages: edited))
        #expect(
            PromptCheckpointPlanner.conversationKey(
                domain: domain, messages: before)
                != PromptCheckpointPlanner.conversationKey(
                    domain: domain, messages: other))
    }

    @Test func anOptionalGridAddsPositionsButNeverOverwritesAMessageBoundary() {
        let ids: [Int32] = [9, 1, 2, 9, 3, 4, 9]
        let messages = [message(.user, "question")]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart,
            gridTokens: 2)
        // Grid points 2 and 4; 0 is the render's start and not a restore
        // point, 6 is inside the id list's last message token.
        #expect(
            boundaries
                == [
                    PromptCheckpointBoundary(kind: .grid, position: 2),
                    PromptCheckpointBoundary(kind: .lastTurnStart, position: 3),
                    PromptCheckpointBoundary(kind: .grid, position: 4),
                    PromptCheckpointBoundary(kind: .grid, position: 6),
                ])
    }

    @Test func gridOnlyOmitsTheMessageBoundaries() {
        let ids: [Int32] = [9, 1, 2, 9, 3, 4, 9]
        let messages = [message(.user, "question")]
        let boundaries = PromptCheckpointPlanner.boundaries(
            promptIDs: ids, messages: messages, imStartID: imStart,
            gridTokens: 2, includeMessageBoundaries: false)
        #expect(!boundaries.contains { $0.kind == .lastTurnStart })
        #expect(
            boundaries
                == [
                    PromptCheckpointBoundary(kind: .grid, position: 2),
                    PromptCheckpointBoundary(kind: .grid, position: 4),
                    PromptCheckpointBoundary(kind: .grid, position: 6),
                ])
    }
}
