import Foundation
import SwiftUI
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

/// A row that cannot be continued has to say which of the two very different
/// things went wrong, and what it would take to fix. A greyed-out row says
/// neither, which is why none of these states is one.
@Suite struct ConversationSidebarPresentationTests {
    private let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private func meta(_ title: String = "Explain RoPE",
                      kvTokens: Int? = 1_200,
                      updatedAt: Date = Date()) -> ConversationMeta {
        ConversationMeta(
            id: UUID(), title: title, createdAt: updatedAt, updatedAt: updatedAt,
            turnCount: 4, kvTokens: kvTokens, identity: identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "onDemand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 4_096))
    }

    private var calendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        return calendar
    }

    /// The boundaries, against an injected `now`. A grouping only checked at
    /// whatever time the suite happened to run is a grouping nobody has checked.
    @Test func theDayBoundariesLandWhereTheHeadingsSayTheyDo() {
        let calendar = calendar
        let now = calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 31, hour: 14))!
        func group(daysAgo: Int, hour: Int = 12) -> ConversationDateGroup {
            let day = calendar.date(byAdding: .day, value: -daysAgo, to: now)!
            let at = calendar.date(bySettingHour: hour, minute: 0, second: 0, of: day)!
            return ConversationDateGroup.group(for: at, now: now, calendar: calendar)
        }
        // One minute after midnight today is still Today; one minute before it
        // is Yesterday, even though they are two minutes apart.
        #expect(group(daysAgo: 0, hour: 0) == .today)
        #expect(group(daysAgo: 1, hour: 23) == .yesterday)
        #expect(group(daysAgo: 2) == .previousSevenDays)
        #expect(group(daysAgo: 7) == .previousSevenDays)
        #expect(group(daysAgo: 8) == .older)
        // A clock change or a restored backup can date a chat in the future.
        // "Older" would be the one place it certainly does not belong.
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: now)!
        #expect(ConversationDateGroup.group(
            for: tomorrow, now: now, calendar: calendar) == .today)
    }

    @Test func sectionsAreOrderedNewestFirstWithinAndBetween() {
        let now = Date()
        let today = meta("today", updatedAt: now.addingTimeInterval(-60))
        let earlierToday = meta("earlier", updatedAt: now.addingTimeInterval(-3_600))
        let old = meta("old", updatedAt: now.addingTimeInterval(-60 * 60 * 24 * 30))
        let sections = ConversationDateGroup.sections([old, earlierToday, today], now: now)
        #expect(sections.map(\.group) == [.today, .older])
        #expect(sections[0].entries.map(\.title) == ["today", "earlier"])
    }

    /// Exact numbers, because "needs more context" is not something a user can
    /// act on and the picker only offers certain sizes.
    @Test func eachStateHasItsOwnSubtitle() {
        #expect(ConversationRowPresentation.subtitle(
            meta(kvTokens: 6_120), state: .continuable).contains("6,120"))

        // What it holds, then the context it needs rounded up to a size the
        // picker actually offers. The raw requirement is not a thing anyone can
        // select, so the row names the option instead.
        #expect(ConversationRowPresentation.subtitle(
            meta(kvTokens: 6_120), state: .needsContext(required: 6_376))
            == "6,120 tokens \u{00B7} needs 8K context")

        #expect(ConversationRowPresentation.subtitle(
            meta(), state: .cannotReplay(reason: .differentModel))
            == "recorded with a different model")
        #expect(ConversationRowPresentation.subtitle(
            meta(), state: .cannotReplay(reason: .differentTemplate))
            == "recorded with a different chat format")
        #expect(ConversationRowPresentation.subtitle(
            meta(kvTokens: nil), state: .cannotReplay(reason: .tokenCountUnknown))
            == "no replay record")

        let missingImage = ConversationMeta(
            id: UUID(), title: "roofs", createdAt: Date(), updatedAt: Date(),
            turnCount: 2, kvTokens: nil, imageCount: 1,
            identity: identity,
            session: ConversationSessionSettings(
                contextTokens: 8_192, expertCacheSlots: 16,
                visionResidencyPolicy: "on-demand"),
            sampling: ConversationSampling(
                temperature: 0.2, topKEnabled: true, topK: 64,
                topPEnabled: true, topP: 0.95, maxNewTokens: 8_192),
            imageWriteFailed: true)
        let state = ConversationContinuability.evaluate(
            meta: missingImage, currentContext: 8_192, identity: identity)
        #expect(ConversationRowPresentation.subtitle(missingImage, state: state)
            == "an image was not saved")
    }

    /// Raising the context reloads the model and ends the chat that is open.
    /// Finding that out afterwards is the failure this copy exists to prevent.
    @Test func theNoticeNamesTheContextAndWhatRaisingItCosts() throws {
        let content = try #require(ConversationStateNotice.content(
            for: .needsContext(required: 6_376), currentContext: 4_096))
        #expect(content.message.contains("6,376"))
        #expect(content.message.contains("4,096"))
        #expect(content.message.contains("8K"))
        #expect(content.message.contains("reloads the model"))
        #expect(content.message.contains("ends the chat"))
        #expect(content.requiredContext == 6_376)
        // Re-reading is not offered here: the conversation replays fine, it
        // just does not fit, and a re-read would throw away a working lineage.
        #expect(!content.offersReread)
    }

    /// A re-read is a different conversation and has to be named as one
    /// wherever it appears. Presenting it as "Continue" is the one thing it
    /// must never do.
    @Test func theUnreplayableNoticeOffersARereadAndSaysItIsOne() throws {
        let content = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .differentModel), currentContext: 8_192))
        #expect(content.offersReread)
        #expect(content.requiredContext == nil)
        #expect(content.message.contains("re-read"))
        #expect(content.message.contains("different conversation"))
    }

    @Test func theMissingImageNoticeNamesTheStorageFailure() throws {
        let content = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .imageRecordMissing),
            currentContext: 8_192))
        #expect(content.offersReread)
        #expect(content.message.contains("could not be saved"))
    }

    @Test func aNewerFormatNoticeRefusesReplayAndNamesTheRequiredUpgrade() throws {
        let content = try #require(ConversationStateNotice.content(
            for: .cannotReplay(reason: .newerFormat), currentContext: 8_192))
        #expect(!content.offersReread)
        #expect(content.message.contains("newer history format"))
        #expect(content.message.contains("saved files are unchanged"))
    }

    @Test func aContinuableChatShowsNoNotice() {
        #expect(ConversationStateNotice.content(
            for: .continuable, currentContext: 8_192) == nil)
    }
}

@Suite struct WindowControlsPresentationTests {
    /// The toggle's help has to describe what clicking it will do, which is the
    /// opposite of the state it is in.
    @Test func theToggleHelpNamesTheActionNotTheState() {
        #expect(WindowControlsPresentation.sidebarToggleHelp(isVisible: true)
            == "Hide Sidebar")
        #expect(WindowControlsPresentation.sidebarToggleHelp(isVisible: false)
            == "Show Sidebar")
        #expect(WindowControlsPresentation.sidebarToggleAccessibilityLabel(isVisible: true)
            == "Hide Sidebar")
        #expect(WindowControlsPresentation.newChatHelp == "New Chat")
    }

    /// The Inspector toggle is the sidebar toggle's mirror, so it has to name
    /// the action the same way round.
    @Test func theInspectorToggleHelpNamesTheActionNotTheState() {
        #expect(WindowControlsPresentation.inspectorToggleHelp(isVisible: true)
            == "Hide Inspector")
        #expect(WindowControlsPresentation.inspectorToggleHelp(isVisible: false)
            == "Show Inspector")
        #expect(WindowControlsPresentation.inspectorToggleAccessibilityLabel(isVisible: false)
            == "Show Inspector")
    }
}

@Suite struct TransientPopoverPresentationTests {
    @MainActor
    @Test func promptTipsDismissesThroughTransientAppKitBehavior() {
        let coordinator = TransientPopoverButton(systemImage: "lightbulb", help: "Prompt Tips") {
            EmptyView()
        }.makeCoordinator()
        #expect(TransientPopoverPresentation.behavior == .transient)
        #expect(coordinator.popover.behavior == .transient)
    }
}
