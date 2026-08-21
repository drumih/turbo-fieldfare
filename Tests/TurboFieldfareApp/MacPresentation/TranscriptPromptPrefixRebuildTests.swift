import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

/// The image strip is built asynchronously and applied when it lands, so the
/// identifier the controller is told about decides whether it ever gets drawn.
///
/// The coordinator used to record the final identifier up front, alongside an
/// empty prefix, and only then start the build. When the built prefix arrived it
/// carried that same identifier — so prompt, identifier, response and finalized
/// state all matched, `needsRebuild` was false, `storage.length == 0` was false,
/// and the strip was dropped. These tests pin both halves of the rule the fix
/// relies on.
@Suite @MainActor struct TranscriptPromptPrefixRebuildTests {
    private func makePrefix(_ text: String) -> NSAttributedString {
        NSAttributedString(string: text)
    }

    /// The failure mode, stated as a rule: same identifier, different prefix, no
    /// rebuild. This is why the identifier must not be recorded before the
    /// prefix it names exists.
    @Test func areusedIdentifierDoesNotRedrawAChangedPrefix() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        _ = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: NSAttributedString(),
            promptPrefixIdentifier: "image-a")

        let late = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: makePrefix("[image]\n"),
            promptPrefixIdentifier: "image-a")

        #expect(late.mutation == .none,
                "reusing the identifier redrew, so the coordinator's contract can be relaxed")
        #expect(!storage.string.contains("[image]"))
    }

    /// The contract the coordinator now honours: the identifier stays empty
    /// until the prefix exists, so the build landing is a change and rebuilds.
    @Test func anidentifierArrivingWithItsPrefixRedraws() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        let initial = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: NSAttributedString(),
            promptPrefixIdentifier: "")
        #expect(initial.mutation == .rebuilt)
        #expect(!storage.string.contains("[image]"))

        let built = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: makePrefix("[image]\n"),
            promptPrefixIdentifier: "image-a")

        #expect(built.mutation == .rebuilt,
                "the built image strip never reached the transcript")
        #expect(storage.string.contains("[image]"))
        #expect(storage.string.contains("describe this"))
    }

    /// A transcript that is already finished is the case with no rescue: there
    /// is no first token to flip the prefill placeholder and force a rebuild, so
    /// the late prefix has to redraw on its own.
    @Test func afinishedTurnStillPicksUpItsImagesWhenTheyArriveLate() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        _ = controller.synchronize(
            storage: storage, prompt: "describe this", response: "a cat",
            isTerminal: true, promptPrefix: NSAttributedString(),
            promptPrefixIdentifier: "")

        let built = controller.synchronize(
            storage: storage, prompt: "describe this", response: "a cat",
            isTerminal: true, promptPrefix: makePrefix("[image]\n"),
            promptPrefixIdentifier: "image-a")

        #expect(built.mutation != .none)
        #expect(storage.string.contains("[image]"),
                "a finished image turn re-rendered with no images at all")
    }

    /// Changing the attachments changes the identifier, so the old strip must
    /// come off even before the replacement is built.
    @Test func clearingTheAttachmentsClearsTheStrip() {
        let controller = InstructionTranscriptDocumentController()
        let storage = NSMutableAttributedString()

        _ = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: makePrefix("[image]\n"),
            promptPrefixIdentifier: "image-a")
        #expect(storage.string.contains("[image]"))

        let cleared = controller.synchronize(
            storage: storage, prompt: "describe this", response: "",
            isTerminal: false, promptPrefix: NSAttributedString(),
            promptPrefixIdentifier: "")

        #expect(cleared.mutation == .rebuilt)
        #expect(!storage.string.contains("[image]"))
    }
}
