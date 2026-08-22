import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct MemoryFootprintExplanationTests {
    private static let chargedBytes: UInt64 = 168_296_448

    @Test func chargedLineNamesExpertSlotsNeverTheExpertPool() {
        let text = MemoryFootprintExplanation.text(chargedBytes: Self.chargedBytes)
        #expect(text.contains("expert slots"))
        // "experts" would claim the 12.9 GB routed pool is inside the footprint.
        // Only the 16 pread slot buffers are.
        #expect(!text.contains("experts"))
        #expect(text.contains("the KV cache, working buffers and expert slots."))
    }

    @Test func theExplanationIsTheChargedLineAndTheExclusion() {
        let text = MemoryFootprintExplanation.text(chargedBytes: Self.chargedBytes)
        #expect(text == """
            \(MetricFormat.memory(Self.chargedBytes)) charged to the app \u{2014} the KV cache, \
            working buffers and expert slots.

            The model's weights and the image tower are not counted here: the GPU reads them \
            straight from disk, and macOS keeps them as cached files it can drop at any moment, \
            charged to no app. To see them: Activity Monitor > Memory > Cached Files.
            """)
    }

    /// The tower is wording, never a live figure: a number in this popover
    /// invites adding it to the charged headline, and the diagnostics section
    /// already carries the measured "Image tower mapped" row.
    @Test func theTowerAppearsAsWordingWithNoFigure() {
        let text = MemoryFootprintExplanation.text(chargedBytes: Self.chargedBytes)
        #expect(text.contains("image tower"))
        #expect(!text.contains("Image tower held"))
        let charged = MetricFormat.memory(Self.chargedBytes)
        let figures = text.components(separatedBy: .newlines)
            .filter { $0.rangeOfCharacter(from: .decimalDigits) != nil }
        #expect(figures.allSatisfy { $0.contains(charged) },
                "a figure other than the charged bytes appeared: \(figures)")
    }

    @Test func anUnsampledFootprintStillExplainsTheExclusion() {
        let text = MemoryFootprintExplanation.text(chargedBytes: nil)
        #expect(text.hasPrefix("\u{2014} charged to the app"))
        #expect(text.contains(MemoryFootprintExplanation.exclusionNote))
    }

    @Test func theExclusionNoteCarriesNoFigureSoPeakRowsCanReuseIt() {
        let note = MemoryFootprintExplanation.exclusionNote
        #expect(!note.contains("charged to the app"))
        #expect(note.contains("Activity Monitor > Memory > Cached Files."))
        #expect(note.rangeOfCharacter(from: .decimalDigits) == nil)
    }
}
