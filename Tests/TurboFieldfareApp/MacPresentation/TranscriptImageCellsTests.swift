import AppKit
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

/// A turn's inline image row has to say how many pictures the turn had, not how
/// many of them happen to be readable right now.
@Suite struct TranscriptImageCellsTests {
    private func attachment(_ name: String) -> ChatImage {
        .staged(StagedImage(
            fileURL: URL(fileURLWithPath: "/tmp/\(name)"),
            displayName: name, encodedBytes: 1, sha256: name))
    }

    private func image(_ side: CGFloat) -> NSImage {
        NSImage(size: NSSize(width: side, height: side))
    }

    /// The regression. A stored conversation can lose a file to anything
    /// outside this app, and dropping the cell rendered the turn as one that
    /// never had a picture — under an answer discussing it. The reader could
    /// not tell the two apart.
    @Test func anUnreadableImageKeepsItsCell() {
        let attachments = [attachment("a"), attachment("b"), attachment("c")]
        let stand = image(160)
        let cells = TranscriptImageCells.images(
            for: attachments,
            load: { $0.displayName == "b" ? nil : self.image(10) },
            unreadable: { stand })

        #expect(cells.count == 3, "the unreadable image was dropped")
        #expect(cells[1] === stand)
        #expect(cells[0] !== stand)
        #expect(cells[2] !== stand)
    }

    /// And the stand-in is only reached when it is needed: building it for
    /// every turn would decode a symbol on each redraw of a transcript that has
    /// nothing wrong with it.
    @Test func areadableImageNeverBuildsTheStandIn() {
        var built = 0
        let cells = TranscriptImageCells.images(
            for: [attachment("a"), attachment("b")],
            load: { _ in self.image(10) },
            unreadable: { built += 1; return self.image(160) })

        #expect(cells.count == 2)
        #expect(built == 0)
    }

    @Test func noAttachmentsMakeNoCells() {
        #expect(TranscriptImageCells.images(
            for: [], load: { _ in nil }, unreadable: { self.image(160) }).isEmpty)
    }
}
