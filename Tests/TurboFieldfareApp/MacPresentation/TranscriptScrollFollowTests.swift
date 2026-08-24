import Testing
import CoreGraphics
@testable import TurboFieldfareMacPresentation

/// The transcript's auto-follow, which has now been wrong in both directions:
/// pinned so hard the reader could not scroll during prefill, then released so
/// eagerly that a prompt with several images stayed scrolled off the top.
@Suite struct TranscriptScrollFollowTests {
    /// Images finish laying out after the scroll and grow the document. The
    /// view is left above the bottom through nobody's doing, and the follow
    /// must re-anchor rather than conclude the reader took over. This is the
    /// reported bug: several images, and the newest turn never comes into view.
    @Test func growingContentKeepsTheFollowAlive() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        // First scroll: the document is short, the view sits at the bottom.
        let firstScroll = follow.shouldScrollToBottom(origin: 0, documentHeight: 400)
        #expect(firstScroll)
        follow.recordScroll(origin: 0, documentHeight: 400)

        // Four thumbnails land, each taller than the last tick's document.
        var height: CGFloat = 400
        for _ in 0..<4 {
            height += 260
            let keepsFollowing = follow.shouldScrollToBottom(
                origin: 0, documentHeight: height)
            #expect(keepsFollowing,
                    "the follow gave up when an image grew the document to \(height)")
            // The view stays pinned to the bottom of the taller document.
            follow.recordScroll(origin: height - 400, documentHeight: height)
        }
        #expect(follow.isFollowing)
    }

    /// The other direction: a reader scrolling back must end the follow at
    /// once, even mid-prefill, and even though no live-scroll notification is
    /// posted for keyboard or scrollbar movement.
    @Test func aReaderScrollingEndsTheFollow() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        _ = follow.shouldScrollToBottom(origin: 600, documentHeight: 1_000)
        follow.recordScroll(origin: 600, documentHeight: 1_000)

        // Same document, different origin: nothing but a reader can do that.
        #expect(follow.readerMoved(origin: 200, documentHeight: 1_000))
        let afterReaderScroll = follow.shouldScrollToBottom(
            origin: 200, documentHeight: 1_000)
        #expect(!afterReaderScroll)
        #expect(!follow.isFollowing, "the follow survived a reader scroll")
        // And it stays ended once the document starts growing again.
        let afterGrowth = follow.shouldScrollToBottom(origin: 200, documentHeight: 1_400)
        #expect(!afterGrowth)
    }

    /// Growth and a reader scroll in the same tick resolve as growth. The
    /// alternative is worse: dropping the follow on a coincidence puts the user
    /// back in the bug this exists for, and the next tick catches the scroll
    /// anyway once the document settles.
    @Test func simultaneousGrowthAndScrollFavoursTheFollow() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        follow.recordScroll(origin: 100, documentHeight: 1_000)
        #expect(!follow.readerMoved(origin: 40, documentHeight: 1_600))
        let stillFollows = follow.shouldScrollToBottom(origin: 40, documentHeight: 1_600)
        #expect(stillFollows)
    }

    /// Sub-pixel jitter is not a scroll. Layout rounding moves the origin by
    /// fractions, and a follow that gave up on that would be no follow at all.
    @Test func roundingIsNotAScroll() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        follow.recordScroll(origin: 100, documentHeight: 1_000)
        #expect(!follow.readerMoved(origin: 100.2, documentHeight: 1_000.1))
        let survivesJitter = follow.shouldScrollToBottom(
            origin: 100.2, documentHeight: 1_000.1)
        #expect(survivesJitter)
    }

    /// A new run always wins: it is the thing the user just asked for, and the
    /// previous run's geometry says nothing about this one.
    @Test func aNewRunRestartsTheFollowFromNoBaseline() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        follow.recordScroll(origin: 800, documentHeight: 2_000)
        _ = follow.shouldScrollToBottom(origin: 100, documentHeight: 2_000)
        #expect(!follow.isFollowing)

        follow.beginRun()
        #expect(follow.isFollowing)
        // No baseline yet, so nothing counts as a reader scroll.
        #expect(!follow.readerMoved(origin: 0, documentHeight: 50))
        let scrollsForNewRun = follow.shouldScrollToBottom(origin: 0, documentHeight: 50)
        #expect(scrollsForNewRun)
    }

    /// Ending on the answer's first text is what hands control back; a follow
    /// that never ended is the "cannot scroll during generation" complaint.
    @Test func endStopsTheFollowOutright() {
        var follow = TranscriptScrollFollow()
        follow.beginRun()
        follow.end()
        #expect(!follow.isFollowing)
        let afterEnd = follow.shouldScrollToBottom(origin: 0, documentHeight: 10_000)
        #expect(!afterEnd)
    }
}

/// The composer's attachment tiles. Aspect-fitting them gave a row of
/// mismatched heights — a wide screenshot came out a third the height of a
/// portrait photo — and the remove badge, pinned to the tile's corner, floated
/// clear of the short ones.
@Suite struct AttachmentTileGeometryTests {
    private let tile = CGSize(width: 48, height: 48)

    /// What fitting did, kept as the reason the view no longer uses it.
    @Test func fittingProducesMismatchedHeights() {
        let wide = TranscriptImageTile.fittedSize(
            CGSize(width: 3_024, height: 1_000), within: tile)
        let tall = TranscriptImageTile.fittedSize(
            CGSize(width: 1_000, height: 3_024), within: tile)
        #expect(wide.height < tall.height / 2,
                "these two fitted to comparable heights, so the row was never ragged")
    }

    /// Filling gives every attachment the same tile, whatever its shape, and
    /// covers it completely so no background shows through.
    @Test func fillingGivesEveryShapeTheSameTile() {
        for source in [CGSize(width: 3_024, height: 1_000),
                       CGSize(width: 1_000, height: 3_024),
                       CGSize(width: 512, height: 512),
                       CGSize(width: 8_000, height: 6_000)] {
            let filled = TranscriptImageTile.filledSize(source, within: tile)
            #expect(filled.width >= tile.width - 0.01
                        && filled.height >= tile.height - 0.01,
                    "\(source) left part of the tile uncovered: \(filled)")
            // Aspect preserved: one dimension matches exactly, the other spills.
            let ratio = filled.width / filled.height
            let sourceRatio = source.width / source.height
            #expect(abs(ratio - sourceRatio) < 0.01,
                    "\(source) was distorted to \(filled)")
        }
    }

    /// A degenerate image must not produce a zero or infinite tile.
    @Test func adegenerateSourceStillFillsTheTile() {
        let filled = TranscriptImageTile.filledSize(.zero, within: tile)
        #expect(filled == tile)
    }

    /// Fitting bounds an image; it does not enlarge one. The transcript draws
    /// at the returned size, so an unclamped scale turned a 64 pt icon into a
    /// 240 pt canvas of the same 64 pixels — a 3.75x blur on every small
    /// attachment.
    @Test func fittingNeverEnlargesASourceSmallerThanTheTile() {
        let transcriptTile = CGSize(width: 360, height: 240)
        let icon = CGSize(width: 64, height: 64)
        #expect(TranscriptImageTile.fittedSize(icon, within: transcriptTile) == icon,
                "a source already inside the tile has to come back at its own size")

        // One axis under, one axis over: the oversized axis still decides the
        // scale, so this must shrink rather than pass through unchanged.
        let wideAndShort = CGSize(width: 720, height: 100)
        let fitted = TranscriptImageTile.fittedSize(wideAndShort, within: transcriptTile)
        #expect(fitted.width <= transcriptTile.width + 0.01
                    && fitted.height <= transcriptTile.height + 0.01,
                "\(wideAndShort) came back as \(fitted), outside the tile")
        #expect(fitted.width < wideAndShort.width,
                "an over-wide source still has to be scaled down")
    }
}
