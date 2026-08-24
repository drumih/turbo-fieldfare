import CoreGraphics

/// Decides whether the transcript should keep itself pinned to the bottom.
///
/// A new run is scrolled into view and held there until its answer starts,
/// because the images in the prompt finish laying out after the first scroll
/// and push the content down. The hard part is telling that growth apart from
/// the reader scrolling back to look at something: both leave the view above
/// the bottom, and treating the two the same is what broke it in each
/// direction. Pinning on every tick made it impossible to scroll during
/// prefill; ending the follow whenever the view was not at the bottom meant a
/// prompt with several images stayed scrolled off the top, which is the case
/// the follow exists for.
///
/// The discriminator is what moved. Images growing the document change its
/// height and leave the scroll origin alone; a reader moves the origin while
/// the height stays put.
public struct TranscriptScrollFollow: Equatable, Sendable {
    /// Anything smaller is rounding, not a scroll.
    public static let tolerance: CGFloat = 0.5

    public private(set) var isFollowing = false
    private var lastOrigin: CGFloat?
    private var lastHeight: CGFloat?

    public init() {}

    /// Starts a new run's follow. The previous run's geometry is not a baseline
    /// for this one.
    public mutating func beginRun() {
        isFollowing = true
        lastOrigin = nil
        lastHeight = nil
    }

    /// Ends the follow: the answer has text, the run is over, or the reader
    /// scrolled in a way the platform reported directly.
    public mutating func end() {
        isFollowing = false
    }

    /// Records where a programmatic scroll left the view. Without this the next
    /// comparison reads our own scroll as the reader's.
    public mutating func recordScroll(origin: CGFloat, documentHeight: CGFloat) {
        lastOrigin = origin
        lastHeight = documentHeight
    }

    /// True when the view moved and the document did not, which no layout pass
    /// can cause.
    public func readerMoved(origin: CGFloat, documentHeight: CGFloat) -> Bool {
        guard let lastOrigin, let lastHeight else { return false }
        guard abs(documentHeight - lastHeight) < Self.tolerance else { return false }
        return abs(origin - lastOrigin) > Self.tolerance
    }

    /// The whole decision for one tick: whether to scroll now, having first let
    /// the reader take over if they moved the view.
    public mutating func shouldScrollToBottom(
        origin: CGFloat, documentHeight: CGFloat
    ) -> Bool {
        guard isFollowing else { return false }
        if readerMoved(origin: origin, documentHeight: documentHeight) {
            isFollowing = false
            return false
        }
        return true
    }
}
