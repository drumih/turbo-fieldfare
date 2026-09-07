import Foundation

/// The message the composer handed over, carried as one value from the click to
/// the commit or back to the composer.
///
/// Its images are the composer's staged files, moved rather than copied: the
/// composer stops drawing them the moment this exists, and whichever stage
/// fails hands both halves back together. The prompt and the images used to
/// travel separately through the composer, the live fields and the screen, and
/// three defects were one half of the send path not knowing what the other had
/// already done with them — a replay overwriting the message that started it, a
/// hand-back that dropped the pictures, a second Generate starting a second
/// replay of the same chat.
public struct PreparedTurn: Equatable, Sendable {
    public let prompt: String
    public let images: [StagedImage]
    public init(prompt: String, images: [StagedImage]) {
        self.prompt = prompt
        self.images = images
    }

    /// The same message carried by different files.
    ///
    /// A turn that reaches the model is carried by the retained hard links, not
    /// by the composer's own copies: those are deleted at the hand-off, so
    /// handing them back after a rewind would give the user thumbnails with no
    /// files behind them.
    func carrying(_ images: [StagedImage]) -> PreparedTurn {
        PreparedTurn(prompt: prompt, images: images)
    }
}
