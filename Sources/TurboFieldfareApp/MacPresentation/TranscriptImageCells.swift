import AppKit
import TurboFieldfareAppCore

/// Which images a turn's inline row is built from.
///
/// One cell per attachment, always. An image whose file is gone or will not
/// decode used to be dropped, and the turn then rendered as one that never had
/// a picture — under an answer that discusses it — so the transcript stopped
/// agreeing with the context the model was actually given. A conversation can
/// lose a stored file to anything outside this app, and the reader has to be
/// able to tell "there was a picture here" from "there was not".
public enum TranscriptImageCells {
    public static func images(
        for attachments: [ChatImage],
        load: (ChatImage) -> NSImage?,
        unreadable: () -> NSImage
    ) -> [NSImage] {
        attachments.map { load($0) ?? unreadable() }
    }
}
