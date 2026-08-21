import CoreGraphics

/// Sizing for attached-image thumbnails.
///
/// The composer shows a row of them, and they have to read as one row. Fitting
/// each image inside the tile preserved its shape but gave every attachment a
/// different height, so the row was ragged and the remove badge — pinned to the
/// tile corner — sat clear of the short ones. Filling the tile and cropping the
/// overflow is what makes a row of mixed screenshots and photos look like a row.
public enum TranscriptImageTile {
    /// Largest size that fits inside `tile` with the aspect ratio kept, never
    /// larger than the source. Used by the transcript, where images are laid out
    /// inline and may be any shape.
    ///
    /// The clamp is what stops a 64 pt icon being drawn into a 240 pt canvas:
    /// scaling up an attachment smaller than the tile only makes it blurry, and
    /// "fits inside the tile" was never meant to mean "fills it".
    public static func fittedSize(_ source: CGSize, within tile: CGSize) -> CGSize {
        guard source.width > 0, source.height > 0 else { return tile }
        let scale = min(tile.width / source.width, tile.height / source.height, 1)
        return CGSize(width: max(1, source.width * scale),
                      height: max(1, source.height * scale))
    }

    /// Smallest size that covers `tile` with the aspect ratio kept. The excess
    /// is cropped by the tile, which is the point.
    public static func filledSize(_ source: CGSize, within tile: CGSize) -> CGSize {
        guard source.width > 0, source.height > 0 else { return tile }
        let scale = max(tile.width / source.width, tile.height / source.height)
        return CGSize(width: max(1, source.width * scale),
                      height: max(1, source.height * scale))
    }
}
