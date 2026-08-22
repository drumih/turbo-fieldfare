import AppKit
import ImageIO
import UniformTypeIdentifiers

/// The one place an attached image file is decoded for display.
///
/// Both call sites go through here — the composer's thumbnail row and the
/// transcript's inline copy of what was sent — so the decode budget is one rule
/// rather than one rule per caller. It sits in the presentation module rather
/// than beside the view because the view lives in the app executable, which no
/// test target can reach, and a budget no test can reach is a budget that
/// drifts back out.
public enum TranscriptImageLoader {
    /// What a source is measured against before ImageIO is asked for anything.
    ///
    /// The numbers come from the caller rather than from an import so the app
    /// keeps a single mapping site onto the runtime's `VisionImageLimits` (see
    /// Runtime/Vision/Preprocessing/ImageMetadataReader.swift) and so the rule
    /// can be exercised against a small image instead of a 50 MP one.
    public struct Budget: Sendable {
        let maximumSourcePixels: Int
        let maximumSourceDimension: Int
        /// One ceiling for every format, matching `VisionImageLimits`: a
        /// source the model would refuse should not draw either.
        let maximumDecodedBytes: Int
        /// The decoders this is allowed to reach. Every other limit here was
        /// copied from `VisionImageLimits` except this one, so a camera RAW or
        /// EXR the runtime refuses still ran through ImageIO's decoder for it
        /// inside the process holding the model.
        let allowedTypeIdentifiers: Set<String>

        public init(
            maximumSourcePixels: Int,
            maximumSourceDimension: Int,
            maximumDecodedBytes: Int,
            allowedTypeIdentifiers: Set<String>
        ) {
            self.maximumSourcePixels = maximumSourcePixels
            self.maximumSourceDimension = maximumSourceDimension
            self.maximumDecodedBytes = maximumDecodedBytes
            self.allowedTypeIdentifiers = allowedTypeIdentifiers
        }
    }

    /// A thumbnail bounded by `maximumPixelSize`, or nil when the file cannot be
    /// read or its decode would exceed `budget`.
    ///
    /// Nil is the refusal, not an error: both callers already draw their
    /// placeholder for an image that would not load, so a file the process
    /// cannot afford costs a tile rather than the process.
    ///
    /// `cacheKey` identifies the *contents*, so callers pass the attachment's
    /// digest rather than its path. Without it every redraw re-decoded the file:
    /// the transcript rebuilds its prefix whenever the coordinator is recreated,
    /// and `kCGImageSourceCreateThumbnailFromImageAlways` costs a full decode
    /// for most formats. Omitting the key skips the cache entirely.
    public static func thumbnail(
        at url: URL,
        maximumPixelSize: Int,
        budget: Budget,
        cacheKey: String? = nil
    ) -> NSImage? {
        let entry = cacheKey.map { "\($0)-\(maximumPixelSize)" as NSString }
        if let entry, let cached = cache.storage.object(forKey: entry) { return cached }
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(url as CFURL, sourceOptions) else {
            return nil
        }
        // ImageIO sniffs the bytes, so this is the real format rather than the
        // extension's claim. Checked before the budget because an unsupported
        // format is refused whatever its size.
        guard let typeIdentifier = CGImageSourceGetType(source) as String?,
              budget.allowedTypeIdentifiers.contains(typeIdentifier) else {
            return nil
        }
        guard fitsDecodeBudget(source, budget: budget) else { return nil }
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0, thumbnailOptions as CFDictionary) else { return nil }
        let result = NSImage(cgImage: image, size: .zero)
        if let entry {
            // NSCache so the decoded copies are the first thing released under
            // pressure. They are a redraw convenience beside a resident model,
            // never something worth holding against it.
            cache.storage.setObject(result, forKey: entry,
                                    cost: image.width * image.height * 4)
        }
        return result
    }

    /// Decoded copies only. Refusals are not cached: a file that failed the
    /// budget is cheap to re-refuse, and caching nil would need a second map.
    ///
    /// `NSCache` is already thread-safe; the wrapper exists only to say so to
    /// the compiler, since decoding runs off the main thread.
    private final class ThumbnailCache: @unchecked Sendable {
        let storage = NSCache<NSString, NSImage>()

        init() {
            storage.countLimit = 32
            storage.totalCostLimit = 64 * 1_024 * 1_024
        }
    }

    private static let cache = ThumbnailCache()

    /// Drops every decoded copy, so one case's images cannot satisfy the next
    /// one's lookup.
    ///
    /// This is a test hook, and deliberately not called when the app releases
    /// the files these were decoded from: the release path lives in `AppModel`,
    /// in `TurboFieldfareAppCore`, which this module depends on rather than the
    /// other way round, so it cannot reach here. What bounds the cache in the
    /// app is `ThumbnailCache` itself — 32 objects, 64 MB, and NSCache evicts
    /// under memory pressure, which is the normal state beside a resident
    /// model. Decoded copies of deleted files can therefore outlive them, up to
    /// that ceiling. Say so rather than claim a call that does not exist.
    public static func clearCache() {
        cache.storage.removeAllObjects()
    }

    /// `kCGImageSourceCreateThumbnailFromImageAlways` costs whatever the full
    /// decode costs for any format ImageIO cannot decode subsampled, so the
    /// source is measured from its properties — which decode nothing — before a
    /// thumbnail is asked for. A 50 MP PNG would otherwise spend about 200 MB
    /// inside the process that also holds the ~1.4 GB model, to draw a tile a
    /// few hundred pixels across.
    private static func fitsDecodeBudget(
        _ source: CGImageSource,
        budget: Budget
    ) -> Bool {
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) else {
            return false
        }
        let dictionary = properties as NSDictionary
        guard let width = (dictionary[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (dictionary[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0,
              width <= budget.maximumSourceDimension,
              height <= budget.maximumSourceDimension else {
            return false
        }
        let (pixels, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        guard !pixelOverflow, pixels <= budget.maximumSourcePixels else { return false }
        // Four channels is the decode target; source depth drives how much
        // ImageIO materialises on the way there. Every step is overflow-checked
        // because the depth comes from the file, not from us.
        let depth = (dictionary[kCGImagePropertyDepth] as? NSNumber)?.intValue ?? 8
        let (bytesPerPixel, depthOverflow) = max(1, depth / 8)
            .multipliedReportingOverflow(by: 4)
        guard !depthOverflow else { return false }
        let (decodedBytes, byteOverflow) = pixels.multipliedReportingOverflow(
            by: bytesPerPixel)
        return !byteOverflow && decodedBytes <= budget.maximumDecodedBytes
    }

}
