import Foundation
import Testing
import UniformTypeIdentifiers
@testable import TurboFieldfare

/// The formats a caller may attach and the formats the pack can decode have to
/// be one list.
///
/// The picker and the pasteboard admitted anything conforming to `public.image`
/// — which includes camera RAW, EXR and WebP — while `ImageMetadataReader`
/// accepts seven identifiers. Everything in between was copied at up to 64 MB,
/// decoded in-process for a thumbnail beside a loaded model, and only then
/// refused by the run.
@Suite struct VisionImageContentTypesTests {
    @Test func thecontentTypeListMatchesTheIdentifierAllowlist() {
        let limits = VisionImageLimits()
        #expect(Set(limits.allowedContentTypes.map(\.identifier))
                    == limits.allowedTypeIdentifiers,
                "the UTType list and the identifier allowlist disagree")
    }

    /// The gate has to be narrower than `public.image`, which is the whole
    /// reason it exists.
    @Test func formatsThePackCannotDecodeAreOutsideTheList() {
        let limits = VisionImageLimits()
        for excluded in [UTType("com.adobe.raw-image"),
                         UTType("public.camera-raw-image"),
                         UTType("com.ilm.openexr-image"),
                         UTType.webP,
                         UTType.bmp,
                         UTType.pdf] {
            guard let excluded else { continue }
            #expect(!limits.allowedTypeIdentifiers.contains(excluded.identifier),
                    "\(excluded.identifier) is admitted but cannot be decoded")
            // Everything excluded here still conforms to public.image, which is
            // why the old gate let it through.
            #expect(!limits.allowedContentTypes.contains(excluded))
        }
    }

    /// The formats the pack does decode have to survive the round trip to
    /// `UTType`, or the picker would silently offer fewer than it supports.
    @Test func everySupportedIdentifierResolvesToAContentType() {
        let limits = VisionImageLimits()
        #expect(limits.allowedContentTypes.count == limits.allowedTypeIdentifiers.count,
                "an identifier did not resolve to a UTType and was dropped from the picker")
        for expected in [UTType.jpeg, .png, .heic, .tiff, .gif] {
            #expect(limits.allowedContentTypes.contains(expected),
                    "\(expected.identifier) is decodable but the picker would not offer it")
        }
    }

    /// Stable ordering, so the picker's type list does not shuffle between
    /// launches from `Set` iteration order.
    @Test func thecontentTypeListIsStablyOrdered() {
        let limits = VisionImageLimits()
        #expect(limits.allowedContentTypes == limits.allowedContentTypes)
        #expect(limits.allowedContentTypes.map(\.identifier)
                    == limits.allowedContentTypes.map(\.identifier).sorted())
    }
}
