import Foundation

extension Collection where Element == UInt8 {
    // The rule is deliberately not followed for this one declaration; the doc
    // comment below gives the reason. A disable/enable pair (rather than
    // `disable:next`) is used because a disable comment between the doc comment
    // and the declaration would orphan the doc comment.
    // swiftlint:disable optional_data_string_conversion
    /// Lossy UTF-8 text for a byte collection.
    ///
    /// This is deliberate, and it is the reason SwiftLint's
    /// `optional_data_string_conversion` rule is suppressed here rather than
    /// followed. Two families of callers need the replacement behavior of
    /// `String(decoding:as:)`:
    ///
    ///   * the streaming detokenizer and the JSON byte parser, which split a
    ///     multi-byte scalar across chunk boundaries and must emit U+FFFD for a
    ///     truncated sequence instead of failing the stream (`GFDetokenizer.drain`
    ///     documents exactly that contract);
    ///   * the server's own encoders and SSE frame builders, whose bytes are
    ///     valid UTF-8 by construction, so the failable `String(bytes:encoding:)`
    ///     initializer the rule prefers would only add an unreachable `nil`
    ///     branch.
    ///
    /// The rule's premise — that a `Data` to `String` conversion should surface
    /// invalid UTF-8 — is the right default for data arriving from outside, and
    /// the places that decode such input (the repacker's manifest and tokenizer
    /// reads) do use the throwing `JSONDecoder`/`String(bytes:encoding:)` paths.
    /// It is the wrong default for a byte stream that is allowed to be
    /// incomplete, so the intent is named here once instead of being re-decided
    /// at 43 call sites.
    public var lossyUTF8String: String { String(decoding: self, as: UTF8.self) }
    // swiftlint:enable optional_data_string_conversion
}
