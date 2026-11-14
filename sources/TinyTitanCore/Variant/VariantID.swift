import Foundation

/// The identity of one engine variant: `<model-id>-<bits>bit` (7.0 spec §3).
///
/// A variant is the unit of code, test, tuning and release. Its id is parsed
/// once, at load, and an unknown or malformed id is refused rather than
/// defaulted. The width is part of the identity: 4-bit and 8-bit of the same
/// model are two variants, not one variant with a flag.
public struct VariantID: Hashable, Sendable, Comparable, CustomStringConvertible {
    public let raw: String
    /// The quantization width this variant reads (2, 4 or 8). Validated at init.
    public let bits: Int

    /// Parses and validates a variant id. Refuses anything that is not
    /// lowercase `<model-id>-<2|4|8>bit`; 6-bit is refused by name (it is a
    /// withdrawn legacy format, spec §6).
    public init(_ raw: String) throws {
        guard !raw.isEmpty, raw == raw.lowercased() else {
            throw VariantIDError("variant id must be non-empty lowercase: \(raw)")
        }
        guard let suffix = raw.split(separator: "-").last, suffix.hasSuffix("bit") else {
            throw VariantIDError("variant id must end in -<bits>bit: \(raw)")
        }
        let width = suffix.dropLast(3)
        guard let bits = Int(width), [2, 4, 8].contains(bits) else {
            throw VariantIDError(
                "variant width must be 2, 4 or 8 (6-bit is withdrawn): \(raw)")
        }
        // The model-id part must be present (something before the width suffix).
        guard raw.count > suffix.count + 1 else {
            throw VariantIDError("variant id has no model-id part: \(raw)")
        }
        self.raw = raw
        self.bits = bits
    }

    public var description: String { raw }

    public static func < (lhs: VariantID, rhs: VariantID) -> Bool {
        lhs.raw < rhs.raw
    }
}

/// A refused variant id, carrying the reason so the load path can name it.
public struct VariantIDError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}
