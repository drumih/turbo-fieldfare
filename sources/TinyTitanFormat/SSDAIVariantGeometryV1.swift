import Foundation

/// Geometry an engine variant needs that the shared `arch` fields do not
/// describe: named scalars and named integer lists, stored, hashed and
/// round-tripped by the format and interpreted only by the variant that wrote
/// the install (TT-121).
///
/// The 6.x `arch` block grew one optional field per family feature
/// (hyper-connections, the QSA indexer, PLE, MoVA) — the union shape the 7.0
/// design retires. A variant's own geometry lives here instead, keyed by names
/// the variant defines, so the format carries no model knowledge and a new
/// family adds no field to it. The variant validates every key it reads and
/// refuses an install that lacks one; the format checks only that the block is
/// well formed.
package struct SSDAIVariantGeometryV1: Codable, Equatable, Sendable {
    package let scalars: [String: Double]
    package let lists: [String: [Int]]

    package init(scalars: [String: Double] = [:], lists: [String: [Int]] = [:]) {
        self.scalars = scalars
        self.lists = lists
    }

    /// Non-empty keys, finite scalars, and no name used as both a scalar and a list.
    package func validate(field: String) throws {
        for (key, value) in scalars where key.isEmpty || !value.isFinite {
            throw TinyTitanFormatError.invalid(
                field: "\(field).scalars.\(key)", reason: "empty key or non-finite value")
        }
        for key in lists.keys where key.isEmpty || scalars[key] != nil {
            throw TinyTitanFormatError.invalid(
                field: "\(field).lists.\(key)", reason: "empty key or also a scalar")
        }
    }
}
