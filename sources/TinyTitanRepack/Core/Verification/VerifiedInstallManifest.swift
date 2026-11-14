import Foundation
import TinyTitanFormat
import TinyTitanCore

// The verifier's local manifest mirror: only the fields this tool needs,
// decoded without the format layer's structural validation.
//
// Split out of `VerifiedInstallTool.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// The five structs widened from `private` to internal because the loader that
// returns them stays in VerifiedInstallTool.swift.
struct ManifestFileEntry: Decodable {
    let size: UInt64
    let sha256: String
}

struct Manifest: Decodable {
    let files: [String: ManifestFileEntry]
    let expertsPerLayer: Int
    let numLayers: Int
    let expertStride: UInt64
    let sourceSnapshotHash: String?
    /// Decoded through `SSDAIManifestQuantV1`, whose hand-written `Codable`
    /// keeps the open set of per-tensor width keys. A synthesised decoder
    /// would drop them, which is the bug this whole check exists to catch.
    let quant: SSDAIManifestQuantV1?
}

struct PackedExpertsLayout: Decodable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    /// `lzfse` when the layer files hold frames and `.idx` sidecars, absent for
    /// a raw install. The verifier decodes it as a plain string: an unknown
    /// value is refused inside the check rather than failing the decode with a
    /// message about JSON.
    let compression: String?
    let layers: [Layer]
}

struct Layer: Decodable {
    let layer: Int
    let file: String
    let experts: [Expert]
    /// Present when this layer's expert count differs from the install-wide
    /// one; nil means it uses the install-wide value.
    let expertsPerLayer: Int?
}

struct Expert: Decodable {
    let expert: Int?
    let offset: UInt64
    let size: UInt64
    /// The slices that make up one expert's blob, in the format layer's own type.
    ///
    /// Required rather than optional: `SSDAIExpertV1` declares it non-optional and
    /// the runtime decodes every `layout.json` through that type, so a layout
    /// without it could not load. Certifying one here would be the exact thing
    /// this tool exists to prevent -- and the per-tensor sizes are what make the
    /// declared routed-expert width checkable at all.
    let tensors: [String: SSDAISubTensorV1]
}
