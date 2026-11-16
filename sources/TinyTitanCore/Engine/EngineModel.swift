import Foundation

/// The model-facing half of the engine contract (TT-097 spec §4-D1): exactly the
/// members `ServerModelSession` reads off `Model`, promoted to `TinyTitanCore` so
/// both the 6.x runtime and every `EngineVariant`'s identical `Model` can satisfy
/// the *same* nominal protocol and one shared session can run either engine.
///
/// Model-free by construction: `ArchConfig` is a Core type and every other member
/// is a Foundation primitive, so this carries no model name and no family switch
/// (`core-purity` holds). Signatures are copied verbatim from the 6.x
/// `Runtime/Inference/Model.swift` / `Model+ExpertProfile.swift` declarations, so
/// adding conformance is a declaration only — no behavior changes.
public protocol EngineModel: Sendable {
    /// Resolved geometry + layer conventions, read from the install manifest.
    var config: ArchConfig { get }
    /// The catalog identity this model was loaded under.
    var modelID: String { get }
    /// The pinned source snapshot hash the receipt was bound to, when known.
    var sourceSnapshotHash: String? { get }
    /// The bit width routed experts are stored at (4 or 8); dense models have no
    /// routed experts and this is reported for the streaming surfaces that ask.
    var routedExpertWeightBits: Int { get }
    /// The number of residency slots the expert cache is sized for, when a cache
    /// exists (nil for a model that streams without one).
    func routedExpertCacheSlotCount() -> Int?
    /// Persist the expert-usage profile gathered during generation.
    func saveExpertUsageProfile()
}
