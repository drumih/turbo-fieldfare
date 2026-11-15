import Foundation

/// The tensor names a Qwen4-lineage Flash model needs beyond `TensorSchema`.
///
/// The hyper-connection, QSA-indexer, Gated-DeltaNet and PLE roles are not part
/// of the shared schema: only these families have them, and putting twenty
/// optional fields on `TensorSchema` for one lineage would invite every other
/// family to compile against names it never carries.
///
/// The set is a protocol rather than one enum shared by both families because
/// Qwen 3.8 Flash and Qwen 4.0 Flash are deliberately separate contracts. The
/// names agree today -- the 4.0 path was copied from the 3.8 one -- but a
/// checkpoint may rename or move a tensor without touching the other, and a
/// single shared namespace would make that a silent cross-family change. Each
/// family states its own names; `Model.flashTensors` picks by family.
protocol FlashTensorSet {
    static var pleConstantsFile: String { get }
    static var ngramTableFile: String { get }

    // Hyper-connections (gated residual)
    static func attnMixDown(_ layer: Int) -> String
    static func attnMixUp(_ layer: Int) -> String
    static func attnInject(_ layer: Int) -> String
    static func mlpMixDown(_ layer: Int) -> String
    static func mlpMixUp(_ layer: Int) -> String
    static func mlpInject(_ layer: Int) -> String
    static var mixerDown: String { get }
    static var mixerUp: String { get }

    // QSA indexer (full-attention layers only)
    static func indexerQProj(_ layer: Int) -> String
    static func indexerKProj(_ layer: Int) -> String
    static func indexerQNorm(_ layer: Int) -> String
    static func indexerKNorm(_ layer: Int) -> String

    // Per-head attention norms
    static func qNorm(_ layer: Int) -> String
    static func kNorm(_ layer: Int) -> String

    // Gated DeltaNet (linear-attention layers only)
    static func gdnQKV(_ layer: Int) -> String
    static func gdnZ(_ layer: Int) -> String
    static func gdnA(_ layer: Int) -> String
    static func gdnB(_ layer: Int) -> String
    static func gdnConv(_ layer: Int) -> String
    static func gdnALog(_ layer: Int) -> String
    static func gdnDtBias(_ layer: Int) -> String
    static func gdnNorm(_ layer: Int) -> String
    static func gdnOut(_ layer: Int) -> String

    // PLE n-gram block (one layer only)
    static func pleConv(_ layer: Int) -> String
    static func pleKeyProj(_ layer: Int) -> String
    static func pleValueProj(_ layer: Int) -> String
    static func pleNormKey(_ layer: Int) -> String
    static func pleNormQuery(_ layer: Int) -> String
    static func pleNormConv(_ layer: Int) -> String
}
