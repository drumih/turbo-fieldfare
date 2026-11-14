import Foundation

/// A variant's install/convert recipe (7.0 spec §3, §9): the tensor renames, the
/// bf16 keeps, and the head/embedding width overrides. This is the Swift home of
/// what 6.x scattered across per-model `KEEP_BF16`/`HEAD_BITS` constants in the
/// Python converters. It is used only by install/convert, never by the runtime.
public struct ModelRecipe: Sendable {
    public let variantID: VariantID
    /// Checkpoint tensor name -> the variant's tensor stem.
    public let renames: [String: String]
    /// Tensor stems kept at the checkpoint's own bf16 rather than quantized.
    public let keepBF16Stems: Set<String>
    /// Optional width overrides for the head and embedding (nil = default spec).
    public let headBits: Int?
    public let embeddingBits: Int?

    public init(
        variantID: VariantID,
        renames: [String: String],
        keepBF16Stems: Set<String>,
        headBits: Int?,
        embeddingBits: Int?
    ) {
        self.variantID = variantID
        self.renames = renames
        self.keepBF16Stems = keepBF16Stems
        self.headBits = headBits
        self.embeddingBits = embeddingBits
    }
}
