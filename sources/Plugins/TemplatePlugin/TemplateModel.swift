import Foundation
import TinyTitanCore

/// The model side of the engine (`EngineModel`): identity and geometry. A real
/// plugin reads its install's manifest and maps its weights here.
///
/// unchecked-invariant: every stored property is immutable after init.
public final class TemplateModel: EngineModel, @unchecked Sendable {
    /// The template's vocabulary: token ids `0 ..< vocabulary`.
    public static let vocabulary = 64

    public static let architecture = ArchConfig(
        hiddenSize: 1, intermediateSize: 0, moeIntermediateSize: 0, numHeads: 1,
        numKVHeads: 1, numFullKVHeads: 1, headDim: 1, fullHeadDim: 1,
        vocabSize: vocabulary, slidingWindow: 0, finalLogitSoftcap: 0, ropeTheta: 1,
        fullRopeTheta: 1, partialRotaryFactor: 0, numLayers: 1, numExperts: 0,
        topKExperts: 0, tieWordEmbeddings: false, attentionKEqV: false,
        fullAttentionLayerMask: [1], hiddenActivation: "none", family: TemplatePlugin.family)

    public let config = TemplateModel.architecture
    public let modelID: String
    public let sourceSnapshotHash: String? = nil
    public let routedExpertWeightBits = 16

    public init(directory: URL) {
        modelID = Self.modelID(directory)
    }

    static func modelID(_ directory: URL) -> String {
        "tinytitan-template-" + directory.lastPathComponent
    }

    public func routedExpertCacheSlotCount() -> Int? { nil }
    public func saveExpertUsageProfile() {}
}
