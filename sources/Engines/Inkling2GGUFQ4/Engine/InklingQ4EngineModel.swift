import Foundation
import Metal
import TinyTitanCore

/// The Inkling 2 Q4 install as the shared session's model (TT-122): the
/// variant's decoder, with its experts streamed into the slots the session's
/// load plan resolved, behind the Core `EngineModel` contract.
///
/// unchecked-invariant: every stored property is immutable after init; the
/// decoder lives in `InklingQ4Runner`, which the session serialises.
package final class InklingQ4EngineModel: EngineModel, @unchecked Sendable {
    package let config: ArchConfig
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let routedExpertWeightBits = 4
    let directory: URL
    let slots: Int

    /// Reads the install's identity; the decoder, which needs the planned
    /// window, is built by `makeDecoder` when the session builds its runner.
    package init(directory: URL, slots: Int) throws {
        let manifest = try SSDAIManifestReader.read(directoryURL: directory)
        config = Self.architecture(manifest.arch)
        modelID = manifest.modelID
        sourceSnapshotHash = manifest.sourceSnapshotHash
        self.directory = directory
        self.slots = slots
    }

    /// The decoder over this install, with its experts streamed into
    /// `slots` per layer and its caches sized for `maxContext` positions.
    package func makeDecoder(
        context: MetalContext, maxContext: Int, maxChunk: Int
    ) throws -> InklingQ4Decoder {
        try InklingQ4Decoder(
            directory: directory, maxContext: maxContext,
            maxChunk: min(maxChunk, maxContext), experts: .slots(slots), context: context)
    }

    package func routedExpertCacheSlotCount() -> Int? { slots }

    /// Inkling keeps no expert-usage profile: its prefetch is the lookahead
    /// router, measured to beat a usage table (TT-121, the 250-record database).
    package func saveExpertUsageProfile() {}

    /// The install's architecture as the session reads it. Only the generic
    /// fields are meaningful to the session (vocabulary, experts, softcap); the
    /// graph reads its own geometry from the manifest's `variantGeometry`.
    package static func architecture(_ arch: ManifestArch) -> ArchConfig {
        ArchConfig(
            hiddenSize: arch.hiddenSize,
            intermediateSize: arch.ffnIntermediate,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map { UInt8(clamping: $0) },
            hiddenActivation: arch.hiddenActivation,
            family: "inkling2")
    }
}
