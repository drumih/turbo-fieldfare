import Foundation
import TinyTitanFormat

// The wire-to-model conversion for `manifest.json`: the `SSDAIManifest*V1`
// package types become the runtime's `Manifest` and its parts.
//
// Split out of `ManifestReader.swift` (2026-10-25) as pure code motion under
// the 500-line-per-file rule; the K2 fields are the keys that pushed the reader
// over it, and this block was already a cohesive seam at the end of the file.
// The initializers are `internal` rather than `fileprivate` because the caller
// (`ManifestReader.decode`) stays behind in the other file.

extension ManifestFileEntry {
    package init(wire: SSDAIManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

extension ManifestArch {
    package init(wire: SSDAIManifestArchV1) {
        self.init(
            hiddenSize: wire.hiddenSize,
            ffnIntermediate: wire.ffnIntermediate,
            moeIntermediateSize: wire.moeIntermediateSize,
            numHeads: wire.numHeads,
            numKVHeads: wire.numKVHeads,
            numFullKVHeads: wire.numFullKVHeads,
            headDim: wire.headDim,
            fullHeadDim: wire.fullHeadDim,
            vocabSize: wire.vocabSize,
            slidingWindow: wire.slidingWindow,
            finalLogitSoftcap: wire.finalLogitSoftcap,
            ropeTheta: wire.ropeTheta,
            fullRopeTheta: wire.fullRopeTheta,
            partialRotaryFactor: wire.partialRotaryFactor,
            numLayers: wire.numLayers,
            numExperts: wire.numExperts,
            topKExperts: wire.topKExperts,
            tieWordEmbeddings: wire.tieWordEmbeddings,
            attentionKEqV: wire.attentionKEqV,
            hiddenActivation: wire.hiddenActivation,
            fullAttentionLayerMask: wire.fullAttentionLayerMask,
            hcCount: wire.hcCount,
            hcLowRank: wire.hcLowRank,
            indexerNumHeads: wire.indexerNumHeads,
            indexerNumKVHeads: wire.indexerNumKVHeads,
            indexerHeadDim: wire.indexerHeadDim,
            indexerBudget: wire.indexerBudget,
            indexerCompressRatio: wire.indexerCompressRatio,
            pleLayerIndices: wire.pleLayerIndices,
            pleEmbedDim: wire.pleEmbedDim,
            pleConvKernelSize: wire.pleConvKernelSize,
            pleNgramSize: wire.pleNgramSize,
            pleVocabSizeBase: wire.pleVocabSizeBase,
            pleHeadsPerNgram: wire.pleHeadsPerNgram,
            pleVocabDivisor: wire.pleVocabDivisor,
            routerNormTopK: wire.routerNormTopK,
            quantGroupSize: wire.quantGroupSize,
            attnOutputGate: wire.attnOutputGate,
            attentionScale: wire.attentionScale,
            embeddingScaledBySqrtHidden: wire.embeddingScaledBySqrtHidden,
            routerScaled: wire.routerScaled,
            ffnSandwichNorms: wire.ffnSandwichNorms,
            sharedExpertGated: wire.sharedExpertGated,
            ropeNeoxSubdim: wire.ropeNeoxSubdim,
            normGroups: wire.normGroups,
            attentionGate: wire.attentionGate,
            routerHasBias: wire.routerHasBias,
            routerScaling: wire.routerScaling,
            movaNumExperts: wire.movaNumExperts,
            movaTopKExperts: wire.movaTopKExperts,
            movaRouterScaling: wire.movaRouterScaling,
            movaRouterHasBias: wire.movaRouterHasBias,
            movaActivation: wire.movaActivation,
            denseLayerIndices: wire.denseLayerIndices,
            linearNumKHeads: wire.linearNumKHeads,
            linearNumVHeads: wire.linearNumVHeads,
            linearKeyHeadDim: wire.linearKeyHeadDim,
            linearValueHeadDim: wire.linearValueHeadDim,
            linearConvKernelSize: wire.linearConvKernelSize)
    }
}

extension ManifestQuantSlot {
    package init(wire: SSDAIManifestQuantSlotV1) {
        self.init(
            weightBits: wire.weightBits, scheme: wire.scheme,
            scaleType: wire.scaleType, biasType: wire.biasType,
            groupSize: wire.groupSize)
    }
}

extension ManifestQuant {
    package init(wire: SSDAIManifestQuantV1) {
        self.init(
            embedding: ManifestQuantSlot(wire: wire.embedding),
            attention: ManifestQuantSlot(wire: wire.attention),
            router: ManifestQuantSlot(wire: wire.router),
            sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
            routedExpert: ManifestQuantSlot(wire: wire.routedExpert))
    }
}

extension Manifest {
    package init(wire: SSDAIManifestV1) {
        self.init(
            magic: wire.magic,
            versionMajor: wire.versionMajor,
            versionMinor: wire.versionMinor,
            flags: wire.flags,
            modelID: wire.modelID,
            sourceSnapshotHash: wire.sourceSnapshotHash,
            arch: ManifestArch(wire: wire.arch),
            quant: wire.quant.map(ManifestQuant.init(wire:)),
            quantOverrides: wire.quant?.overrides?.mapValues(\.weightBits) ?? [:],
            quantPromoted: Set(wire.quant?.promoted ?? []),
            files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
            expertsPerLayer: wire.expertsPerLayer,
            numLayers: wire.numLayers,
            expertStride: wire.expertStride)
    }
}
