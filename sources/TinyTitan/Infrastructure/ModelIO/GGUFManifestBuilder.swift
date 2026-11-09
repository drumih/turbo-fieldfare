import Foundation
import TinyTitanFormat

package enum GGUFManifestError: Error, Equatable, CustomStringConvertible {
    case missingMetadata(String)
    case mismatch(field: String, file: String, preset: String)
    case unsupportedBits(Int)
    case unsupportedGroupSize(Int)

    package var description: String {
        switch self {
        case .missingMetadata(let key):
            return "GGUF metadata has no \(key), so the manifest cannot be built"
        case .mismatch(let field, let file, let preset):
            return
                "GGUF says \(field) = \(file); the runtime's preset says \(preset)"
        case .unsupportedBits(let bits):
            return "the manifest quant slots are written at 4 bits; \(bits) is not supported"
        case .unsupportedGroupSize(let size):
            return "the affine writer packs groups of 64; the preset says \(size)"
        }
    }
}

/// Builds the `manifest.json` a converted install needs, from the GGUF's own
/// geometry and the runtime's preset for the family.
///
/// The split is deliberate and is the whole point of this type:
///
/// - **The file is authoritative for geometry.** Every field that has a GGUF
///   key is read from the file and required to agree with the preset. A
///   checkpoint whose layer count, expert count, head geometry or extension
///   shapes differ from what the runtime executes is refused here, not loaded
///   and run wrongly.
/// - **The preset is authoritative for conventions.** `attnOutputGate`,
///   `attentionScale`, the partial-RoPE pairing, the hyper-connection/ indexer /
///   PLE shapes and the layer mask are family knowledge that a GGUF does not
///   carry in a form the runtime reads, so they come from `ArchConfig`.
///
/// `files` is a parameter, not something this type invents: sizes and SHA-256
/// values describe bytes that do not exist until the conversion writes them,
/// and a fabricated hash would be worse than a missing manifest.
package enum GGUFManifestBuilder {
    package static func manifest(
        from model: GGUFModel,
        arch preset: ArchConfig,
        modelID: String,
        files: [String: SSDAIManifestFileV1],
        expertStride: UInt64,
        bits: Int = 4,
        groupSize: Int = 64,
        flags: [String: Bool] = [:]
    ) throws -> SSDAIManifestV1 {
        guard bits == 4 else { throw GGUFManifestError.unsupportedBits(bits) }
        guard groupSize == 64 else { throw GGUFManifestError.unsupportedGroupSize(groupSize) }
        try crossCheck(model, preset: preset)

        // The split mirrors the engine's 4-bit installs: experts, embedding and
        // shared experts at 4 bits, attention and router kept at 8. These slots
        // are a claim about bytes, so the resident writer must emit exactly
        // this; the manifest is shippable only once the two agree.
        let slot = SSDAIManifestQuantSlotV1(
            weightBits: bits, scheme: "affine", scaleType: "BF16",
            biasType: "BF16", groupSize: groupSize)
        let quant = SSDAIManifestQuantV1(
            embedding: slot,
            attention: SSDAIManifestQuantSlotV1(
                weightBits: 8, scheme: "affine", scaleType: "BF16",
                biasType: "BF16", groupSize: groupSize),
            router: SSDAIManifestQuantSlotV1(
                weightBits: 8, scheme: "affine", scaleType: "BF16",
                biasType: "BF16", groupSize: groupSize),
            sharedExpert: slot,
            routedExpert: slot,
            overrides: nil,
            promoted: nil)

        let arch = SSDAIManifestArchV1(
            hiddenSize: preset.hiddenSize,
            ffnIntermediate: preset.intermediateSize,
            moeIntermediateSize: preset.moeIntermediateSize,
            numHeads: preset.numHeads,
            numKVHeads: preset.numKVHeads,
            numFullKVHeads: preset.numFullKVHeads,
            headDim: preset.headDim,
            fullHeadDim: preset.fullHeadDim,
            vocabSize: preset.vocabSize,
            slidingWindow: preset.slidingWindow,
            finalLogitSoftcap: preset.finalLogitSoftcap,
            ropeTheta: preset.ropeTheta,
            fullRopeTheta: preset.fullRopeTheta,
            partialRotaryFactor: preset.partialRotaryFactor,
            numLayers: preset.numLayers,
            numExperts: preset.numExperts,
            topKExperts: preset.topKExperts,
            tieWordEmbeddings: preset.tieWordEmbeddings,
            attentionKEqV: preset.attentionKEqV,
            hiddenActivation: preset.hiddenActivation,
            fullAttentionLayerMask: preset.fullAttentionLayerMask.map(Int.init),
            family: preset.family.rawValue,
            hcCount: preset.hyperConnections.enabled ? preset.hyperConnections.count : nil,
            hcLowRank: preset.hyperConnections.enabled ? preset.hyperConnections.lowRank : nil,
            indexerNumHeads: preset.sparseIndexer.enabled ? preset.sparseIndexer.numHeads : nil,
            indexerNumKVHeads: preset.sparseIndexer.enabled
                ? preset.sparseIndexer.numKVHeads : nil,
            indexerHeadDim: preset.sparseIndexer.enabled ? preset.sparseIndexer.headDim : nil,
            indexerBudget: preset.sparseIndexer.enabled ? preset.sparseIndexer.budget : nil,
            indexerCompressRatio: preset.sparseIndexer.enabled
                ? preset.sparseIndexer.compressRatio : nil,
            pleLayerIndices: preset.ple.enabled ? preset.ple.layerIndices : nil,
            pleEmbedDim: preset.ple.enabled ? preset.ple.embedDim : nil,
            pleConvKernelSize: preset.ple.enabled ? preset.ple.convKernelSize : nil,
            pleNgramSize: preset.ple.enabled ? preset.ple.ngramSize : nil,
            pleVocabSizeBase: preset.ple.enabled ? preset.ple.vocabSizeBase : nil,
            pleHeadsPerNgram: preset.ple.enabled ? preset.ple.headsPerNgram : nil,
            pleVocabDivisor: preset.ple.enabled ? preset.ple.vocabDivisor : nil,
            routerNormTopK: preset.routerNormTopK,
            quantGroupSize: preset.quantGroupSize,
            linearNumKHeads: preset.linearAttention.numKHeads,
            linearNumVHeads: preset.linearAttention.numVHeads,
            linearKeyHeadDim: preset.linearAttention.keyHeadDim,
            linearValueHeadDim: preset.linearAttention.valueHeadDim,
            linearConvKernelSize: preset.linearAttention.convKernelSize,
            attnOutputGate: preset.attnOutputGate,
            attentionScale: preset.attentionScale,
            embeddingScaledBySqrtHidden: preset.embeddingScaledBySqrtHidden,
            routerScaled: preset.routerScaled,
            ffnSandwichNorms: preset.ffnSandwichNorms,
            sharedExpertGated: preset.sharedExpertGated,
            ropeNeoxSubdim: preset.ropeNeoxSubdim,
            normGroups: preset.normGroups,
            attentionGate: preset.attentionGate.rawValue,
            routerHasBias: preset.routerHasBias,
            routerScaling: preset.routerScaling,
            movaNumExperts: preset.mova.numExperts > 0 ? preset.mova.numExperts : nil,
            movaTopKExperts: preset.mova.numExperts > 0 ? preset.mova.topKExperts : nil,
            movaRouterScaling: preset.mova.numExperts > 0 ? preset.mova.routerScaling : nil,
            movaRouterHasBias: preset.mova.numExperts > 0 ? preset.mova.routerHasBias : nil,
            movaActivation: preset.mova.numExperts > 0 ? preset.mova.activation : nil,
            denseLayerIndices: preset.denseLayerIndices.isEmpty
                ? nil : preset.denseLayerIndices)

        return SSDAIManifestV1(
            flags: flags,
            modelID: modelID,
            sourceSnapshotHash: nil,
            arch: arch,
            quant: quant,
            files: files,
            expertsPerLayer: preset.numExperts,
            numLayers: preset.numLayers,
            expertStride: expertStride)
    }

    /// Every geometry field that has a GGUF key is compared with the preset.
    /// Absent keys are an error rather than a skip: this builder exists for a
    /// checkpoint written by a tool that emits them.
    static func crossCheck(_ model: GGUFModel, preset: ArchConfig) throws {
        func integer(_ key: String, _ expected: Int, _ field: String) throws {
            guard let value = model.architectureValue(key)?.integer else {
                throw GGUFManifestError.missingMetadata(key)
            }
            guard value == Int64(expected) else {
                throw GGUFManifestError.mismatch(
                    field: field, file: String(value), preset: String(expected))
            }
        }
        try integer("block_count", preset.numLayers, "numLayers")
        try integer("embedding_length", preset.hiddenSize, "hiddenSize")
        try integer(
            "expert_shared_feed_forward_length", preset.intermediateSize, "ffnIntermediate")
        try integer(
            "expert_feed_forward_length", preset.moeIntermediateSize,
            "moeIntermediateSize")
        try integer("attention.head_count", preset.numHeads, "numHeads")
        try integer("attention.head_count_kv", preset.numKVHeads, "numKVHeads")
        try integer("attention.key_length", preset.headDim, "headDim")
        try integer("attention.value_length", preset.fullHeadDim, "fullHeadDim")
        try integer("expert_count", preset.numExperts, "numExperts")
        try integer("expert_used_count", preset.topKExperts, "topKExperts")
        try integer(
            "rope.dimension_count",
            Int((preset.partialRotaryFactor * Double(preset.headDim)).rounded()),
            "rope.dimension_count")

        guard let rope = model.architectureValue("rope.freq_base")?.double else {
            throw GGUFManifestError.missingMetadata("rope.freq_base")
        }
        guard abs(rope - preset.ropeTheta) < 1e-6 else {
            throw GGUFManifestError.mismatch(
                field: "ropeTheta", file: "\(rope)", preset: "\(preset.ropeTheta)")
        }
        if let tokens = model.metadata["tokenizer.ggml.tokens"]?.strings {
            guard tokens.count == preset.vocabSize else {
                throw GGUFManifestError.mismatch(
                    field: "vocabSize", file: "\(tokens.count)",
                    preset: "\(preset.vocabSize)")
            }
        }

        if preset.hyperConnections.enabled {
            try integer(
                "hyper_connection.count", preset.hyperConnections.count, "hcCount")
            try integer(
                "hyper_connection.low_rank", preset.hyperConnections.lowRank, "hcLowRank")
        }
        if preset.sparseIndexer.enabled {
            try integer(
                "attention.indexer.head_count", preset.sparseIndexer.numHeads,
                "indexerNumHeads")
            try integer(
                "attention.indexer.key_length", preset.sparseIndexer.headDim,
                "indexerHeadDim")
            try integer(
                "attention.indexer.top_k", preset.sparseIndexer.budget, "indexerBudget")
        }
        if preset.ple.enabled {
            // The GGUF key is the table's row width, which is the PLE head
            // width (embedDim / ngramHeads = 2560 / 16 = 160), not the
            // injection width the preset calls embedDim.
            try integer(
                "embedding_length_per_layer_input", preset.ple.headDim,
                "pleHeadDim")
            try integer("ple.conv_kernel", preset.ple.convKernelSize, "pleConvKernelSize")
            try integer("ple.ngram_size", preset.ple.ngramSize, "pleNgramSize")
            try integer("ple.heads_per_ngram", preset.ple.headsPerNgram, "pleHeadsPerNgram")
        }
    }
}
