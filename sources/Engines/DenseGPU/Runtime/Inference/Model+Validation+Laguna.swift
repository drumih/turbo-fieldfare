import Foundation
import TinyTitanCore

/// Laguna-S-2.1's resident schema and its executable-geometry bounds.
///
/// It cannot use `validateExecutableGeometry`: that gate refuses any
/// sliding-window layer and any hidden size above the MoE kernels' 2,816
/// tile, both of which are true of Laguna and both of which the probe graph
/// handles (windowed decode attention; the MoE stage allocates its scratch
/// from the hidden size instead of a compile-time tile). The serving runner is
/// refused separately, so bypassing that gate cannot reach a Qwen-shaped graph.
extension Model {
    static func validateExecutablePerLayerHeadGeometry(_ config: ArchConfig) throws {
        guard config.perLayerHeadGeometry.enabled else {
            throw ModelError.unsupportedArchitecture(
                detail: "Laguna family without its per-layer head geometry")
        }
        guard config.hiddenSize <= maximumDenseThreadgroupTileWidth else {
            throw ModelError.unsupportedArchitecture(
                detail: "hiddenSize \(config.hiddenSize) exceeds the "
                    + "\(maximumDenseThreadgroupTileWidth)-element dense tile")
        }
        guard config.fullHeadDim <= Attention.maxHeadDim else {
            throw ModelError.unsupportedArchitecture(
                detail: "head dimension \(config.fullHeadDim) exceeds the "
                    + "\(Attention.maxHeadDim)-element attention tile")
        }
        let most = config.perLayerHeadGeometry.headsPerLayer.max() ?? 0
        guard most <= Attention.maxQHeads else {
            throw ModelError.unsupportedArchitecture(
                detail: "\(most) query heads exceeds the \(Attention.maxQHeads)-head "
                    + "split-KV scratch")
        }
        guard config.perLayerHeadGeometry.headsPerLayer.count == config.numLayers else {
            throw ModelError.unsupportedArchitecture(
                detail:
                    "Laguna declares \(config.perLayerHeadGeometry.headsPerLayer.count) per-layer "
                    + "head counts for \(config.numLayers) layers")
        }
        guard config.slidingWindow > 0,
            config.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 })
        else {
            throw ModelError.unsupportedArchitecture(
                detail: "Laguna needs a positive sliding window and a 0/1 layer mask")
        }
        for (layer, heads) in config.perLayerHeadGeometry.headsPerLayer.enumerated() {
            guard heads > 0, heads % config.numKVHeads == 0 else {
                throw ModelError.unsupportedArchitecture(
                    detail: "layer \(layer) has \(heads) query heads against "
                        + "\(config.numKVHeads) KV heads")
            }
        }
    }

    static func validateLagunaSchema(
        checks: RuntimeSchemaChecks,
        layout: PackedExpertsLayout,
        config: ArchConfig,
        quant: ManifestQuant,
        overrides: [String: Int]
    ) throws {
        func slot(_ name: String, _ fallback: ManifestQuantSlot) -> ManifestQuantSlot {
            quant.slot(forTensorNamed: name, overrides: overrides, fallback: fallback)
        }
        let hidden = config.hiddenSize
        let headDim = config.fullHeadDim
        let kvDim = config.numFullKVHeads * headDim

        try checks.requireAffine(
            LagunaTensors.embedding, rows: config.vocabSize, columns: hidden,
            slot: slot(LagunaTensors.embedding, quant.embedding))
        if !config.tieWordEmbeddings {
            try checks.requireAffine(
                LagunaTensors.lmHead, rows: config.vocabSize, columns: hidden,
                slot: slot(LagunaTensors.lmHead, quant.embedding))
        }
        try checks.requireBF16(LagunaTensors.finalNorm, count: hidden)

        let dense = Set(config.denseLayerIndices)
        for layer in 0..<config.numLayers {
            let heads = config.perLayerHeadGeometry.headsPerLayer[layer]
            let queryDim = heads * headDim
            try checks.requireBF16(LagunaTensors.inputNorm(layer), count: hidden)
            try checks.requireBF16(LagunaTensors.postAttnNorm(layer), count: hidden)
            try checks.requireBF16(LagunaTensors.qNorm(layer), count: headDim)
            try checks.requireBF16(LagunaTensors.kNorm(layer), count: headDim)
            try checks.requireAffine(
                LagunaTensors.qProj(layer), rows: queryDim, columns: hidden,
                slot: slot(LagunaTensors.qProj(layer), quant.attention))
            try checks.requireAffine(
                LagunaTensors.kProj(layer), rows: kvDim, columns: hidden,
                slot: slot(LagunaTensors.kProj(layer), quant.attention))
            try checks.requireAffine(
                LagunaTensors.vProj(layer), rows: kvDim, columns: hidden,
                slot: slot(LagunaTensors.vProj(layer), quant.attention))
            try checks.requireAffine(
                LagunaTensors.oProj(layer), rows: hidden, columns: queryDim,
                slot: slot(LagunaTensors.oProj(layer), quant.attention))
            // The gate is one value per head, not per channel.
            try checks.requireAffine(
                LagunaTensors.attnGate(layer), rows: heads, columns: hidden,
                slot: slot(LagunaTensors.attnGate(layer), quant.attention))

            if dense.contains(layer) {
                try validateLagunaDenseMLP(
                    layer: layer, checks: checks, config: config,
                    quant: quant, slot: slot)
            } else {
                try validateLagunaSparseLayer(
                    layer: layer, checks: checks, config: config,
                    quant: quant, slot: slot)
            }
        }

        try validateRoutedExpertLayout(
            checks: checks, layout: layout, config: config, quant: quant)
    }

    /// Layer 0's plain SwiGLU MLP, `intermediateSize` (12,288) wide.
    private static func validateLagunaDenseMLP(
        layer: Int,
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        slot: (String, ManifestQuantSlot) -> ManifestQuantSlot
    ) throws {
        let names = [
            LagunaTensors.denseGate(layer), LagunaTensors.denseUp(layer),
            LagunaTensors.denseDown(layer),
        ]
        try checks.requireAffine(
            names[0], rows: config.intermediateSize, columns: config.hiddenSize,
            slot: slot(names[0], quant.sharedExpert))
        try checks.requireAffine(
            names[1], rows: config.intermediateSize, columns: config.hiddenSize,
            slot: slot(names[1], quant.sharedExpert))
        try checks.requireAffine(
            names[2], rows: config.hiddenSize, columns: config.intermediateSize,
            slot: slot(names[2], quant.sharedExpert))
    }

    /// A sparse layer: the bf16 256-expert router and its selection bias, and
    /// the always-on shared expert at `moeIntermediateSize` (1,024).
    private static func validateLagunaSparseLayer(
        layer: Int,
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        slot: (String, ManifestQuantSlot) -> ManifestQuantSlot
    ) throws {
        let router = LagunaTensors.moeRouter(layer)
        try checks.requireBF16(router, count: config.numExperts * config.hiddenSize)
        try checks.requireBF16(
            LagunaTensors.moeRouterBias(layer), count: config.numExperts)
        let names = [
            LagunaTensors.sharedGate(layer), LagunaTensors.sharedUp(layer),
            LagunaTensors.sharedDown(layer),
        ]
        try checks.requireAffine(
            names[0], rows: config.moeIntermediateSize, columns: config.hiddenSize,
            slot: slot(names[0], quant.sharedExpert))
        try checks.requireAffine(
            names[1], rows: config.moeIntermediateSize, columns: config.hiddenSize,
            slot: slot(names[1], quant.sharedExpert))
        try checks.requireAffine(
            names[2], rows: config.hiddenSize, columns: config.moeIntermediateSize,
            slot: slot(names[2], quant.sharedExpert))
    }
}
