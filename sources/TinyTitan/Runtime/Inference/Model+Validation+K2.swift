import Foundation
import TinyTitanFormat
import TinyTitanCore

// K2-Horizon's resident-tensor schema.
//
// The family's install is not a Qwen install with different names: it has a
// separate attention gate projection, a bf16 MoE router with a selection bias,
// 64 resident MoVA value experts per sparse layer, a shared expert beside the
// routed experts, and three dense layers whose MLP is `mlp.gate/up/down` rather
// than a shared-expert stage. The generic per-layer checks would ask for
// `q_norm`, a packed [query ; gate] q_proj and one v_proj per layer, none of
// which this checkpoint has, so the family gets its own schema read.
//
// The packed-expert layout is still validated by the shared
// `validateRoutedExpertLayout`, extended to know that a dense layer carries no
// experts rather than treating that as a missing file.
extension Model {
    /// The K2 install's resident tensors and packed-expert layout.
    static func validateK2Schema(
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
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim

        try checks.requireAffine(
            K2HorizonTensors.embedding,
            rows: config.vocabSize, columns: hidden,
            slot: slot(K2HorizonTensors.embedding, quant.embedding))
        if !config.tieWordEmbeddings {
            try checks.requireAffine(
                K2HorizonTensors.lmHead,
                rows: config.vocabSize, columns: hidden,
                slot: slot(K2HorizonTensors.lmHead, quant.embedding))
        }
        try checks.requireBF16(K2HorizonTensors.finalNorm, count: hidden)

        let denseLayers = Set(config.denseLayerIndices)
        for layer in 0..<config.numLayers {
            try checks.requireBF16(K2HorizonTensors.inputNorm(layer), count: hidden)
            try checks.requireBF16(K2HorizonTensors.postAttnNorm(layer), count: hidden)
            // Every layer has the same attention shape, gate projection
            // included; only the value path differs (v_proj vs MoVA).
            try checks.requireAffine(
                K2HorizonTensors.qProj(layer),
                rows: queryDim, columns: hidden,
                slot: slot(K2HorizonTensors.qProj(layer), quant.attention))
            try checks.requireAffine(
                K2HorizonTensors.kProj(layer),
                rows: kvDim, columns: hidden,
                slot: slot(K2HorizonTensors.kProj(layer), quant.attention))
            try checks.requireAffine(
                K2HorizonTensors.oProj(layer),
                rows: hidden, columns: queryDim,
                slot: slot(K2HorizonTensors.oProj(layer), quant.attention))
            try checks.requireAffine(
                K2HorizonTensors.attnGate(layer),
                rows: queryDim, columns: hidden,
                slot: slot(K2HorizonTensors.attnGate(layer), quant.attention))

            if denseLayers.contains(layer) {
                try validateK2DenseLayer(
                    layer: layer, checks: checks, config: config,
                    quant: quant, slot: slot, kvDim: kvDim)
            } else {
                try validateK2SparseLayer(
                    layer: layer, checks: checks, config: config,
                    quant: quant, slot: slot, kvDim: kvDim)
            }
        }

        try validateRoutedExpertLayout(
            checks: checks, layout: layout, config: config, quant: quant)
    }

    /// The three dense layers 0-2: `v_proj` and a plain SwiGLU MLP.
    private static func validateK2DenseLayer(
        layer: Int,
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        slot: (String, ManifestQuantSlot) -> ManifestQuantSlot,
        kvDim: Int
    ) throws {
        func ffn(_ name: String) -> ManifestQuantSlot {
            slot(name, quant.sharedExpert)
        }
        try checks.requireAffine(
            K2HorizonTensors.vProj(layer),
            rows: kvDim, columns: config.hiddenSize,
            slot: slot(K2HorizonTensors.vProj(layer), quant.attention))
        let gate = K2HorizonTensors.denseGate(layer)
        let up = K2HorizonTensors.denseUp(layer)
        let down = K2HorizonTensors.denseDown(layer)
        try checks.requireAffine(
            gate, rows: config.intermediateSize, columns: config.hiddenSize,
            slot: ffn(gate))
        try checks.requireAffine(
            up, rows: config.intermediateSize, columns: config.hiddenSize,
            slot: ffn(up))
        try checks.requireAffine(
            down, rows: config.hiddenSize, columns: config.intermediateSize,
            slot: ffn(down))
    }

    /// The 45 sparse layers: MoVA's value router and 64 resident experts, the
    /// bf16 MoE router with its bias, and the single shared expert.
    private static func validateK2SparseLayer(
        layer: Int,
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        slot: (String, ManifestQuantSlot) -> ManifestQuantSlot,
        kvDim: Int
    ) throws {
        let moves = config.mova
        guard moves.enabled else {
            throw ModelError.indexCorrupt(
                detail: "layer \(layer) is sparse but the architecture declares no MoVA")
        }
        let router = K2HorizonTensors.vRouter(layer)
        try checks.requireAffineOrBF16(
            router, rows: moves.numExperts, columns: config.hiddenSize,
            slot: slot(router, quant.router))
        try checks.requireBF16(
            K2HorizonTensors.vRouterBias(layer), count: moves.numExperts)
        for expert in 0..<moves.numExperts {
            let name = K2HorizonTensors.vExpert(layer, expert)
            try checks.requireAffine(
                name, rows: kvDim, columns: config.hiddenSize,
                slot: slot(name, quant.attention))
        }

        let moeRouter = K2HorizonTensors.moeRouter(layer)
        try checks.requireAffineOrBF16(
            moeRouter, rows: config.numExperts, columns: config.hiddenSize,
            slot: slot(moeRouter, quant.router))
        try checks.requireBF16(
            K2HorizonTensors.moeRouterBias(layer), count: config.numExperts)

        func ffn(_ name: String) -> ManifestQuantSlot {
            slot(name, quant.sharedExpert)
        }
        let sharedGate = K2HorizonTensors.sharedGate(layer)
        let sharedUp = K2HorizonTensors.sharedUp(layer)
        let sharedDown = K2HorizonTensors.sharedDown(layer)
        try checks.requireAffine(
            sharedGate,
            rows: config.moeIntermediateSize, columns: config.hiddenSize,
            slot: ffn(sharedGate))
        try checks.requireAffine(
            sharedUp,
            rows: config.moeIntermediateSize, columns: config.hiddenSize,
            slot: ffn(sharedUp))
        try checks.requireAffine(
            sharedDown,
            rows: config.hiddenSize, columns: config.moeIntermediateSize,
            slot: ffn(sharedDown))
    }
}
