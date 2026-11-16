import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

/// The packed-expert layout's per-layer rule, which is what makes K2's mixed
/// install legal: layers 0-2 are dense and carry no experts, the other 45 carry
/// 100 each. Before this rule the shared validator treated every empty layer as
/// a missing blob, and it treated a dense layer carrying experts as fine.
@Suite("K2 routed layout")
struct K2RoutedLayoutTests {
    private static let hidden = 64
    private static let expertWidth = 128

    private static func quant() -> ManifestQuant {
        let slot = ManifestQuantSlot(
            weightBits: 4, scheme: "affine", scaleType: "bf16",
            biasType: "bf16", groupSize: 64)
        return ManifestQuant(
            embedding: slot, attention: slot, router: slot,
            sharedExpert: slot, routedExpert: slot)
    }

    private static func config() -> ArchConfig {
        ArchConfig(
            hiddenSize: hidden, intermediateSize: expertWidth,
            moeIntermediateSize: expertWidth,
            numHeads: 4, numKVHeads: 1, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 16, vocabSize: 128,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 1, numLayers: 4,
            numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [1, 1, 1, 1], hiddenActivation: "silu",
            family: "k2_horizon", attnOutputGate: true, attentionScale: 0.25,
            embeddingScaledBySqrtHidden: false, routerScaled: false,
            ffnSandwichNorms: false, sharedExpertGated: true,
            ropeNeoxSubdim: true, routerNormTopK: true,
            normGroups: 2, attentionGate: .softplus,
            routerHasBias: true, routerScaling: 2.5,
            denseLayerIndices: [0, 1, 2],
            mova: MoVAConfig(
                numExperts: 4, topKExperts: 2,
                routerScaling: 2.5, routerHasBias: true, activation: "silu"))
    }

    private static func checks() -> RuntimeSchemaChecks {
        RuntimeSchemaChecks(
            residentIndex: ResidentIndex(
                header: ResidentIndexHeader(
                    indexSize: 0, residentSize: 0, entryCount: 0),
                entries: [:]),
            quant: quant())
    }

    private static func expert(id: Int, size: UInt64 = 13_824) -> ExpertEntry {
        let slot = quant().routedExpert
        var offset: UInt64 = 0
        var subTensors: [String: SubTensorEntry] = [:]
        func add(_ role: String, _ rows: Int, _ columns: Int) {
            let weightBytes = UInt64(rows * columns * slot.weightBits / 8)
            let groups = columns / slot.groupSize
            let auxBytes = UInt64(rows * groups * 2)
            subTensors[role] = SubTensorEntry(
                offset: offset, size: weightBytes, dtype: "U32",
                shape: [UInt32(rows), UInt32(columns)], bits: slot.weightBits)
            offset += weightBytes
            for suffix in ["scales", "biases"] {
                subTensors["\(role)_\(suffix)"] = SubTensorEntry(
                    offset: offset, size: auxBytes, dtype: "BF16",
                    shape: [UInt32(rows), UInt32(groups)], bits: nil)
                offset += auxBytes
            }
        }
        add("gate", expertWidth, hidden)
        add("up", expertWidth, hidden)
        add("down", hidden, expertWidth)
        return ExpertEntry(
            expert: id, offset: 0, size: size, subTensors: subTensors)
    }

    @Test("Dense layers may be empty; a sparse layer's experts are validated")
    func allowsDenseLayers() throws {
        let layout = PackedExpertsLayout(
            expertStride: 13_824, numLayers: 4, expertsPerLayer: 2,
            compression: nil,
            layers: [
                LayerLayout(layer: 0, file: "layer_00.bin", experts: [], expertsPerLayer: 0),
                LayerLayout(layer: 1, file: "layer_01.bin", experts: [], expertsPerLayer: 0),
                LayerLayout(layer: 2, file: "layer_02.bin", experts: [], expertsPerLayer: 0),
                LayerLayout(
                    layer: 3, file: "layer_03.bin",
                    experts: [Self.expert(id: 0), Self.expert(id: 1)],
                    expertsPerLayer: nil),
            ])
        try Model.validateRoutedExpertLayout(
            checks: Self.checks(), layout: layout,
            config: Self.config(), quant: Self.quant())
    }

    @Test("A sparse layer with no experts is still refused")
    func refusesEmptySparseLayer() {
        let layout = PackedExpertsLayout(
            expertStride: 13_824, numLayers: 4, expertsPerLayer: 2,
            compression: nil,
            layers: [
                LayerLayout(layer: 0, file: "layer_00.bin", experts: [], expertsPerLayer: 0),
                LayerLayout(layer: 3, file: "layer_03.bin", experts: [], expertsPerLayer: 0),
            ])
        #expect(throws: ModelError.self) {
            try Model.validateRoutedExpertLayout(
                checks: Self.checks(), layout: layout,
                config: Self.config(), quant: Self.quant())
        }
    }

    @Test("A dense layer carrying experts is refused")
    func refusesExpertsOnDenseLayer() {
        let layout = PackedExpertsLayout(
            expertStride: 13_824, numLayers: 4, expertsPerLayer: 2,
            compression: nil,
            layers: [
                LayerLayout(
                    layer: 0, file: "layer_00.bin",
                    experts: [Self.expert(id: 0), Self.expert(id: 1)],
                    expertsPerLayer: nil),
                LayerLayout(layer: 3, file: "layer_03.bin", experts: [], expertsPerLayer: 0),
            ])
        #expect(throws: ModelError.self) {
            try Model.validateRoutedExpertLayout(
                checks: Self.checks(), layout: layout,
                config: Self.config(), quant: Self.quant())
        }
    }
}
