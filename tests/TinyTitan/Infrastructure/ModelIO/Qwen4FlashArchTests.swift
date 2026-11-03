import Testing

@testable import TinyTitan

/// Pins the Qwen 4.0 Flash runtime preset.
///
/// The preset starts as the 3.8 preview's verified geometry because the 4.0
/// checkpoint is not published; the runtime also reads a differently-shaped
/// 4.0 install from its manifest. What this suite guarantees is that the two
/// families are distinct values with distinct presets, so a later edit to one
/// cannot move the other by accident.
@Suite struct Qwen4FlashArchTests {
    @Test func geometryMatchesTheStartingContract() {
        let arch = ArchConfig.qwen4Flash
        #expect(arch.family == .qwen4flash)
        #expect(arch.numLayers == 48)
        #expect(arch.hiddenSize == 2560)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.numExperts == 512)
        #expect(arch.topKExperts == 10)
        #expect(arch.moeIntermediateSize == 640)
        #expect(arch.intermediateSize == 640)
        #expect(arch.routerNormTopK)

        let fullLayers = arch.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 1 }.map(\.offset)
        #expect(fullLayers == Array(stride(from: 3, to: 48, by: 4)))
        #expect(arch.fullAttentionLayerMask.count == 48)

        #expect(arch.attnOutputGate)
        #expect(arch.attentionScale == 0.0625)
        #expect(arch.linearAttention.qkvDim == 10_240)
        #expect(arch.linearAttention.valueDim == 6_144)
        #expect(arch.hyperConnections.enabled)
        #expect(arch.hyperConnections.count == 4)
        #expect(arch.hyperConnections.lowRank == 320)
        #expect(arch.sparseIndexer.enabled)
        #expect(arch.sparseIndexer.budget == 2048)
        #expect(arch.ple.enabled)
        #expect(arch.ple.layerIndices == [1])
        #expect(arch.ple.ngramHeads == 16)
    }

    @Test func registryResolvesTheFamily() {
        #expect(ArchConfig.knownArchitectures[.qwen4flash] == .qwen4Flash)
        #expect(ArchConfig.knownArchitectures[.qwen4flashMTP] == .qwen4FlashMTP)
        #expect(ArchConfig.knownArchitectures[.qwen38flash] == .qwen38FlashNext)
    }

    @Test func theTwoFlashPresetsAreSeparateValues() {
        let four = ArchConfig.qwen4Flash
        let threeEight = ArchConfig.qwen38FlashNext
        // Today they agree field-for-field; what must not agree is the family.
        #expect(four.family != threeEight.family)
        #expect(four.numLayers == threeEight.numLayers)
        // Each family name reaches its own schema; the strings agree today.
        #expect(
            TensorSchema.schema(for: .qwen4flash).embedding
                == TensorSchema.schema(for: .qwen38flash).embedding)
        #expect(
            TensorSchema.schema(for: .qwen4flashMTP).finalNorm
                == TensorSchema.schema(for: .qwen4flash).finalNorm)
    }

    @Test func draftCarriesTheTargetGeometryButNoNgramBlock() {
        let draft = ArchConfig.qwen4FlashMTP
        #expect(draft.family == .qwen4flashMTP)
        #expect(draft.numLayers == 1)
        #expect(draft.fullAttentionLayerMask == [1])
        #expect(!draft.ple.enabled)
        #expect(draft.hyperConnections.count == 4)
        #expect(draft.vocabSize == 248_320)
    }
}
