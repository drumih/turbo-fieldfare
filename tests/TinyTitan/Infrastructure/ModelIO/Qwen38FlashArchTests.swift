import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// Pins the Qwen3.8-Flash-Next geometry against the checkpoint facts recorded
/// in docs/qwen38-flash-next-port.md, so a later edit cannot silently drift
/// from the verified config.json.
@Suite struct Qwen38FlashArchTests {
    @Test func geometryMatchesTheVerifiedCheckpointConfig() {
        let arch = ArchConfig.qwen38FlashNext
        #expect(arch.modelFamily == .qwen38flash)
        #expect(arch.numLayers == 48)
        #expect(arch.hiddenSize == 2560)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.numExperts == 512)
        #expect(arch.topKExperts == 10)
        #expect(arch.moeIntermediateSize == 640)
        #expect(arch.intermediateSize == 640)
        #expect(arch.routerNormTopK)
        // 64, not the 32 first assumed from the Vontra checkpoints: the
        // pinned artifact declares group_size 64 and decodes correctly at 64
        // (benchmark/tinytitan_quant_fidelity.py, within 1.02x of the format floor).
        #expect(arch.quantGroupSize == 64)

        // (3x linear -> 1x QSA) x 12: full attention on every 4th layer.
        let fullLayers = arch.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 1 }.map(\.offset)
        #expect(fullLayers == Array(stride(from: 3, to: 48, by: 4)))
        #expect(arch.fullAttentionLayerMask.count == 48)

        // QSA geometry: 24 Q / 2 KV heads, dim 256, packed q+gate.
        #expect(arch.numHeads == 24)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.fullHeadDim == 256)
        #expect(arch.attnOutputGate)
        #expect(arch.attentionScale == 0.0625)

        // GDN: 16 QK / 48 V heads, dim 128 -> packed qkv rows 10,240.
        #expect(arch.linearAttention.qkvDim == 10_240)
        #expect(arch.linearAttention.valueDim == 6_144)

        // Hyper-connections: 4 streams, rank-320 mixers.
        #expect(arch.hyperConnections.enabled)
        #expect(arch.hyperConnections.count == 4)
        #expect(arch.hyperConnections.lowRank == 320)

        // Indexer: dense attention is exact only within the budget window.
        #expect(arch.sparseIndexer.enabled)
        #expect(arch.sparseIndexer.budget == 2048)
        #expect(arch.sparseIndexer.compressRatio == 4)

        // PLE: 16 hashed lookups per token, 160-wide rows, at layer index 1.
        #expect(arch.ple.enabled)
        #expect(arch.ple.layerIndices == [1])
        #expect(arch.ple.ngramHeads == 16)
        #expect(arch.ple.headDim == 160)
        #expect(arch.ple.seed == 1234)
    }

    @Test func registryResolvesTheFamilyAndLoadingFailsClosed() {
        #expect(ArchConfig.knownArchitectures[.qwen38flash] == .qwen38FlashNext)
        // Existing architectures keep plain residuals and group 64.
        #expect(!ArchConfig.qwen36_35B_A3B.hyperConnections.enabled)
        #expect(!ArchConfig.qwen36_35B_A3B.sparseIndexer.enabled)
        #expect(!ArchConfig.qwen36_35B_A3B.ple.enabled)
        #expect(ArchConfig.qwen36_35B_A3B.quantGroupSize == 64)
        #expect(!ArchConfig.qwen36_35B_A3B.routerNormTopK)
    }
}
