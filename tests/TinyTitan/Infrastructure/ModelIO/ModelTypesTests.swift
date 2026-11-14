import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

@Suite struct ModelTypesTests {

    @Test func archConfigQwen36BaselineMatchesDocs() {
        let a = ArchConfig.qwen36_35B_A3B
        #expect(a.hiddenSize == 2048)
        #expect(a.intermediateSize == 512)
        #expect(a.moeIntermediateSize == 512)
        #expect(a.numLayers == 40)
        #expect(a.numExperts == 256)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 248_320)
        #expect(a.tieWordEmbeddings == false)
        #expect(a.finalLogitSoftcap == 0.0)
        #expect(a.fullAttentionLayerMask.count == 40)
        // Mask values: 0 = sliding-window, 1 = full attention, 2 = gated-DeltaNet
        // linear. Count the full-attention (== 1) layers.
        let fullCount = a.fullAttentionLayerMask.filter { $0 == 1 }.count
        #expect(fullCount == 10, "Qwen 3.6 has 10 full-attention layers, got \(fullCount)")
        // Mask flags every 4th layer (3, 7, ..., 39) as full; the rest linear.
        for L in stride(from: 3, to: 40, by: 4) {
            #expect(a.fullAttentionLayerMask[L] == 1, "layer \(L) should be full-attention")
        }
        #expect(a.fullAttentionLayerMask[0] == 2)
        #expect(a.attnOutputGate)
        #expect(a.sharedExpertGated)
        #expect(a.ropeNeoxSubdim)
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2048", actual: "4096")
        #expect(e1.description.contains("2048") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }
}
