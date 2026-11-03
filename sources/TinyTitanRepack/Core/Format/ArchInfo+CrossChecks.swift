import Foundation

// Cross-checks between a config and the family it claims to be.
//
// Split out of `ArchInfo.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. Both widened from
// `private` to internal because their callers are the loaders, which now live
// in ArchInfo+Loaders.swift.
extension ArchInfo {

    static func crossCheckQwen38FlashNext(
        _ a: ArchInfo,
        configPath: String
    ) throws {
        guard a.hiddenSize == 2560, a.numLayers == 48 else { return }
        var problems: [String] = []
        func want(_ ok: Bool, _ what: String) { if !ok { problems.append(what) } }
        want(a.numExperts == 512, "num_experts 512")
        want(a.topKExperts == 10, "num_experts_per_tok 10")
        want(a.moeIntermediateSize == 640, "moe_intermediate_size 640")
        want(a.vocabSize == 248_320, "vocab_size 248320")
        want(a.numHeads == 24 && a.numKVHeads == 2, "24 Q / 2 KV heads")
        want(a.headDim == 256, "head_dim 256")
        want(
            a.linearNumKHeads == 16 && a.linearNumVHeads == 48,
            "GDN 16 K / 48 V heads")
        want(
            a.hcCount == 4 && a.hcLowRank == 320,
            "hyper-connections 4 x 320")
        want(
            a.indexerBudget == 2048 && a.indexerCompressRatio == 4,
            "indexer budget 2048 / compress 4")
        want(
            a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12,
            "12 full-attention layers")
        want(a.pleLayerIndices == [1], "ple_layer_ids [2] (1-based)")
        guard problems.isEmpty else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen4_exp config does not match the supported "
                    + "Qwen3.8-Flash-Next contract: expected "
                    + problems.joined(separator: ", "))
        }
    }

    /// Production Qwen3.5-MoE 35B-A3B contract (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    static func crossCheckProductionQwen35MoE(
        _ a: ArchInfo,
        configPath: String
    ) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the supported "
                    + "35B-A3B architecture contract")
        }
    }

    /// Production Ornith-1.5-122B contract: the same Qwen3.5-MoE text
    /// architecture as the 35B-A3B at hidden 3072 / 48 layers, 256 experts at
    /// top-8, a 1024-wide expert and shared expert, 32 Q / 2 KV heads and
    /// 16 GDN K / 64 V heads. The fields are read from the checkpoint's
    /// `text_config`; this check refuses a 3072/48 config that does not agree
    /// with the published geometry, which is the failure a silent re-shape
    /// would otherwise produce. Other shapes are exempt, as for the 35B check.
    static func crossCheckOrnith122(
        _ a: ArchInfo,
        configPath: String
    ) throws {
        guard a.hiddenSize == 3072, a.numLayers == 48 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 3072,
            intermediateSize: 1024,
            moeIntermediateSize: 1024,
            numHeads: 32,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 48,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 64,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the supported "
                    + "Ornith-1.5-122B architecture contract")
        }
    }

    /// Qwen 4.0 Flash's production contract.
    ///
    /// Copied from the 3.8 preview's check and kept separate: when the 4.0
    /// checkpoint lands, a shape change is reconciled here and in
    /// `ArchConfig.qwen4Flash` only, and cannot loosen the 3.8 gate. The
    /// numbers are the preview's until the published config says otherwise.
    static func crossCheckQwen4Flash(
        _ a: ArchInfo,
        configPath: String
    ) throws {
        guard a.hiddenSize == 2560, a.numLayers == 48 else { return }
        var problems: [String] = []
        func want(_ ok: Bool, _ what: String) { if !ok { problems.append(what) } }
        want(a.numExperts == 512, "num_experts 512")
        want(a.topKExperts == 10, "num_experts_per_tok 10")
        want(a.moeIntermediateSize == 640, "moe_intermediate_size 640")
        want(a.vocabSize == 248_320, "vocab_size 248320")
        want(a.numHeads == 24 && a.numKVHeads == 2, "24 Q / 2 KV heads")
        want(a.headDim == 256, "head_dim 256")
        want(
            a.linearNumKHeads == 16 && a.linearNumVHeads == 48,
            "GDN 16 K / 48 V heads")
        want(
            a.hcCount == 4 && a.hcLowRank == 320,
            "hyper-connections 4 x 320")
        want(
            a.indexerBudget == 2048 && a.indexerCompressRatio == 4,
            "indexer budget 2048 / compress 4")
        want(
            a.fullAttentionLayerMask.filter { $0 == 1 }.count == 12,
            "12 full-attention layers")
        want(a.pleLayerIndices == [1], "ple_layer_ids [2] (1-based)")
        guard problems.isEmpty else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen4_omni config does not match the supported "
                    + "Qwen 4.0 Flash contract: expected "
                    + problems.joined(separator: ", "))
        }
    }
}
