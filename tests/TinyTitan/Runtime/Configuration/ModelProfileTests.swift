import Darwin
import Testing

@testable import TinyTitan

/// One profile per (model, width), resolved family -> table -> environment.
@Suite struct ModelProfileTests {
    static let shipped: [(String, ModelFamily)] = [
        ("qwen3.6-35b-a3b", .qwen36), ("ornith-1.5-35b-a3b", .qwen36),
        ("ornith-1.5-122b", .qwen36),
        ("qwen-agentworld", .qwen36), ("kat-coder-v2.5", .qwen36),
        ("qwen3.8-flash-next", .qwen38flash),
    ]

    @Test func everyShippedInstallHasItsOwnRow() {
        for (id, family) in Self.shipped {
            for bits in [4, 8] {
                let p = ModelProfile.resolve(
                    modelID: id, family: family, weightBits: bits, environment: [:])
                #expect(p.isTabled, "\(id) \(bits)-bit falls back to its family")
                #expect(p.key == ModelProfile.Key(id, bits))
            }
        }
        #expect(ModelProfile.table.count == 13)
    }

    @Test func theBf16DenseOrnithInheritsTheFourBitRowsTuning() {
        // It is a different install, not a second width: the 4-bit experts and
        // their cache are the same, so every value this table tunes is the
        // plain 4-bit row's. The row is still its own key, because sharing one
        // would make the two installs indistinguishable to the log and to a
        // later re-tune of either.
        let dense = ModelProfile.resolve(
            modelID: "ornith-1.5-35b-a3b-bf16dense", family: .qwen36, weightBits: 4,
            environment: [:])
        let plain = ModelProfile.resolve(
            modelID: "ornith-1.5-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(dense.isTabled)
        #expect(dense.key != plain.key)
        #expect(dense.expertCacheBudgetBytes == plain.expertCacheBudgetBytes)
        #expect(dense.prefetchDepth == plain.prefetchDepth)
        #expect(dense.prefillChunkTokens == plain.prefillChunkTokens)
        #expect(dense.sampling == plain.sampling)
        #expect(dense.routerTopKSimd == plain.routerTopKSimd)
        #expect(dense.attentionSimdPartial == plain.attentionSimdPartial)
        #expect(dense.keepExpertCacheWired == plain.keepExpertCacheWired)
    }

    @Test func modelsSharingAFamilyResolveIndependently() {
        // Same family, different keys: editing one row cannot move the other.
        let a = ModelProfile.resolve(
            modelID: "qwen-agentworld", family: .qwen36, weightBits: 4, environment: [:])
        let q = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(a.key != q.key)
        #expect(ModelProfile.table[a.key] != nil && ModelProfile.table[q.key] != nil)
    }

    @Test func tabledValuesMatchWhatWasMeasured() {
        let q38 = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4, environment: [:])
        #expect(q38.expertCacheBudgetBytes == 12 << 30)
        // Depth 1 since 2026-09-21: the ring was re-measured and wins at both
        // prompt lengths now (7-token +15.7%, ~500-token +14.6%), which
        // supersedes the 2026-09-05 decision to leave it off.
        #expect(q38.prefetchDepth == 1)
        #expect(q38.keepExpertCacheWired)
        // 2026-10-22: the 4,096 reservation costs decode on the 24 GB host;
        // 2,048 measured 5-16% faster decode at the same prefill on chat-like
        // prompts and the generated text stayed byte-identical.
        #expect(q38.prefillChunkTokens == 2_048)
        let q38b = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 8, environment: [:])
        #expect(q38b.expertCacheBudgetBytes == Int(9.5 * Double(1 << 30)))
        // Inferred from the 4-bit A/B (see the 8-bit row comment), not measured.
        #expect(q38b.prefetchDepth == 1)
        // The 8-bit row keeps 4,096: its prefill is slower and the long-prompt
        // case is the one it is chosen for.
        #expect(q38b.prefillChunkTokens == 4_096)
        #expect(q38.sampling.temperature == 1.0 && q38.sampling.topP == 0.95)
        #expect(!q38.hcFused && !q38.qsaGPUSelect)
        let q36 = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 8, environment: [:])
        #expect(q36.expertCacheBudgetBytes == 12 << 30)
        #expect(q36.keepExpertCacheWired)
        let q36four = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(q36four.expertCacheBudgetBytes == 10 << 30)
        #expect(q36.prefetchDepth == 1)
        #expect(q36.prefillChunkTokens == 4_096)
        #expect(q36.sampling == GenerationDefaults.house)
    }

    @Test func samplingRowsFollowTheirSeries() {
        let qwen36Series = GenerationDefaults.Sampling(
            temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95)
        let qwen38Series = GenerationDefaults.Sampling(
            temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95)
        let expected: [(String, ModelFamily, GenerationDefaults.Sampling)] = [
            ("qwen3.6-35b-a3b", .qwen36, qwen36Series),
            ("qwen-agentworld", .qwen36, qwen36Series),
            // Ornith-1.5-122B's `generation_config.json` asks for 0.6 / top-k
            // 20 / top-p 0.95, which happens to equal the house pair. Stated so
            // that a later house change cannot move it.
            ("ornith-1.5-122b", .qwen36, qwen36Series),
            // A fine-tune does not inherit its base's sampling: KAT-Coder's own
            // generation_config.json asks for 1.0, so its rows must not be
            // swept along by the 0.6 the rest of the qwen36 series uses.
            ("kat-coder-v2.5", .qwen36, qwen38Series),
            ("qwen3.8-flash-next", .qwen38flash, qwen38Series),
            // Not a Qwen-named series: it keeps the house values until its own
            // card is checked.
            ("ornith-1.5-35b-a3b", .qwen36, GenerationDefaults.house),
        ]
        for (id, family, sampling) in expected {
            for bits in [4, 8] {
                let p = ModelProfile.resolve(
                    modelID: id, family: family, weightBits: bits, environment: [:])
                #expect(p.sampling == sampling, "\(id) \(bits)-bit")
            }
        }
    }

    @Test func qwen4FallsBackToItsOwnFamilyUntilItsRowsAreMeasured() {
        // No ModelProfile row is claimed for a model with no install and no
        // measurements; what is pinned is that the family defaults are the 4.0
        // ones (12 GiB / depth 1 / the thinking row), not the 35B house values.
        let p = ModelProfile.resolve(
            modelID: "qwen4-omni-120b-think", family: .qwen4flash,
            weightBits: 4, environment: [:])
        #expect(!p.isTabled)
        #expect(p.expertCacheBudgetBytes == 12 << 30)
        #expect(p.prefetchDepth == 1)
        #expect(p.sampling == GenerationDefaults.qwen4Thinking)

        let draft = ModelProfile.resolve(
            modelID: "qwen4-omni-120b-think-mtp", family: .qwen4flashMTP,
            weightBits: 4, environment: [:])
        #expect(!draft.isTabled)
        #expect(draft.expertCacheBudgetBytes == 12 << 30)
    }

    @Test func unknownModelFallsBackToItsFamily() {
        let p = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b-mtp-4bit", family: .qwen36MTP, weightBits: 4, environment: [:]
        )
        #expect(!p.isTabled)
        let f = RuntimeConfiguration.decodeTuning(family: .qwen36MTP, weightBits: 4)
        #expect(p.expertCacheBudgetBytes == f.expertCacheBudgetBytes)
        #expect(p.prefetchDepth == f.prefetchDepth)
        #expect(p.prefillChunkTokens == nil)
        #expect(p.sampling == GenerationDefaults.forFamily(.qwen36MTP))
    }

    @Test func environmentOverridesTheTable() {
        let env = [
            "TINYTITAN_ROUTER_TOPK_SIMD": "0", "TINYTITAN_HC_FUSED": "1",
            "TINYTITAN_PREDICTIVE_PREFETCH": "1",
            "TINYTITAN_QSA_GPU_SELECT": "verify",
        ]
        let p = ModelProfile.resolve(
            modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: env)
        #expect(!p.routerTopKSimd)
        #expect(p.hcFused)
        #expect(p.qsaGPUSelect)
        let off = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4,
            environment: ["TINYTITAN_PREDICTIVE_PREFETCH": "0"])
        #expect(off.prefetchDepth == 0)
    }

    @Test func theRowDecidesWhetherTheCacheStaysWired() {
        // The tri-state `TINYTITAN_KEEP_WIRED` override is gone: it measured a
        // wash on decode (-0.37%) and the row's own value is the decision, so
        // every streaming row keeps its cache wired and a dense row does not.
        let streaming = ModelProfile.resolve(
            modelID: "qwen3.8-flash-next", family: .qwen38flash,
            weightBits: 4, environment: [:])
        #expect(streaming.keepExpertCacheWired, "the row wires it")
        let dense = ModelProfile.resolve(
            modelID: "qwen3.5-4b", family: .qwen35Dense,
            weightBits: 4, environment: [:])
        #expect(!dense.keepExpertCacheWired, "a dense install has no routed-expert cache")
    }

    @Test func summaryNamesTheKeyAndEveryKnob() {
        let p = ModelProfile.resolve(
            modelID: "qwen-agentworld", family: .qwen36, weightBits: 8, environment: [:])
        for needle in [
            "model=qwen-agentworld", "bits=8", "tabled", "budget=", "prefetch=1",
            "chunk=4096", "topk_simd=true", "hc_fused=false", "keep_wired=true",
        ] {
            #expect(p.summary.contains(needle), Comment(rawValue: needle))
        }
    }
}
