import Foundation

/// One tuning profile per (model, routed-expert width): everything the
/// runtime chooses for a model that is not architecture -- the expert-cache
/// budget, the prefetch depth, the prefill chunk, the sampling defaults and
/// the kernel switches -- so each install is tuned on its own and a change
/// to one never touches another.
///
/// Resolution order, last wins:
///   1. the family default (`RuntimeConfiguration.decodeTuning`,
///      `GenerationDefaults.forFamily`, the family's chunk);
///   2. the entry for this (modelID, width) in `ModelProfile.table`;
///   3. the environment switches, so an experiment can still override a
///      shipped value without editing the table.
///
/// The table spells every entry out in full, even where it equals the family
/// default, precisely so that editing one row cannot change another. The
/// resolved profile is logged once at load under TINYTITAN_RUNNER_STATS.
public struct ModelProfile: Sendable, Equatable {
    public struct Key: Hashable, Sendable {
        public let modelID: String
        public let weightBits: Int
        public init(_ modelID: String, _ weightBits: Int) {
            self.modelID = modelID
            self.weightBits = weightBits
        }
    }

    public let key: Key
    public let family: ModelFamily
    /// Target bytes for the routed-expert slot cache (before the RAM clamp).
    public var expertCacheBudgetBytes: Int
    /// Speculative expert reads in flight; 0 disables prefetch.
    public var prefetchDepth: Int
    /// Prefill chunk in tokens; nil takes the front end's fallback.
    public var prefillChunkTokens: Int?
    public var sampling: GenerationDefaults.Sampling
    /// One-simdgroup router top-k (both k == 8 and k != 8).
    public var routerTopKSimd: Bool
    /// Simdgroup-per-key sparse decode attention (keep-mask layers only).
    public var attentionSimdPartial: Bool
    /// Fused hyper-connection gates (hyper-connection families only).
    public var hcFused: Bool
    /// GPU-side QSA key selection (sparse-attention families only).
    public var qsaGPUSelect: Bool
    /// Keep the routed-expert cache wired through prefill instead of
    /// unpinning it at prefill start and re-wiring it on the first decode
    /// token. Measured on Qwen3.8 4-bit: the re-wire faults a swapped-out
    /// 12 GiB cache back in, 1.6-4.7 s per request; holding it is a wash on
    /// decode throughput and prefill time. The row carries the measured default
    /// and is the whole decision: the env override that used to force it either
    /// way measured a wash on decode (-0.37%) and is gone, so a row that wires
    /// the cache keeps it wired.
    public var keepExpertCacheWired: Bool

    /// The shipped entries. Measured values, each on its own install.
    public static let table:
        [Key: (
            budget: Int, prefetch: Int, chunk: Int?,
            sampling: GenerationDefaults.Sampling,
            topKSimd: Bool, attnSimd: Bool,
            hcFused: Bool, qsaSelect: Bool, keepWired: Bool
        )] = [
            // The 35B rows take prefetch depth 1 and hold the expert cache wired
            // through prefill (2026-09-05, on the repaired ring). Measured per
            // install, five interleaved pairs on Qwen 3.6 and three on the other
            // two, 512-token generations:
            //
            //   prefetch depth 1   4-bit            8-bit
            //     Qwen 3.6         +1.8%            +11.3% (+9.8% on a rerun)
            //     Ornith 1.5       +1.8%            +12.6%
            //     AgentWorld       +1.4%            +11.4%
            //
            // Every interval excludes zero. Depth 2 gives only +2.7% where depth 1
            // gives +9.8%, so the ring stays one read deep, as it was at 4.7.
            // The 8-bit gain is the one the clogged ring had been hiding since
            // 2026-09-04: those rows asked for depth 1 and got nothing.
            //
            // Keeping the cache wired is worth +0.9% / +1.6% on 512-token
            // generations and +7.1% / +5.1% on 48-token ones (Qwen 3.6 4/8-bit),
            // where the first decode token no longer faults a swapped-out 10-12
            // GiB cache back in. Measured on Qwen 3.6; the other two installs
            // have identical geometry and cache sizes and take it by inference.
            // Qwen 3.6 35B-A3B, slot A/B 2026-09-05 (essay, interleaved, swap
            // sampled): 4-bit 128 slots 19.50 / 20.45, 160 (10 GiB) 20.55 / 21.04
            // with swap flat, 192 (12 GiB) 21.61 / 21.60 but 1.5 GB pushed to swap
            // on first contact -- 160 is the 24 GB default, 192 is one budget
            // setting away. 8-bit 64 slots 9.85 / 9.72, 96 (12 GiB) 11.13 / 11.19,
            // swap flat: the 8-bit hit rate was 79% at 64 (37% of the token in
            // exposed expert reads) and the trace simulation halves the misses
            // at 96. Prefetch one deep still pays at 8-bit; utility-tier depth 2
            // measured a wash at both widths.
            //
            // Sampling: the Qwen 3.5/3.6 series runs at temperature 0.6 / top-p
            // 0.95. The rows state it rather than borrow `house`, which happens
            // to hold the same numbers today: a later house change must not move
            // a model off its series' settings.
            Key("qwen3.6-35b-a3b", 4): (
                10 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            Key("qwen3.6-35b-a3b", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            // Ornith 1.5, same geometry, measured on its own 2026-09-05: 4-bit
            // 128 slots 19.91 / 20.41 vs 160 20.84 / 21.02; 8-bit 64 slots
            // 8.69 / 9.12 vs 96 10.83 / 10.86, swap flat on every arm.
            Key("ornith-1.5-35b-a3b", 4): (
                10 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false, true
            ),
            // The bf16-dense variant of the row above: the same 4-bit routed
            // experts and the same cache geometry, with attention, GDN and the
            // shared expert held at the checkpoint's bf16. Everything this row
            // tunes is on the expert path, so the values are inherited from
            // that measured install by inference -- the dense blocks are bigger
            // and read every token, which is why the variant decodes slower,
            // not why it would want a different cache.
            Key("ornith-1.5-35b-a3b-bf16dense", 4): (
                10 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false, true
            ),
            Key("ornith-1.5-35b-a3b", 8): (
                12 << 30, 1, 4_096, GenerationDefaults.house, true, true, false, false, true
            ),
            // AgentWorld, measured on its own 2026-09-05: 4-bit 128 slots 20.52 /
            // 20.50 vs 160 21.11 / 20.92; 8-bit 64 slots 9.31 / 9.25 vs 96
            // 11.15 / 11.21, swap flat. Residency and barrier execution both
            // lose on it. AgentWorld is a Qwen 3.6 fine-tune (manifest family
            // qwen36, same geometry; its template is Qwen 3.6's plus an audio
            // branch), so it takes the Qwen 3.6 sampling.
            Key("qwen-agentworld", 4): (
                10 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            Key("qwen-agentworld", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            // KAT-Coder-V2.5-Dev: a Qwen3.6-35B-A3B fine-tune with the same
            // geometry, so the cache budget, prefetch depth and wired cache are
            // taken from the rows above **by inference, not measured on this
            // install** -- the same inheritance the AgentWorld comment records.
            // What is not inherited is the sampling: the checkpoint's own
            // `generation_config.json` specifies temperature 1.0 with top-k 20 and
            // top-p 0.95, so stating 0.6 here (the Qwen 3.6 series setting) would
            // quietly run a coding model at a temperature its authors did not ask
            // for.
            Key("kat-coder-v2.5", 4): (
                10 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            Key("kat-coder-v2.5", 8): (
                12 << 30, 1, 4_096,
                GenerationDefaults.Sampling(
                    temperature: 1.0, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            // Qwen3.8-Flash-Next: 96 slots (12 GiB) still climbing; its card
            // specifies temperature 1.0 / top-p 0.95. The fused hyper-connection
            // gates and the GPU key select are measured washes and stay off.
            // 4-bit, 2026-09-21: prefetch depth 1, ON. The 2026-09-05 decision to
            // leave the ring off came from 512-token story runs where it lost. On
            // this engine it now wins at both prompt lengths measured: the 7-token
            // prompt 3.993 -> 4.621 tok/s decode (+15.7%), the ~500-token prompt
            // 3.627 -> 4.158 (+14.6%), with the expert hit rate up 4.5-4.7 points,
            // misses down 11-17% and io_hidden up. The ring turns speculative reads
            // into hits *before* demand rather than merely warming pages, and the
            // response is byte-identical (golden re-checked). The cache stays wired
            // through prefill (see keepExpertCacheWired).
            //
            // Chunk 2,048 (2026-10-22): the 4,096 reservation is held through
            // decode and costs it. An 851-token prompt prefilled in the same
            // ~30 s at both sizes, and 2,048 decoded 5.51/4.90 tok/s against
            // 4,769/4,645 (+15.6%/+5.5%, two interleaved rounds); a 200-token
            // chat prompt measured 6.11/6.79 against 5.58/6.09. The 4,096
            // chunk still wins the long-prefill case (~11% on 10k in the
            // comment on `defaultPrefillChunkTokens`), so an operator running
            // 10k prompts should pass `--prefill-chunk 4096`; the family
            // fallback and the 8-bit row keep it.
            //
            // Sampling is the *thinking* row here; the request's thinking mode
            // selects between it and the instruct row at validation time
            // (`GenerationDefaults.forFamily(_:thinking:)`).
            Key("qwen3.8-flash-next", 4): (
                12 << 30, 1, 2_048,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, true
            ),
            // 8-bit: 32 slots (8 GiB) 2.05 / 2.06 tok/s; 40 slots (9.5 GiB) 2.18 /
            // 2.27 with swap falling; 48 (13 GiB) 2.24-2.33 but ~1 GB of swap
            // growth per run on this 24 GB machine. 40 is the no-paging middle.
            // Prefetch depth 1 here is inferred from the 4-bit A/B, not measured:
            // the ring is family-level and width-independent, and this install is
            // not present to A/B. The depth-0 this replaces was inferred the same
            // way from the 2026-09-05 measurement.
            Key("qwen3.8-flash-next", 8): (
                Int(9.5 * Double(1 << 30)), 1, 4_096,
                GenerationDefaults.qwen38Thinking,
                true, true, false, false, true
            ),
            // Ornith-1.5-122B: the same qwen36 graph at hidden 3072 and 48
            // layers, 256 experts top-8, so a routed-expert slot is 5.3 MB at
            // 4-bit and 10.0 MB at 8-bit (against 2.5/4.9 MB for the 35B). Its
            // resident body is 4.30 GB at 4-bit and 6.58 GB at 8-bit, so at the
            // 12 GB process target the 4-bit install keeps 24 of 256 experts a
            // layer (9.4%) and the 8-bit 8 (3.1%); the counts above the target
            // are recorded on each row. The inferred prefetch depth 1 follows
            // qwen38: at this width the routed-expert read dominates the token.
            // Sampling is the checkpoint's own `generation_config.json`
            // (temperature 0.6, top-k 20, top-p 0.95), which is the house pair
            // -- stated rather than borrowed so a later house change cannot move
            // this model.
            Key("ornith-1.5-122b", 4): (
                // 24 slots x 5,308,416 B x 48 layers = 6,115,295,232 B
                // (~5.70 GiB): the remainder of the 12 GB process target after
                // the 4.0 GB resident body. Measured 2026-10-27 at
                // `--ram-budget 12G`: 10.2 GiB estimate, 10.63 GB peak RSS,
                // 2.720 tok/s at 128 generated tokens and 2.43-2.86 at 256
                // across the records and arms logged in .build/speeds-4bit-*.
                // This row is a cache *budget*, not a process cap: on a 24 GiB
                // host `affordableExpertCacheBudget` grows it to a third of RAM
                // (8.59 GB, 32 slots, 12.10 GiB estimate, 12.85 GB peak RSS), so
                // holding the 12 GB target needs `--ram-budget 12G` -- which is
                // what the launcher passes on a 24 GB Mac. 40 slots (3.411) and
                // 48 (3.574) also exceed the target; 64 fell to 0.682 (swap).
                6_115_295_232, 1, 2_048,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
            Key("ornith-1.5-122b", 8): (
                // 8 slots x 10,027,008 B x 48 layers = 3,850,371,072 B
                // (~3.59 GiB): the largest cache that fits the 12 GB target
                // beside the 6.1 GB 8-bit resident body. Measured 2026-10-27 at
                // `--ram-budget 12G`: 10.22 GiB estimate, 10.78 GB peak RSS,
                // 1.048 tok/s at 128 tokens and 1.00-1.24 at 256. Like the
                // 4-bit row it is a budget, so the third-of-RAM floor grows it
                // to 7.17 GB (16 slots, 13.80 GiB estimate, 14.48 GB peak RSS)
                // on this 24 GiB host -- pass the target to stay at 12 GB.
                // 16 slots measured 1.056 tok/s at ~15 GB -- no faster for
                // twice the footprint; 24 slots was killed by the OS.
                3_850_371_072, 1, 2_048,
                GenerationDefaults.Sampling(
                    temperature: 0.6, topK: GenerationDefaults.topK, topP: 0.95),
                true, true, false, false, true
            ),
        ]

    /// Environment switches applied last. Read once per process.
    public static let environment = ProcessInfo.processInfo.environment

    public static func resolve(
        modelID: String, family: ModelFamily, weightBits: Int,
        environment env: [String: String] = environment
    ) -> ModelProfile {
        let familyTuning = RuntimeConfiguration.decodeTuning(family: family, weightBits: weightBits)
        var profile = ModelProfile(
            key: Key(modelID, weightBits), family: family,
            expertCacheBudgetBytes: familyTuning.expertCacheBudgetBytes,
            prefetchDepth: familyTuning.prefetchDepth,
            prefillChunkTokens: nil,
            sampling: GenerationDefaults.forFamily(family),
            routerTopKSimd: true, attentionSimdPartial: true,
            hcFused: false, qsaGPUSelect: false,
            keepExpertCacheWired: false)
        if let row = table[profile.key] {
            profile.expertCacheBudgetBytes = row.budget
            profile.prefetchDepth = row.prefetch
            profile.prefillChunkTokens = row.chunk
            profile.sampling = row.sampling
            profile.routerTopKSimd = row.topKSimd
            profile.attentionSimdPartial = row.attnSimd
            profile.hcFused = row.hcFused
            profile.qsaGPUSelect = row.qsaSelect
            profile.keepExpertCacheWired = row.keepWired
        }
        if let v = env["TINYTITAN_ROUTER_TOPK_SIMD"] { profile.routerTopKSimd = v != "0" }
        if let v = env["TINYTITAN_ATTN_SIMD_PARTIAL"] { profile.attentionSimdPartial = v != "0" }
        if let v = env["TINYTITAN_HC_FUSED"] { profile.hcFused = v == "1" }
        if let v = env["TINYTITAN_QSA_GPU_SELECT"] {
            profile.qsaGPUSelect = v == "1" || v == "verify"
        }
        if let v = env["TINYTITAN_PREDICTIVE_PREFETCH"] {
            profile.prefetchDepth = v == "1" ? max(1, profile.prefetchDepth) : 0
        }
        // The ring depth itself has no override: every value but 1 measured a
        // loss (docs/qwen38-prefetch-predictor-study.md, Lever 5/8), so the
        // profile row's depth is the only source.
        return profile
    }

    public static func resolve(identity: ManifestIdentity) -> ModelProfile {
        resolve(modelID: identity.modelID, family: identity.family, weightBits: identity.weightBits)
    }

    /// Whether the table names this install, as opposed to falling back to
    /// its family. Draft-head sidecars and unknown ids fall back.
    public var isTabled: Bool { Self.table[key] != nil }

    /// One line for the log, so a benchmark records what ran.
    public var summary: String {
        "profile model=\(key.modelID) bits=\(key.weightBits) family=\(family.rawValue) "
            + (isTabled ? "tabled" : "family-default") + " "
            + "budget=\(expertCacheBudgetBytes >> 20)MiB prefetch=\(prefetchDepth) "
            + "chunk=\(prefillChunkTokens.map(String.init) ?? "fallback") "
            + "sampling=\(sampling.temperature)/\(sampling.topK)/\(sampling.topP) "
            + "topk_simd=\(routerTopKSimd) attn_simd=\(attentionSimdPartial) "
            + "hc_fused=\(hcFused) qsa_select=\(qsaGPUSelect) keep_wired=\(keepExpertCacheWired)"
    }
}
