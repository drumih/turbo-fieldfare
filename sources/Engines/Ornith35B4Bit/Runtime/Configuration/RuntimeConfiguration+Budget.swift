import Foundation
import TinyTitanCore

extension RuntimeConfiguration {
    /// Decode defaults that are not one number across the catalogue.
    ///
    /// Both settings below are governed by the same quantity: how much of the
    /// token is expert I/O. A speculative read is only ever a bet that the SSD
    /// has service to spare, so it pays where I/O dominates and costs where it
    /// does not.
    public struct DecodeTuning: Sendable, Equatable {
        /// Target bytes for the routed-expert slot cache.
        public let expertCacheBudgetBytes: Int
        /// Speculative expert reads allowed in flight; 0 disables prefetch.
        public let prefetchDepth: Int

        public init(expertCacheBudgetBytes: Int, prefetchDepth: Int) {
            self.expertCacheBudgetBytes = expertCacheBudgetBytes
            self.prefetchDepth = prefetchDepth
        }
    }

    /// The measured optimum for a family at a given routed-expert width.
    ///
    /// Measured 2026-08-30, 512-token continuous prose, fresh process per run,
    /// configs interleaved with the order reversed on alternate repetitions:
    ///
    ///     qwen38flash 4-bit   5.735 -> 6.957 tok/s  (+21.3%)  12 GiB + depth 1
    ///     qwen36      8-bit  11.994 -> 12.659       ( +5.5%)   8 GiB + depth 1
    ///     ornith      8-bit  11.173 -> 11.837       ( +5.9%)   8 GiB + depth 1
    ///     qwen36/ornith 4-bit                        (regress)  8 GiB, no prefetch
    ///
    /// The 4-bit 35B pair is the instructive one. Expert I/O there is only
    /// ~7 ms of a ~44 ms token, so there is almost nothing for a speculative
    /// read to recover, and the read still costs SSD service and ring
    /// bookkeeping -- it measured a regression at 7 runs per config. Ornith and
    /// Qwen 3.6 share the `qwen36` family and measured the same, so keying on
    /// family rather than model id is correct here rather than merely
    /// convenient.
    public static func decodeTuning(
        family: ModelFamily,
        weightBits: Int
    ) -> DecodeTuning {
        switch (family, weightBits) {
        case (.qwen38flash, _), (.qwen38flashMTP, _):
            // 96 slots. The only family whose working set justifies the extra
            // 4 GiB: 512 experts at top-10 spread far wider than 128 at top-8,
            // so it is still climbing at 96 where the 35B families have
            // flattened above 90% hit rate.
            return DecodeTuning(expertCacheBudgetBytes: 12 << 30, prefetchDepth: 1)
        case (.qwen4flash, _), (.qwen4flashMTP, _):
            // Same 512-expert/top-10 geometry until the 4.0 checkpoint measures otherwise.
            return DecodeTuning(expertCacheBudgetBytes: 12 << 30, prefetchDepth: 1)
        case (.qwen36, 8), (.qwen36MTP, 8):
            return DecodeTuning(
                expertCacheBudgetBytes: defaultExpertCacheBudgetBytes,
                prefetchDepth: 1)
        default:
            return DecodeTuning(
                expertCacheBudgetBytes: defaultExpertCacheBudgetBytes,
                prefetchDepth: 0)
        }
    }

    /// A tuned budget the machine can actually hold.
    ///
    /// `decodeTuning` returns what measured fastest on a 24 GiB machine. A third
    /// of physical memory is the ceiling because the slot cache is not the only
    /// resident claim -- dense weights, the KV cache and the prompt cache all
    /// have to fit beside it. Without this, a 12 GiB default aimed at
    /// qwen38flash would be handed unchanged to a 16 GiB Mac.
    ///
    /// A third rather than a half, and `defaultExpertCacheBudgetBytes` is the
    /// evidence: 8 GiB is a third of the 24 GiB machine those budgets were tuned
    /// on, so a third reproduces the tuned value exactly where it was tuned and
    /// scales down where a constant could not. At a half, an 8 GB mini is handed
    /// 64 slots -- 4.22 GiB of cache against a 70.8 MB slot -- and pages:
    /// measured swap 855 -> 1610 MB and 5.576 tok/s, against a flat swap and
    /// 7.289 tok/s at the 40 slots a third selects. The failure is the one
    /// `expertCacheSlots` already documents for 8-bit, where an over-budget
    /// cache cost 4.8x throughput with the hit rate *falling*, so it is a paging
    /// problem rather than a cache one.
    ///
    /// Verified end to end on the machine that found it: the server with no cache
    /// flag decodes 9.27 tok/s against 2.26 before, at 4.59 GB resident with swap
    /// flat, which is the explicit-40 reference (9.34) within noise.
    ///
    /// The cost is on the large machine, and it is small but real. An install whose
    /// profile table asks for **more** than a third -- qwen3.6 35B-A3B 4-bit asks
    /// for 10 GiB, and a 24 GB Mac passes a half (12 GiB) untouched -- does lose
    /// slots to this ceiling. Measured on a 24 GB M3 with the same binary and an
    /// interleaved A/B, three pairs, 256 tokens at temperature 0:
    ///
    ///      default (128 slots, 8 GiB)   16.374 / 16.624 / 16.666 tok/s
    ///      --expert-cache-slots 160     16.750 / 16.777 / 16.761 tok/s
    ///
    /// 16.55 against 16.76, about **-1.3% decode**, and time to first token moved
    /// the other way (1.27-1.32 s against 1.44-1.56 s) because less cache is
    /// wired. The tuned 10 GiB was not paging there -- peak footprint 14.25 GB,
    /// zero swaps.
    ///
    /// So the third is floored at the tune: a machine with at least the 24 GiB
    /// the budgets were measured on keeps `wanted`, and only a smaller machine
    /// is cut. The floor became safe when the QSA indexer started growing on
    /// demand (45f55bb): before it, the indexer's 0.94 GiB of full-context key
    /// buffers sat on top of qwen38's tuned 12 GiB - 96 slots - and the pair
    /// paged at the shipped 262144 context even on the 24 GB tuning machine
    /// (3.052/3.201 tok/s mirrored, against 4.481-4.551 at the clamped 64).
    /// Grown on demand, the same 96 slots fit and measure 5.156 tok/s at full
    /// context, greedy output byte-identical to the 64-slot run
    /// (docs/qwen38-decode-12gb-measurements.md §5o/§5p). Below the floor the
    /// small-machine evidence above still stands and the third still applies.
    ///
    /// Above the tune the third is then taken as a floor as well as a ceiling:
    /// a machine that can hold more than the tuned budget gets a third of what
    /// it has, because §5o's verdict is that the slots which collapsed on
    /// 24 GB did so on physical working set, not on budget policy — so the
    /// machine with the working set to spare is the case the tune left on the
    /// table. On a 64 GiB Mac qwen38 4-bit moves from 96 slots to 160 (21.3
    /// GiB) and 128 GiB reaches the ladder's 256 top rung; that buys both
    /// prefill and decode expert reads, which is the LMCache rule (size the
    /// resident tier to the machine, not to one machine class) applied at home.
    /// A third never contradicts the half-of-RAM line the launcher warns on,
    /// because a third of a machine is below half of it by construction.
    /// **Not measured on the larger machines**: this repository's install is
    /// the 24 GiB M3, where `wanted` is already above a third and the rule is a
    /// deliberate no-op; the arithmetic is pinned by test instead.
    public static func affordableExpertCacheBudget(
        _ wanted: Int,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        guard physicalMemory > 0 else { return wanted }
        let third = Int(physicalMemory / 3)
        if physicalMemory >= 24 << 30 { return max(wanted, third) }
        return min(wanted, third)
    }

    /// Parses a RAM budget such as `2G`, `512M`, `8GiB` or a plain byte count.
    ///
    /// Accepts the sizes users actually type. Returns nil for anything
    /// unparseable or non-positive, so a typo becomes an argument error rather
    /// than a silently tiny cache.
    public static func parseBudgetBytes(_ text: String) -> Int? {
        let raw = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard !raw.isEmpty else { return nil }
        let multipliers: [(String, Int)] = [
            ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
            ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
            ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10),
        ]
        for (suffix, scale) in multipliers where raw.hasSuffix(suffix) {
            let number = String(raw.dropLast(suffix.count))
                .trimmingCharacters(in: .whitespaces)
            guard let value = Double(number), value > 0 else { return nil }
            let bytes = value * Double(scale)
            guard bytes.isFinite, bytes >= 1, bytes < Double(Int.max) else { return nil }
            return Int(bytes)
        }
        guard let plain = Int(raw), plain > 0 else { return nil }
        return plain
    }

    /// Slots that fit `budgetBytes`, snapped to the nearest supported count.
    ///
    /// Deriving from the stride rather than hard-coding a number per quantisation
    /// keeps 4-bit and 8-bit on the same rule: 1 GiB lands on 16 slots at a
    /// 1.688 MiB stride and 8 slots at 3.188 MiB, which are the measured optima
    /// for each.
    public static func expertCacheSlots(
        expertStrideBytes: UInt64,
        layers: Int,
        budgetBytes: Int = defaultExpertCacheBudgetBytes
    ) -> Int {
        guard expertStrideBytes > 0, layers > 0 else {
            return Self.minimumExpertCacheSlots
        }
        let perSlot = Double(expertStrideBytes) * Double(layers)
        let wanted = Double(budgetBytes) / perSlot
        // The nearest rung, and the earliest one wins a tie -- the rule
        // `min(by:)` applies, walked over the list so no branch has to pretend
        // it can be empty.
        var choice = Self.minimumExpertCacheSlots
        for candidate in allowedExpertCacheSlots
        where abs(Double(candidate) - wanted) < abs(Double(choice) - wanted) {
            choice = candidate
        }
        // Nearest, then step down until the footprint honours the budget.
        //
        // Rounding to nearest alone can overshoot, and the overshoot grows with
        // the expert stride: the wider the model, the further past the budget
        // the nearest rung lands. Qwen3.8 at 8-bit wanted 51.4 slots and was
        // handed 64 -- a 16.1 GiB cache against a 12 GiB budget, 34% over. On a
        // 24 GiB machine that put the process into swap and cost 4.8x
        // throughput: 0.42 tok/s at 64 slots against 2.01 at 32, with the hit
        // rate *falling* 80% -> 67.7%, which is how a paging problem looks when
        // it is mistaken for a cache problem. The same arithmetic at 4-bit
        // wants 96.9 and gets 96, so it never showed on the model this was
        // tuned against.
        //
        // 1.15 is not a new number: `chosenCountStaysNearTheRequestedBudget`
        // has always asserted the footprint stays within 15% of the budget. It
        // simply never sampled a 48-layer model at a 12 GiB budget, so the one
        // configuration that broke the contract went unmeasured.
        //
        // Stepping down is the safe direction, and §5o measured both sides on
        // this machine: at the 32K cap 96 slots gave 4.929/4.966 against 64's
        // 4.543/4.175, so the rung below the top costs ~12 %; 112 fell to
        // 3.622 (-26 %) and 128 to 1.473 (-70 %). A rung past the working set
        // does not buy its I/O back -- it pages.
        let ceiling = Double(budgetBytes) * 1.15
        // `last(where:)` reads the list's order as a promise -- the largest rung
        // below the current one -- so it takes a sorted copy rather than the
        // literal's spelling. The literal is written ascending and
        // `expertCacheSlotFloorIsTheSmallestRung` pins that, because the
        // nearest-rung walk above settles a tie on the earlier element, which is
        // the smaller rung only while the list climbs.
        let rungs = allowedExpertCacheSlots.sorted()
        while Double(choice) * perSlot > ceiling,
            let smaller = rungs.last(where: { $0 < choice })
        {
            choice = smaller
        }
        return choice
    }
    /// Bytes that are resident before the expert cache is allocated.
    ///
    /// `--ram-budget` names what the whole server may hold, so the cache gets
    /// the remainder after the weights and the runtime. The weights are the
    /// manifest's `model_weights.bin` -- lm head, embeddings, attention and the
    /// shared experts; `packed_experts/` and the n-gram table are streamed and
    /// are not resident.
    ///
    /// The reserve is measured on the 24 GiB M3 with Qwen3.8 4-bit: the server
    /// loads at 3.75 GiB against a 3.22 GiB weight file, and the smallest cache
    /// (8 slots, 0.99 GiB) takes it to 4.74 GiB -- exactly the cache delta -- so
    /// the non-weight floor is 0.53 GiB. That is the Metal context and its
    /// pipelines, the tokenizer, the expert layout table, the prefetch ring,
    /// per-sequence scratch and the KV for a short request. 512 MiB is that
    /// measurement rounded with a little slack.
    ///
    /// The KV is the part that grows: this model needs 49,152 B a token at
    /// 8-bit (48 layers x 2 x 2 KV heads x 256), so 512 MiB covers the runtime
    /// plus about 10,000 tokens of context and the full 262,144-token window
    /// would add 12 GiB. The server prints the floor it used, and a long context
    /// is what takes the estimate past the target.
    public static let residentRuntimeReserveBytes = 512 << 20

    public static func residentFloorBytes(residentWeightBytes: Int) -> Int {
        max(0, residentWeightBytes) + residentRuntimeReserveBytes
    }

    /// The smallest `--ram-budget` the server accepts.
    ///
    /// A streaming install cannot stay under less. The resident weight file is
    /// 2.5-4.5 GiB depending on family and width and the expert cache has an
    /// 8-slot floor, so the smallest real footprint is about 4.7 GiB on
    /// Qwen3.8 4-bit. The flag used to accept 1G and 2G and quietly land on that
    /// floor; it now refuses, because a number the process cannot stay under is
    /// worse than an error naming the floor. Values at or above this one are
    /// honoured; values below it are an argument error.
    public static let minimumProcessTargetBytes = 4 << 30

    /// The largest supported slot count whose cache fits `cacheBytes`, never
    /// below the smallest rung.
    ///
    /// This is the process-target path, and it steps *down* rather than to the
    /// nearest rung: `expertCacheSlots` rounds to nearest and tolerates 15% over
    /// its budget, which is right for a tuned profile and wrong for a number the
    /// user asked the whole server to stay under. The smallest rung is the floor
    /// because a cache smaller than a layer's top-k cannot place its experts.
    public static func expertCacheSlotsFitting(
        expertStrideBytes: UInt64,
        layers: Int,
        cacheBytes: Int
    ) -> Int {
        guard expertStrideBytes > 0, layers > 0 else {
            return Self.minimumExpertCacheSlots
        }
        guard cacheBytes > 0 else { return Self.minimumExpertCacheSlots }
        let perSlot = Double(expertStrideBytes) * Double(layers)
        let fitting = allowedExpertCacheSlots.filter {
            Double($0) * perSlot <= Double(cacheBytes)
        }
        // `fitting` really can be empty -- a budget below the smallest rung
        // leaves nothing that fits -- so this fallback is not defensive code for
        // an impossible branch; only the rung it names is. `max()` rather than
        // `last` because the largest rung that fits is the answer, whatever
        // order the list is written in.
        return fitting.max() ?? Self.minimumExpertCacheSlots
    }

}
