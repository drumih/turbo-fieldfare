import Foundation
import Metal
import Synchronization
import TinyTitanCore

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition penalty — the one policy that needs `history` random access — as
/// a single in-place CPU pass over the (shared) logits before the front-end,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows mlx-lm's sampler order: Top-P is computed from the
/// model's full probability distribution, Top-K caps that surviving set, and
/// temperature is applied only to the final categorical draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let softcapTiled: LogitSoftcapSoftmaxTiled?
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    private let samplerPath: RuntimeSamplerPath
    let vocab: Int
    private let logitSoftcap: Float

    /// Incremental repetition-penalty history (R25): id -> occurrence count,
    /// carried across `sample` calls within one generation. The penalty is
    /// applied once per distinct id (HF convention), so the counts themselves
    /// are bookkeeping; the dict replaces the per-token `Set(history)` build.
    private var penaltyFrequency: [Int32: Int] = [:]
    /// Whether the current generation's prompt has been folded into
    /// `penaltyFrequency` yet.
    private var penaltyHistorySeeded = false
    /// How many ids of the caller's history have been folded into
    /// `penaltyFrequency`. The incremental path uses it to fold exactly the ids
    /// appended since the last call.
    private var penaltyHistoryCount = 0

    /// Monotonic counter combined with the clock so two samples in the same
    /// nanosecond still draw distinct non-deterministic seeds (R34).
    private static let nondeterministicSeedCounter = Atomic<UInt64>(0)

    /// Which front-end the last `sample` encoded, so `lastRowHadFiniteLogit`
    /// reads the buffer the GPU actually wrote. Reading at encode time would
    /// return the constructor's sentinel: the dispatch has not run yet.
    private enum FrontEnd { case singleThreadgroup, tiled }
    private var lastFrontEnd: FrontEnd = .singleThreadgroup

    init(
        context: MetalContext, vocab: Int = 262_144,
        logitSoftcap: Float = 30.0
    ) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context)
        self.softcapTiled = try LogitSoftcapSoftmaxTiled(context: context, vocab: vocab)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab)
        self.samplerPath = try RuntimeSamplerPath.environmentValue()
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap, in a `.storageModeShared` buffer (the
    /// repetition-penalty path edits it in place). `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(
        commandBuffer: MTLCommandBuffer,
        logits: MTLBuffer,
        probs: MTLBuffer,
        history: [Int32],
        config: GenerationConfig,
        position: Int,
        outToken: MTLBuffer
    ) throws -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty =
            (config.repetitionPenalty != 1.0
                || config.presencePenalty != 0) && !history.isEmpty
        if appliedPenalty {
            applyPenaltiesInPlace(
                logits: logits,
                history: history,
                repetition: config.repetitionPenalty,
                presence: config.presencePenalty)
        }
        // Structured output: floor every token the grammar no longer accepts,
        // in the shared logits buffer, before the softcap+softmax front-end
        // reads it. The write is host-side and the buffer is the previous
        // token's, already completed, so this is the same safe moment the
        // repetition penalty uses. The constraint is advanced by the decode
        // loop once the token it is about to allow has actually been chosen.
        if let constraint = config.constraint {
            let mask = constraint.allowedMask()
            guard !mask.isEmpty else { throw GeneratorError.constrainedDecodeStalled }
            mask.apply(
                toLogits: logits.contents().bindMemory(to: Float16.self, capacity: vocab),
                count: vocab)
        }
        // The tiled front-end follows the same path selection as the Top-K
        // half: `generic` forces the single-threadgroup pair so an A/B
        // measures both halves of the sampler, not one.
        if samplerPath == .tiled, let softcapTiled {
            try softcapTiled.encode(
                commandBuffer: commandBuffer,
                logits: logits, probs: probs, v: v,
                softcap: logitSoftcap)
            lastFrontEnd = .tiled
        } else {
            try softcap.encode(
                commandBuffer: commandBuffer,
                logits: logits, probs: probs, v: v,
                softcap: logitSoftcap)
            lastFrontEnd = .singleThreadgroup
        }

        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        // The tiled reduction serves every k it can reconstruct from a
        // top-64-per-tile stage 1, which is all of 1...64 — not just 64. The
        // production default is Top-K 20, so gating on `== 64` sent every
        // shipped token to the generic kernel instead, and that kernel takes
        // k full passes over a 262,144-entry vocabulary from a single
        // 256-thread threadgroup. Measured at the published 4-bit profile,
        // widening this gate is worth 16.656 -> 21.454 tok/s (+28.8%), with
        // the head_logits->embed gap falling from 15.45 ms to 1.41 ms/token.
        // k > 64, k == 0 (top-k disabled, k becomes 256), and greedy stay on
        // the generic path, which remains the reference implementation.
        if samplerPath == .tiled,
            config.temperature > 0,
            let requestedK = config.topK,
            (1...64).contains(requestedK)
        {
            try topK64Kernel.encode(
                commandBuffer: commandBuffer,
                probs: probs,
                outToken: outToken,
                temperature: config.temperature,
                topP: config.topP ?? 1.0,
                seed: seed,
                topK: UInt32(requestedK))
        } else {
            try sampleKernel.encode(
                commandBuffer: commandBuffer,
                probs: probs, outToken: outToken, v: v,
                temperature: isGreedy ? 0.0 : config.temperature,
                topK: UInt32(config.topK ?? 0),
                topP: config.topP ?? 1.0,
                seed: seed,
                position: UInt32(position))
        }

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    /// Whether the row the last `sample` ran on carried at least one finite
    /// logit. Valid once that command buffer has completed.
    ///
    /// `false` means the softmax wrote an all-zero probability row, so the id
    /// this sample returned is the kernel's in-range fallback and not an answer
    /// drawn from a distribution. The kernels keep that fallback on purpose --
    /// it is the invariant that keeps an out-of-range id out of the vocabulary
    /// -- so the emptiness has to be reported by whoever can see it, which is
    /// here. `softcap_value` already stops one NaN logit from emptying the row;
    /// this catches the row where every logit is NaN, which no folding in the
    /// kernel can turn into a distribution.
    var lastRowHadFiniteLogit: Bool {
        switch lastFrontEnd {
        case .singleThreadgroup: return softcap.rowMax.isFinite
        case .tiled: return softcapTiled?.rowMax.isFinite ?? true
        }
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention: for each token id seen in `history`, a positive logit is
    /// divided by `penalty`, a negative logit multiplied. Edits the shared
    /// `logits` buffer in place — no full-buffer copy, only the unique history
    /// entries are touched.
    ///
    /// The penalty must act on the POST-softcap logit (HF applies it to the
    /// model's output logits). For architectures with a logit softcap the raw
    /// logits can reach deep into tanh saturation, where dividing the raw value
    /// by 1.1 moves the capped logit by ~nothing — the penalty silently no-ops
    /// on exactly the high-confidence tokens that form repetition loops. So:
    /// softcap the raw value, penalize, and invert through atanh so the
    /// downstream softcap+softmax kernel reproduces the penalized capped logit.
    /// (Qwen 3.6 has no logit softcap, so the 0.0 branch is the production
    /// path.)
    ///
    /// Incremental history (R25): the first call of a generation folds the
    /// prompt into `penaltyFrequency`; each later call only adds the single
    /// token appended since the last call (the history's last element). The
    /// logit edit still runs for every distinct id every call because the
    /// logits buffer is fresh per token.
    /// Both history-reading penalties, in one pass over the shared logits.
    ///
    /// `repetition` is the HF-style multiplicative penalty; `presence` is the
    /// OpenAI-style additive one, subtracted once per distinct id. They share
    /// the frequency table and the host-side window before the softcap front
    /// end, so applying them together costs one walk of the table rather than
    /// two. Both work inside the softcap's space, which is where the repeated
    /// logit already lived for the repetition penalty.
    private func applyPenaltiesInPlace(
        logits: MTLBuffer,
        history: [Int32],
        repetition: Float,
        presence: Float
    ) {
        if !penaltyHistorySeeded {
            for id in history where id >= 0 && Int(id) < vocab {
                penaltyFrequency[id, default: 0] += 1
            }
            penaltyHistorySeeded = true
            penaltyHistoryCount = history.count
        } else if history.count > penaltyHistoryCount {
            // Every id appended since the last call, not just the final one.
            // Folding only `history.last` was correct for the one production
            // caller, which appends exactly one token between samples — but a
            // speculative path that appends two would silently fold one, and the
            // repetition penalty is what keeps a decode loop from repeating
            // itself. For that caller this is the same single id as before.
            for id in history[penaltyHistoryCount...]
            where id >= 0 && Int(id) < vocab {
                penaltyFrequency[id, default: 0] += 1
            }
            penaltyHistoryCount = history.count
        } else if history.count < penaltyHistoryCount {
            // The caller replaced the history rather than appending to it, so the
            // frequency table describes a sequence that is no longer in the
            // buffer. `resetPenaltyHistory` is the supported way to say this; a
            // table that still penalizes absent ids is a quieter wrong answer than
            // a re-seed, so re-seed.
            penaltyFrequency.removeAll(keepingCapacity: true)
            for id in history where id >= 0 && Int(id) < vocab {
                penaltyFrequency[id, default: 0] += 1
            }
            penaltyHistoryCount = history.count
        }

        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        let limit = logitSoftcap * 0.9999
        for (id, _) in penaltyFrequency {
            guard id >= 0 && Int(id) < vocab else { continue }
            let i = Int(id)
            var value = Float(ptr[i])
            if repetition != 1.0 {
                if logitSoftcap > 0 {
                    let capped = logitSoftcap * tanhf(value / logitSoftcap)
                    // A saturated negative logit times the penalty can leave the
                    // softcap's open interval; clamp inside it so atanh stays
                    // finite.
                    let scaled = capped > 0 ? capped / repetition : capped * repetition
                    value = logitSoftcap * atanhf(max(min(scaled, limit), -limit) / logitSoftcap)
                } else {
                    value = value > 0 ? value / repetition : value * repetition
                }
            }
            if presence != 0 {
                if logitSoftcap > 0 {
                    let capped = logitSoftcap * tanhf(value / logitSoftcap)
                    value =
                        logitSoftcap
                        * atanhf(max(min(capped - presence, limit), -limit) / logitSoftcap)
                } else {
                    value -= presence
                }
            }
            ptr[i] = Float16(value)
        }
    }

    /// Clear the incremental penalty history. The scratch sampler outlives a
    /// single generation, so the caller resets it at the start of each one.
    func resetPenaltyHistory() {
        penaltyFrequency.removeAll(keepingCapacity: true)
        penaltyHistorySeeded = false
        penaltyHistoryCount = 0
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E37_79B9_7F4A_7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        // Combine the clock with a monotonic counter (R34) so two samples in
        // the same nanosecond still draw distinct seeds.
        let counter = Self.nondeterministicSeedCounter
            .wrappingAdd(1, ordering: .relaxed).newValue
        let raw =
            (UInt64(bitPattern: Int64(t.tv_nsec)) &+ counter) &* 0x9E37_79B9_7F4A_7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E37_79B9_7F4A_7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
