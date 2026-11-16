import Foundation

/// Choosing the next token, on the CPU.
///
/// Greedy is the default and the only mode the memory work uses: distilling
/// a session and checking a claim both want the model's best answer, and a
/// deterministic side-engine is one whose output can be compared between
/// runs. But a CPU-served model is a served model, and a client that sends a
/// temperature expects it to mean something.
///
/// Deliberately the simple algorithms. Sampling 248,320 logits costs a
/// fraction of the 1.9 GB of weight reads that produced them, so the clever
/// version would save nothing measurable and could get the distribution
/// wrong.
public struct CPUSampler: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    /// The two penalties the wire and a model's published sampling row carry.
    /// Neutral at their defaults -- presence 0, repetition 1 -- so a request
    /// that does not ask for one is sampled exactly as before.
    public var presencePenalty: Float
    public var repetitionPenalty: Float
    /// A seed makes the draw reproducible across runs, which is what makes a
    /// regression visible; nil takes the clock. This is the same rule the GPU
    /// path's `seedFor` applies to `GenerationConfig.seed`.
    public var seed: UInt64?

    public init(
        temperature: Float = 0, topP: Float = 1, topK: Int = 0,
        presencePenalty: Float = 0, repetitionPenalty: Float = 1,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    public var isGreedy: Bool { temperature <= 0 }

    /// unchecked-invariant: `state` is only touched from `next()`, and a
    /// sampler belongs to one generation, which is one task.
    public final class Generator: @unchecked Sendable {
        private var state: UInt64
        init(seed: UInt64) { state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }
        func next() -> Float {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Float(state >> 40) / Float(1 << 24)
        }
    }

    public func makeGenerator() -> Generator {
        Generator(seed: seed ?? UInt64(Date().timeIntervalSince1970 * 1000))
    }

    public func pick(
        _ logits: [Float], history: [Int32] = [], using generator: Generator
    ) -> Int {
        var logits = logits
        applyPenalties(&logits, history: history)
        guard !isGreedy else {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            trace(logits, chosen: best)
            return best
        }
        // Top-k first, because it bounds the sort; top-p then trims what is
        // left by mass. Both are no-ops at their defaults.
        let limit = topK > 0 ? min(topK, logits.count) : logits.count
        var order = Array(logits.indices)
        if limit < logits.count {
            order.sort { logits[$0] > logits[$1] }
            order = Array(order.prefix(limit))
        } else {
            order.sort { logits[$0] > logits[$1] }
        }
        let peak = logits[order[0]]
        var weights = [Float]()
        weights.reserveCapacity(order.count)
        var total: Float = 0
        for index in order {
            let value = expf((logits[index] - peak) / max(temperature, 1e-4))
            weights.append(value)
            total += value
        }
        var cutoff = order.count
        if topP < 1 {
            var mass: Float = 0
            for position in weights.indices {
                mass += weights[position] / total
                if mass >= topP {
                    cutoff = position + 1
                    break
                }
            }
        }
        var remaining: Float = 0
        for position in 0..<cutoff { remaining += weights[position] }
        let target = generator.next() * remaining
        var running: Float = 0
        for position in 0..<cutoff {
            running += weights[position]
            if running >= target {
                trace(logits, chosen: order[position])
                return order[position]
            }
        }
        trace(logits, chosen: order[0])
        return order[0]
    }

    /// The penalties, applied the way the GPU sampler applies them
    /// (`Sampler.applyPenaltiesInPlace`, minus its softcap branch because this
    /// engine has no softcap): a logit the history already contains is divided
    /// by the repetition factor while positive and multiplied by it while
    /// negative, then has the presence penalty subtracted. Each distinct id is
    /// penalized once whatever its count in the history, and out-of-vocabulary
    /// ids are skipped rather than trapping -- the history is model output, and
    /// a tokenizer that emits an id past `logits.count` must not abort a
    /// generation.
    private func applyPenalties(_ logits: inout [Float], history: [Int32]) {
        guard !history.isEmpty, presencePenalty != 0 || repetitionPenalty != 1 else { return }
        for id in Set(history) {
            guard id >= 0, Int(id) < logits.count else { continue }
            var value = logits[Int(id)]
            if repetitionPenalty != 1 {
                value = value > 0 ? value / repetitionPenalty : value * repetitionPenalty
            }
            if presencePenalty != 0 {
                value -= presencePenalty
            }
            logits[Int(id)] = value
        }
    }

    /// `TINYTITAN_LOGIT_TRACE=1`: the top-2 of this step's logits, for
    /// engine-agreement work (TT-002). The GPU path prints the same line from
    /// `sampleOnce`; a greedy argmax alone hides how close the decision was.
    /// Lines arrive in generation order, so the Nth is generated token N.
    private func trace(_ logits: [Float], chosen: Int) {
        guard ProcessInfo.processInfo.environment["TINYTITAN_LOGIT_TRACE"] == "1" else { return }
        var first = -Float.greatestFiniteMagnitude
        var second = first
        var firstID = 0
        var secondID = 0
        for index in logits.indices {
            let value = logits[index]
            if value > first {
                second = first
                secondID = firstID
                first = value
                firstID = index
            } else if value > second {
                second = value
                secondID = index
            }
        }
        FileHandle.standardError.write(
            Data(
                String(
                    format: "[logit] chosen=%d top1=%d:%.4f top2=%d:%.4f margin=%.4f\n",
                    chosen, firstID, first, secondID, second, first - second
                ).utf8))
    }
}
