import Foundation

/// CPU reference for the K3 GPU sampler (`sampling_k3.metal`). Mirrors the
/// kernel's logic exactly — same max-subtracted fp32 softmax, same
/// lowest-index tie-breaks, same xorshift64*/splitmix64 streams, same
/// truncation order (top-p cut on the descending top-k list, then temperature
/// reweight, then the seeded CDF walk) — so seeded GPU draws can be asserted
/// against it bit-for-bit whenever the distribution has clear margins.
///
/// `seed` is the FINAL per-position seed (the house `Sampler.seedFor`
/// mixing is applied by the caller, exactly as the GPU sampler does).
public enum K3SamplerReference {

    // MARK: - PRNG (bit-identical to sampling_k3.metal)

    public static func xorshift64(_ s: inout UInt64) -> UInt64 {
        s ^= s << 13
        s ^= s >> 7
        s ^= s << 17
        return s &* 2_685_821_657_736_338_717
    }

    public static func uniform01(_ s: inout UInt64) -> Float {
        let bits = UInt32(xorshift64(&s) >> 40)
        return Float(bits) * (1.0 / 16_777_216.0)
    }

    public static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E37_79B9_7F4A_7C15
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    public static func gumbel(seed: UInt64, position: UInt32, row: UInt32) -> Float {
        let key = seed ^ (UInt64(position) &* 0xD2B7_4407_B1CE_6E93) ^ UInt64(row)
        let r = splitmix64(key)
        let bits = UInt32(r >> 40)
        let u = (Float(bits) + 0.5) * (1.0 / 16_777_216.0)
        return -log(-log(u))
    }

    // MARK: - Distribution

    /// Max-subtracted fp32 softmax (the kernel's online reduction computes
    /// the same values up to last-ulp reduction order).
    public static func softmax(_ logits: [Float]) -> [Float] {
        guard let m = logits.max() else { return [] }
        var probs = logits.map { exp($0 - m) }
        let sum = probs.reduce(0, +)
        let inv = 1.0 / sum
        for i in probs.indices { probs[i] *= inv }
        return probs
    }

    /// Argmax with lowest-index tie-break (the kernel's simd_min rule).
    public static func argmax(_ probs: [Float]) -> UInt32 {
        var bestV: Float = -.infinity
        var bestI: UInt32 = 0
        for (i, p) in probs.enumerated() where p > bestV {
            bestV = p
            bestI = UInt32(i)
        }
        return bestI
    }

    /// HF repetition penalty, host-side, no softcap (K3 rule): positive
    /// logits divide by `penalty`, negative ones multiply.
    public static func applyRepetitionPenalty(logits: inout [Float],
                                              history: [Int32],
                                              penalty: Float) {
        guard penalty != 1.0 else { return }
        var seen = Set<Int32>()
        for id in history {
            guard id >= 0 && Int(id) < logits.count, seen.insert(id).inserted else { continue }
            let i = Int(id)
            logits[i] = logits[i] > 0 ? logits[i] / penalty : logits[i] * penalty
        }
    }

    /// Full sampling pipeline. `topK == 0` disables top-k; `topP >= 1`
    /// disables nucleus truncation; `temperature == 0` is greedy.
    public static func sample(logits: [Float],
                              temperature: Float,
                              topK: Int = 0,
                              topP: Float = 1.0,
                              seed: UInt64,
                              position: UInt32,
                              repetitionPenalty: Float = 1.0,
                              history: [Int32] = []) -> UInt32 {
        var edited = logits
        applyRepetitionPenalty(logits: &edited, history: history,
                               penalty: repetitionPenalty)
        let probs = softmax(edited)

        if temperature <= 0 {
            return argmax(probs)
        }

        // Fast path: no truncation. Gumbel-argmax with the kernel's stream.
        if topK == 0 && (topP <= 0 || topP >= 1) {
            let invT = 1.0 / temperature
            var bestV: Float = -.infinity
            var bestI: UInt32 = 0xFFFF_FFFF
            for (i, p) in probs.enumerated() where p > 0 {
                let s = invT * log(p) + gumbel(seed: seed, position: position,
                                               row: UInt32(i))
                if s > bestV {
                    bestV = s
                    bestI = UInt32(i)
                }
            }
            return bestI == 0xFFFF_FFFF ? 0 : bestI
        }

        // Truncating path: k repeated argmax passes (lowest-index ties).
        let v = probs.count
        let k = min(min(topK == 0 ? v : topK, 256), v)
        var topkVal = [Float](repeating: 0, count: k)
        var topkIdx = [UInt32](repeating: 0, count: k)
        var claimed = Set<UInt32>()
        claimed.reserveCapacity(k)
        for slot in 0..<k {
            var bestV: Float = -.infinity
            var bestI: UInt32 = 0xFFFF_FFFF
            for (i, p) in probs.enumerated() {
                if claimed.contains(UInt32(i)) { continue }
                if p > bestV {
                    bestV = p
                    bestI = UInt32(i)
                }
            }
            topkVal[slot] = bestV
            topkIdx[slot] = bestI
            claimed.insert(bestI)
        }

        var kept = 0
        // Drop trailing invalid slots (kernel rule: they sort last).
        while kept < k && topkIdx[kept] != 0xFFFF_FFFF && topkVal[kept].isFinite {
            kept += 1
        }
        // Top-p cut on the descending masses (still full-vocab normalized).
        if topP > 0 && topP < 1 {
            var cum: Float = 0
            var cut = kept
            for i in 0..<kept {
                cum += topkVal[i]
                if cum >= topP { cut = i + 1; break }
            }
            kept = cut
        }

        let invT = 1.0 / temperature
        if temperature != 1.0 {
            for i in 0..<kept { topkVal[i] = pow(topkVal[i], invT) }
        }

        var surviving: Float = 0
        for i in 0..<kept { surviving += topkVal[i] }

        var rng = seed
        _ = xorshift64(&rng)
        let u = uniform01(&rng) * surviving

        var run: Float = 0
        var picked = topkIdx[0]
        for i in 0..<kept {
            run += topkVal[i]
            if u <= run { picked = topkIdx[i]; break }
        }
        return picked
    }
}
