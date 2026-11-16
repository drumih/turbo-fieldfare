import Foundation

/// The vocabulary ids a constrained decode may pick from, as a bitset.
///
/// A grammar hands one of these to the sampler for every token position; the
/// sampler writes a floor into every logit the mask does not allow, exactly the
/// way it already writes the repetition penalty in place. One bit per
/// vocabulary id keeps the whole mask at 32 KiB for a 262,144-entry vocabulary
/// -- small enough to keep for a generation, cheap enough to rebuild per
/// position.
///
/// The mask is a *set*, not a distribution: it says which ids are possible,
/// never how likely they are. Ranking is the sampler's job and stays untouched.
public struct LogitMask: Sendable, Equatable {
    /// Ids at or above this are not in the vocabulary and are never allowed.
    public let vocab: Int
    private var words: [UInt64]

    public init(vocab: Int) {
        let count = max(0, vocab)
        self.vocab = count
        self.words = [UInt64](repeating: 0, count: (count + 63) / 64)
    }

    public mutating func removeAll() {
        for index in words.indices { words[index] = 0 }
    }

    public mutating func insert(_ id: Int32) {
        let index = Int(id)
        guard index >= 0, index < vocab else { return }
        words[index >> 6] |= 1 << UInt64(index & 63)
    }

    public func contains(_ id: Int32) -> Bool {
        let index = Int(id)
        guard index >= 0, index < vocab else { return false }
        return words[index >> 6] & (1 << UInt64(index & 63)) != 0
    }

    /// How many ids the mask allows.
    public var count: Int { words.reduce(0) { $0 + $1.nonzeroBitCount } }

    /// No id is allowed. A constrained decode in that state cannot continue.
    public var isEmpty: Bool { words.allSatisfy { $0 == 0 } }

    /// Whether the mask allows every id (nothing to do).
    public var isEverything: Bool { count == vocab }
}

/// The value written into a disallowed logit, per dtype.
///
/// FP16 -- the dtype of the GPU logits buffer -- cannot hold infinity, so the
/// floor is its largest finite magnitude. With the logit softcap every raw
/// value is folded through `softcap * tanh(x / softcap)`, which maps this floor
/// onto exactly `-softcap`, while any finite `x` maps strictly inside
/// `(-softcap, softcap)`. The separation therefore holds for every raw logit
/// down to the point where `tanh` saturates in FP16 (about 300 for the shipped
/// softcap of 30); below that a genuine logit ties with the floor, which is the
/// softcap's own resolution limit rather than something the mask introduces.
/// Model logits in the sampled row sit well inside that range.
public enum LogitMaskFloor {
    public static let half: Float16 = -Float16.greatestFiniteMagnitude
    public static let single: Float = -Float.greatestFiniteMagnitude
}

extension LogitMask {
    /// Floor every disallowed entry of an FP16 logits buffer in place, on the
    /// host, before the softmax front-end is encoded. Safe because the buffer
    /// belongs to the completed forward pass of the previous token.
    public func apply(toLogits logits: UnsafeMutablePointer<Float16>, count: Int) {
        for index in 0..<min(count, vocab) where !contains(Int32(index)) {
            logits[index] = LogitMaskFloor.half
        }
    }

    /// Floor every disallowed entry of a CPU logits row in place.
    public func apply(toLogits logits: inout [Float]) {
        for index in 0..<min(logits.count, vocab) where !contains(Int32(index)) {
            logits[index] = LogitMaskFloor.single
        }
    }
}
