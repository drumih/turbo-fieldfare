import Foundation

/// The stable log-softmax / cross-entropy formulation from arXiv:2610.31642.
///
/// The standard `-z[target] + log(sum(exp(z)))` loses the target's margin once
/// it passes the precision cliff: adding `1` to a tiny `exp(-margin)` rounds the
/// term away in the working precision, which is what makes the gradient collapse
/// in training (Softmax Collapse) and makes `q - p` lose precision near the
/// teacher (Softmax Cancellation). No surface in this tree consumes gradients
/// today, but the forward quantities are what a logprobs or perplexity surface
/// would report, so the formulation is adopted before such a surface exists.
///
/// The margin is taken against the log-sum-exp of the *other* entries, with the
/// argmax excluded, and the loss is `softplus(-margin)`. The paper's own case is
/// pinned by `StableLogSoftmaxTests`: `z = [9, -9, -9]`, target `0`, where the
/// standard float implementation returns `0` and this returns `3.046e-8`.
public enum StableLogSoftmax {
    /// `log softmax(logits)`, shifted by the maximum for overflow safety.
    public static func logSoftmax(_ logits: [Float]) -> [Float] {
        guard let peak = logits.max(), peak.isFinite else {
            return logits.map { _ in -.infinity }
        }
        var total = 0.0
        for value in logits { total += exp(Double(value) - Double(peak)) }
        let offset = Double(peak) + log(total)
        return logits.map { Float(Double($0) - offset) }
    }

    /// `-log softmax(logits)[target]`, in the stable margin form.
    ///
    /// Returns `infinity` for a target outside the vocabulary or a non-finite
    /// logit; a target that is the only token has loss zero.
    public static func negativeLogLikelihood(_ logits: [Float], target: Int) -> Double {
        guard logits.indices.contains(target) else { return .infinity }
        guard logits[target].isFinite else { return .infinity }
        guard logits.count > 1 else { return 0 }
        var peakOther = -Double.infinity
        for (index, value) in logits.enumerated() where index != target {
            let candidate = Double(value)
            guard candidate.isFinite else { return .infinity }
            if candidate > peakOther { peakOther = candidate }
        }
        guard peakOther.isFinite else { return .infinity }
        var total = 0.0
        for (index, value) in logits.enumerated() where index != target {
            total += exp(Double(value) - peakOther)
        }
        let logSumOther = peakOther + log(total)
        let margin = Double(logits[target]) - logSumOther
        return softplus(-margin)
    }

    /// `log(1 + exp(x))`, evaluated without overflow in either tail.
    private static func softplus(_ x: Double) -> Double {
        if x > 0 { return x + log1p(exp(-x)) }
        return log1p(exp(x))
    }
}
