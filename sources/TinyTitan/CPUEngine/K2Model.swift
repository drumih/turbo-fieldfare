import Foundation
import TinyTitanCore

/// K2-Horizon on the CPU.
///
/// This is the first runnable path for the architecture: one token at a time,
/// with the snapshot's matrices read through the same 4/8-bit GEMV kernels the
/// dense CPU engine uses. The sparse layers read only their routed experts, so
/// a token touches about 4.5B parameters rather than the 36B stored.
///
/// The layer shape comes from `modeling_k2_horizon.py`:
///
/// - layers in `mlp_only_layers` are dense: standard GQA attention with a
///   softplus gate, and a SwiGLU MLP;
/// - the other layers use MoVA attention (the value states come from four
///   routed value experts) and a sparse MoE (100 experts at top-8 plus a
///   shared expert), both with a sigmoid router whose bias only affects
///   selection, top-k weights normalized and scaled by 2.5;
/// - every norm is a grouped RMS norm over `layernorm_num_groups` groups.
///
/// The numbers are the checkpoint's; nothing here is tuned.
public final class K2Model {

    public let snapshot: K2Snapshot
    public let configuration: K2Snapshot.Configuration
    /// Rows of each GEMV to split across the performance cores.
    public var threads: Int

    /// Per-layer KV cache, flat `[position][kvHeads * headDim]`.
    var keys: [[Float]] = []
    var values: [[Float]] = []
    public private(set) var position = 0

    public init(snapshot: K2Snapshot, threads: Int? = nil) throws {
        self.snapshot = snapshot
        configuration = snapshot.configuration
        self.threads = threads ?? Int8AffineGEMV.preferredThreads
        keys = Array(repeating: [], count: configuration.layers)
        values = Array(repeating: [], count: configuration.layers)
        guard configuration.hiddenSize % configuration.normGroups == 0 else {
            throw ModelError.internalInconsistency(
                detail: "hidden size \(configuration.hiddenSize) is not divisible by "
                    + "\(configuration.normGroups) norm groups")
        }
        guard configuration.heads % configuration.keyValueHeads == 0 else {
            throw ModelError.internalInconsistency(
                detail: "\(configuration.heads)/\(configuration.keyValueHeads) heads is not whole")
        }
    }

    public func reset() {
        for layer in keys.indices {
            keys[layer].removeAll(keepingCapacity: true)
            values[layer].removeAll(keepingCapacity: true)
        }
        position = 0
    }

    /// One token, returning the logits over the vocabulary.
    @discardableResult
    public func step(token: Int, needsLogits: Bool = true) throws -> [Float] {
        var hidden = try snapshot.row("model.embed_tokens.weight", token)
        for layer in 0..<configuration.layers {
            var normalized = groupedNorm(
                hidden, weight: try snapshot.floats(layerName(layer, "input_layernorm.weight")))
            let dense = configuration.denseLayers.contains(layer)
            let mixed =
                dense
                ? try attention(layer: layer, x: normalized)
                : try movaAttention(layer: layer, x: normalized)
            for index in hidden.indices { hidden[index] += mixed[index] }

            normalized = groupedNorm(
                hidden,
                weight: try snapshot.floats(layerName(layer, "post_attention_layernorm.weight")))
            let feed =
                dense
                ? try denseMLP(layer: layer, x: normalized)
                : try sparseMoE(layer: layer, x: normalized)
            for index in hidden.indices { hidden[index] += feed[index] }
        }
        position += 1
        guard needsLogits else { return [] }
        let final = groupedNorm(hidden, weight: try snapshot.floats("model.norm.weight"))
        return try project("lm_head.weight", final)
    }

    /// Greedy continuation, for benchmarks and the correctness checks.
    public func generate(
        prompt: [Int], maximumTokens: Int, onToken: ((Int) -> Bool)? = nil
    ) throws -> [Int] {
        precondition(!prompt.isEmpty, "a generation needs a prompt")
        var logits: [Float] = []
        for (index, token) in prompt.enumerated() {
            logits = try step(token: token, needsLogits: index == prompt.count - 1)
        }
        var produced: [Int] = []
        for _ in 0..<maximumTokens {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            if configuration.eosTokens.contains(best) { break }
            produced.append(best)
            if let onToken, !onToken(best) { break }
            logits = try step(token: best)
        }
        return produced
    }

    // MARK: - shared

    func layerName(_ layer: Int, _ suffix: String) -> String {
        "model.layers.\(layer).\(suffix)"
    }

    /// `y = W x` through the quantized GEMV kernels.
    func project(_ name: String, _ x: [Float]) throws -> [Float] {
        let matrix = try snapshot.matrix(name)
        guard matrix.columns == x.count else {
            throw ModelError.internalInconsistency(
                detail: "\(name): \(matrix.columns) columns for \(x.count) activations")
        }
        var out = [Float](repeating: 0, count: matrix.rows)
        try x.withUnsafeBufferPointer { input in
            try out.withUnsafeMutableBufferPointer { output in
                guard let inputBase = input.baseAddress, let outputBase = output.baseAddress else {
                    return
                }
                try CPUOps.gemv(
                    matrix, x: inputBase, out: outputBase, threads: threads)
            }
        }
        return out
    }

    /// `y = W x` for a tensor the converter kept at bf16 (the routers): a
    /// dense matvec over the widened floats, not the quantized kernels.
    func projectFloat(_ name: String, _ x: [Float]) throws -> [Float] {
        let values = try snapshot.floats(name)
        guard !x.isEmpty, values.count % x.count == 0 else {
            throw ModelError.internalInconsistency(
                detail: "\(name): \(values.count) values for \(x.count) activations")
        }
        let rows = values.count / x.count
        var out = [Float](repeating: 0, count: rows)
        values.withUnsafeBufferPointer { weights in
            x.withUnsafeBufferPointer { input in
                for row in 0..<rows {
                    var total: Float = 0
                    let base = row * x.count
                    for column in 0..<x.count {
                        total += weights[base + column] * input[column]
                    }
                    out[row] = total
                }
            }
        }
        return out
    }

    /// Grouped RMS norm: one variance per group of `hidden / groups`, then the
    /// per-element weight.
    func groupedNorm(_ x: [Float], weight: [Float]) -> [Float] {
        let groups = max(1, configuration.normGroups)
        let width = x.count / groups
        var out = [Float](repeating: 0, count: x.count)
        for group in 0..<groups {
            let base = group * width
            var sum: Float = 0
            for index in 0..<width {
                let value = x[base + index]
                sum += value * value
            }
            let scale = 1 / (sum / Float(width) + configuration.epsilon).squareRoot()
            for index in 0..<width {
                out[base + index] = x[base + index] * scale * weight[base + index]
            }
        }
        return out
    }

    /// `sigmoid` and `silu` in one place; the scalar versions are hot enough to
    /// keep inline.
    @inline(__always)
    func sigmoid(_ x: Float) -> Float { 1 / (1 + expf(-x)) }

    @inline(__always)
    func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

    /// The checkpoint's `softplus(., beta = ln 2)`.
    func softplusGate(_ x: Float) -> Float {
        let beta = Float(log(2.0))
        let scaled = beta * x
        return (max(scaled, 0) + log1pf(expf(-abs(scaled)))) / beta
    }

    /// Top-k indices by score, descending; ties keep the lower index, which is
    /// PyTorch's `topk` order.
    func topK(_ scores: [Float], _ k: Int) -> [Int] {
        scores.indices.sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }.prefix(k).map { $0 }
    }
}
