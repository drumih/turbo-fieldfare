import Foundation

// The family-specific configuration structs: hyper-connection residuals, the
// sparse indexer, the hashed n-gram block and Gated-DeltaNet dimensions.
//
// Split out of `ModelTypes.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
/// Hyper-connection residual configuration (Qwen3.8-Flash-Next). The residual
/// stream is `count` parallel 2560-wide streams; each sublayer mixes them to
/// one block input through a low-rank gate and injects its output back into
/// every stream with learned per-stream weights. Zeroed for plain-residual
/// architectures.
public struct HyperConnectionConfig: Sendable, Equatable {
    public let count: Int
    public let lowRank: Int

    public init(count: Int, lowRank: Int) {
        self.count = count
        self.lowRank = lowRank
    }

    public static let none = HyperConnectionConfig(count: 0, lowRank: 0)
    public var enabled: Bool { count > 0 }
}

/// Qwen Sparse Attention indexer configuration. Each full-attention layer
/// scores mean-pooled key blocks with a small MQA head set and keeps the
/// `budget` highest-scoring visible tokens (plus the incomplete tail block).
/// Dense attention is exact whenever a query sees at most `budget` keys plus
/// the tail — the runtime's dense path is gated on that window. Zeroed for
/// dense-attention architectures.
public struct SparseIndexerConfig: Sendable, Equatable {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let budget: Int
    public let compressRatio: Int

    public init(
        numHeads: Int, numKVHeads: Int, headDim: Int,
        budget: Int, compressRatio: Int
    ) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.budget = budget
        self.compressRatio = compressRatio
    }

    public static let none = SparseIndexerConfig(
        numHeads: 0, numKVHeads: 0, headDim: 0, budget: 0, compressRatio: 0)
    public var enabled: Bool { budget > 0 }
}

/// Hashed n-gram "per-layer embedding" configuration (Qwen3.8-Flash-Next).
/// At each layer in `layerIndices` (0-based), every token gathers
/// `(ngramSize - 1) * headsPerNgram` rows from a prime-partitioned hashed
/// table and injects them into the hyper streams through key/value
/// projections and a dilated depthwise conv. The multipliers, per-head vocab
/// sizes, and offsets ship as checkpoint tensors and are read, not
/// re-derived; `seed` is recorded for validation only. Zeroed when absent.
public struct PLEConfig: Sendable, Equatable {
    public let layerIndices: [Int]
    public let embedDim: Int
    public let convKernelSize: Int
    public let ngramSize: Int
    public let vocabSizeBase: Int
    public let headsPerNgram: Int
    public let vocabDivisor: Int
    public let seed: Int

    public init(
        layerIndices: [Int], embedDim: Int, convKernelSize: Int,
        ngramSize: Int, vocabSizeBase: Int, headsPerNgram: Int,
        vocabDivisor: Int, seed: Int
    ) {
        self.layerIndices = layerIndices
        self.embedDim = embedDim
        self.convKernelSize = convKernelSize
        self.ngramSize = ngramSize
        self.vocabSizeBase = vocabSizeBase
        self.headsPerNgram = headsPerNgram
        self.vocabDivisor = vocabDivisor
        self.seed = seed
    }

    public static let none = PLEConfig(
        layerIndices: [], embedDim: 0, convKernelSize: 0, ngramSize: 0,
        vocabSizeBase: 0, headsPerNgram: 0, vocabDivisor: 0, seed: 0)
    public var enabled: Bool { !layerIndices.isEmpty }
    /// Lookups per token: (ngramSize - 1) n-gram orders x headsPerNgram.
    public var ngramHeads: Int { (ngramSize - 1) * headsPerNgram }
    /// Row width of the hashed table.
    public var headDim: Int { ngramHeads > 0 ? embedDim / ngramHeads : 0 }
}

/// The elementwise function a family applies to the attention output gate.
///
/// `attnOutputGate` says whether a gate exists; this says which function turns
/// the gate projection into a multiplier. Qwen's families use `sigmoid`, where
/// reading a K2 install as `softplus` would still produce bounded, positive
/// gates and fluent nonsense rather than a shape error -- so it is declared
/// per family and read from the manifest, never assumed.
public enum AttentionGateFunction: String, Sendable, Equatable {
    case none
    case sigmoid
    case silu
    case softplus
}

/// K2-Horizon's MoVA (mixture of value experts) attention block.
///
/// Every sparse layer replaces `v_proj` with a router over `numExperts` linear
/// value experts: the selected `topKExperts` outputs pass through `activation`
/// before the weighted sum. Routing is sigmoid, the router's bias is added only
/// to the selection scores, and the chosen weights are scaled by
/// `routerScaling`. Zeroed for architectures without MoVA.
public struct MoVAConfig: Sendable, Equatable {
    public let numExperts: Int
    public let topKExperts: Int
    public let routerScaling: Double
    public let routerHasBias: Bool
    public let activation: String

    public init(
        numExperts: Int,
        topKExperts: Int,
        routerScaling: Double,
        routerHasBias: Bool,
        activation: String
    ) {
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.routerScaling = routerScaling
        self.routerHasBias = routerHasBias
        self.activation = activation
    }

    public static let none = MoVAConfig(
        numExperts: 0, topKExperts: 0,
        routerScaling: 1, routerHasBias: false, activation: "silu")
    public var enabled: Bool { numExperts > 0 }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    /// Nonlinearity applied to `z` before it scales the normalized delta
    /// readout. Qwen 3.6 (like Qwen3-Next) uses silu; Qwen3.8-Flash-Next uses
    /// sigmoid, which its config states as `output_gate_type`. Both are
    /// smooth and positive-ish, so picking the wrong one produces confident
    /// nonsense rather than an error -- it is declared per family, never
    /// defaulted from the other one.
    public enum OutputGate: String, Sendable, Equatable {
        case silu
        case sigmoid
    }

    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int
    public let outputGate: OutputGate

    public init(
        numKHeads: Int, numVHeads: Int,
        keyHeadDim: Int, valueHeadDim: Int,
        convKernelSize: Int,
        outputGate: OutputGate = .silu
    ) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
        self.outputGate = outputGate
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}
