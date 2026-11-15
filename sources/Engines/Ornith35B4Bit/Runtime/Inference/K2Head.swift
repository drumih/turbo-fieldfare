import Foundation
import Metal
import TinyTitanCore

/// K2-Horizon's input and output ends: the embedding lookup, the grouped final
/// norm and the untied `lm_head` GEMV.
///
/// The layer loop needs both ends before it can produce a first token; keeping
/// them together also makes the scale convention (K2's embedding is not
/// multiplied by `sqrt(hidden)`) a single readable place.
final class K2Head {
    private let config: ArchConfig
    private let model: Model
    private let rms: RMSNorm
    private let gemv4: AffineQuantGEMV
    private let gemv8: AffineQuantGEMV
    private let bf16: BF16GEMV
    private let embed4: AffineQuantEmbeddingLookup
    private let embed8: AffineQuantEmbeddingLookup
    private let normed: MTLBuffer
    private let logits: MTLBuffer

    init(model: Model, context: MetalContext) throws {
        self.model = model
        self.config = model.config
        self.rms = try RMSNorm(context: context)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)
        self.bf16 = try BF16GEMV(context: context)
        self.embed4 = try AffineQuantEmbeddingLookup(context: context, weightBits: 4)
        self.embed8 = try AffineQuantEmbeddingLookup(context: context, weightBits: 8)
        func buffer(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count * MemoryLayout<UInt16>.stride, 2),
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.normed = try buffer(config.hiddenSize, "k2.head.normed")
        self.logits = try buffer(config.vocabSize, "k2.head.logits")
    }

    /// Embedding lookup into `out` (`hiddenSize` halfs). K2 does not scale the
    /// embedding by `sqrt(hidden)`.
    func encodeEmbedding(
        commandBuffer: MTLCommandBuffer,
        tokenID: UInt32,
        out: MTLBuffer, outOffset: Int = 0,
        scale: Float = 1
    ) throws {
        let view = try model.embedding()
        let bits = weightBits(view, rows: config.vocabSize, columns: config.hiddenSize)
        let lookup = bits == 8 ? embed8 : embed4
        try lookup.encode(
            commandBuffer: commandBuffer,
            table: view.buffer, tableOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            out: out, outOffset: outOffset,
            tokenId: tokenID, d: UInt32(config.hiddenSize),
            outScale: scale
                * (config.embeddingScaledBySqrtHidden
                    ? Float(config.hiddenSize).squareRoot() : 1),
            vocab: UInt32(config.vocabSize))
    }

    /// Grouped final norm plus the untied head; writes `vocabSize` halfs.
    func encodeLogits(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0
    ) throws {
        let groups = max(1, config.normGroups)
        let norm = try model.finalNorm()
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: hidden, xOffset: hiddenOffset,
            weight: norm.buffer, weightOffset: Int(norm.offset),
            out: normed,
            groupDim: UInt32(config.hiddenSize / groups), numGroups: groups,
            eps: 1e-6)
        let head = try model.lmHead()
        try gemv(
            commandBuffer: commandBuffer, view: head,
            x: normed, y: logits,
            rows: config.vocabSize, columns: config.hiddenSize)
    }

    /// The logits buffer written by the last `encodeLogits`.
    func readLogits() -> [Float] {
        let base = logits.contents()
        return (0..<config.vocabSize).map {
            Float(
                base.load(
                    fromByteOffset: $0 * MemoryLayout<UInt16>.stride,
                    as: Float16.self))
        }
    }

    private func gemv(
        commandBuffer: MTLCommandBuffer,
        view: TensorView,
        x: MTLBuffer,
        y: MTLBuffer,
        rows: Int,
        columns: Int
    ) throws {
        if view.dtype == 1 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: view.buffer, weightsOffset: Int(view.offset),
                x: x, y: y, m: UInt32(rows), n: UInt32(columns))
            return
        }
        let bits = weightBits(view, rows: rows, columns: columns)
        let gemv = bits == 8 ? gemv8 : gemv4
        try gemv.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, y: y, m: UInt32(rows), n: UInt32(columns))
    }

    private func weightBits(
        _ view: TensorView, rows: Int, columns: Int
    ) -> Int {
        Int(view.length) * 8 / (rows * columns)
    }
}
