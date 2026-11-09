import Foundation
import Metal

/// The model's tail: collapse the 4-stream residual with the output
/// hyper-connection (read only - there is no block after it), then project the
/// 2560-wide vector through the Q6_K head to logits.
///
/// This family has no plain final norm; `output_hc_norm/down/up` is the mixer
/// that collapses the streams before `lm_head`, which is the same read path the
/// per-layer gates use with no inject.
package final class GGUFOutputStage {
    private let hc: HyperConnection
    private let head: Q6KGEMV
    private let arena: GGUFResidentArena
    private let headWeights: MTLBuffer
    private let vocab: Int
    private let hidden: Int
    private let streams: Int
    private let mixed: MTLBuffer
    package let logits: MTLBuffer

    package init(
        context: MetalContext, arch: ArchConfig, model: GGUFModel, arena: GGUFResidentArena
    ) throws {
        self.streams = arch.hyperConnections.count
        self.hidden = arch.hiddenSize
        self.vocab = arch.vocabSize
        self.arena = arena
        self.hc = try HyperConnection(
            context: context, dim: arch.hiddenSize, streams: arch.hyperConnections.count,
            lowRank: arch.hyperConnections.lowRank, weightBits: 4)
        self.head = try Q6KGEMV(context: context)

        guard let location = model.tensor(named: "output.weight") else {
            throw MetalError.invalidState("missing output.weight")
        }
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
        guard
            let headBuffer = context.device.makeBuffer(
                bytes: [UInt8](data), length: data.count, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("output.weight") }
        self.headWeights = headBuffer
        guard
            let mixed = context.device.makeBuffer(
                length: arch.hiddenSize * 2, options: .storageModeShared),
            let logits = context.device.makeBuffer(
                length: arch.vocabSize * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("output stage scratch") }
        self.mixed = mixed
        self.logits = logits
    }

    /// `streams` is the 4-stream residual (streams x hidden, half).
    package func encode(
        commandBuffer: MTLCommandBuffer,
        streamsBuffer: MTLBuffer,
        eps: Float = 1e-6
    ) throws {
        func weights(_ suffix: String) throws -> HyperConnection.Weights {
            guard let entry = arena.entries["output_hc_" + suffix] else {
                throw MetalError.invalidState("missing output_hc_\(suffix)")
            }
            return HyperConnection.Weights(
                weights: arena.buffer, weightsOffset: Int(entry.offset),
                scales: arena.buffer, biases: arena.buffer, isBF16: true)
        }
        guard let normEntry = arena.entries["output_hc_norm.weight"] else {
            throw MetalError.invalidState("missing output_hc_norm.weight")
        }
        try hc.encodeRead(
            commandBuffer: commandBuffer, streamsBuffer: streamsBuffer,
            hcNorm: arena.buffer, hcNormOffset: Int(normEntry.offset),
            down: try weights("down.weight"), up: try weights("up.weight"),
            blockInput: mixed, eps: eps)
        try head.encode(
            commandBuffer: commandBuffer, weights: headWeights, x: mixed,
            y: logits, rows: vocab, columns: hidden)
    }
}
