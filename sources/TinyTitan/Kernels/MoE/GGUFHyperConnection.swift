import Foundation
import Metal

/// The gated residual, with its two projections read from the mapped blocks.
///
/// The engine's `HyperConnection` takes affine or bf16 weights, so on this
/// checkpoint it forced a 1.27 GB bf16 copy into the arena. Everything else it
/// does is engine kernels: the grouped RMS norm, the SiLU on the low-rank gate,
/// the stream reduce and the stream inject. This keeps those and replaces only
/// the two GEMVs with the fused ones, so the Q8_0 blocks are read in place.
package final class GGUFHyperConnection {
    private let rms: RMSNorm
    private let bf16: BF16GEMV
    private let fused: GGUFExpertGEMV
    private let elementwise: Elementwise
    private let arena: GGUFResidentArena
    private let mapped: GGUFMappedShards
    private let dim: Int
    private let streams: Int
    private let lowRank: Int

    private let normed: MTLBuffer
    private let low: MTLBuffer
    private let mix: MTLBuffer
    private let inject: MTLBuffer

    package init(
        context: MetalContext, arch: ArchConfig, arena: GGUFResidentArena,
        mapped: GGUFMappedShards
    ) throws {
        self.rms = try RMSNorm(context: context)
        self.bf16 = try BF16GEMV(context: context)
        self.fused = try GGUFExpertGEMV(context: context)
        self.elementwise = try Elementwise(context: context)
        self.arena = arena
        self.mapped = mapped
        self.dim = arch.hiddenSize
        self.streams = arch.hyperConnections.count
        self.lowRank = arch.hyperConnections.lowRank
        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: bytes, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            return buffer
        }
        let wide = dim * streams * 2
        self.normed = try make(wide, "hc normed")
        self.mix = try make(wide, "hc mix")
        self.low = try make(lowRank * 2, "hc low")
        self.inject = try make(max(streams, 8) * 2, "hc inject")
    }

    private enum Sublayer: String {
        case attention = "attn"
        case ffn = "ffn"
    }

    private func arenaOffset(_ prefix: String, _ suffix: String) throws -> Int {
        guard let entry = arena.entries[prefix + suffix] else {
            throw MetalError.invalidState("missing \(prefix + suffix) in the arena")
        }
        return Int(entry.offset)
    }

    private func mappedWeight(_ prefix: String, _ suffix: String) throws -> GGUFMappedShards.Weight
    {
        guard let weight = try mapped.weight(named: prefix + suffix) else {
            throw MetalError.invalidState("missing \(prefix + suffix) in the shards")
        }
        return weight
    }

    /// Bytes of the normalised residual a read leaves for its write.
    package var normedBytes: Int { dim * streams * 2 }

    /// Residual -> block input, matching `HyperConnection.encodeRead`. The
    /// normalised residual lands in `normedOut` when given (a caller that
    /// interleaves several reads before their writes keeps one per token),
    /// else in the stage's own buffer.
    package func encodeRead(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        sublayer: String,
        streamsBuffer: MTLBuffer,
        out: MTLBuffer,
        normedOut: MTLBuffer? = nil,
        eps: Float = 1e-6
    ) throws {
        let normed = normedOut ?? self.normed
        let prefix = "blk.\(layer).hc_\(sublayer)_"
        let wide = dim * streams
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: streamsBuffer,
            weight: arena.buffer,
            weightOffset: try arenaOffset(prefix, "norm.weight"),
            out: normed,
            groupDim: UInt32(dim), numGroups: streams, eps: eps)
        let down = try mappedWeight(prefix, "down.weight")
        try fused.encode(
            commandBuffer: commandBuffer, weights: down.buffer,
            weightsOffset: down.offset, x: normed, y: low,
            rows: lowRank, columns: wide, type: down.type)
        try elementwise.encodeSilu(
            commandBuffer: commandBuffer, x: low, out: low, count: lowRank,
            inScale: 1 / Float(streams))
        let up = try mappedWeight(prefix, "up.weight")
        try fused.encode(
            commandBuffer: commandBuffer, weights: up.buffer,
            weightsOffset: up.offset, x: low, y: mix,
            rows: wide, columns: lowRank, type: up.type)
        try elementwise.encodeHCMixReduce(
            commandBuffer: commandBuffer, mix: mix, normed: normed,
            out: out, dim: dim, streams: streams, inScale: 1)
    }

    /// Block output -> residual, matching `HyperConnection.encodeWrite`. Must
    /// follow its matching read: it consumes the normalised residual that read
    /// left, in `normed` when the read was given one.
    package func encodeWrite(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        sublayer: String,
        streamsBuffer: MTLBuffer,
        blockOut: MTLBuffer,
        normed: MTLBuffer? = nil
    ) throws {
        let normed = normed ?? self.normed
        let prefix = "blk.\(layer).hc_\(sublayer)_"
        // The inject projection is tiny (streams x wide) and F32 in the file;
        // the arena carries its bf16 copy, which BF16GEMV reads.
        let injectWeight = try arenaOffset(prefix, "inject.weight")
        try bf16.encode(
            commandBuffer: commandBuffer, weights: arena.buffer,
            weightsOffset: injectWeight, x: normed, y: inject,
            m: UInt32(streams), n: UInt32(dim * streams))
        try elementwise.encodeHCInject(
            commandBuffer: commandBuffer, streams: streamsBuffer,
            blockOut: blockOut, inject: inject,
            dim: dim, streamCount: streams, inScale: 1 / Float(streams))
    }
}
