import Foundation
import Metal

/// The per-layer n-gram injection, driven from the checkpoint: the 16 rows this
/// token's n-grams hash to are gathered from the IQ4_NL table and decoded on the
/// host (2,560 values, cheaper than a dispatch), then the engine's PLE block
/// applies the key/value projections, the signed-sqrt gate and the dilated
/// convolution to the hyper streams.
///
/// The block's projections are bf16 in the arena and its norms and conv taps are
/// bfloat in the kernel, so nothing here needs a GGUF-specific kernel.
package final class GGUFPLEStage {
    private let block: PLEBlock
    private let setup: GGUFPLESetup
    private let arena: GGUFResidentArena
    private let model: GGUFModel
    private let tableFile: FileHandle
    private let tableFileOffset: UInt64
    private let rowBytes: Int
    private let layer: Int

    package init(
        context: MetalContext, arch: ArchConfig, model: GGUFModel, arena: GGUFResidentArena
    ) throws {
        self.setup = try GGUFPLESetup.read(from: model)
        guard let pleLayer = setup.layerIndices.first else {
            throw MetalError.invalidState("the checkpoint declares no PLE layer")
        }
        self.layer = pleLayer
        self.block = try PLEBlock(
            context: context, dim: arch.hiddenSize, streams: arch.hyperConnections.count,
            embedDim: setup.embedDim, kernelSize: setup.convKernelSize,
            dilation: setup.ngramSize, maxRows: 1)
        self.arena = arena
        self.model = model
        guard let table = model.tensor(named: "per_layer_token_embd.weight") else {
            throw MetalError.invalidState("missing per_layer_token_embd.weight")
        }
        let shard = model.shards[table.shardIndex]
        self.tableFile = try FileHandle(forReadingFrom: shard.url)
        self.tableFileOffset = shard.dataOffset + table.tensor.offset
        self.rowBytes = Int(table.tensor.dimensions[0]) / 32 * 18
    }

    deinit { try? tableFile.close() }

    /// The layer this block runs on (0-based).
    package var layerIndex: Int { layer }

    /// Clears the block's causal-conv history, for a new sequence: the conv
    /// carries rows from token to token, so a second prompt run without this
    /// starts from the first prompt's tail.
    package func resetState() {
        block.resetState()
    }

    /// The hash context for the token at `position` of `sequence`: the token,
    /// then up to `ngram_size - 1` predecessors, newest first - the order
    /// `PLEHash.rows(context:)` reads. A predecessor before the sequence start
    /// is simply absent, which the hash resolves to EOS, as llama.cpp's
    /// `llm_graph_input_qwen4exp_ple::set_input` does.
    ///
    /// The window comes from the checkpoint's `ple.ngram_size` (3 on
    /// Qwen3.8-Flash-Next), so a caller never hardcodes how far back it reaches.
    package func history(of sequence: [Int32], at position: Int) -> [Int32] {
        Self.window(of: sequence, at: position, ngramSize: setup.ngramSize)
    }

    /// The rule `history(of:at:)` applies, for an explicit n-gram size.
    package static func window(of sequence: [Int32], at position: Int, ngramSize: Int) -> [Int32] {
        precondition(ngramSize >= 1, "an n-gram covers at least the token itself")
        precondition(
            position >= 0 && position < sequence.count,
            "position \(position) is outside a \(sequence.count)-token sequence")
        let oldest = max(0, position - (ngramSize - 1))
        return Array(sequence[oldest...position].reversed())
    }

    /// Gathers the token's rows into the block's embedding buffer as fp16.
    package func gather(context tokens: [Int32]) throws {
        let rows = setup.rows(context: tokens)
        var embedding = [Float](repeating: 0, count: setup.embedDim)
        for (index, row) in rows.enumerated() {
            try tableFile.seek(toOffset: tableFileOffset + UInt64(row) * UInt64(rowBytes))
            guard let data = try tableFile.read(upToCount: rowBytes) else {
                throw MetalError.invalidState("short read of PLE row \(row)")
            }
            let values = try GGUFDequant.decode(type: .iq4NL, bytes: [UInt8](data))
            let base = index * 160
            for (offset, value) in values.enumerated() { embedding[base + offset] = value }
        }
        var bits: [UInt16] = []
        bits.reserveCapacity(embedding.count)
        for value in embedding { bits.append(Float16(value).bitPattern) }
        bits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            memcpy(block.embedding.contents(), base, raw.count)
        }
    }

    /// The gathered rows, as fp16 bit patterns - what `gather` just wrote.
    package func gatheredBits() -> [UInt16] {
        let pointer = block.embedding.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<setup.embedDim).map { pointer[$0] }
    }

    package func encode(
        commandBuffer: MTLCommandBuffer, streamsBuffer: MTLBuffer, eps: Float = 1e-6
    ) throws {
        let prefix = "blk.\(layer)."
        func offset(_ suffix: String) throws -> Int {
            guard let entry = arena.entries[prefix + suffix] else {
                throw MetalError.invalidState("missing \(prefix + suffix) in the arena")
            }
            return Int(entry.offset)
        }
        func projection(_ suffix: String) throws -> PLEBlock.Projection {
            PLEBlock.Projection(
                weights: arena.buffer, weightsOffset: try offset(suffix),
                scales: arena.buffer, scalesOffset: 0,
                biases: arena.buffer, biasesOffset: 0, isBF16: true)
        }
        let weights = PLEBlock.Weights(
            keyProj: try projection("ple_key.weight"),
            valueProj: try projection("ple_value.weight"),
            normKey: PLEBlock.Vector(
                buffer: arena.buffer, offset: try offset("ple_norm_key.weight")),
            normQuery: PLEBlock.Vector(
                buffer: arena.buffer, offset: try offset("ple_norm_query.weight")),
            normConv: PLEBlock.Vector(
                buffer: arena.buffer, offset: try offset("ple_norm_conv.weight")),
            conv1d: PLEBlock.Vector(buffer: arena.buffer, offset: try offset("ple_conv1d.weight")))
        try block.encodeDecode(
            commandBuffer: commandBuffer, streamsBuffer: streamsBuffer,
            weights: weights, eps: eps)
    }
}
