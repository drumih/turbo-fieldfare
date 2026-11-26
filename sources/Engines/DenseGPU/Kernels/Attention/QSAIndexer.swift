import Foundation
import Metal
import TinyTitanCore

/// Qwen Sparse Attention's block indexer.
///
/// A small head set (4 heads of 128 against the attention's 24 of 256) scores
/// mean-pooled blocks of keys, and the attention is then restricted to the
/// highest-scoring blocks plus the ragged tail of the current query's own
/// block. Below `QSAExactness.maximumExactVisibleKeys` the selection is
/// vacuous and dense attention computes the same thing; past it, this is what
/// makes the model faithful rather than merely fluent.
///
/// Two caches, both per QSA layer:
///
/// - the **raw** indexer keys, cached before their norm and rope exactly as
///   the reference does, because a block's pooled vector is the mean of raw
///   members and normalizing first would change what is averaged;
/// - the **pooled** block vectors, post-norm and post-rope. Only the block
///   holding the newest token changes on a decode step, so a step repools one
///   block and leaves the history alone.
///
/// The block's rope position is `blockIndex * compressRatio` -- the position
/// of its first member, not the query's.
final class QSAIndexer {
    /// Cells each layer's key buffers are sized for before any growth.
    ///
    /// 8192 for the same reason the KV cache starts at
    /// `KVCacheManager.initialCapacityTokens`: it holds an ordinary
    /// conversation without a single grow, and the per-layer cost is 2.5 MB
    /// against the 84 MB a full 262144-cell reservation wires on first use.
    /// That reservation is what kept 96 expert-cache slots from fitting at the
    /// shipped context on a 24 GB machine — measured in
    /// `docs/qwen38-decode-12gb-measurements.md` §5o.
    static let initialCapacityCells = 8_192

    struct Weights {
        let queryProjection: TensorView
        let keyProjection: TensorView
        let queryNorm: TensorView
        let keyNorm: TensorView
    }

    let heads: Int
    let headDim: Int
    let compressRatio: Int
    let budget: Int
    let ropeTheta: Float

    let ctx: MetalContext
    private let poolPSO: MTLComputePipelineState
    let poolRangePSO: MTLComputePipelineState
    private let scorePSO: MTLComputePipelineState
    let scoreRowsPSO: MTLComputePipelineState
    /// Decode selection on the GPU; nil when the kernel is unavailable.
    let selectPSO: MTLComputePipelineState?
    let rms: RMSNorm
    let rope: RoPE
    let gemv: SlotGEMV

    /// `[heads * headDim]` query scratch, normed and roped in place.
    private let queryBuf: MTLBuffer
    /// One token's raw key, before it joins the cache.
    private let keyBuf: MTLBuffer
    /// Per layer: `[capacity, headDim]` raw keys and
    /// `[capacity / compressRatio + 1, headDim]` pooled blocks.
    var rawKeys: [Int: MTLBuffer] = [:]
    var pooled: [Int: MTLBuffer] = [:]
    /// `[n_blocks]` float scores, and the `[capacity]` byte selection.
    var scoresBuf: MTLBuffer
    var keepBuf: MTLBuffer
    /// Compacted selection: `selectionWidth` ascending key indices per query,
    /// plus how many of them are valid. The mask form costs the attention
    /// kernel one byte-load per *visible* key per query per head, which is
    /// O(context) work to discover an O(budget) answer -- measured at 4.53s per
    /// layer-chunk at 10k context against 1.42s at 2.4k, where context and
    /// budget coincide. Compacted, the loop is the budget regardless of depth.
    var keepIndexBuf: MTLBuffer?
    var keepCountBuf: MTLBuffer?
    private(set) var capacity: Int
    /// Chunk-sized scratch, grown on demand: a prefill chunk needs a query
    /// row and a selection row per token, which decode does not.
    var queryRowsBuf: MTLBuffer?
    var lastScoredBlocks = 0
    /// Blocks scored per query row by the most recent chunk pass.
    var scoredBlocksPerRow: Int { lastScoredBlocks }

    /// Cells a query keeps: the budget plus the tail block's extra members.
    var selectionWidth: Int { budget + compressRatio - 1 }

    init(
        context: MetalContext, config: SparseIndexerConfig,
        budget: Int? = nil,
        ropeTheta: Float, capacity: Int, weightBits: Int = 4
    ) throws {
        precondition(config.enabled, "QSAIndexer requires a configured indexer")
        precondition(capacity > 0)
        self.ctx = context
        self.heads = config.numHeads
        self.headDim = config.headDim
        self.compressRatio = config.compressRatio
        self.budget = budget ?? config.budget
        self.ropeTheta = ropeTheta
        self.capacity = capacity
        self.poolPSO = try context.pipeline("qsa_pool_block")
        self.poolRangePSO = try context.pipeline("qsa_pool_blocks")
        self.scorePSO = try context.pipeline("qsa_block_scores")
        self.scoreRowsPSO = try context.pipeline("qsa_block_scores_rows")
        self.selectPSO = try? context.pipeline("qsa_select_decode")
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.gemv = try SlotGEMV(context: context, weightBits: weightBits)
        let f16 = MemoryLayout<Float16>.stride
        guard
            let query = context.device.makeBuffer(
                length: heads * headDim * f16, options: .storageModeShared),
            let key = context.device.makeBuffer(
                length: headDim * f16, options: .storageModeShared),
            let scores = context.device.makeBuffer(
                length: max(1, capacity / compressRatio + 1)
                    * MemoryLayout<Float>.stride,
                options: .storageModeShared),
            let keep = context.device.makeBuffer(
                length: capacity, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("QSA indexer scratch")
        }
        self.queryBuf = query
        self.keyBuf = key
        self.scoresBuf = scores
        self.keepBuf = keep
    }

    /// The most recent selection and the scores behind it, for A/B-ing two
    /// paths that should agree.
    /// The pooled block cache for one layer, post-norm and post-rope.
    func debugPooled(layer: Int, blocks: Int) -> [Float]? {
        guard let pool = pooled[layer] else { return nil }
        let ptr = pool.contents().bindMemory(
            to: Float16.self,
            capacity: blocks * headDim)
        return (0..<blocks * headDim).map { Float(ptr[$0]) }
    }

    /// The raw indexer keys for one layer, before norm and rope.
    func debugRawKeys(layer: Int, count: Int) -> [Float]? {
        guard let raw = rawKeys[layer] else { return nil }
        let ptr = raw.contents().bindMemory(
            to: Float16.self,
            capacity: count * headDim)
        return (0..<count * headDim).map { Float(ptr[$0]) }
    }

    func debugSnapshot(cells: Int, blocks: Int) -> (keep: [UInt8], scores: [Float]) {
        let k = keepBuf.contents().bindMemory(to: UInt8.self, capacity: cells)
        let sc = scoresBuf.contents().bindMemory(to: Float.self, capacity: blocks)
        return ((0..<cells).map { k[$0] }, (0..<blocks).map { sc[$0] })
    }

    /// The block-score scratch, so a test can drive the selection with known
    /// rankings instead of whatever the scoring kernel happens to produce.
    var scoresForTesting: MTLBuffer { scoresBuf }

    /// The selection buffer the attention reads, or nil when every visible
    /// key is kept and the mask would be a no-op.
    var keepMask: MTLBuffer { keepBuf }

    func reset() {
        rawKeys.removeAll()
        pooled.removeAll()
    }

    func encodePool(
        commandBuffer: MTLCommandBuffer,
        buffers: (raw: MTLBuffer, pooled: MTLBuffer),
        block: Int, first: Int, count: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(poolPSO)
        enc.setBuffer(buffers.raw, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        var d = UInt32(headDim)
        var b = UInt32(block)
        var f = UInt32(first)
        var c = UInt32(count)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&f, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
        let w = min(poolPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: headDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Scores every block against this token's indexer query.
    func encodeScores(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        weights: Weights, layer: Int, position: Int,
        eps: Float
    ) throws {
        let buffers = try layerBuffers(layer, hold: position + 1)
        let query = weights.queryProjection
        try gemv.encode(
            commandBuffer: commandBuffer,
            weights: query.buffer, weightsOffset: Int(query.offset),
            scales: query.buffer, scalesOffset: Int(query.scaleOffset),
            biases: query.buffer, biasesOffset: Int(query.biasOffset),
            x: hidden, xOffset: hiddenOffset,
            y: queryBuf,
            m: UInt32(heads * headDim), n: UInt32(hiddenColumns(query)),
            isBF16: query.dtype == 1)
        try rms.encodeBF16WPerHead(
            commandBuffer: commandBuffer,
            x: queryBuf,
            weight: weights.queryNorm.buffer,
            weightOffset: Int(weights.queryNorm.offset),
            out: queryBuf,
            headDim: UInt32(headDim), numHeads: heads,
            eps: eps)
        try rope.encodeNeoxSubdim(
            commandBuffer: commandBuffer,
            data: queryBuf,
            position: UInt32(position),
            headDim: UInt32(headDim), numHeads: UInt32(heads),
            rotaryDim: UInt32(headDim), theta: ropeTheta)

        let blocks = position / compressRatio + 1
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(scorePSO)
        enc.setBuffer(queryBuf, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        var d = UInt32(headDim)
        var h = UInt32(heads)
        var b = UInt32(blocks)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&h, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 5)
        let w = min(scorePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: blocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

}
