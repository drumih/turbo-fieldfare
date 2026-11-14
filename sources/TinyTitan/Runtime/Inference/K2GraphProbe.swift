import Foundation
import Metal
import TinyTitanCore

/// A K2 decode step over a mixed model: embedding, the dense and sparse layers
/// in order, then the final norm and head.
///
/// The sparse probe drives its own command buffers (its router ids have to
/// cross to the CPU to address the expert blobs), so this graph sequences each
/// layer one at a time and carries the hidden state as halfs between them. That is the probe
/// form of the loop, not the batched serving graph.
package final class K2GraphProbe {
    private let config: ArchConfig
    private let head: K2Head
    private let dense: [Int: K2DenseLayer]
    private let sparse: [Int: K2SparseLayerProbe]
    /// The chunked prefill view of each dense layer, sharing that layer's K/V
    /// cache so a chunk and a later decode step agree.
    private let denseBatch: [Int: K2DenseBatch]
    /// The chunked view of each sparse layer, sharing that layer's K/V cache.
    private let sparseBatch: [Int: K2SparseBatch]
    private let context: MetalContext
    private let queue: MTLCommandQueue
    private let maxSequence: Int
    private let maxChunk: Int
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer
    private let chunkA: MTLBuffer
    private let chunkB: MTLBuffer

    package init(
        model: Model, context: MetalContext, maxSequence: Int,
        maxChunk: Int = 1
    ) throws {
        precondition(maxChunk > 0)
        let config = model.config
        self.config = config
        self.context = context
        self.queue = context.queue
        self.maxSequence = maxSequence
        self.maxChunk = maxChunk
        self.head = try K2Head(model: model, context: context)
        var dense: [Int: K2DenseLayer] = [:]
        var sparse: [Int: K2SparseLayerProbe] = [:]
        for layer in 0..<config.numLayers {
            if config.denseLayerIndices.contains(layer) {
                dense[layer] = try K2DenseLayer(
                    model: model, context: context, layer: layer,
                    maxSequence: maxSequence)
            } else {
                sparse[layer] = try K2SparseLayerProbe(
                    model: model, context: context, layer: layer,
                    maxSequence: maxSequence)
            }
        }
        self.dense = dense
        self.sparse = sparse
        // K2's dense layers are the prefix; the chunk path relies on that order
        // (one command buffer embeds and runs them, then the sparse layers take
        // over token by token).
        if let lastDense = config.denseLayerIndices.max(),
            let firstSparse = (0..<config.numLayers).first(where: {
                !config.denseLayerIndices.contains($0)
            }),
            lastDense > firstSparse
        {
            throw ModelError.internalInconsistency(
                detail: "K2 dense layers must precede the sparse layers")
        }
        var batches: [Int: K2DenseBatch] = [:]
        for (layer, denseLayer) in dense {
            batches[layer] = try K2DenseBatch(
                model: model, context: context, layer: layer,
                maxSequence: maxSequence, maxTokens: maxChunk,
                kCache: denseLayer.kCache, vCache: denseLayer.vCache)
        }
        self.denseBatch = batches
        var sparseBatches: [Int: K2SparseBatch] = [:]
        for (layer, sparseLayer) in sparse {
            sparseBatches[layer] = try K2SparseBatch(
                model: model, context: context, layer: layer,
                maxSequence: maxSequence, maxTokens: maxChunk,
                kCache: sparseLayer.kCache, vCache: sparseLayer.vCache)
        }
        self.sparseBatch = sparseBatches
        let hidden = config.hiddenSize
        func buffer(_ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: hidden * MemoryLayout<UInt16>.stride,
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.hiddenA = try buffer("k2.graph.hiddenA")
        self.hiddenB = try buffer("k2.graph.hiddenB")
        guard
            let chunkA = context.device.makeBuffer(
                length: maxChunk * hidden * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
            let chunkB = context.device.makeBuffer(
                length: maxChunk * hidden * MemoryLayout<UInt16>.stride,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("k2.graph.chunk")
        }
        self.chunkA = chunkA
        self.chunkB = chunkB
    }

    /// Prefill a chunk of tokens at consecutive positions and return the
    /// logits of the last one.
    ///
    /// Every layer runs layer-major over the chunk: the dense prefix through
    /// `K2DenseBatch`, the sparse layers through `K2SparseBatch`, both sharing
    /// the decode layers' K/V caches. The sparse batch still resolves routing
    /// per token and streams each token's experts, but its norm, q/k/gate/o,
    /// attention and shared expert read their weights once per chunk. Positions
    /// are absolute, so a caller may start a chunk at any cached prefix
    /// boundary.
    package func forwardChunk(
        tokenIDs: [UInt32], basePosition: UInt32
    ) async throws -> [Float] {
        let started = Date()
        let d = config.hiddenSize
        let half = MemoryLayout<UInt16>.stride
        let count = tokenIDs.count
        guard count > 0, count <= maxChunk else {
            throw ModelError.internalInconsistency(
                detail: "K2 chunk of \(count) tokens exceeds its \(maxChunk)-token buffer")
        }
        guard Int(basePosition) + count <= maxSequence else {
            throw ModelError.internalInconsistency(
                detail:
                    "K2 chunk positions \(basePosition)..\(basePosition + UInt32(count) - 1) exceed its \(maxSequence)-token cache"
            )
        }
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            for layer in sparse.values { layer.resetTimings() }
        }
        var current = chunkA
        var next = chunkB
        guard let cb = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        for (index, token) in tokenIDs.enumerated() {
            try head.encodeEmbedding(
                commandBuffer: cb, tokenID: token, out: current,
                outOffset: index * d * half)
        }
        for layer in config.denseLayerIndices.sorted() {
            guard let batch = denseBatch[layer] else {
                throw ModelError.internalInconsistency(
                    detail: "dense layer \(layer) has no chunk path")
            }
            try batch.encode(
                commandBuffer: cb, hidden: current, out: next,
                basePosition: Int(basePosition), tokenCount: count)
            swap(&current, &next)
        }
        cb.commit()
        await cb.completed()
        try checkFailure(cb)

        for layer in 0..<config.numLayers where sparseBatch[layer] != nil {
            guard let batch = sparseBatch[layer] else { continue }
            try await batch.encode(
                queue: queue, hidden: current, out: next,
                basePosition: Int(basePosition), tokenCount: count)
            swap(&current, &next)
        }

        // Only the last token's logits are read; its hidden state is the row
        // the decode step would have produced.
        guard let headCB = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        guard let blit = headCB.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: current, sourceOffset: (count - 1) * d * half,
            to: hiddenA, destinationOffset: 0,
            size: d * half)
        blit.endEncoding()
        try head.encodeLogits(commandBuffer: headCB, hidden: hiddenA)
        headCB.commit()
        await headCB.completed()
        try checkFailure(headCB)
        for layer in sparse.values { try layer.checkPendingError() }
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            let fetch = sparse.values.reduce(0.0) { $0 + $1.fetchSeconds }
            let busy = sparse.values.reduce(0.0) { $0 + $1.gpuBusySeconds }
            let total = Date().timeIntervalSince(started)
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "k2 chunk: %d tokens %.3fs fetch %.3fs busy %.3fs\n",
                        count, total, fetch, busy
                    ).utf8))
        }
        return head.readLogits()
    }

    /// The chunk's dense phase has to complete before its rows feed the sparse
    /// layers; a failed buffer would otherwise be read as valid halfs.
    private func checkFailure(_ commandBuffer: MTLCommandBuffer) throws {
        if let error = commandBuffer.error {
            throw ModelError.internalInconsistency(
                detail: "K2 chunk command buffer failed: \(error)")
        }
    }

    package func forward(tokenID: UInt32, position: UInt32) async throws -> [Float] {
        let started = Date()
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            for layer in sparse.values { layer.resetTimings() }
        }
        // The hidden state stays on the GPU for the whole token. A dense layer
        // writes the other ping-pong buffer; a sparse layer writes its own
        // output, which the next layer reads. Only the router ids and the final
        // logits come back to the CPU.
        guard let embed = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeEmbedding(commandBuffer: embed, tokenID: tokenID, out: hiddenA)
        embed.commit()
        var current = hiddenA
        for layer in 0..<config.numLayers {
            if let dense = dense[layer] {
                let out = current === hiddenA ? hiddenB : hiddenA
                guard let cb = queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                try dense.encode(
                    commandBuffer: cb, hidden: current, out: out, position: position)
                cb.commit()
                current = out
            } else if let sparse = sparse[layer] {
                try await sparse.prefetch(
                    queue: queue, input: current, position: Int(position))
                try await sparse.submitExperts(queue: queue)
                current = sparse.output
            } else {
                throw ModelError.internalInconsistency(
                    detail: "layer \(layer) is neither dense nor sparse")
            }
        }
        guard let headCB = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeLogits(commandBuffer: headCB, hidden: current)
        headCB.commit()
        await headCB.completed()
        // The expert phases are submitted without waits; surface any failure
        // they reported before the token's logits are trusted.
        for layer in sparse.values { try layer.checkPendingError() }
        if ProcessInfo.processInfo.environment["TINYTITAN_K2_TIMING"] == "1" {
            let fetch = sparse.values.reduce(0.0) { $0 + $1.fetchSeconds }
            let encode = sparse.values.reduce(0.0) { $0 + $1.encodeSeconds }
            let busy = sparse.values.reduce(0.0) { $0 + $1.gpuBusySeconds }
            let prefetch = sparse.values.reduce(0.0) { $0 + $1.prefetchBusySeconds }
            let experts = sparse.values.reduce(0.0) { $0 + $1.expertBusySeconds }
            let total = Date().timeIntervalSince(started)
            let rest = total - fetch - encode - busy
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "k2 timing: total %.3fs fetch %.3fs busy %.3fs (pre %.3fs experts %.3fs) encode %.3fs other %.3fs\n",
                        total, fetch, busy, prefetch, experts, encode, rest
                    ).utf8))
        }
        return head.readLogits()
    }

}
