import Foundation
import Metal

/// Layer-major prompt prefill: every token of a chunk through layer 0, then
/// every token through layer 1, and so on.
///
/// Token-at-a-time prefill pays two host waits per token per layer (the router
/// read-back, then the routed experts) and reads each token's experts on its
/// own. Layer-major order keeps every dependency - token t at layer l+1 needs
/// only layer l at tokens up to t, and the GDN state and KV cache advance in
/// token order within a layer - while it lets one command buffer carry a whole
/// chunk's token mixers, and lets the routed half go expert-major: every
/// distinct expert the chunk selected at a layer is read from the SSD once,
/// and serves every token that chose it.
///
/// Each (token, expert) pair runs the decode route's kernels, its output is
/// kept in its own row, and each token's weighted sum then adds its rows in
/// selection order with the decode route's rounding - so the logits are
/// bit-identical to the token-at-a-time loop; `GGUFBatchedPrefillTests` holds
/// that.
///
/// The one piece of state a stage carries from a call to the next is the
/// hyper-connection's normalised residual, which a write consumes from its
/// read. The attention read and write stay adjacent per token; the FFN write
/// runs after every token's FFN read, so each token keeps its own copy
/// (`ffnNormed`) - without it every token's FFN write used the last token's.
extension GGUFDecoder {
    /// Tokens per chunk. A longer chunk lets more tokens share each expert
    /// read; the per-token scratch is about `(topK + 6) * hidden * 2` bytes.
    package static let prefillChunkTokens = 128

    /// Per-token state a chunk needs between the two halves of a layer.
    struct PrefillScratch {
        let streams: [MTLBuffer]
        let ffnInput: [MTLBuffer]
        let ffnOutput: [MTLBuffer]
        let routerLogits: [MTLBuffer]
        /// The FFN hyper-connection read's normalised residual, which its
        /// write consumes after every other token's read has run.
        let ffnNormed: [MTLBuffer]
        /// Every token's FFN input side by side (token t at `t * hidden`),
        /// for the pair dispatches that mix tokens.
        let ffnInputs: MTLBuffer
        /// One hidden-wide row per (token, selection): token t's k-th
        /// expert output at row `t * topK + k`.
        let expertRows: MTLBuffer

        init(
            device: MTLDevice, tokens: Int, hidden: Int, wideBytes: Int, experts: Int,
            topK: Int, normedBytes: Int
        ) throws {
            func buffer(_ bytes: Int) throws -> MTLBuffer {
                guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
                else { throw MetalError.bufferAllocationFailed("batched prefill scratch") }
                return buffer
            }
            func make(_ bytes: Int) throws -> [MTLBuffer] {
                try (0..<tokens).map { _ in try buffer(bytes) }
            }
            self.streams = try make(wideBytes)
            self.ffnInput = try make(hidden * 2)
            self.ffnOutput = try make(hidden * 2)
            self.routerLogits = try make(experts * MemoryLayout<Float>.size)
            self.ffnNormed = try make(normedBytes)
            self.ffnInputs = try buffer(tokens * hidden * 2)
            self.expertRows = try buffer(tokens * topK * hidden * 2)
        }
    }

    /// Runs `sequence[range]` as prompt tokens at their absolute positions and
    /// returns the logits of the range's last token. Earlier positions must
    /// already have run (through this or `decode`), so the GDN state and KV
    /// cache hold them.
    package func prefillBatched(sequence: [Int32], range: Range<Int>) throws -> [Float] {
        precondition(!range.isEmpty, "batched prefill needs at least one token")
        let scratch = try PrefillScratch(
            device: context.device, tokens: range.count, hidden: hidden,
            wideBytes: wideCount * 2, experts: arch.numExperts, topK: arch.topKExperts,
            normedBytes: hc.normedBytes)
        var timing = GGUFDecodeTiming()
        for (index, position) in range.enumerated() {
            try embed(token: sequence[position])
            try run { commandBuffer in
                try elementwise.encodeHCBroadcast(
                    commandBuffer: commandBuffer, streams: streams, dim: hidden,
                    streamCount: arch.hyperConnections.count)
                try copy(
                    commandBuffer, from: streams, to: scratch.streams[index], at: 0,
                    bytes: wideCount * 2)
            }
        }
        for layer in 0..<arch.numLayers {
            try prefillMixers(
                layer, sequence: sequence, range: range, scratch: scratch, timing: &timing)
            try prefillRouted(layer, tokens: range.count, scratch: scratch, timing: &timing)
        }
        if GGUFDecodeTiming.enabled { timing.report() }
        let last = range.count - 1
        try run { commandBuffer in
            try output.encode(commandBuffer: commandBuffer, streamsBuffer: scratch.streams[last])
        }
        let pointer = output.logits.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<arch.vocabSize).map { Float(Float16(bitPattern: pointer[$0])) }
    }

    /// The first half of a layer for every token of the chunk: the PLE block
    /// (per token, as its gather is a host write), then every token mixer,
    /// FFN read, router and shared expert in one command buffer.
    private func prefillMixers(
        _ layer: Int, sequence: [Int32], range: Range<Int>, scratch: PrefillScratch,
        timing: inout GGUFDecodeTiming
    ) throws {
        if layer == ple.layerIndex {
            for (index, position) in range.enumerated() {
                try ple.gather(context: pleHistory(of: sequence, at: position))
                try run { commandBuffer in
                    try ple.encode(
                        commandBuffer: commandBuffer, streamsBuffer: scratch.streams[index])
                }
            }
        }
        let maskValue = arch.fullAttentionLayerMask[layer]
        guard maskValue == 1 || maskValue == 2 else {
            throw MetalError.invalidState(
                "layer \(layer) has attention mask \(maskValue), which this decoder does not handle"
            )
        }
        let layerStart = GGUFDecodeTiming.now()
        let commandBuffer = try makeCommandBuffer()
        for (index, position) in range.enumerated() {
            let tokenStreams = scratch.streams[index]
            try hc.encodeRead(
                commandBuffer: commandBuffer, layer: layer, sublayer: "attn",
                streamsBuffer: tokenStreams, out: blockInput)
            if maskValue == 2 {
                try gdn.encode(
                    commandBuffer: commandBuffer, layer: layer, x: blockInput, out: attnOut)
            } else {
                try fullAttention.encode(
                    commandBuffer: commandBuffer, layer: layer, position: position,
                    x: blockInput, out: attnOut)
            }
            try hc.encodeWrite(
                commandBuffer: commandBuffer, layer: layer, sublayer: "attn",
                streamsBuffer: tokenStreams, blockOut: attnOut)
            let ffnInput = scratch.ffnInput[index]
            try hc.encodeRead(
                commandBuffer: commandBuffer, layer: layer, sublayer: "ffn",
                streamsBuffer: tokenStreams, out: ffnInput, normedOut: scratch.ffnNormed[index])
            try copy(
                commandBuffer, from: ffnInput, to: scratch.ffnInputs, at: index * hidden * 2,
                bytes: hidden * 2)
            try encodeZero(commandBuffer, scratch.ffnOutput[index], bytes: hidden * 2)
            try router.encode(
                commandBuffer: commandBuffer, layer: layer, x: ffnInput,
                into: scratch.routerLogits[index])
            try shared.encode(
                commandBuffer: commandBuffer, layer: layer, x: ffnInput,
                out: scratch.ffnOutput[index])
        }
        let waitStart = GGUFDecodeTiming.now()
        try finish(commandBuffer)
        timing.add("  wait(attn)", since: waitStart)
        timing.add("attn+router+shared", since: layerStart)
    }

    /// The routed half of a layer, expert-major: the chunk's distinct experts
    /// in batches the cache can hold, each batch read once and run for every
    /// (token, selection) that chose one of its experts; then each token's
    /// weighted sum and FFN write.
    private func prefillRouted(
        _ layer: Int, tokens: Int, scratch: PrefillScratch, timing: inout GGUFDecodeTiming
    ) throws {
        let routedStart = GGUFDecodeTiming.now()
        let topK = arch.topKExperts
        let selections = scratch.routerLogits.prefix(tokens).map { buffer in
            let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
            let logits = Array(UnsafeBufferPointer(start: pointer, count: arch.numExperts))
            return GGUFRouter.select(logits: logits, topK: topK)
        }
        for batch in Self.expertBatches(selections.map(\.experts), capacity: cache.capacity) {
            let fetchStart = GGUFDecodeTiming.now()
            let fetched = try cache.experts(layer: layer, experts: batch)
            timing.add("  expert fetch", since: fetchStart)
            let byExpert = Dictionary(uniqueKeysWithValues: zip(batch, fetched))
            var pairs: [GGUFMoERoutedStage.RoutedPair] = []
            for (token, selection) in selections.enumerated() {
                for (rank, expert) in selection.experts.enumerated() {
                    guard let buffers = byExpert[expert] else { continue }
                    pairs.append(
                        GGUFMoERoutedStage.RoutedPair(
                            expert: buffers, xOffset: token * hidden, row: token * topK + rank))
                }
            }
            let routedBuffer = try makeCommandBuffer()
            try routed.encodePairs(
                commandBuffer: routedBuffer, pairs: pairs, x: scratch.ffnInputs,
                rows: scratch.expertRows)
            let routedWait = GGUFDecodeTiming.now()
            try finish(routedBuffer)
            timing.add("  wait(routed)", since: routedWait)
        }
        let writeBuffer = try makeCommandBuffer()
        for (token, selection) in selections.enumerated() {
            try routed.encodeWeightedSum(
                commandBuffer: writeBuffer, rows: scratch.expertRows,
                rowsOffset: token * topK * hidden * 2, weights: selection.weights,
                out: scratch.ffnOutput[token], outOffset: 0)
            try hc.encodeWrite(
                commandBuffer: writeBuffer, layer: layer, sublayer: "ffn",
                streamsBuffer: scratch.streams[token], blockOut: scratch.ffnOutput[token],
                normed: scratch.ffnNormed[token])
        }
        try finish(writeBuffer)
        timing.add("routed", since: routedStart)
    }

    /// The distinct experts of `selections`, ascending, in batches of at most
    /// `capacity`: each batch is one cache fetch, so every expert the chunk
    /// selected is read once per layer however many tokens chose it.
    package static func expertBatches(_ selections: [[Int]], capacity: Int) -> [[Int]] {
        precondition(capacity > 0, "a batch holds at least one expert")
        let distinct = Array(Set(selections.joined())).sorted()
        return stride(from: 0, to: distinct.count, by: capacity).map {
            Array(distinct[$0..<min(distinct.count, $0 + capacity)])
        }
    }

    private func copy(
        _ commandBuffer: MTLCommandBuffer, from source: MTLBuffer, to target: MTLBuffer,
        at offset: Int, bytes: Int
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(from: source, sourceOffset: 0, to: target, destinationOffset: offset, size: bytes)
        blit.endEncoding()
    }
}
