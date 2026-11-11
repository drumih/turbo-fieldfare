import Foundation
import Metal

/// Layer-major prompt prefill: every token of a chunk through layer 0, then
/// every token through layer 1, and so on.
///
/// Token-at-a-time prefill pays two host waits per token per layer (the router
/// read-back, then the routed experts) and fetches each token's experts on its
/// own. Layer-major order keeps every dependency - token t at layer l+1 needs
/// only layer l at tokens up to t, the GDN state and KV cache advance in token
/// order within a layer - while it lets one command buffer carry a whole
/// chunk's token mixers, and one expert fetch serve every token of a group that
/// selected the same expert. Each token still runs exactly the kernels, in the
/// order, that `decode(token:position:history:)` runs, so the logits are
/// bit-identical to the token-at-a-time loop; `GGUFBatchedPrefillTests` holds
/// that.
///
/// The one piece of state a stage carries from a call to the next is the
/// hyper-connection's normalised residual, which a write consumes from its
/// read. The attention read and write stay adjacent per token; the FFN write
/// runs after every token's FFN read, so each token keeps its own copy
/// (`ffnNormed`) - without it every token's FFN write used the last token's.
extension GGUFDecoder {
    /// Tokens per chunk: the per-token scratch is a few KiB, and the expert
    /// groups inside a layer are bounded by the cache, not by this.
    package static let prefillChunkTokens = 64

    /// Per-token state a chunk needs between the two halves of a layer.
    struct PrefillScratch {
        let streams: [MTLBuffer]
        let ffnInput: [MTLBuffer]
        let ffnOutput: [MTLBuffer]
        let routerLogits: [MTLBuffer]
        /// The FFN hyper-connection read's normalised residual, which its
        /// write consumes after every other token's read has run.
        let ffnNormed: [MTLBuffer]

        init(
            device: MTLDevice, tokens: Int, wideBytes: Int, hiddenBytes: Int, experts: Int,
            normedBytes: Int
        ) throws {
            func make(_ bytes: Int) throws -> [MTLBuffer] {
                try (0..<tokens).map { _ in
                    guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
                    else { throw MetalError.bufferAllocationFailed("batched prefill scratch") }
                    return buffer
                }
            }
            self.streams = try make(wideBytes)
            self.ffnInput = try make(hiddenBytes)
            self.ffnOutput = try make(hiddenBytes)
            self.routerLogits = try make(experts * MemoryLayout<Float>.size)
            self.ffnNormed = try make(normedBytes)
        }
    }

    /// Runs `sequence[range]` as prompt tokens at their absolute positions and
    /// returns the logits of the range's last token. Earlier positions must
    /// already have run (through this or `decode`), so the GDN state and KV
    /// cache hold them.
    package func prefillBatched(sequence: [Int32], range: Range<Int>) throws -> [Float] {
        precondition(!range.isEmpty, "batched prefill needs at least one token")
        let scratch = try PrefillScratch(
            device: context.device, tokens: range.count, wideBytes: wideCount * 2,
            hiddenBytes: hidden * 2, experts: arch.numExperts, normedBytes: hc.normedBytes)
        var timing = GGUFDecodeTiming()
        for (index, position) in range.enumerated() {
            try embed(token: sequence[position])
            try run { commandBuffer in
                try elementwise.encodeHCBroadcast(
                    commandBuffer: commandBuffer, streams: streams, dim: hidden,
                    streamCount: arch.hyperConnections.count)
                try copy(
                    commandBuffer, from: streams, to: scratch.streams[index], bytes: wideCount * 2)
            }
        }
        for layer in 0..<arch.numLayers {
            try prefillLayer(
                layer, sequence: sequence, range: range, scratch: scratch, timing: &timing)
        }
        if GGUFDecodeTiming.enabled { timing.report() }
        let last = range.count - 1
        try run { commandBuffer in
            try output.encode(commandBuffer: commandBuffer, streamsBuffer: scratch.streams[last])
        }
        let pointer = output.logits.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<arch.vocabSize).map { Float(Float16(bitPattern: pointer[$0])) }
    }

    /// One layer for every token of the chunk: the PLE block (per token, as
    /// its gather is a host write), then all token mixers, FFN reads, routers
    /// and shared experts in one command buffer, then the routed experts in
    /// groups whose union fits the cache.
    private func prefillLayer(
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
            try hc.encodeRead(
                commandBuffer: commandBuffer, layer: layer, sublayer: "ffn",
                streamsBuffer: tokenStreams, out: scratch.ffnInput[index],
                normedOut: scratch.ffnNormed[index])
            try encodeZero(commandBuffer, scratch.ffnOutput[index], bytes: hidden * 2)
            try router.encode(
                commandBuffer: commandBuffer, layer: layer, x: scratch.ffnInput[index],
                into: scratch.routerLogits[index])
            try shared.encode(
                commandBuffer: commandBuffer, layer: layer, x: scratch.ffnInput[index],
                out: scratch.ffnOutput[index])
        }
        let waitStart = GGUFDecodeTiming.now()
        try finish(commandBuffer)
        timing.add("  wait(attn)", since: waitStart)
        timing.add("attn+router+shared", since: layerStart)

        let routedStart = GGUFDecodeTiming.now()
        let selections = scratch.routerLogits.prefix(range.count).map { buffer in
            let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
            let logits = Array(UnsafeBufferPointer(start: pointer, count: arch.numExperts))
            return GGUFRouter.select(logits: logits, topK: arch.topKExperts)
        }
        for group in Self.expertGroups(selections.map(\.experts), capacity: cache.capacity) {
            let union = Array(Set(group.flatMap { selections[$0].experts })).sorted()
            let fetchStart = GGUFDecodeTiming.now()
            let fetched = try cache.experts(layer: layer, experts: union)
            timing.add("  expert fetch", since: fetchStart)
            let byExpert = Dictionary(uniqueKeysWithValues: zip(union, fetched))
            let routedBuffer = try makeCommandBuffer()
            for index in group {
                let selection = selections[index]
                let experts = try selection.experts.map { expert in
                    guard let buffers = byExpert[expert] else {
                        throw MetalError.invalidState(
                            "expert \(expert) missing from its group fetch")
                    }
                    return buffers
                }
                try routed.encode(
                    commandBuffer: routedBuffer,
                    experts: experts,
                    x: scratch.ffnInput[index], weights: selection.weights,
                    out: scratch.ffnOutput[index])
                try hc.encodeWrite(
                    commandBuffer: routedBuffer, layer: layer, sublayer: "ffn",
                    streamsBuffer: scratch.streams[index], blockOut: scratch.ffnOutput[index],
                    normed: scratch.ffnNormed[index])
            }
            let routedWait = GGUFDecodeTiming.now()
            try finish(routedBuffer)
            timing.add("  wait(routed)", since: routedWait)
        }
        timing.add("routed", since: routedStart)
    }

    /// Splits tokens (by index, in order) into consecutive groups whose
    /// selected experts together fit `capacity` cache slots, so one fetch
    /// serves the whole group. Order is kept because the groups run in turn.
    package static func expertGroups(_ selections: [[Int]], capacity: Int) -> [[Int]] {
        var groups: [[Int]] = []
        var current: [Int] = []
        var union = Set<Int>()
        for (index, experts) in selections.enumerated() {
            let merged = union.union(experts)
            if !current.isEmpty, merged.count > capacity {
                groups.append(current)
                current = []
                union = Set(experts)
            } else {
                union = merged
            }
            current.append(index)
        }
        if !current.isEmpty { groups.append(current) }
        return groups
    }

    private func copy(
        _ commandBuffer: MTLCommandBuffer, from source: MTLBuffer, to target: MTLBuffer, bytes: Int
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(from: source, sourceOffset: 0, to: target, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }
}
