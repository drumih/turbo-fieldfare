import Foundation
import Metal

/// The as-is forward pass: one token through all 48 layers, composed from the
/// stages that were each verified against a CPU reference on this checkpoint.
///
/// Per layer: hyper-connection read into the block input, the layer's attention
/// (GDN or full), hyper-connection write; then the same for the MoE block, whose
/// output is the routed experts plus the shared expert. The PLE block runs on
/// its declared layer before that layer's attention read. Finally the output
/// hyper-connection collapses the streams and the head produces logits.
package final class GGUFDecoder {
    private let context: MetalContext
    private let arch: ArchConfig
    private let arena: GGUFResidentArena
    private let mapped: GGUFMappedShards

    private let hc: GGUFHyperConnection
    private let elementwise: Elementwise
    private let gdn: GGUFGDNDecodeStage
    private let fullAttention: GGUFFullAttentionDecodeStage
    private let router: GGUFRouter
    private let cache: GGUFExpertBufferCache
    private let routed: GGUFMoERoutedStage
    private let shared: GGUFSharedExpertStage
    /// Internal, not private: the prefill extension derives the n-gram window
    /// from it.
    let ple: GGUFPLEStage
    private let output: GGUFOutputStage
    private let embedGather: GGUFEmbeddingGather
    private let embedTable: MTLBuffer
    private let convert: MTLComputePipelineState

    private let streams: MTLBuffer
    private let blockInput: MTLBuffer
    /// The token mixer's block output, and the FFN's: separate, because the
    /// shared and routed stages accumulate into theirs.
    private let attnOut: MTLBuffer
    private let ffnOut: MTLBuffer
    private let embedBF16: MTLBuffer
    private let embedHalf: MTLBuffer
    private let rowIndex: MTLBuffer

    /// Opt-in parity capture (`TINYTITAN_GGUF_DUMP_DIR`); nil on the normal path.
    private let stageDump: GGUFStageDump?

    package let hidden: Int
    package let model: GGUFModel
    /// The experts each layer selected on the last decode, for routing studies.
    package private(set) var lastRouting: [[Int]] = []

    package init(context: MetalContext, arch: ArchConfig, model: GGUFModel, maxSeq: Int) throws {
        self.context = context
        self.arch = arch
        self.model = model
        let directory = (model.shards.first?.url ?? URL(fileURLWithPath: "."))
            .deletingLastPathComponent()
        let source = try GGUFModelSource.open(directoryURL: directory)
        // The embedding and head tables are held as their blocks (the gather
        // and the Q6_K head read them directly), and the shared experts are
        // preloaded by their own stage: including them in the arena as bf16
        // would duplicate ~2.5 GB of the 24 GB this machine has.
        // The GDN stage reads its three Q6_K matrices through the mapped
        // shards, so they do not belong in the bf16 arena either.
        let gdnMatrixSuffixes = [
            "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        ]
        func isGDNAndMatrix(_ name: String) -> Bool {
            guard name.hasPrefix("blk.") else { return false }
            let parts = name.split(separator: ".")
            guard parts.count >= 3, let layer = Int(parts[1]) else { return false }
            guard layer < arch.numLayers else { return false }
            return gdnMatrixSuffixes.contains(parts.dropFirst(2).joined(separator: "."))
        }
        func isHyperConnectionProjection(_ name: String) -> Bool {
            (name.contains(".hc_attn_down.weight") || name.contains(".hc_attn_up.weight")
                || name.contains(".hc_ffn_down.weight") || name.contains(".hc_ffn_up.weight"))
        }
        let arenaNames = source.residentNames.filter { name in
            name != "token_embd.weight" && name != "output.weight"
                && !name.contains("_shexp.") && !isGDNAndMatrix(name)
                && !isHyperConnectionProjection(name)
        }
        self.arena = try GGUFResidentArena.build(
            names: arenaNames, from: model, context: context)
        let mappedShards = try GGUFMappedShards(model: model, device: context.device)
        self.mapped = mappedShards
        self.hc = try GGUFHyperConnection(
            context: context, arch: arch, arena: arena, mapped: mappedShards)
        self.elementwise = try Elementwise(context: context)
        self.gdn = try GGUFGDNDecodeStage(
            context: context, arch: arch, arena: arena, mapped: mappedShards)
        self.fullAttention = try GGUFFullAttentionDecodeStage(
            context: context, arch: arch, arena: arena, mapped: mappedShards,
            maxSeq: maxSeq)
        self.router = try GGUFRouter(context: context, arch: arch, model: model)
        // Experts go through the slot cache, not the mapping: the GPU stalls on
        // mapped expert pages when it faults them mid-dispatch (measured
        // wait(routed) 1.7-2.3 s), while a pread into a reused slot is paid once
        // on the host and the GPU then reads clean pages (measured 90 ms).
        //
        // Capacity stays top-k: a 480-slot cache (one token's whole working
        // set) was measured and changed nothing (1.78 s/token against 1.66),
        // because successive tokens select different experts - the cost is the
        // per-token read, not a repeated one, and the extra 1.3 GB buys no hit.
        self.cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: arch.topKExperts)
        self.routed = try GGUFMoERoutedStage(context: context, arch: arch)
        self.shared = try GGUFSharedExpertStage(context: context, arch: arch, model: model)
        self.ple = try GGUFPLEStage(context: context, arch: arch, model: model, arena: arena)
        self.output = try GGUFOutputStage(
            context: context, arch: arch, model: model, arena: arena)
        self.embedGather = try GGUFEmbeddingGather(context: context)
        self.convert = try context.pipeline("gguf_bf16_to_half")
        self.hidden = arch.hiddenSize

        guard let location = model.tensor(named: "token_embd.weight") else {
            throw MetalError.invalidState("missing token_embd.weight")
        }
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
        guard
            let table = context.device.makeBuffer(
                bytes: [UInt8](data), length: data.count, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("token embedding table") }
        self.embedTable = table

        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: bytes, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            memset(buffer.contents(), 0, bytes)
            return buffer
        }
        self.streams = try make(arch.hyperConnections.count * hidden * 2, "streams")
        self.blockInput = try make(hidden * 2, "block input")
        self.attnOut = try make(hidden * 2, "attention out")
        self.ffnOut = try make(hidden * 2, "ffn out")
        self.embedBF16 = try make(hidden * 2, "embed bf16")
        self.embedHalf = try make(hidden * 2, "embed half")
        self.rowIndex = try make(4, "row index")
        self.stageDump = GGUFStageDump.fromEnvironment()
    }

    /// Runs one token and returns its logits. `position` is the token's index
    /// in the sequence; `history` is the PLE hash context - the token, then its
    /// predecessors newest first. `decode(sequence:position:)` derives it from
    /// the sequence and is what a caller normally uses.
    package func decode(token: Int32, position: Int, history: [Int32]) throws -> [Float] {
        let capture = try stageDump?.capture(position: position)
        try embed(token: token)
        try ple.gather(context: history)
        try run { commandBuffer in
            try elementwise.encodeHCBroadcast(
                commandBuffer: commandBuffer, streams: streams, dim: hidden,
                streamCount: arch.hyperConnections.count)
        }
        try capture?.write("hc_init", half: streams, count: wideCount)

        var timing = GGUFDecodeTiming()
        for layer in 0..<arch.numLayers {
            try decodeLayer(layer, position: position, capture: capture, timing: &timing)
        }
        if GGUFDecodeTiming.enabled { timing.report() }

        try run { commandBuffer in
            try output.encode(commandBuffer: commandBuffer, streamsBuffer: streams)
        }
        let pointer = output.logits.contents().assumingMemoryBound(to: UInt16.self)
        let logits = (0..<arch.vocabSize).map { Float(Float16(bitPattern: pointer[$0])) }
        try capture?.write("logits", values: logits)
        return logits
    }

    /// The residual's element count: every hyper stream of the hidden width.
    private var wideCount: Int { arch.hyperConnections.count * hidden }

    /// Embedding: gather the token's Q6_K row as bf16, convert to half, and
    /// place it in stream 0 for the broadcast.
    private func embed(token: Int32) throws {
        rowIndex.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(bitPattern: token)
        try run { commandBuffer in
            try embedGather.encode(
                commandBuffer: commandBuffer, table: embedTable, rows: rowIndex, count: 1,
                out: embedBF16)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            encoder.setComputePipelineState(convert)
            encoder.setBuffer(embedBF16, offset: 0, index: 0)
            encoder.setBuffer(embedHalf, offset: 0, index: 1)
            var count = UInt32(hidden)
            encoder.setBytes(&count, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: hidden, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
        memcpy(streams.contents(), embedHalf.contents(), hidden * 2)
    }

    /// One layer: the PLE block on its layer, the hyper-connection read, the
    /// token mixer (GDN or full attention) and its write; then the FFN read,
    /// router and shared expert, a host wait for the router's top-k, the routed
    /// experts and the FFN write.
    ///
    /// The FFN output has its own buffer, zeroed on the GPU in the same command
    /// buffer: the shared and routed stages accumulate into it, and the
    /// attention output must not be part of that sum.
    private func decodeLayer(
        _ layer: Int, position: Int, capture: GGUFStageDump.Capture?,
        timing: inout GGUFDecodeTiming
    ) throws {
        let tag = String(format: "l%02d", layer)
        if layer == ple.layerIndex {
            try run { commandBuffer in
                try ple.encode(commandBuffer: commandBuffer, streamsBuffer: streams)
            }
            try capture?.write(tag + ".ple", half: streams, count: wideCount)
        }
        // Mask values: 1 full attention, 2 GDN linear attention (the checkpoint
        // has 12 of the former: layers 3, 7, ... 47).
        let maskValue = arch.fullAttentionLayerMask[layer]
        guard maskValue == 1 || maskValue == 2 else {
            throw MetalError.invalidState(
                "layer \(layer) has attention mask \(maskValue), which this decoder does not handle"
            )
        }
        let layerStart = GGUFDecodeTiming.now()
        var commandBuffer = try makeCommandBuffer()
        try hc.encodeRead(
            commandBuffer: commandBuffer, layer: layer, sublayer: "attn",
            streamsBuffer: streams, out: blockInput)
        try checkpoint(&commandBuffer, capture, tag + ".attn_read", blockInput, hidden)
        if maskValue == 2 {
            try gdn.encode(commandBuffer: commandBuffer, layer: layer, x: blockInput, out: attnOut)
        } else {
            try fullAttention.encode(
                commandBuffer: commandBuffer, layer: layer, position: position,
                x: blockInput, out: attnOut)
        }
        try checkpoint(&commandBuffer, capture, tag + ".attn_out", attnOut, hidden)
        try hc.encodeWrite(
            commandBuffer: commandBuffer, layer: layer, sublayer: "attn",
            streamsBuffer: streams, blockOut: attnOut)
        try checkpoint(&commandBuffer, capture, tag + ".attn_res", streams, wideCount)
        try hc.encodeRead(
            commandBuffer: commandBuffer, layer: layer, sublayer: "ffn",
            streamsBuffer: streams, out: blockInput)
        try checkpoint(&commandBuffer, capture, tag + ".ffn_read", blockInput, hidden)
        try encodeZero(commandBuffer, ffnOut, bytes: hidden * 2)
        try router.encode(commandBuffer: commandBuffer, layer: layer, x: blockInput)
        try shared.encode(commandBuffer: commandBuffer, layer: layer, x: blockInput, out: ffnOut)
        let waitStart = GGUFDecodeTiming.now()
        try finish(commandBuffer)
        timing.add("  wait(attn)", since: waitStart)
        timing.attentionWait(layer: layer, since: waitStart)
        timing.gpuAttentionSeconds += commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        timing.add("attn+router+shared", since: layerStart)

        let routedStart = GGUFDecodeTiming.now()
        let selection = router.select(topK: arch.topKExperts)
        if layer == 0 { lastRouting = [] }
        lastRouting.append(selection.experts)
        try capture?.write(tag + ".topk", values: selection.experts.map(Float.init))
        try capture?.write(tag + ".topk_weights", values: selection.weights)
        let fetchStart = GGUFDecodeTiming.now()
        _ = try cache.experts(layer: layer, experts: selection.experts)
        timing.add("  expert fetch", since: fetchStart)
        var routedBuffer = try makeCommandBuffer()
        try routed.encode(
            commandBuffer: routedBuffer, cache: cache, layer: layer, x: blockInput,
            experts: selection.experts, weights: selection.weights, out: ffnOut)
        try checkpoint(&routedBuffer, capture, tag + ".ffn_out", ffnOut, hidden)
        try hc.encodeWrite(
            commandBuffer: routedBuffer, layer: layer, sublayer: "ffn",
            streamsBuffer: streams, blockOut: ffnOut)
        let routedWait = GGUFDecodeTiming.now()
        try finish(routedBuffer)
        timing.add("  wait(routed)", since: routedWait)
        timing.gpuRoutedSeconds += routedBuffer.gpuEndTime - routedBuffer.gpuStartTime
        timing.add("routed", since: routedStart)
        try capture?.write(tag + ".out", half: streams, count: wideCount)
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        return commandBuffer
    }

    private func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
    }

    /// One command buffer around `body`, committed and waited on.
    private func run(_ body: (MTLCommandBuffer) throws -> Void) throws {
        let commandBuffer = try makeCommandBuffer()
        try body(commandBuffer)
        try finish(commandBuffer)
    }

    /// A stage boundary. With no capture it is free; with one, the work so far
    /// is committed and waited on, the stage's output is written, and a fresh
    /// command buffer continues - same kernels, same order, same numerics.
    private func checkpoint(
        _ commandBuffer: inout MTLCommandBuffer, _ capture: GGUFStageDump.Capture?,
        _ stage: String, _ buffer: MTLBuffer, _ count: Int
    ) throws {
        guard let capture else { return }
        try finish(commandBuffer)
        try capture.write(stage, half: buffer, count: count)
        commandBuffer = try makeCommandBuffer()
    }

    /// Zeroes `bytes` of `buffer` in GPU order, so the clear lands after every
    /// earlier write in the same command buffer and before every later one.
    private func encodeZero(_ commandBuffer: MTLCommandBuffer, _ buffer: MTLBuffer, bytes: Int)
        throws
    {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.fill(buffer: buffer, range: 0..<bytes, value: 0)
        blit.endEncoding()
    }
}
