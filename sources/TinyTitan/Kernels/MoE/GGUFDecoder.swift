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
///
/// The stages and scratch are internal rather than private because the
/// decoder is split across files (`+Prefill`, `+BatchedPrefill`) that drive the
/// same stages in a different order; nothing outside the decoder uses them.
package final class GGUFDecoder {
    let context: MetalContext
    let arch: ArchConfig
    let arena: GGUFResidentArena
    let mapped: GGUFMappedShards

    let hc: GGUFHyperConnection
    let elementwise: Elementwise
    let gdn: GGUFGDNDecodeStage
    let fullAttention: GGUFFullAttentionDecodeStage
    let router: GGUFRouter
    let cache: GGUFExpertBufferCache
    let routed: GGUFMoERoutedStage
    let shared: GGUFSharedExpertStage
    let ple: GGUFPLEStage
    let output: GGUFOutputStage
    let embedGather: GGUFEmbeddingGather
    let embedTable: MTLBuffer
    let convert: MTLComputePipelineState

    let streams: MTLBuffer
    let blockInput: MTLBuffer
    /// The token mixer's block output, and the FFN's: separate, because the
    /// shared and routed stages accumulate into theirs.
    let attnOut: MTLBuffer
    let ffnOut: MTLBuffer
    let embedBF16: MTLBuffer
    let embedHalf: MTLBuffer
    let rowIndex: MTLBuffer

    /// Opt-in parity capture (`TINYTITAN_GGUF_DUMP_DIR`); nil on the normal path.
    let stageDump: GGUFStageDump?
    /// Opt-in routing log (`TINYTITAN_GGUF_ROUTING_LOG=<file>`): one line per
    /// token, the routed experts of every layer (`,` within a layer, `;`
    /// between layers), appended. The input for expert-cache policy studies.
    let routingLog: FileHandle?

    package let hidden: Int
    package let model: GGUFModel
    /// The experts each layer selected on the last decode, for routing studies.
    package private(set) var lastRouting: [[Int]] = []

    /// `ramBudgetBytes` is the process target (`--ram-budget`); nil sizes the
    /// expert cache by `TINYTITAN_GGUF_EXPERT_CACHE_GB` or the measured default.
    package init(
        context: MetalContext, arch: ArchConfig, model: GGUFModel, maxSeq: Int,
        ramBudgetBytes: Int? = nil
    ) throws {
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
        // Under a RAM budget the attention, GDN and hyper-connection matrices
        // are copied in rather than mapped: the budget already counts them,
        // and mapped pages the system evicts are faulted back in by the GPU
        // mid-dispatch. Measured at --ram-budget 10G on the 24 GB M3 with the
        // system in swap: wait(attn) 0.9-1.9 s a token mapped.
        let mappedShards = try GGUFMappedShards(
            model: model, device: context.device,
            residency: ramBudgetBytes == nil ? .mapped : .resident)
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
        // A cache sized for reuse does not pay: a 480-slot cache (one token's
        // whole working set) was measured and changed nothing (1.78 s/token
        // against 1.66), because successive tokens select different experts.
        self.cache = try GGUFExpertBufferCache(
            model: model, device: context.device,
            capacity: try Self.expertSlotsPerLayer(
                model: model, arch: arch, ramBudgetBytes: ramBudgetBytes,
                arenaBytes: arena.buffer.length))
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
        self.routingLog = Self.openRoutingLog(
            ProcessInfo.processInfo.environment["TINYTITAN_GGUF_ROUTING_LOG"])
        if ramBudgetBytes != nil {
            // The budget counts these bytes; pin them so the system cannot
            // swap them out between dispatches (see ResidentMemoryLock).
            ResidentMemoryLock.lock(cache.poolBuffers, label: "the expert cache")
            ResidentMemoryLock.lock([arena.buffer, embedTable], label: "the resident weights")
        }
    }

    /// Runs one token and returns its logits. `position` is the token's index
    /// in the sequence; `history` is the PLE hash context - the token, then its
    /// predecessors newest first. `decode(sequence:position:)` derives it from
    /// the sequence and is what a caller normally uses.
    package func decode(token: Int32, position: Int, history: [Int32]) throws -> [Float] {
        let capture = try stageDump?.capture(position: position)
        var timing = GGUFDecodeTiming()
        let headStart = GGUFDecodeTiming.now()
        try embed(token: token)
        try ple.gather(context: history)
        // One command buffer is carried from layer to layer: a layer's routed
        // experts and FFN write go into the buffer that then takes the next
        // layer's mixer, router and shared expert, so a token waits once a
        // layer (for the router's top-k) instead of twice.
        var commandBuffer = try makeCommandBuffer()
        try elementwise.encodeHCBroadcast(
            commandBuffer: commandBuffer, streams: streams, dim: hidden,
            streamCount: arch.hyperConnections.count)
        try checkpoint(&commandBuffer, capture, "hc_init", streams, wideCount)
        timing.add("embed+ple", since: headStart)

        for layer in 0..<arch.numLayers {
            try decodeLayer(
                layer, position: position, commandBuffer: &commandBuffer, capture: capture,
                timing: &timing)
        }

        let outputStart = GGUFDecodeTiming.now()
        try output.encode(commandBuffer: commandBuffer, streamsBuffer: streams)
        try finish(commandBuffer)
        timing.gpuRoutedSeconds += commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        timing.add("last routed + output head", since: outputStart)
        let readStart = GGUFDecodeTiming.now()
        let pointer = output.logits.contents().assumingMemoryBound(to: UInt16.self)
        let logits = (0..<arch.vocabSize).map { Float(Float16(bitPattern: pointer[$0])) }
        timing.add("logits read", since: readStart)
        if GGUFDecodeTiming.enabled {
            timing.report()
            let stats = cache.statistics
            FileHandle.standardError.write(
                Data(
                    ("  experts (cumulative): hits \(stats.hits), misses \(stats.misses), "
                        + "\(cache.capacity) slots/layer, "
                        + "\(cache.residentBytes / 1_048_576) MiB\n").utf8))
        }
        try capture?.write("logits", values: logits)
        if let routingLog {
            let line = lastRouting.map { $0.map(String.init).joined(separator: ",") }
                .joined(separator: ";")
            routingLog.write(Data((line + "\n").utf8))
        }
        return logits
    }

    /// Expert slots per layer: the RAM budget (`TINYTITAN_GGUF_EXPERT_CACHE_GB`,
    /// default a twelfth of physical memory) divided by what one slot costs
    /// across all layers, at least two top-k sets (the pinned layer and the one
    /// being fetched) and at most every expert.
    ///
    /// The default is measured, not guessed. The pool is wired Metal memory, so
    /// every GiB it takes is a GiB the mmap'd attention and GDN weights lose to
    /// the page cache. On the 24 GB M-series Mac (essay prompt, 80 tokens, two
    /// rounds, 2026-11-11): 2 GiB 2.07/2.46 tok/s, 3 GiB 1.07/2.46, 4 GiB
    /// 0.75/2.34 with wait(attn) doubling to 199 ms; the hit rate rose 52% -> 64%
    /// and bought nothing. 6 GiB (the old quarter) drove free memory to 28% and
    /// the GPU into page-fault stalls. A twelfth is 2 GiB there.
    ///
    /// With a process target (`ramBudgetBytes`, the CLI's `--ram-budget`, the
    /// launcher's `--ram`) the cache gets what the target leaves after the rest
    /// of the model: every non-expert tensor as stored (the mapped matrices are
    /// paged in on use and count against the process), the bf16 arena built
    /// from some of them, and a working margin for KV, scratch and the
    /// runtime. A target that leaves less than the two-set floor is refused
    /// with the floor named, rather than silently exceeded.
    static func expertSlotsPerLayer(
        model: GGUFModel, arch: ArchConfig, ramBudgetBytes: Int?, arenaBytes: Int
    ) throws -> Int {
        let perSlot = try GGUFExpertBufferCache.bytesPerSlotAcrossLayers(model: model)
        let floor = 2 * arch.topKExperts
        let budget: Double
        if let ramBudgetBytes {
            let resident = residentBytesOutsideTheExpertCache(
                model: model, arenaBytes: arenaBytes)
            let required = resident + floor * perSlot
            guard ramBudgetBytes >= required else {
                throw MetalError.invalidState(
                    "--ram-budget \(ramBudgetBytes / 1_048_576) MiB is below this model's floor of "
                        + "\(required / 1_048_576) MiB (\(resident / 1_048_576) MiB outside the "
                        + "expert cache plus \(floor) expert slots per layer)")
            }
            budget = Double(ramBudgetBytes - resident)
        } else if let value = ProcessInfo.processInfo.environment["TINYTITAN_GGUF_EXPERT_CACHE_GB"]
            .flatMap(Double.init), value > 0
        {
            budget = value * 1_073_741_824
        } else {
            budget = Double(ProcessInfo.processInfo.physicalMemory) / 12
        }
        let slots = Int(budget / Double(perSlot))
        return min(arch.numExperts, max(floor, slots))
    }

    /// What the process holds besides the expert pools: the non-expert
    /// tensors as stored, the bf16 arena, and a 768 MiB margin (KV cache,
    /// scratch, Metal and the runtime itself).
    ///
    /// The PLE n-gram table is left out: it is 26.8 GiB of this checkpoint's
    /// 77.2, and the stage reads a handful of its rows per token from the
    /// mapping, so what it costs the process is those pages, not the table.
    /// Counting it would put every RAM target under 33 GiB below the floor.
    static func residentBytesOutsideTheExpertCache(model: GGUFModel, arenaBytes: Int) -> Int {
        let notResident = model.tensors.filter { name, _ in
            name.contains("_exps.") || name.hasSuffix("per_layer_token_embd.weight")
        }
        .reduce(0) { $0 + Int($1.value.tensor.byteCount) }
        let resident = Int(model.totalTensorBytes) - notResident
        return resident + arenaBytes + 768 * 1_048_576
    }

    private static func openRoutingLog(_ path: String?) -> FileHandle? {
        guard let path, !path.isEmpty else { return nil }
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = FileHandle(forWritingAtPath: path) else { return nil }
        handle.seekToEndOfFile()
        return handle
    }

    /// The residual's element count: every hyper stream of the hidden width.
    var wideCount: Int { arch.hyperConnections.count * hidden }

    /// Embedding: gather the token's Q6_K row as bf16, convert to half, and
    /// place it in stream 0 for the broadcast.
    func embed(token: Int32) throws {
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
    /// `commandBuffer` arrives holding the previous layer's uncommitted routed
    /// work (or the embedding broadcast) and leaves holding this layer's: the
    /// one wait per layer is the router read-back.
    ///
    /// The FFN output has its own buffer, zeroed on the GPU in the same command
    /// buffer: the shared and routed stages accumulate into it, and the
    /// attention output must not be part of that sum.
    func decodeLayer(
        _ layer: Int, position: Int, commandBuffer: inout MTLCommandBuffer,
        capture: GGUFStageDump.Capture?, timing: inout GGUFDecodeTiming
    ) throws {
        let tag = String(format: "l%02d", layer)
        if layer == ple.layerIndex {
            try ple.encode(commandBuffer: commandBuffer, streamsBuffer: streams)
            try checkpoint(&commandBuffer, capture, tag + ".ple", streams, wideCount)
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
        let fetched = try cache.experts(layer: layer, experts: selection.experts)
        timing.add("  expert fetch", since: fetchStart)
        commandBuffer = try makeCommandBuffer()
        try routed.encode(
            commandBuffer: commandBuffer, experts: fetched, x: blockInput,
            weights: selection.weights, out: ffnOut)
        try checkpoint(&commandBuffer, capture, tag + ".ffn_out", ffnOut, hidden)
        try hc.encodeWrite(
            commandBuffer: commandBuffer, layer: layer, sublayer: "ffn",
            streamsBuffer: streams, blockOut: ffnOut)
        try checkpoint(&commandBuffer, capture, tag + ".out", streams, wideCount)
        timing.add("routed", since: routedStart)
    }

    func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        return commandBuffer
    }

    func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
    }

    /// One command buffer around `body`, committed and waited on.
    func run(_ body: (MTLCommandBuffer) throws -> Void) throws {
        let commandBuffer = try makeCommandBuffer()
        try body(commandBuffer)
        try finish(commandBuffer)
    }

    /// A stage boundary. With no capture it is free; with one, the work so far
    /// is committed and waited on, the stage's output is written, and a fresh
    /// command buffer continues - same kernels, same order, same numerics.
    func checkpoint(
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
    func encodeZero(_ commandBuffer: MTLCommandBuffer, _ buffer: MTLBuffer, bytes: Int)
        throws
    {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.fill(buffer: buffer, range: 0..<bytes, value: 0)
        blit.endEncoding()
    }
}
