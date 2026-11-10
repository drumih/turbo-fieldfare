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
    private let ple: GGUFPLEStage
    private let output: GGUFOutputStage
    private let embedGather: GGUFEmbeddingGather
    private let embedTable: MTLBuffer
    private let convert: MTLComputePipelineState

    private let streams: MTLBuffer
    private let blockInput: MTLBuffer
    private let moeOut: MTLBuffer
    private let embedBF16: MTLBuffer
    private let embedHalf: MTLBuffer
    private let rowIndex: MTLBuffer

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
        self.moeOut = try make(hidden * 2, "moe out")
        self.embedBF16 = try make(hidden * 2, "embed bf16")
        self.embedHalf = try make(hidden * 2, "embed half")
        self.rowIndex = try make(4, "row index")
    }

    /// Runs one token and returns its logits. `position` is the token's index
    /// in the sequence; `history` is the token followed by its predecessors,
    /// which is what the PLE n-grams hash.
    package func decode(token: Int32, position: Int, history: [Int32]) throws -> [Float] {
        // Embedding: gather the row, convert to half, stream 0, broadcast.
        rowIndex.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(bitPattern: token)
        guard let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try embedGather.encode(
            commandBuffer: cb, table: embedTable, rows: rowIndex, count: 1, out: embedBF16)
        guard let encoder = cb.makeComputeCommandEncoder() else {
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
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }
        let half = embedHalf.contents().assumingMemoryBound(to: UInt16.self)
        let streamPointer = streams.contents().assumingMemoryBound(to: UInt16.self)
        for index in 0..<hidden { streamPointer[index] = half[index] }

        // PLE rows depend only on the token and its predecessors.
        try ple.gather(context: history)

        var waitByLayer: [(Int, UInt64)] = []
        var gpuAttnSeconds: TimeInterval = 0
        var gpuRoutedSeconds: TimeInterval = 0
        var phaseNanos: [String: UInt64] = [:]
        func mark(_ name: String, _ start: UInt64) {
            phaseNanos[name, default: 0] &+= DispatchTime.now().uptimeNanoseconds - start
        }
        // The router's top-k is read back on the host, so each layer is two
        // command buffers: everything up to and including the router and the
        // shared expert, then the routed experts and the FFN write. That is a
        // wait per layer, which a GPU top-k would remove; correctness first.
        guard let broadcastCB = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try elementwise.encodeHCBroadcast(
            commandBuffer: broadcastCB, streams: streams, dim: hidden,
            streamCount: arch.hyperConnections.count)
        broadcastCB.commit()
        broadcastCB.waitUntilCompleted()
        if let error = broadcastCB.error { throw MetalError.invalidState("\(error)") }

        for layer in 0..<arch.numLayers {
            if layer == ple.layerIndex {
                guard let pleCB = context.queue.makeCommandBuffer() else {
                    throw MetalError.commandEncoderFailed
                }
                try ple.encode(commandBuffer: pleCB, streamsBuffer: streams)
                pleCB.commit()
                pleCB.waitUntilCompleted()
                if let error = pleCB.error { throw MetalError.invalidState("\(error)") }
            }
            let tAttn = DispatchTime.now().uptimeNanoseconds
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            // Mask values: 1 full attention, 2 GDN linear attention (the
            // checkpoint has 12 of the former: layers 3, 7, ... 47).
            let maskValue = arch.fullAttentionLayerMask[layer]
            let isLinear = maskValue == 2
            guard maskValue == 1 || maskValue == 2 else {
                throw MetalError.invalidState(
                    "layer \(layer) has attention mask \(maskValue), which this decoder does not handle"
                )
            }
            let attnPrefix = "blk.\(layer).hc_attn_"
            var t = DispatchTime.now().uptimeNanoseconds
            try encodeRead(cb, prefix: attnPrefix, streamsBuffer: streams, out: blockInput)
            mark("  hc read attn", t)
            t = DispatchTime.now().uptimeNanoseconds
            if isLinear {
                try gdn.encode(commandBuffer: cb, layer: layer, x: blockInput, out: moeOut)
            } else {
                try fullAttention.encode(
                    commandBuffer: cb, layer: layer, position: position,
                    x: blockInput, out: moeOut)
            }
            mark("  attention", t)
            t = DispatchTime.now().uptimeNanoseconds
            try encodeWrite(cb, prefix: attnPrefix, streamsBuffer: streams, blockOut: moeOut)
            mark("  hc write attn", t)
            let ffnPrefix = "blk.\(layer).hc_ffn_"
            try encodeRead(cb, prefix: ffnPrefix, streamsBuffer: streams, out: blockInput)
            memset(moeOut.contents(), 0, hidden * 2)
            t = DispatchTime.now().uptimeNanoseconds
            try router.encode(commandBuffer: cb, layer: layer, x: blockInput)
            mark("  router", t)
            t = DispatchTime.now().uptimeNanoseconds
            try shared.encode(commandBuffer: cb, layer: layer, x: blockInput, out: moeOut)
            mark("  shared", t)
            let tWait = DispatchTime.now().uptimeNanoseconds
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw MetalError.invalidState("\(error)") }
            mark("  wait(attn)", tWait)
            gpuAttnSeconds += cb.gpuEndTime - cb.gpuStartTime
            waitByLayer.append((layer, DispatchTime.now().uptimeNanoseconds - tWait))
            mark("attn+router+shared", tAttn)

            let tRouted = DispatchTime.now().uptimeNanoseconds
            let selection = router.select(topK: arch.topKExperts)
            if layer == 0 { lastRouting = [] }
            lastRouting.append(selection.experts)
            guard let routedCB = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            let tFetch = DispatchTime.now().uptimeNanoseconds
            for expert in selection.experts {
                _ = try cache.expert(layer: layer, expert: expert)
            }
            mark("  expert fetch", tFetch)
            let tEncode = DispatchTime.now().uptimeNanoseconds
            try routed.encode(
                commandBuffer: routedCB, cache: cache, layer: layer, x: blockInput,
                experts: selection.experts, weights: selection.weights, out: moeOut)
            mark("  routed encode", tEncode)
            try encodeWrite(
                routedCB, prefix: ffnPrefix, streamsBuffer: streams, blockOut: moeOut)
            let tRoutedWait = DispatchTime.now().uptimeNanoseconds
            routedCB.commit()
            routedCB.waitUntilCompleted()
            if let error = routedCB.error { throw MetalError.invalidState("\(error)") }
            mark("  wait(routed)", tRoutedWait)
            gpuRoutedSeconds += routedCB.gpuEndTime - routedCB.gpuStartTime
            mark("routed", tRouted)
        }
        print(
            "  GPU: attn \(String(format: "%.0f", gpuAttnSeconds * 1000)) ms, "
                + "routed \(String(format: "%.0f", gpuRoutedSeconds * 1000)) ms")
        for (layer, nanos) in waitByLayer.prefix(4) {
            print("  wait(attn) layer \(layer): \(String(format: "%.0f", Double(nanos) / 1e6)) ms")
        }
        if let last = waitByLayer.last {
            print(
                "  wait(attn) layer \(last.0): \(String(format: "%.0f", Double(last.1) / 1e6)) ms")
        }
        for (name, nanos) in phaseNanos.sorted(by: { $0.value > $1.value }) {
            print("  \(name): \(String(format: "%.1f", Double(nanos) / 1e6)) ms")
        }

        guard let tail = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try output.encode(commandBuffer: tail, streamsBuffer: streams)
        tail.commit()
        tail.waitUntilCompleted()
        if let error = tail.error { throw MetalError.invalidState("\(error)") }

        let pointer = output.logits.contents().assumingMemoryBound(to: UInt16.self)
        return (0..<arch.vocabSize).map { Float(Float16(bitPattern: pointer[$0])) }
    }

    private func encodeRead(
        _ cb: MTLCommandBuffer, prefix: String, streamsBuffer: MTLBuffer, out: MTLBuffer
    ) throws {
        let sublayer = prefix.contains("attn") ? "attn" : "ffn"
        let layer = try layerIndex(from: prefix)
        try hc.encodeRead(
            commandBuffer: cb, layer: layer, sublayer: sublayer,
            streamsBuffer: streamsBuffer, out: out)
    }

    private func encodeWrite(
        _ cb: MTLCommandBuffer, prefix: String, streamsBuffer: MTLBuffer, blockOut: MTLBuffer
    ) throws {
        let sublayer = prefix.contains("attn") ? "attn" : "ffn"
        let layer = try layerIndex(from: prefix)
        try hc.encodeWrite(
            commandBuffer: cb, layer: layer, sublayer: sublayer,
            streamsBuffer: streamsBuffer, blockOut: blockOut)
    }

    private func layerIndex(from prefix: String) throws -> Int {
        let parts = prefix.split(separator: ".")
        guard parts.count >= 2, let layer = Int(parts[1]) else {
            throw MetalError.invalidState("cannot read a layer from \(prefix)")
        }
        return layer
    }
}
