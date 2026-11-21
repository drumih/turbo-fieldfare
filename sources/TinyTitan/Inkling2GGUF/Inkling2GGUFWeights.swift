import Foundation
import Metal
import TinyTitanCore

/// One mapped tensor: its bytes on the GPU and the GEMV shape (`columns` is
/// GGML's fast axis ne0, `rows` the product of the rest).
struct Inkling2GGUFTensor {
    let operand: GGUFGemvIds.Operand
    let type: GGUFType
    let rows: Int
    let columns: Int
    let byteCount: Int
}

/// Every tensor of one stored layer the decode graph reads.
struct Inkling2GGUFLayerWeights {
    let index: Int
    let isSliding: Bool
    let attnQ: Inkling2GGUFTensor
    let attnK: Inkling2GGUFTensor
    let attnV: Inkling2GGUFTensor
    let attnOutput: Inkling2GGUFTensor
    let attnGate: Inkling2GGUFTensor
    let attnRel: Inkling2GGUFTensor
    let attnSink: Inkling2GGUFTensor
    let qClip: Inkling2GGUFTensor
    let relBias: Inkling2GGUFTensor
    let relClip: Inkling2GGUFTensor
    let kConv: Inkling2GGUFTensor
    let vConv: Inkling2GGUFTensor
    let attnPostConv: Inkling2GGUFTensor
    let ffnPostConv: Inkling2GGUFTensor
    /// Sliding layers: one input norm. Full layers: Q, K and V input norms.
    let inputNorms: [Inkling2GGUFTensor]
    /// Full layers only: input-dependent K and V convolution taps.
    let kConvProj: Inkling2GGUFTensor?
    let vConvProj: Inkling2GGUFTensor?
    let attnResQuery: Inkling2GGUFTensor?
    let ffnResQuery: Inkling2GGUFTensor
    let ffnNorm: Inkling2GGUFTensor
    let router: Inkling2GGUFTensor
    let routerBias: Inkling2GGUFTensor
    let gateUp: Inkling2GGUFTensor
    let down: Inkling2GGUFTensor
    let gateUpScale: Inkling2GGUFTensor?
    let downScale: Inkling2GGUFTensor?
    /// Loop message controllers (attn, k, v, mlp): down, gain, forget.
    let loopDown: [Inkling2GGUFTensor]
    let loopGain: [Inkling2GGUFTensor]
    let loopForget: [Inkling2GGUFTensor]
}

/// The mapped weights of an Inkling2 GGUF. Tensors are no-copy Metal buffers
/// over the file's pages (`GGUFMappedShards`); `prefault` reads them once so
/// the first token does not take the page faults on the GPU.
final class Inkling2GGUFWeights {
    let manifest: Inkling2GGUFManifest
    let layers: [Inkling2GGUFLayerWeights]
    let tokenEmbedding: Inkling2GGUFTensor
    let tokenNorm: Inkling2GGUFTensor
    let output: Inkling2GGUFTensor
    let outputNorm: Inkling2GGUFTensor
    let outputResQuery: Inkling2GGUFTensor
    let loopResQuery: Inkling2GGUFTensor?
    let loopKVResQuery: Inkling2GGUFTensor?
    private let shards: GGUFMappedShards

    init(manifest: Inkling2GGUFManifest, device: MTLDevice) throws {
        self.manifest = manifest
        let shards = try GGUFMappedShards(model: manifest.model, device: device)
        self.shards = shards
        func tensor(_ name: String) throws -> Inkling2GGUFTensor {
            try Self.map(name, model: manifest.model, shards: shards)
        }
        func optional(_ name: String) throws -> Inkling2GGUFTensor? {
            manifest.model.tensor(named: name) == nil ? nil : try tensor(name)
        }
        tokenEmbedding = try tensor("token_embd.weight")
        tokenNorm = try tensor("token_embd_norm.weight")
        output = try tensor("output.weight")
        outputNorm = try tensor("output_norm.weight")
        outputResQuery = try tensor("output_res_query.weight")
        loopResQuery = try optional("loop_res_query.weight")
        loopKVResQuery = try optional("loop_kv_res_query.weight")
        var layers: [Inkling2GGUFLayerWeights] = []
        for layer in manifest.layers {
            let p = "blk.\(layer.index)."
            let looped =
                manifest.loopPasses > 1
                && layer.index >= manifest.loopStart && layer.index < manifest.loopEnd
            let modules = looped ? ["attn", "k", "v", "mlp"] : []
            layers.append(
                Inkling2GGUFLayerWeights(
                    index: layer.index, isSliding: layer.isSliding,
                    attnQ: try tensor(p + "attn_q.weight"),
                    attnK: try tensor(p + "attn_k.weight"),
                    attnV: try tensor(p + "attn_v.weight"),
                    attnOutput: try tensor(p + "attn_output.weight"),
                    attnGate: try tensor(p + "attn_gate.weight"),
                    attnRel: try tensor(p + "attn_rel.weight"),
                    attnSink: try tensor(p + "attn_sink_proj.weight"),
                    qClip: try tensor(p + "attn_q_clip.weight"),
                    relBias: try tensor(p + "attn_rel_bias.weight"),
                    relClip: try tensor(p + "attn_rel_clip.weight"),
                    kConv: try tensor(p + "attn_k_conv.weight"),
                    vConv: try tensor(p + "attn_v_conv.weight"),
                    attnPostConv: try tensor(p + "attn_post_conv.weight"),
                    ffnPostConv: try tensor(p + "ffn_post_conv.weight"),
                    inputNorms: layer.isSliding
                        ? [try tensor(p + "attn_norm.weight")]
                        : try ["q", "k", "v"].map { try tensor(p + "attn_\($0)_inp_norm.weight") },
                    kConvProj: layer.isSliding ? nil : try tensor(p + "attn_k_conv_proj.weight"),
                    vConvProj: layer.isSliding ? nil : try tensor(p + "attn_v_conv_proj.weight"),
                    attnResQuery: try optional(p + "attn_res_query.weight"),
                    ffnResQuery: try tensor(p + "ffn_res_query.weight"),
                    ffnNorm: try tensor(p + "ffn_norm.weight"),
                    router: try tensor(p + "ffn_gate_inp.weight"),
                    routerBias: try tensor(p + "exp_probs_b.bias"),
                    gateUp: try tensor(p + "ffn_gate_up_exps.weight"),
                    down: try tensor(p + "ffn_down_exps.weight"),
                    gateUpScale: try optional(p + "ffn_gate_up_exps.scale"),
                    downScale: try optional(p + "ffn_down_exps.scale"),
                    loopDown: try modules.map { try tensor(p + "loop_\($0)_down.weight") },
                    loopGain: try modules.map { try tensor(p + "loop_\($0)_gain.weight") },
                    loopForget: try modules.map { try tensor(p + "loop_\($0)_forget.weight") }))
        }
        self.layers = layers
    }

    /// Bytes of every mapped tensor.
    var mappedBytes: Int { shards.mappedBytes }

    /// Reads every mapped page once (sequentially, on `threads` CPU threads) so
    /// the GPU does not fault them in during the first tokens. Returns the
    /// checksum of the touched bytes so the reads cannot be optimised away.
    @discardableResult
    func prefault(threads: Int = 8) -> UInt64 {
        var tensors: [Inkling2GGUFTensor] = [
            tokenEmbedding, tokenNorm, output, outputNorm, outputResQuery,
        ]
        tensors += [loopResQuery, loopKVResQuery].compactMap { $0 }
        for layer in layers {
            tensors += [
                layer.attnQ, layer.attnK, layer.attnV, layer.attnOutput, layer.attnGate,
                layer.attnRel, layer.attnSink, layer.gateUp, layer.down, layer.router,
            ]
        }
        // Page addresses as integers: the workers only read them.
        let pages = tensors.flatMap { tensor -> [UInt] in
            let base =
                UInt(bitPattern: tensor.operand.buffer.contents()) + UInt(tensor.operand.offset)
            return stride(from: 0, to: tensor.byteCount, by: 16_384).map { base + UInt($0) }
        }
        let sums = PrefaultSums(count: threads)
        let chunk = (pages.count + threads - 1) / threads
        DispatchQueue.concurrentPerform(iterations: threads) { worker in
            var sum: UInt64 = 0
            for index in (worker * chunk)..<min(pages.count, (worker + 1) * chunk) {
                if let page = UnsafeRawPointer(bitPattern: pages[index]) {
                    sum &+= UInt64(page.load(as: UInt8.self))
                }
            }
            sums.store(sum, at: worker)
        }
        return sums.total
    }

    private static func map(
        _ name: String, model: GGUFModel, shards: GGUFMappedShards
    ) throws -> Inkling2GGUFTensor {
        guard let location = model.tensor(named: name) else {
            throw Inkling2GGUFManifest.Error.missingTensor(name)
        }
        let tensor = location.tensor
        let columns = Int(tensor.dimensions[0])
        let rows = tensor.dimensions.dropFirst().reduce(1) { $0 * Int($1) }
        let weight: GGUFMappedShards.Weight
        if tensor.dimensions.count >= 2, let mapped = try shards.weight(named: name) {
            weight = mapped
        } else {
            let shard = model.shards[location.shardIndex]
            weight = try shards.weight(
                key: name, url: shard.url, fileOffset: shard.dataOffset + tensor.offset,
                byteCount: tensor.byteCount, type: tensor.type, rows: rows, columns: columns)
        }
        return Inkling2GGUFTensor(
            operand: .init(weight.buffer, offset: weight.offset), type: tensor.type,
            rows: rows, columns: columns, byteCount: Int(tensor.byteCount))
    }
}

// Per-worker checksums of `Inkling2GGUFWeights.prefault`.
// unchecked-invariant: each worker of one concurrentPerform writes only its own
// slot, and `total` is read after concurrentPerform has returned, so no slot is
// ever read and written concurrently.
private final class PrefaultSums: @unchecked Sendable {
    private let slots: UnsafeMutableBufferPointer<UInt64>

    init(count: Int) {
        slots = .allocate(capacity: count)
        slots.initialize(repeating: 0)
    }

    deinit { slots.deallocate() }

    func store(_ value: UInt64, at index: Int) { slots[index] = value }

    var total: UInt64 { slots.reduce(0, &+) }
}
