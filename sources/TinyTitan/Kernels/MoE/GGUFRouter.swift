import Foundation
import Metal

/// The as-is router for one layer, in F32 end to end.
///
/// The checkpoint stores `ffn_gate_inp.weight` as F32 (2560 x 512), and this
/// router computes in that precision: F32 weights (5.2 MB a layer, 252 MB
/// resident, 126 MB more than a bf16 copy) and F32 logits. Measured against
/// llama.cpp on "The capital of France is" (2026-11-11), every expert-set
/// difference between the two engines sits at the top-10 boundary, where the
/// 10th and 11th probabilities differ by ~1e-4; the count of differing layers
/// was the same with the earlier bf16 weights and half logits (10/48 at
/// position 0, 27/48 at position 4), so the flips come from the activations
/// upstream, not from the router's rounding. F32 is kept because it is the
/// checkpoint's own precision and removes one source of divergence for free.
/// Selection stays on the host: 512 logits are cheaper to rank there than to
/// ship a top-k kernel, and the checkpoint's `norm_topk_prob` semantics are a
/// few lines.
package final class GGUFRouter {
    private let pipeline: MTLComputePipelineState
    private let hidden: Int
    private let expertCount: Int
    private let logits: MTLBuffer
    private let layers: [MTLBuffer]

    package init(context: MetalContext, arch: ArchConfig, model: GGUFModel) throws {
        self.pipeline = try context.pipeline("gguf_f32_gemv_rows")
        self.hidden = arch.hiddenSize
        self.expertCount = arch.numExperts
        precondition(arch.hiddenSize % 4 == 0, "the F32 router GEMV reads float4 rows")
        var built: [MTLBuffer] = []
        built.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let name = "blk.\(layer).ffn_gate_inp.weight"
            guard let location = model.tensor(named: name) else {
                throw MetalError.invalidState("missing \(name)")
            }
            guard location.tensor.type == .f32 else {
                throw MetalError.invalidState("\(name) is \(location.tensor.type), expected F32")
            }
            let byteCount = Int(location.tensor.byteCount)
            guard byteCount == arch.numExperts * arch.hiddenSize * 4 else {
                throw MetalError.invalidState("\(name) holds \(byteCount) bytes")
            }
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            guard let data = try handle.read(upToCount: byteCount), data.count == byteCount
            else { throw MetalError.invalidState("short read of \(name)") }
            guard
                let buffer = data.withUnsafeBytes({ raw -> MTLBuffer? in
                    guard let base = raw.baseAddress else { return nil }
                    return context.device.makeBuffer(
                        bytes: base, length: raw.count, options: .storageModeShared)
                })
            else { throw MetalError.bufferAllocationFailed("router weights") }
            built.append(buffer)
        }
        self.layers = built
        guard
            let logitsBuffer = context.device.makeBuffer(
                length: expertCount * MemoryLayout<Float>.size, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("router logits") }
        self.logits = logitsBuffer
    }

    /// Encodes the router GEMV for `layer`; the logits land in the shared
    /// buffer `select` reads after the command buffer completes.
    package func encode(
        commandBuffer: MTLCommandBuffer, layer: Int, x: MTLBuffer
    ) throws {
        try encode(commandBuffer: commandBuffer, layer: layer, x: x, into: logits)
    }

    /// Encodes the router GEMV for `layer` into `output` (float logits, one per
    /// expert): what a batched prefill uses to keep each token's logits apart.
    package func encode(
        commandBuffer: MTLCommandBuffer, layer: Int, x: MTLBuffer, into output: MTLBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(layers[layer], offset: 0, index: 0)
        encoder.setBuffer(x, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var columns = UInt32(hidden)
        var rows = UInt32(expertCount)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: expertCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// The logits `encode` last wrote, after its command buffer completed.
    package func readLogits() -> [Float] {
        let pointer = logits.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: expertCount))
    }

    /// Softmax over all experts, then the top-k renormalised when the
    /// checkpoint asks for `norm_topk_prob` (this one does).
    package func select(topK: Int) -> (experts: [Int], weights: [Float]) {
        Self.select(logits: readLogits(), topK: topK)
    }

    /// The selection rule on explicit logits: softmax, the `topK` largest
    /// probabilities (ties to the lower index), renormalised to sum to one.
    package static func select(logits: [Float], topK: Int) -> (experts: [Int], weights: [Float]) {
        precondition(topK > 0 && topK <= logits.count, "top-\(topK) of \(logits.count)")
        let maximum = logits.max() ?? 0
        var values = logits.map { exp($0 - maximum) }
        let sum = values.reduce(0, +)
        for index in values.indices { values[index] /= sum }
        let ranked = values.indices.sorted {
            values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1
        }
        let chosen = Array(ranked.prefix(topK))
        let total = chosen.reduce(Float(0)) { $0 + values[$1] }
        return (chosen, chosen.map { values[$0] / total })
    }
}
