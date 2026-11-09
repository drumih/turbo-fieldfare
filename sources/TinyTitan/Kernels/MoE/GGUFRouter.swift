import Foundation
import Metal

/// The as-is router for one layer: the checkpoint stores it as F32
/// (`ffn_gate_inp.weight`, 2560 x 512), so it is converted to bf16 once at
/// construction and dotted with `BF16GEMV`. Selection stays on the host: 512
/// logits are cheaper to rank there than to ship a top-k kernel, and the
/// checkpoint's `norm_topk_prob` semantics are a few lines.
package final class GGUFRouter {
    private let bf16: BF16GEMV
    private let hidden: Int
    private let expertCount: Int
    private let logits: MTLBuffer
    private let layers: [MTLBuffer]

    package init(context: MetalContext, arch: ArchConfig, model: GGUFModel) throws {
        self.bf16 = try BF16GEMV(context: context)
        self.hidden = arch.hiddenSize
        self.expertCount = arch.numExperts
        var built: [MTLBuffer] = []
        built.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let name = "blk.\(layer).ffn_gate_inp.weight"
            guard let location = model.tensor(named: name) else {
                throw MetalError.invalidState("missing \(name)")
            }
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let data = try handle.read(upToCount: Int(location.tensor.byteCount)) ?? Data()
            let floats: [Float] = data.withUnsafeBytes { raw in
                let count = raw.count / MemoryLayout<Float>.size
                let pointer = raw.bindMemory(to: Float.self)
                return (0..<count).map { pointer[$0] }
            }
            var bits: [UInt16] = []
            bits.reserveCapacity(floats.count)
            for value in floats { bits.append(Quantization.bf16Bits(value)) }
            guard
                let buffer = bits.withUnsafeBytes({ raw in
                    guard let base = raw.baseAddress else { return nil as MTLBuffer? }
                    return context.device.makeBuffer(
                        bytes: base, length: raw.count, options: .storageModeShared)
                })
            else { throw MetalError.bufferAllocationFailed("router weights") }
            built.append(buffer)
        }
        self.layers = built
        guard
            let logitsBuffer = context.device.makeBuffer(
                length: expertCount * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("router logits") }
        self.logits = logitsBuffer
    }

    /// Encodes the router GEMV for `layer`; the logits land in the shared
    /// buffer `select` reads after the command buffer completes.
    package func encode(
        commandBuffer: MTLCommandBuffer, layer: Int, x: MTLBuffer
    ) throws {
        try bf16.encode(
            commandBuffer: commandBuffer, weights: layers[layer], x: x, y: logits,
            m: UInt32(expertCount), n: UInt32(hidden))
    }

    /// Softmax over all experts, then the top-k renormalised when the
    /// checkpoint asks for `norm_topk_prob` (this one does).
    package func select(topK: Int) -> (experts: [Int], weights: [Float]) {
        let pointer = logits.contents().assumingMemoryBound(to: UInt16.self)
        var values = [Float](repeating: 0, count: expertCount)
        var maximum = -Float.infinity
        for index in 0..<expertCount {
            let value = Float(Float16(bitPattern: pointer[index]))
            values[index] = value
            maximum = max(maximum, value)
        }
        var sum: Float = 0
        for index in 0..<expertCount {
            values[index] = exp(values[index] - maximum)
            sum += values[index]
        }
        for index in 0..<expertCount { values[index] /= sum }

        let ranked = (0..<expertCount).sorted { values[$0] > values[$1] }
        let chosen = Array(ranked.prefix(topK))
        var total: Float = 0
        for index in chosen { total += values[index] }
        let weights = chosen.map { values[$0] / total }
        return (chosen, weights)
    }
}
