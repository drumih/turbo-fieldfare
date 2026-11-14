import Foundation
import Metal
import TinyTitanCore

/// A K2 decode step over a dense-only model: embedding, the layer stack, then
/// the final norm and untied head.
///
/// The synthetic install used by the tests is all-dense, so this is the first
/// complete token path on the GPU. Sparse layers are refused by name until the
/// sparse layer (MoVA attention + MoE block) is assembled; a real 36B install
/// therefore still cannot run through this engine.
final class K2DenseGraph {
    private let config: ArchConfig
    private let head: K2Head
    private let layers: [K2DenseLayer]
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer

    init(model: Model, context: MetalContext, maxSequence: Int) throws {
        let config = model.config
        self.config = config
        self.head = try K2Head(model: model, context: context)
        var layers: [K2DenseLayer] = []
        for layer in 0..<config.numLayers {
            guard config.denseLayerIndices.contains(layer) else {
                throw ModelError.unsupportedArchitecture(
                    detail: "K2 sparse layer \(layer) is not implemented on the GPU yet; "
                        + "the dense path stops before it")
            }
            layers.append(
                try K2DenseLayer(
                    model: model, context: context, layer: layer,
                    maxSequence: maxSequence))
        }
        self.layers = layers
        let hiddenSize = config.hiddenSize
        func buffer(_ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: hiddenSize * MemoryLayout<UInt16>.stride,
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.hiddenA = try buffer("k2.graph.hiddenA")
        self.hiddenB = try buffer("k2.graph.hiddenB")
    }

    /// Embed `tokenID`, run every layer at `position`, then write the logits
    /// into the head's buffer (`head.readLogits()` afterwards).
    func encode(
        commandBuffer: MTLCommandBuffer,
        tokenID: UInt32,
        position: UInt32
    ) throws {
        try head.encodeEmbedding(
            commandBuffer: commandBuffer, tokenID: tokenID, out: hiddenA)
        var input = hiddenA
        var output = hiddenB
        for layer in layers {
            try layer.encode(
                commandBuffer: commandBuffer,
                hidden: input, out: output, position: position)
            swap(&input, &output)
        }
        try head.encodeLogits(commandBuffer: commandBuffer, hidden: input)
    }

    func readLogits() -> [Float] { head.readLogits() }
}
