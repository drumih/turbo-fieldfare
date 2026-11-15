import Foundation
import Metal
import TinyTitanCore

/// A Laguna-S-2.1 decode step: embedding, all 48 layers in order, then the
/// final norm and the untied head.
///
/// The sparse layers drive their own command buffers because the router ids
/// have to cross to the CPU between phases, so this graph sequences one layer
/// at a time and carries the hidden state as halfs between them. That is the
/// probe form of the serving loop, not the batched one.
package final class LagunaGraphProbe {
    private let config: ArchConfig
    private let head: K2Head
    private let layers: [LagunaLayer]
    private let context: MetalContext
    private let queue: MTLCommandQueue
    private let hiddenA: MTLBuffer
    private let hiddenB: MTLBuffer
    /// Debug hook: write each layer's hidden state as raw FP16 when
    /// `TINYTITAN_LAGUNA_DUMP` names a directory. Used to compare a real
    /// install against an independent CPU reference; off by default.
    private let dumpDirectory: URL?

    package init(model: Model, context: MetalContext, maxSequence: Int) throws {
        self.config = model.config
        self.context = context
        self.queue = context.queue
        self.head = try K2Head(model: model, context: context)
        self.layers = try (0..<model.config.numLayers).map { layer in
            try LagunaLayer(
                model: model, context: context, layer: layer,
                maxSequence: maxSequence)
        }
        let hidden = model.config.hiddenSize
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
        self.hiddenA = try buffer("laguna.graph.hiddenA")
        self.hiddenB = try buffer("laguna.graph.hiddenB")
        if let path = ProcessInfo.processInfo.environment["TINYTITAN_LAGUNA_DUMP"] {
            let directory = URL(fileURLWithPath: path)
            try? FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            self.dumpDirectory = directory
        } else {
            self.dumpDirectory = nil
        }
    }

    private func dump(
        _ buffer: MTLBuffer, layer: Int, position: UInt32
    ) {
        guard let directory = dumpDirectory else { return }
        let data = Data(
            bytes: buffer.contents(),
            count: config.hiddenSize * MemoryLayout<UInt16>.stride)
        let name = String(format: "layer-%02d-pos-%u.bin", layer, position)
        try? data.write(to: directory.appendingPathComponent(name))
    }

    package func forward(tokenID: UInt32, position: UInt32) async throws -> [Float] {
        guard let embed = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeEmbedding(
            commandBuffer: embed, tokenID: tokenID, out: hiddenA,
            scale: LagunaLayer.residualScale)
        embed.commit()
        await embed.completed()
        dump(hiddenA, layer: -1, position: position)

        var input = hiddenA
        var output = hiddenB
        for (index, layer) in layers.enumerated() {
            try await layer.run(
                queue: queue, hidden: input, out: output, position: Int(position))
            swap(&input, &output)
            dump(input, layer: index, position: position)
        }
        guard let final = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try head.encodeLogits(commandBuffer: final, hidden: input)
        final.commit()
        await final.completed()
        return head.readLogits()
    }
}
