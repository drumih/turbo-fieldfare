import Foundation
import Metal
import TinyTitanCore

/// GPU buffers of one Inkling2 GGUF decode session.
///
/// Recurrent state is per *unrolled* layer visit (48), as in llama.cpp: each
/// pass of the looped core has its own KV cache and convolution history.
/// Sliding-window visits keep a ring of `slidingWindow` positions; full
/// attention visits keep `maxContext`.
final class Inkling2GGUFState {
    /// Convolution channels per visit: K, V, attention output, FFN output.
    let convChannels: Int
    let kvWidth: Int
    let maxContext: Int
    let kCaches: [MTLBuffer]
    let vCaches: [MTLBuffer]
    let capacities: [Int]
    let convStates: [MTLBuffer]
    /// Hidden vectors produced during a token; rows are handed out in order.
    let pool: MTLBuffer
    let poolRows: Int
    let step: MTLBuffer
    let token: MTLBuffer
    let nextToken: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let r: MTLBuffer
    let sinks: MTLBuffer
    let gate: MTLBuffer
    let attention: MTLBuffer
    let dynamicK: MTLBuffer
    let dynamicV: MTLBuffer
    let controlHidden: MTLBuffer
    let controlJoined: MTLBuffer
    let routerLogits: MTLBuffer
    let attentionLogits: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let gateUp: MTLBuffer
    let activation: MTLBuffer
    let down: MTLBuffer
    let logits: MTLBuffer
    let argmaxPartials: MTLBuffer
    static let argmaxGroups = 256

    init(manifest m: Inkling2GGUFManifest, device: MTLDevice, maxContext: Int) throws {
        func buffer(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("Inkling2 GGUF \(label)") }
            memset(made.contents(), 0, made.length)
            return made
        }
        let layer = m.layers[0]
        let attentionWidth = m.headDimension * layer.heads
        kvWidth = m.headDimension * layer.kvHeads
        convChannels = 2 * kvWidth + 2 * m.hidden
        self.maxContext = maxContext
        var kCaches: [MTLBuffer] = []
        var vCaches: [MTLBuffer] = []
        var capacities: [Int] = []
        var convStates: [MTLBuffer] = []
        for visit in 0..<m.unrolledLayers {
            let sliding = m.layers[m.storedLayer(forUnrolled: visit)].isSliding
            let capacity = sliding ? min(m.slidingWindow, maxContext) : maxContext
            capacities.append(capacity)
            kCaches.append(try buffer(capacity * kvWidth * 2, "K cache \(visit)"))
            vCaches.append(try buffer(capacity * kvWidth * 2, "V cache \(visit)"))
            convStates.append(try buffer(convChannels * 3 * 4, "conv state \(visit)"))
        }
        self.kCaches = kCaches
        self.vCaches = vCaches
        self.capacities = capacities
        self.convStates = convStates
        poolRows = 2_048
        pool = try buffer(poolRows * m.hidden * 4, "hidden pool")
        step = try buffer(16, "step")
        token = try buffer(16, "token")
        nextToken = try buffer(16, "next token")
        q = try buffer(attentionWidth * 4, "q")
        k = try buffer(kvWidth * 4, "k")
        v = try buffer(kvWidth * 4, "v")
        r = try buffer(m.relativeLength * layer.heads * 4, "r")
        sinks = try buffer(layer.heads * 4, "sinks")
        gate = try buffer(attentionWidth * 4, "gate")
        attention = try buffer(attentionWidth * 4, "attention")
        dynamicK = try buffer(m.convolutionTaps * layer.kvHeads * 4, "dynamic K")
        dynamicV = try buffer(m.convolutionTaps * layer.kvHeads * 4, "dynamic V")
        controlHidden = try buffer(m.loopGateRank * 4, "control hidden")
        controlJoined = try buffer(2 * m.hidden * 4, "control input")
        routerLogits = try buffer(m.expertCount * 4, "router logits")
        attentionLogits = try buffer(layer.heads * maxContext * 4, "attention logits")
        routeIDs = try buffer(m.expertsPerToken * 4, "route ids")
        routeWeights = try buffer(m.expertsPerToken * 4, "route weights")
        let feedForward = m.layers.map(\.expertFeedForward).max() ?? 0
        gateUp = try buffer(m.expertsPerToken * 2 * feedForward * 4, "gate_up")
        activation = try buffer(m.expertsPerToken * feedForward * 4, "activation")
        down = try buffer(m.expertsPerToken * m.hidden * 4, "down")
        logits = try buffer(m.paddedVocabulary * 4, "logits")
        argmaxPartials = try buffer(Self.argmaxGroups * 8, "argmax partials")
    }

    /// Clears every cache and convolution history for a new sequence.
    func reset() {
        for buffer in kCaches + vCaches + convStates {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

    /// Bytes of recurrent state and scratch held by this session.
    var allocatedBytes: Int {
        (kCaches + vCaches + convStates).reduce(0) { $0 + $1.length } + pool.length
            + attentionLogits.length + logits.length
    }
}
