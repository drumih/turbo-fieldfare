// Provenance: copied from sources/TinyTitan/Inkling2GGUF/Inkling2GGUFState.swift
// (TT-121 phase 2) under the variant-isolation rule.
import Foundation
import Metal
import TinyTitanCore

/// GPU buffers of one Inkling2 GGUF session, sized for chunks of up to
/// `maxChunk` tokens (one at decode).
///
/// Recurrent state is per *unrolled* layer visit (48), as in llama.cpp: each
/// pass of the looped core has its own KV cache and convolution history.
/// Sliding-window visits keep a ring of `slidingWindow + maxChunk` positions —
/// the extra `maxChunk` slots keep a chunk's later tokens from overwriting keys
/// its earlier tokens still attend to; full-attention visits keep `maxContext`.
final class InklingQ4State {
    /// Convolution channels per visit: K, V, attention output, FFN output.
    let convChannels: Int
    let kvWidth: Int
    let maxContext: Int
    let maxChunk: Int
    let kCaches: [MTLBuffer]
    let vCaches: [MTLBuffer]
    let capacities: [Int]
    let convStates: [MTLBuffer]
    /// Hidden vectors of a chunk, in slabs of `maxChunk * hidden` floats:
    /// slabs `0 ..< persistentSlabs` hold values that outlive a layer visit,
    /// the rest a visit's temporaries.
    let pool: MTLBuffer
    let slabFloats: Int
    let persistentSlabs = 40
    let transientSlabs = 24
    let step: MTLBuffer
    let tokens: MTLBuffer
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
    let routeIDs: MTLBuffer
    /// Routing ids rewritten to expert-cache slots, when experts stream.
    let slotIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let gateUp: MTLBuffer
    let activation: MTLBuffer
    let down: MTLBuffer
    let expertLists: MTLBuffer
    let expertCounts: MTLBuffer
    let logits: MTLBuffer
    let argmaxPartials: MTLBuffer
    static let argmaxGroups = 256

    init(
        manifest m: InklingQ4Manifest, device: MTLDevice, maxContext: Int, maxChunk: Int
    ) throws {
        func buffer(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let made = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("Inkling2 GGUF \(label)") }
            memset(made.contents(), 0, made.length)
            return made
        }
        let layer = m.layers[0]
        let heads = layer.heads
        let attentionWidth = m.headDimension * heads
        let t = maxChunk
        kvWidth = m.headDimension * layer.kvHeads
        convChannels = 2 * kvWidth + 2 * m.hidden
        self.maxContext = maxContext
        self.maxChunk = maxChunk
        var kCaches: [MTLBuffer] = []
        var vCaches: [MTLBuffer] = []
        var capacities: [Int] = []
        var convStates: [MTLBuffer] = []
        for visit in 0..<m.unrolledLayers {
            let sliding = m.layers[m.storedLayer(forUnrolled: visit)].isSliding
            let capacity = sliding ? min(m.slidingWindow + maxChunk, maxContext) : maxContext
            capacities.append(capacity)
            kCaches.append(try buffer(capacity * kvWidth * 2, "K cache \(visit)"))
            vCaches.append(try buffer(capacity * kvWidth * 2, "V cache \(visit)"))
            convStates.append(try buffer(convChannels * 3 * 4, "conv state \(visit)"))
        }
        self.kCaches = kCaches
        self.vCaches = vCaches
        self.capacities = capacities
        self.convStates = convStates
        slabFloats = t * m.hidden
        pool = try buffer((persistentSlabs + transientSlabs) * slabFloats * 4, "hidden pool")
        step = try buffer(16, "step")
        tokens = try buffer(t * 4, "tokens")
        nextToken = try buffer(16, "next token")
        q = try buffer(t * attentionWidth * 4, "q")
        k = try buffer(t * kvWidth * 4, "k")
        v = try buffer(t * kvWidth * 4, "v")
        r = try buffer(t * m.relativeLength * heads * 4, "r")
        sinks = try buffer(t * heads * 4, "sinks")
        gate = try buffer(t * attentionWidth * 4, "gate")
        attention = try buffer(t * attentionWidth * 4, "attention")
        dynamicK = try buffer(t * m.convolutionTaps * layer.kvHeads * 4, "dynamic K")
        dynamicV = try buffer(t * m.convolutionTaps * layer.kvHeads * 4, "dynamic V")
        // One slice per concurrent attention-side controller (attn, K, V).
        controlHidden = try buffer(3 * t * m.loopGateRank * 4, "control hidden")
        // One slice per concurrent attention-side controller (attn, K, V).
        controlJoined = try buffer(3 * t * 2 * m.hidden * 4, "control input")
        routerLogits = try buffer(t * m.expertCount * 4, "router logits")
        let pairs = t * m.expertsPerToken
        routeIDs = try buffer(pairs * 4, "route ids")
        slotIDs = try buffer(pairs * 4, "slot ids")
        routeWeights = try buffer(pairs * 4, "route weights")
        let feedForward = m.layers.map(\.expertFeedForward).max() ?? 0
        gateUp = try buffer(pairs * 2 * feedForward * 4, "gate_up")
        activation = try buffer(pairs * feedForward * 4, "activation")
        down = try buffer(pairs * m.hidden * 4, "down")
        expertLists = try buffer(m.expertCount * pairs * 4, "expert lists")
        expertCounts = try buffer(m.expertCount * 4, "expert counts")
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
            + gateUp.length + down.length + activation.length + logits.length
    }
}
