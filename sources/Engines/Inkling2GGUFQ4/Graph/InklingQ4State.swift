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
package final class InklingQ4State {
    /// Per-sequence recurrent state: KV caches and short-convolution histories
    /// for all 48 unrolled layer visits.
    package struct SlotState {
        let kCaches: [MTLBuffer]
        let vCaches: [MTLBuffer]
        let convStates: [MTLBuffer]
    }

    /// Convolution channels per visit: K, V, attention output, FFN output.
    let convChannels: Int
    let kvWidth: Int
    package let kvCachePrecision: KVCachePrecision
    package let kvRowBytes: Int
    package let maxContext: Int
    let maxChunk: Int
    package let slots: Int
    package let slotStates: [SlotState]
    var kCaches: [MTLBuffer] { slotStates[0].kCaches }
    var vCaches: [MTLBuffer] { slotStates[0].vCaches }
    let capacities: [Int]
    var convStates: [MTLBuffer] { slotStates[0].convStates }
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
    /// The next visit's router run on this visit's MoE input: its predicted
    /// routing, which the expert cache prefetches (decode only).
    let lookaheadLogits: MTLBuffer
    let lookaheadIDs: MTLBuffer
    let lookaheadWeights: MTLBuffer
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
        manifest m: InklingQ4Manifest, device: MTLDevice, maxContext: Int, maxChunk: Int,
        slots: Int = 1, kvCachePrecision: KVCachePrecision = .fp16
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
        let slotCount = max(1, slots)
        kvWidth = m.headDimension * layer.kvHeads
        self.kvCachePrecision = kvCachePrecision
        kvRowBytes = InklingQ4LoadPlan.kvRowBytes(kvWidth: kvWidth, precision: kvCachePrecision)
        convChannels = 2 * kvWidth + 2 * m.hidden
        self.maxContext = maxContext
        self.maxChunk = maxChunk
        self.slots = slotCount
        var capacities: [Int] = []
        for visit in 0..<m.unrolledLayers {
            let sliding = m.layers[m.storedLayer(forUnrolled: visit)].isSliding
            capacities.append(sliding ? min(m.slidingWindow + maxChunk, maxContext) : maxContext)
        }
        var slotStates: [SlotState] = []
        slotStates.reserveCapacity(slotCount)
        for slot in 0..<slotCount {
            var kCaches: [MTLBuffer] = []
            var vCaches: [MTLBuffer] = []
            var convStates: [MTLBuffer] = []
            for visit in 0..<m.unrolledLayers {
                let capacity = capacities[visit]
                kCaches.append(try buffer(capacity * kvRowBytes, "K cache s\(slot) v\(visit)"))
                vCaches.append(try buffer(capacity * kvRowBytes, "V cache s\(slot) v\(visit)"))
                convStates.append(try buffer(convChannels * 3 * 4, "conv s\(slot) v\(visit)"))
            }
            slotStates.append(SlotState(kCaches: kCaches, vCaches: vCaches, convStates: convStates))
        }
        self.slotStates = slotStates
        self.capacities = capacities
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
        lookaheadLogits = try buffer(m.expertCount * 4, "lookahead logits")
        lookaheadIDs = try buffer(m.expertsPerToken * 4, "lookahead ids")
        lookaheadWeights = try buffer(m.expertsPerToken * 4, "lookahead weights")
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

    /// Clears every cache and convolution history for one sequence slot.
    package func reset(slot: Int) {
        guard slot >= 0, slot < slotStates.count else { return }
        let s = slotStates[slot]
        for buffer in s.kCaches + s.vCaches + s.convStates {
            memset(buffer.contents(), 0, buffer.length)
        }
    }

    /// Clears every cache and convolution history for all slots.
    package func reset() {
        for slot in 0..<slotStates.count { reset(slot: slot) }
    }

    /// Bytes of recurrent state held for one sequence slot.
    package var perSlotBytes: Int {
        guard let first = slotStates.first else { return 0 }
        return (first.kCaches + first.vCaches + first.convStates).reduce(0) { $0 + $1.length }
    }

    /// Bytes of recurrent state and scratch held by this session.
    package var allocatedBytes: Int {
        perSlotBytes * slotStates.count + pool.length
            + gateUp.length + down.length + activation.length + logits.length
    }
}
