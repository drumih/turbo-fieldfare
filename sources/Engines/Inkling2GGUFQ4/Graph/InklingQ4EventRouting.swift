import Foundation
import Metal
import TinyTitanCore

/// Streamed-expert decode in one command buffer (TT-121 phase 2).
///
/// Each layer visit's router signals a shared event; the GPU then waits on
/// the next value. A listener on a serial queue reads the routed ids, resolves
/// them to slots (reading what is missing), writes the slot ids, starts the
/// lookahead prefetch and signals the GPU on. The token is encoded and
/// committed once instead of being cut at every router, so the GPU never
/// idles waiting for the CPU to encode and commit the rest of the visit.
///
/// Ordering: the GPU signals only after everything encoded before the signal
/// has finished, and waits before anything after it, so the CPU reads a
/// finished `routeIDs` and writes `slotIDs` while no GPU work reads either.
/// The next visit's handler fires only after this visit's experts are done.
/// unchecked-invariant: `event`, `listener` and the buffers are set once
/// in `init`. `failure` is only touched under `lock`. Handlers run either
/// on the decoding thread (`spin`, in `drive`) or on the listener's serial
/// queue while the decoding thread is blocked in `waitUntilCompleted` on
/// the buffer that fires them; one at a time either way, so the cache is
/// never used from two threads at once.
final class InklingQ4EventRouting: @unchecked Sendable {
    let event: MTLSharedEvent
    private let listener: MTLSharedEventListener
    private let cache: InklingQ4ExpertCache
    private let routeIDs: MTLBuffer
    private let slotIDs: MTLBuffer
    private let lookaheadIDs: MTLBuffer
    private let used: Int
    private let lock = NSLock()
    private var failure: Error?
    /// Seconds spent inside handlers, and handlers run (diagnostics).
    private(set) var handlerSeconds: Double = 0
    private(set) var handlers = 0
    /// Resolve on the decoding thread, polling the event (true), or on the
    /// listener's queue when the GPU signals (false).
    var spin = true
    /// Visits encoded and not yet driven (spin mode).
    private var queued: [Visit] = []
    /// The last value handed out; each visit takes two.
    private(set) var value: UInt64

    init(
        device: MTLDevice, cache: InklingQ4ExpertCache, state: InklingQ4State,
        expertsPerToken: Int
    ) throws {
        guard let event = device.makeSharedEvent() else {
            throw MetalError.invalidState("cannot create the routing event")
        }
        self.event = event
        listener = MTLSharedEventListener(
            dispatchQueue: DispatchQueue(label: "inkling-q4.routing", qos: .userInteractive))
        self.cache = cache
        routeIDs = state.routeIDs
        slotIDs = state.slotIDs
        lookaheadIDs = state.lookaheadIDs
        used = expertsPerToken
        value = event.signaledValue
    }

    /// Ends `encoder`, makes `buffer` signal the router's result and wait for
    /// the slots, and registers the handler that resolves them. Returns the
    /// encoder the visit's experts go to.
    func encodeResolve(
        buffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder, layer: Int,
        predicts: Int?, deep: (ids: MTLBuffer, layers: [Int])? = nil
    ) throws -> MTLComputeCommandEncoder {
        encoder.endEncoding()
        let deep = deep.map { Deep(ids: $0.ids, layers: $0.layers) }
        let routed = value + 1
        let resolved = value + 2
        value = resolved
        buffer.encodeSignalEvent(event, value: routed)
        buffer.encodeWaitForEvent(event, value: resolved)
        let visit = Visit(
            routed: routed, resolved: resolved, layer: layer, predicts: predicts, deep: deep)
        if spin {
            queued.append(visit)
        } else {
            event.notify(listener, atValue: routed) { [self] _, _ in handle(visit) }
        }
        guard let next = buffer.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            throw MetalError.commandEncoderFailed
        }
        return next
    }

    /// One visit's resolution: the values it waits for and signals, and what
    /// it resolves and predicts.
    /// unchecked-invariant: `deep` holds a diagnostic scratch buffer the GPU
    /// wrote before `routed` was signalled; only the handler reads it.
    private struct Visit: @unchecked Sendable {
        let routed: UInt64
        let resolved: UInt64
        let layer: Int
        let predicts: Int?
        let deep: Deep?
    }

    /// Resolves one visit and releases the GPU, which always happens: a
    /// failure is kept and thrown once the command buffer ends.
    private func handle(_ visit: Visit) {
        let started = Date()
        do {
            try resolve(layer: visit.layer, predicts: visit.predicts)
            if let deep = visit.deep { logDeep(deep) }
        } catch {
            lock.withLock { if failure == nil { failure = error } }
        }
        handlerSeconds += Date().timeIntervalSince(started)
        handlers += 1
        event.signaledValue = visit.resolved
    }

    /// With `spin`, runs the visits queued for `buffer` (just committed) on
    /// the calling thread, polling the event instead of waiting for a
    /// listener wake-up: the round trip per visit is what streamed decode
    /// pays 48 times a token. Stops early if the buffer fails.
    func drive(_ buffer: MTLCommandBuffer) {
        let visits = queued
        queued = []
        for visit in visits {
            while event.signaledValue < visit.routed {
                if buffer.status == .error { return }
            }
            handle(visit)
        }
    }

    /// The first failure a handler met since the last call, cleared.
    func takeFailure() -> Error? {
        lock.withLock {
            defer { failure = nil }
            return failure
        }
    }

    /// Diagnostic: logs the routers of later visits run on this visit's
    /// input (depth 2 onward), for measuring how far ahead routing predicts.
    /// unchecked-invariant: the buffer is a diagnostic scratch the GPU wrote
    /// before the signal that runs the handler; only the handler reads it.
    private struct Deep: @unchecked Sendable {
        let ids: MTLBuffer
        let layers: [Int]
    }

    private func logDeep(_ deep: Deep) {
        let ids = deep.ids.contents().bindMemory(to: Int32.self, capacity: deep.layers.count * used)
        for (index, layer) in deep.layers.enumerated() {
            let routed = Array(UnsafeBufferPointer(start: ids + index * used, count: used))
            cache.routingLog?.append((layer, routed, index + 2))
        }
    }

    private func resolve(layer: Int, predicts: Int?) throws {
        let routed = Array(
            UnsafeBufferPointer(
                start: routeIDs.contents().bindMemory(to: Int32.self, capacity: used),
                count: used))
        let target = slotIDs.contents().bindMemory(to: Int32.self, capacity: used)
        try cache.resolve(layer: layer, routed: routed) { slots, last in
            // One token routes `used` experts and a layer has at least that
            // many slots, so a decode visit is always a single round.
            guard last else {
                throw MetalError.invalidState("decode visit of layer \(layer) needs two rounds")
            }
            for (index, slot) in slots.enumerated() { target[index] = slot }
        }
        // The experts of `layer` run next; a prediction for the same layer
        // could evict slots they read, so only another layer is prefetched.
        if let predicts, predicts != layer {
            cache.prefetch(
                layer: predicts,
                experts: Array(
                    UnsafeBufferPointer(
                        start: lookaheadIDs.contents().bindMemory(to: Int32.self, capacity: used),
                        count: used)))
        }
    }
}
