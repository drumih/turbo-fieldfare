import Foundation
import Metal
import TinyTitanCore

/// One K2-Horizon sparse layer on a decode step: grouped norm, q/k and the
/// separate gate, MoVA value attention over a plain K/V cache, then the bf16
/// router, the fetched routed experts and the shared expert.
///
/// Probe-scoped like `K2DenseLayer`: the K/V cache is a plain FP16 buffer and
/// the routed experts are fetched between command buffers, because the router's
/// ids have to come back to the CPU to address the blobs. Everything before
/// that read-back -- including the MoVA values, whose selection the shader
/// resolves itself -- is one command buffer; the fetched experts are the
/// second. That two-phase shape is what the full graph will keep.
///
/// unchecked-invariant: the two Metal command-buffer completion handlers run
/// on the queue's own threads and touch only the counters and `pendingError`
/// guarded by `timingLock`; every other stored property is immutable after
/// `init`. `fetchSeconds` and `encodeSeconds` are written by the submitting
/// thread and read after the wait, never from a handler.
final class K2SparseLayerProbe: @unchecked Sendable {
    private let config: ArchConfig
    private let model: Model
    private let layer: Int
    private let maxSequence: Int
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let gemv4: AffineQuantGEMV
    private let gemv8: AffineQuantGEMV
    private let bf16: BF16GEMV
    private let mova: MoVAValueStage
    private let moe: K2MoEStage
    private let fusedMoE: Bool
    private let vExperts: [MoVAValueStage.ExpertWeights]
    private let sharedWeights: K2MoEStage.ExpertWeights

    private let input: MTLBuffer
    private let normed: MTLBuffer
    private let q: MTLBuffer
    private let k: MTLBuffer
    private let v: MTLBuffer
    private let gate: MTLBuffer
    private let attn: MTLBuffer
    private let oOut: MTLBuffer
    private let h1: MTLBuffer
    private let normed2: MTLBuffer
    private let moeOut: MTLBuffer
    private let out: MTLBuffer
    /// Internal so the chunked sparse path can share them: a chunk prefill
    /// must write the rows a later decode step reads.
    let kCache: MTLBuffer
    let vCache: MTLBuffer
    private(set) var fetchSeconds: Double = 0
    private(set) var encodeSeconds: Double = 0
    /// GPU busy time, accumulated from the command buffers' own timestamps in
    /// their completion handlers: with the two phases pipelined there is no
    /// single wait to time, so the buffer's own start/end is the measurement.
    private let timingLock = NSLock()
    private var busyTotal = 0.0
    private var prefetchBusyTotal = 0.0
    private var expertBusyTotal = 0.0
    /// The expert phase is submitted without being awaited, so its failure has
    /// to be carried to the next await rather than thrown from `submitExperts`.
    private var pendingError: Error?

    var gpuBusySeconds: Double {
        timingLock.lock()
        defer { timingLock.unlock() }
        return busyTotal
    }
    /// The same busy time split by command buffer, so a timing line can say
    /// whether the pre-fetch GPU work or the routed experts dominate. The
    /// pre-fetch buffer holds everything up to the router read-back: norm,
    /// q/k/gate, MoVA, attention and the shared expert.
    var prefetchBusySeconds: Double {
        timingLock.lock()
        defer { timingLock.unlock() }
        return prefetchBusyTotal
    }
    var expertBusySeconds: Double {
        timingLock.lock()
        defer { timingLock.unlock() }
        return expertBusyTotal
    }

    /// Rethrow an expert command buffer's asynchronous failure (nil when the
    /// buffers have all succeeded). `prefetch` calls this before reusing the
    /// layer's buffers; the graph calls it once after the head.
    func checkPendingError() throws {
        timingLock.lock()
        let error = pendingError
        timingLock.unlock()
        if let error { throw error }
    }

    /// Zero the per-token counters. The graph times one forward at a time; the
    /// probe is long-lived, so without this the timing line prints cumulative
    /// sums next to a per-token total.
    func resetTimings() {
        fetchSeconds = 0
        encodeSeconds = 0
        timingLock.lock()
        busyTotal = 0
        prefetchBusyTotal = 0
        expertBusyTotal = 0
        pendingError = nil
        timingLock.unlock()
    }

    init(
        model: Model, context: MetalContext, layer: Int,
        maxSequence: Int = 1
    ) throws {
        self.model = model
        self.config = model.config
        self.layer = layer
        self.maxSequence = maxSequence
        guard !config.denseLayerIndices.contains(layer) else {
            throw ModelError.internalInconsistency(
                detail: "layer \(layer) is dense, not a K2 sparse layer")
        }
        let hidden = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let vBits = try Self.weightBits(
            try model.k2VExpert(layer: layer, expert: 0),
            rows: kvDim, columns: hidden)
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.elementwise = try Elementwise(context: context)
        self.gemv4 = try AffineQuantGEMV(context: context, weightBits: 4)
        self.gemv8 = try AffineQuantGEMV(context: context, weightBits: 8)
        self.bf16 = try BF16GEMV(context: context)
        self.mova = try MoVAValueStage(
            context: context, weightBits: vBits,
            hiddenSize: hidden, valueDim: kvDim,
            maxExperts: config.mova.numExperts)
        // The routed experts' own width, not the attention slot's: the 8-bit
        // install stores them at 8 bits and a fixed 4 would unpack half the
        // bytes as nibbles.
        let routedEntry = try model.packedExpertsLayout.expert(
            layer: layer, expert: 0)
        let routedBits = routedEntry.subTensors["gate"]?.bits ?? 4
        self.moe = try K2MoEStage(
            context: context, weightBits: routedBits,
            hiddenSize: hidden, expertWidth: config.moeIntermediateSize,
            numExperts: config.numExperts, topK: config.topKExperts)
        // The fused two-dispatch pass is the default; the per-expert path stays
        // available for A/B runs (`TINYTITAN_K2_FUSED_MOE=0`).
        self.fusedMoE = ProcessInfo.processInfo.environment["TINYTITAN_K2_FUSED_MOE"] != "0"
        self.vExperts = try (0..<config.mova.numExperts).map { expert in
            let view = try model.k2VExpert(layer: layer, expert: expert)
            return MoVAValueStage.ExpertWeights(
                weights: view.buffer, weightsOffset: Int(view.offset),
                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                biases: view.buffer, biasesOffset: Int(view.biasOffset))
        }
        self.sharedWeights = K2MoEStage.ExpertWeights(
            gate: try model.k2SharedGate(layer: layer),
            up: try model.k2SharedUp(layer: layer),
            down: try model.k2SharedDown(layer: layer))

        func buffer(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count * MemoryLayout<UInt16>.stride, 2),
                    options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed(label)
            }
            return buffer
        }
        self.input = try buffer(hidden, "k2.sparse.input")
        self.normed = try buffer(hidden, "k2.sparse.normed")
        self.q = try buffer(queryDim, "k2.sparse.q")
        self.k = try buffer(kvDim, "k2.sparse.k")
        self.v = try buffer(kvDim, "k2.sparse.v")
        self.gate = try buffer(queryDim, "k2.sparse.gate")
        self.attn = try buffer(queryDim, "k2.sparse.attn")
        self.oOut = try buffer(hidden, "k2.sparse.o")
        self.h1 = try buffer(hidden, "k2.sparse.h1")
        self.normed2 = try buffer(hidden, "k2.sparse.normed2")
        self.moeOut = try buffer(hidden, "k2.sparse.moeOut")
        self.out = try buffer(hidden, "k2.sparse.out")
        self.kCache = try buffer(maxSequence * kvDim, "k2.sparse.kCache")
        self.vCache = try buffer(maxSequence * kvDim, "k2.sparse.vCache")
    }

    /// The layer's output buffer, written by the routed-expert phase. The
    /// graph passes it straight to the next layer instead of reading it back.
    var output: MTLBuffer { out }

    /// Run the layer for one token, hidden state in and out through the CPU.
    /// The graph uses `prefetch`/`submitExperts` directly so the hidden state
    /// stays on the GPU; this is the self-contained form the tests drive.
    func run(
        queue: MTLCommandQueue,
        hidden: [Float16],
        position: Int
    ) async throws -> [Float] {
        precondition(hidden.count == config.hiddenSize)
        precondition(position < maxSequence)
        hidden.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                memcpy(input.contents(), base, config.hiddenSize * 2)
            }
        }
        try await prefetch(queue: queue, input: input, position: position)
        let experts = try await submitExperts(queue: queue)
        await experts.completed()
        let base = out.contents()
        return (0..<config.hiddenSize).map {
            Float(
                base.load(
                    fromByteOffset: $0 * MemoryLayout<UInt16>.stride,
                    as: Float16.self))
        }
    }

    /// Phase 1: norm, q/k/gate and the MoVA router, then the value experts
    /// (selection resolved on the GPU), RoPE, cached attention, the gate,
    /// o_proj, both residuals, the post-attention norm, the MoE router and the
    /// shared expert. Awaits the command buffer, because the routed-expert ids
    /// have to come back to the CPU to address the blobs.
    func prefetch(
        queue: MTLCommandQueue,
        input source: MTLBuffer,
        inputOffset: Int = 0,
        position: Int
    ) async throws {
        precondition(position < maxSequence)
        try checkPendingError()
        try await encodePrefetchPhase(
            queue: queue, input: source, inputOffset: inputOffset,
            position: position)
    }

    /// Phase 2: read the router ids, fetch the selected experts, and submit the
    /// routed pass. The returned command buffer is committed but not awaited,
    /// so the next layer's phase 1 can be submitted behind it; the queue order
    /// is the dependency.
    @discardableResult
    func submitExperts(queue: MTLCommandQueue) async throws -> MTLCommandBuffer {
        let d = config.hiddenSize
        let fetchStart = Date()
        let ids = moe.selectedExpertIDs()
        guard
            let plan = try model.planRoutedExperts(layer: layer, experts: ids)
        else {
            throw ModelError.internalInconsistency(
                detail: "K2 MoE could not plan expert fetches for layer \(layer)")
        }
        // A plan with no misses is fully resident; reading the slots directly
        // skips the async load operation, whose hop cost ~0.3 ms per layer --
        // 45 of them is most of a decode token's idle time. The cache planner
        // has already updated recency, so nothing else is owed.
        let blobs: [TensorView]
        if plan.misses.isEmpty {
            blobs = try model.routedExpertBuffers(for: plan)
        } else {
            blobs = try await model.fetchRoutedExperts(plan: plan)
        }
        var experts: [K2MoEStage.ExpertWeights] = []
        for (index, id) in plan.experts.enumerated() {
            let entry = try model.packedExpertsLayout.expert(
                layer: layer, expert: id)
            experts.append(
                try K2MoEStage.expertWeights(blob: blobs[index], entry: entry))
        }
        fetchSeconds += Date().timeIntervalSince(fetchStart)

        let phaseStart = Date()
        guard let third = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        // `moeOut` already holds the shared expert from phase 1, so the routed
        // pass adds to it in place.
        if fusedMoE {
            try moe.encodeFusedExperts(
                commandBuffer: third, hidden: normed2,
                experts: experts, shared: nil, out: moeOut)
        } else {
            try moe.encodeExperts(
                commandBuffer: third, hidden: normed2,
                experts: experts, shared: nil, out: moeOut)
        }
        try copy(third, from: h1, to: out, count: d)
        try elementwise.encodeResidualAdd(
            commandBuffer: third, hidden: out, delta: moeOut, count: d)
        encodeSeconds += Date().timeIntervalSince(phaseStart)
        // The completion handler is what records the GPU time now; the caller
        // does not wait here.
        third.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            let busy = max(0, buffer.gpuEndTime - buffer.gpuStartTime)
            self.timingLock.lock()
            self.busyTotal += busy
            self.expertBusyTotal += busy
            if let error = buffer.error { self.pendingError = error }
            self.timingLock.unlock()
        }
        third.commit()
        return third
    }

    private func encodePrefetchPhase(
        queue: MTLCommandQueue,
        input source: MTLBuffer,
        inputOffset: Int,
        position: Int
    ) async throws {
        let d = config.hiddenSize
        let queryDim = config.numHeads * config.fullHeadDim
        let kvDim = config.numFullKVHeads * config.fullHeadDim
        let groups = max(1, config.normGroups)
        let phaseStart = Date()
        guard let first = queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        let inputNorm = try model.inputNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: first,
            x: source, xOffset: inputOffset, weight: inputNorm.buffer,
            weightOffset: Int(inputNorm.offset),
            out: normed,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6)
        try gemv(first, try model.qProj(layer: layer), x: normed, y: q, rows: queryDim, columns: d)
        try gemv(first, try model.kProj(layer: layer), x: normed, y: k, rows: kvDim, columns: d)
        try gemv(
            first, try model.k2AttentionGate(layer: layer), x: normed, y: gate, rows: queryDim,
            columns: d)
        let vRouter = try model.k2VRouter(layer: layer)
        let vRouterBias = try model.k2VRouterBias(layer: layer)
        try mova.encodeRouter(
            commandBuffer: first, hidden: normed,
            routerWeight: vRouter.buffer,
            routerWeightOffset: Int(vRouter.offset),
            routerBias: vRouterBias.buffer,
            routerBiasOffset: Int(vRouterBias.offset),
            numExperts: config.mova.numExperts,
            scaling: Float(config.mova.routerScaling))
        try mova.encodeRoutedValues(
            commandBuffer: first, hidden: normed,
            experts: vExperts, out: v)
        let rotaryDim = UInt32(
            (Double(config.fullHeadDim) * config.partialRotaryFactor).rounded())
        try rope.encodeNeoxSubdim(
            commandBuffer: first, data: q,
            position: UInt32(position), headDim: UInt32(config.fullHeadDim),
            numHeads: UInt32(config.numHeads), rotaryDim: rotaryDim,
            theta: Float(config.fullRopeTheta))
        try rope.encodeNeoxSubdim(
            commandBuffer: first, data: k,
            position: UInt32(position), headDim: UInt32(config.fullHeadDim),
            numHeads: UInt32(config.numFullKVHeads), rotaryDim: rotaryDim,
            theta: Float(config.fullRopeTheta))
        let cacheBytes = position * kvDim * MemoryLayout<UInt16>.stride
        try copy(first, from: k, to: kCache, toOffset: cacheBytes, count: kvDim)
        try copy(first, from: v, to: vCache, toOffset: cacheBytes, count: kvDim)
        try attention.encodeFull(
            commandBuffer: first,
            q: q, k: kCache, v: vCache, out: attn,
            headDim: UInt32(config.fullHeadDim),
            numQHeads: UInt32(config.numHeads),
            numKVHeads: UInt32(config.numFullKVHeads),
            seqLen: UInt32(position + 1), scale: Float(config.attentionScale))
        try elementwise.encodeSoftplusGateMul(
            commandBuffer: first, out: attn, gate: gate, count: queryDim)
        try gemv(
            first, try model.oProj(layer: layer), x: attn, y: oOut, rows: d, columns: queryDim)
        try copy(first, from: source, fromOffset: inputOffset, to: h1, count: d)
        try elementwise.encodeResidualAdd(
            commandBuffer: first, hidden: h1, delta: oOut, count: d)
        let postNorm = try model.postAttnNorm(layer: layer)
        try rms.encodeBF16WGrouped(
            commandBuffer: first,
            x: h1, weight: postNorm.buffer,
            weightOffset: Int(postNorm.offset),
            out: normed2,
            groupDim: UInt32(d / groups), numGroups: groups,
            eps: 1e-6)
        let moeRouter = try model.k2MoeRouter(layer: layer)
        let moeBias = try model.k2MoeRouterBias(layer: layer)
        try moe.encodeRouter(
            commandBuffer: first, hidden: normed2,
            routerWeight: moeRouter.buffer,
            routerWeightOffset: Int(moeRouter.offset),
            routerBias: moeBias.buffer,
            routerBiasOffset: Int(moeBias.offset),
            scaling: Float(config.routerScaling))
        try moe.encodeSharedExpert(
            commandBuffer: first, hidden: normed2, shared: sharedWeights,
            out: moeOut)
        encodeSeconds += Date().timeIntervalSince(phaseStart)
        first.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            let busy = max(0, buffer.gpuEndTime - buffer.gpuStartTime)
            self.timingLock.lock()
            self.busyTotal += busy
            self.prefetchBusyTotal += busy
            self.timingLock.unlock()
        }
        first.commit()
        await first.completed()
    }

    private func gemv(
        _ commandBuffer: MTLCommandBuffer,
        _ view: TensorView, x: MTLBuffer, y: MTLBuffer,
        rows: Int, columns: Int
    ) throws {
        if view.dtype == 1 {
            try bf16.encode(
                commandBuffer: commandBuffer,
                weights: view.buffer, weightsOffset: Int(view.offset),
                x: x, y: y, m: UInt32(rows), n: UInt32(columns))
            return
        }
        let bits = Int(view.length) * 8 / (rows * columns)
        let encoder = bits == 8 ? gemv8 : gemv4
        try encoder.encode(
            commandBuffer: commandBuffer,
            weights: view.buffer, weightsOffset: Int(view.offset),
            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
            biases: view.buffer, biasesOffset: Int(view.biasOffset),
            x: x, y: y, m: UInt32(rows), n: UInt32(columns))
    }

    private func copy(
        _ commandBuffer: MTLCommandBuffer,
        from: MTLBuffer, fromOffset: Int = 0,
        to: MTLBuffer, toOffset: Int = 0, count: Int
    ) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: from, sourceOffset: fromOffset,
            to: to, destinationOffset: toOffset,
            size: count * MemoryLayout<UInt16>.stride)
        blit.endEncoding()
    }

    private static func weightBits(
        _ view: TensorView, rows: Int, columns: Int
    ) throws -> Int {
        let bits = Int(view.length) * 8 / (rows * columns)
        guard bits == 4 || bits == 8 else {
            throw ModelError.internalInconsistency(
                detail: "K2 value expert is \(bits)-bit; expected 4 or 8")
        }
        return bits
    }
}
