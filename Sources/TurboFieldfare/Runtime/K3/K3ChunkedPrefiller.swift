import Foundation
import Metal
import TurboFieldfareFormat

/// Read-only products the chunked prefiller reuses from `K3ForwardRunner`
/// (see `K3ForwardRunner.prefillShared`). The prefiller owns no weights.
struct K3PrefillShared {
    let scoreVectors: MTLBuffer        // fp32 fused AttnRes score vectors
    let convWeights: MTLBuffer         // fp32 [kdaOrd][3][P][4]
    let mlaKTPlane: MTLBuffer          // fp16 absorb kT planes [mlaOrd][H][L][N]
    let mlaVPlane: MTLBuffer           // fp16 absorb v planes [mlaOrd][H][V][L]
    let routerBiasByLayer: [Int: [Float]]
    let embed: K3Embed
    let lmHead: K3LMHeadGEMV
    let rmsNorm: RMSNorm
    let util: K3UtilityKernels
    let attnRes: K3AttnRes
    let logitsBuffer: MTLBuffer        // shared with the decode runner
}

/// Swift mirror of `K3PrefillPair` in prefill_k3.metal (16 bytes).
struct K3PrefillPairMSL {
    var globalPair: UInt32
    var token: UInt32
    var poolSlot: UInt32
    var weight: Float
}

/// Stage-E1 chunked prefill for Kimi K3 (docs/K3_DATAFLOW.md). Replays the
/// prompt `[chunkTokens x hidden]` rows at a time, turning every per-token
/// GEMV into a matmul (`K3PrefillGEMM`, NAX tensor-ops where available), the
/// KDA delta rule into a GPU-resident serial recurrence over the chunk, MLA
/// into batched absorbed causal attention, and the MoE into a grouped MXFP4
/// pass over double-buffered expert tiles.
///
/// The per-position math (reduction structure, fp32 accumulation, fp16
/// storage points) matches the decode path kernel-for-kernel, so the chunk
/// leaves `K3State` — KDA recurrent state, conv state, MLA latent cache —
/// bit-comparable to serial replay up to the GEMM/attention reduction order.
///
/// Command-buffer discipline mirrors `K3ForwardRunner`: one buffer accumulates
/// the embed + layers until a sync point, the only syncs being each MoE
/// layer's router readback (plus the double-buffer expert-tile waits) and the
/// end-of-chunk wait that publishes the last position's logits.
///
/// Memory: the scratch rows are sized `allocT = roundUp(chunkTokens, 64)` so
/// the NAX tensor op never reads past an allocation on a partial chunk. The
/// expert tile pool is `2 * tileSlots` blobs (double-buffered prefetch, the
/// `PrefillRoutedTileScheduler` maxPendingDepth=1 discipline); at the
/// canonical 17,547,264-byte stride and 16 slots/half that is ~560 MB.
final class K3ChunkedPrefiller {
    let model: K3Model
    let context: MetalContext
    let config: K3ArchConfig
    let state: K3State
    let shared: K3PrefillShared
    let streaming: K3ExpertStreaming
    let chunkTokens: Int

    private let gemm: K3PrefillGEMM
    private let kernels: K3PrefillKernels

    /// Chunk scratch row count, padded so a partial chunk never lets the 64-
    /// row tensor-op tile read past an allocation.
    private let allocT: Int
    private let maxBlocks: Int

    /// Expert tile pool: `2 * tileSlots` blobs, double-buffered.
    private let tileSlots: Int
    private let poolSlots: Int
    private let poolPointer: UnsafeMutableRawPointer
    private let pool: MTLBuffer
    private var layerFiles: [Int: K3ExpertLayerFile] = [:]

    /// Per-chunk wall time (ms) for the most recent `prefill` — stats hook.
    private(set) var lastChunkMillis: [Double] = []
    private var captureActivationTrace = false
    private(set) var lastRouterDiagnostics: K3RouterActivationDiagnostics?
    private(set) var lastDiagnosticHeadInput: [Float]?
    /// Final prompt position's AttnRes block list, captured only when the
    /// existing activation trace is enabled. Lifecycle regression hook.
    private(set) var lastDiagnosticAttnResBlocks: [[Float]]?

    var usingTensorOps: Bool { gemm.usingTensorOps }

    // Scratch. Everything storageModeShared; fp16 unless noted.
    private let streamA, streamB, prefix, hPre, normed, attnOut, mlpOut: MTLBuffer
    private let blocks: MTLBuffer
    private let kdaQ, kdaK, kdaV, kdaGate, kdaFA, kdaFB, kdaONorm: MTLBuffer
    private let kdaBeta16: MTLBuffer
    private let kdaConvQ, kdaConvK, kdaConvV, kdaZ, kdaStepOut: MTLBuffer  // fp32
    private let kdaBeta32: MTLBuffer                                       // fp32
    private let mlaQA, mlaQLat, mlaQ, mlaKVA, mlaGate, mlaOut: MTLBuffer
    private let mlaQAbs, mlaOutLat: MTLBuffer                              // fp32
    private let x32: MTLBuffer                                             // fp32
    private let routerScores: MTLBuffer                                    // fp32
    private let xLat, yLatNormed: MTLBuffer
    private let yLat: MTLBuffer                                            // fp32
    private let sharedGate, sharedUp, sharedH, sharedOut: MTLBuffer
    private let denseGate, denseUp, denseH: MTLBuffer
    private let headLast, headNormed: MTLBuffer
    private let tokenIDs: MTLBuffer                                        // int32
    /// Per-half pair-record buffers (double-buffered with the expert pool:
    /// tile h's GPU work may still be in flight while the CPU writes tile
    /// h+1's records — one buffer per pool half keeps the two disjoint).
    private let pairsBuf: [MTLBuffer]                                      // K3PrefillPairMSL
    private let pairH, pairOut: MTLBuffer                                  // fp32

    init(model: K3Model, context: MetalContext, state: K3State,
         shared: K3PrefillShared, streaming: K3ExpertStreaming, chunkTokens: Int,
         forceFallback: Bool? = nil) throws {
        precondition(K3PrefillMode.isAllowedChunkTokens(chunkTokens),
                     "chunkTokens \(chunkTokens) not in \(K3PrefillMode.allowedChunkTokens)")
        self.model = model
        self.context = context
        self.config = model.config
        self.state = state
        self.shared = shared
        self.streaming = streaming
        self.chunkTokens = chunkTokens
        self.gemm = try K3PrefillGEMM(context: context, forceFallback: forceFallback)
        self.kernels = try K3PrefillKernels(context: context)
        self.allocT = ((chunkTokens + K3PrefillGEMM.tileM - 1) / K3PrefillGEMM.tileM)
            * K3PrefillGEMM.tileM
        self.maxBlocks = K3AttnRes.maxBlocks
        self.tileSlots = K3ExpertStreaming.defaultSlotsPerBank
        self.poolSlots = 2 * tileSlots

        let c = model.config
        let device = context.device
        let f16 = MemoryLayout<Float16>.stride
        let f32 = MemoryLayout<Float>.stride
        func buf(_ elements: Int, _ stride: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(elements, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        let T = allocT
        let H = c.hiddenSize
        let P = c.kdaChannels

        streamA = try buf(T * H, f16, "k3.pf.stream_a")
        streamB = try buf(T * H, f16, "k3.pf.stream_b")
        prefix = try buf(T * H, f16, "k3.pf.prefix")
        hPre = try buf(T * H, f16, "k3.pf.h_pre")
        normed = try buf(T * H, f16, "k3.pf.normed")
        attnOut = try buf(T * H, f16, "k3.pf.attn_out")
        mlpOut = try buf(T * H, f16, "k3.pf.mlp_out")
        blocks = try buf(T * maxBlocks * H, f16, "k3.pf.blocks")

        kdaQ = try buf(T * P, f16, "k3.pf.kda_q")
        kdaK = try buf(T * P, f16, "k3.pf.kda_k")
        kdaV = try buf(T * P, f16, "k3.pf.kda_v")
        kdaGate = try buf(T * P, f16, "k3.pf.kda_gate")
        kdaFA = try buf(T * c.kdaDecayLowRankSize, f16, "k3.pf.kda_fa")
        kdaFB = try buf(T * P, f16, "k3.pf.kda_fb")
        kdaONorm = try buf(T * P, f16, "k3.pf.kda_onorm")
        kdaBeta16 = try buf(T * c.kdaNumHeads, f16, "k3.pf.kda_beta16")
        kdaConvQ = try buf(T * P, f32, "k3.pf.kda_conv_q")
        kdaConvK = try buf(T * P, f32, "k3.pf.kda_conv_k")
        kdaConvV = try buf(T * P, f32, "k3.pf.kda_conv_v")
        kdaZ = try buf(T * P, f32, "k3.pf.kda_z")
        kdaStepOut = try buf(T * P, f32, "k3.pf.kda_step_out")
        kdaBeta32 = try buf(T * c.kdaNumHeads, f32, "k3.pf.kda_beta32")

        mlaQA = try buf(T * c.mlaQLoraRank, f16, "k3.pf.mla_qa")
        mlaQLat = try buf(T * c.mlaQLoraRank, f16, "k3.pf.mla_qlat")
        mlaQ = try buf(T * c.mlaQElements, f16, "k3.pf.mla_q")
        mlaKVA = try buf(T * c.mlaCacheRowElements, f16, "k3.pf.mla_kva")
        mlaGate = try buf(T * c.mlaOutputElements, f16, "k3.pf.mla_gate")
        mlaOut = try buf(T * c.mlaOutputElements, f16, "k3.pf.mla_out")
        mlaQAbs = try buf(T * c.mlaNumHeads * c.mlaCacheRowElements, f32, "k3.pf.mla_qabs")
        mlaOutLat = try buf(T * c.mlaNumHeads * c.mlaKVLoraRank, f32, "k3.pf.mla_outlat")

        x32 = try buf(T * H, f32, "k3.pf.x32")
        routerScores = try buf(T * c.moeNumExperts, f32, "k3.pf.router_scores")
        xLat = try buf(T * c.moeLatentBottleneckSize, f16, "k3.pf.x_lat")
        yLat = try buf(T * c.moeLatentBottleneckSize, f32, "k3.pf.y_lat")
        yLatNormed = try buf(T * c.moeLatentBottleneckSize, f16, "k3.pf.y_lat_normed")
        sharedGate = try buf(T * c.moeSharedExpertIntermediateSize, f16, "k3.pf.sh_gate")
        sharedUp = try buf(T * c.moeSharedExpertIntermediateSize, f16, "k3.pf.sh_up")
        sharedH = try buf(T * c.moeSharedExpertIntermediateSize, f16, "k3.pf.sh_h")
        sharedOut = try buf(T * H, f16, "k3.pf.sh_out")
        denseGate = try buf(T * c.denseMLPIntermediateSize, f16, "k3.pf.dense_gate")
        denseUp = try buf(T * c.denseMLPIntermediateSize, f16, "k3.pf.dense_up")
        denseH = try buf(T * c.denseMLPIntermediateSize, f16, "k3.pf.dense_h")
        headLast = try buf(H, f16, "k3.pf.head_last")
        headNormed = try buf(H, f16, "k3.pf.head_normed")

        tokenIDs = try buf(T, MemoryLayout<Int32>.stride, "k3.pf.token_ids")
        let maxPairs = T * c.moeTopKExperts
        pairsBuf = try [
            buf(maxPairs * MemoryLayout<K3PrefillPairMSL>.stride, 1, "k3.pf.pairs0"),
            buf(maxPairs * MemoryLayout<K3PrefillPairMSL>.stride, 1, "k3.pf.pairs1"),
        ]
        pairH = try buf(maxPairs * c.moeExpertIntermediateSize, f32, "k3.pf.pair_h")
        pairOut = try buf(maxPairs * c.moeLatentBottleneckSize, f32, "k3.pf.pair_out")

        // Double-buffered expert tile pool (pread into the raw pointer, GPU
        // reads the wrapped buffer).
        let poolBytes = poolSlots * Int(c.expertStride)
        var raw: UnsafeMutableRawPointer?
        let result = posix_memalign(&raw, K3ExpertStreaming.slotAlignment, poolBytes)
        guard result == 0, let poolPtr = raw else {
            throw StreamerError.allocFailed(errno: result)
        }
        nonisolated(unsafe) let captured = poolPtr
        guard let poolBuf = device.makeBuffer(bytesNoCopy: poolPtr, length: poolBytes,
                                              options: .storageModeShared,
                                              deallocator: { _, _ in free(captured) }) else {
            free(poolPtr)
            throw StreamerError.bufferWrapFailed
        }
        poolBuf.label = "k3.pf.expert_pool"
        self.poolPointer = poolPtr
        self.pool = poolBuf
    }

    func enableActivationTrace() {
        captureActivationTrace = true
        lastRouterDiagnostics = nil
        lastDiagnosticHeadInput = nil
        lastDiagnosticAttnResBlocks = nil
    }

    // MARK: - Entry

    /// Prefill the whole prompt in chunks of `chunkTokens`. Advances
    /// `state.position` by `tokens.count` and leaves the last position's fp32
    /// logits in `shared.logitsBuffer`.
    func prefill(tokens: [Int32]) throws {
        precondition(!tokens.isEmpty)
        precondition(state.position + tokens.count <= state.maxContext,
                     "prefill would exceed maxContext")
        lastChunkMillis = []
        if captureActivationTrace {
            lastRouterDiagnostics = nil
            lastDiagnosticHeadInput = nil
            lastDiagnosticAttnResBlocks = nil
        }
        var offset = 0
        while offset < tokens.count {
            let count = min(chunkTokens, tokens.count - offset)
            let start = state.position
            let emitHead = (offset + count == tokens.count)
            let t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try prefillChunk(Array(tokens[offset..<(offset + count)]),
                             chunkStart: start, emitHead: emitHead)
            state.advance(by: count)
            lastChunkMillis.append(
                Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0) / 1e6)
            offset += count
        }
    }

    // MARK: - Per-chunk forward

    private func prefillChunk(_ tokens: [Int32], chunkStart: Int, emitHead: Bool) throws {
        let c = config
        let T = tokens.count
        let H = c.hiddenSize
        let eps = Float(c.rmsNormEpsilon)
        let rowBytes = T * H * MemoryLayout<Float16>.stride

        // Stage the token ids for the batched embed gather.
        let tokenPtr = tokenIDs.contents().bindMemory(to: Int32.self, capacity: T)
        for (i, tok) in tokens.enumerated() { tokenPtr[i] = tok }

        var blockCount = 0
        var prefixValid = false

        guard var cb = context.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        var committed = false
        func unwind() {
            if !committed { cb.commit() }
            cb.waitUntilCompleted()
        }

        do {
            let emb = model.embedding
            kernels.encodeEmbedGather(commandBuffer: cb,
                                      table: emb.buffer, tableOffset: Int(emb.offset),
                                      scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                      biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                      tokens: tokenIDs, out: streamA,
                                      tokenCount: T, hidden: H)
            var stream = streamA

            for layer in 0..<c.numLayers {
                // Preserve the official `prefix_sum`: pre-attn AttnRes
                // changes the attention input only. The incoming stream is
                // what a boundary appends and a non-boundary residual carries.
                let incoming = stream
                // 1. Pre-attn AttnRes (skipped on the empty block list).
                if blockCount > 0 {
                    let out = stream === streamA ? streamB : streamA
                    kernels.encodeAttnRes(commandBuffer: cb,
                                          blocks: blocks, prefix: stream,
                                          scoreVector: shared.scoreVectors,
                                          scoreVectorOffset: scoreVectorOffset(layer: layer, site: 0)
                                              * MemoryLayout<Float>.stride,
                                          out: out, tokens: T, hidden: H,
                                          numBlocks: blockCount, maxBlocks: maxBlocks, eps: eps)
                    stream = out
                }
                // 2. Boundary append / prefix capture.
                if c.isAttnResBoundary(layer0: layer) {
                    kernels.encodeBlockAppend(commandBuffer: cb, src: incoming, dst: blocks,
                                              tokens: T, hidden: H, slot: blockCount,
                                              maxBlocks: maxBlocks)
                    blockCount += 1
                    prefixValid = false
                } else {
                    if incoming !== prefix {
                        blitCopy(commandBuffer: cb, from: incoming, to: prefix,
                                 destinationOffset: 0, bytes: rowBytes)
                    }
                    prefixValid = true
                }
                // 3. input_layernorm -> attention -> o_proj (into attnOut).
                kernels.encodeRMSNormBF16W(commandBuffer: cb, x: stream,
                                           weight: model.inputNorm(layer: layer).buffer,
                                           weightOffset: Int(model.inputNorm(layer: layer).offset),
                                           out: normed, tokens: T, d: H, eps: eps)
                if c.isKDA(layer0: layer) {
                    try encodeKDAPrefill(layer: layer, tokens: T, commandBuffer: cb)
                } else {
                    try encodeMLAPrefill(layer: layer, tokens: T, chunkStart: chunkStart,
                                         commandBuffer: cb)
                }
                // 4. Prefix update (attention).
                encodePrefixAccumulate(commandBuffer: cb, value: attnOut,
                                       prefixValid: &prefixValid, rowBytes: rowBytes)
                // 5. Pre-mlp AttnRes over the post-append block list.
                kernels.encodeAttnRes(commandBuffer: cb,
                                      blocks: blocks, prefix: prefix,
                                      scoreVector: shared.scoreVectors,
                                      scoreVectorOffset: scoreVectorOffset(layer: layer, site: 1)
                                          * MemoryLayout<Float>.stride,
                                      out: hPre, tokens: T, hidden: H,
                                      numBlocks: blockCount, maxBlocks: maxBlocks, eps: eps)
                // 6. post_attention_layernorm -> MoE / dense.
                kernels.encodeRMSNormBF16W(commandBuffer: cb, x: hPre,
                                           weight: model.postAttnNorm(layer: layer).buffer,
                                           weightOffset: Int(model.postAttnNorm(layer: layer).offset),
                                           out: normed, tokens: T, d: H, eps: eps)
                if c.isMoE(layer0: layer) {
                    try encodeMoEPrefill(layer: layer, tokens: T,
                                         commandBuffer: &cb, committed: &committed,
                                         prefixValid: &prefixValid, rowBytes: rowBytes)
                } else {
                    try encodeDensePrefill(layer: layer, tokens: T, commandBuffer: cb)
                    encodePrefixAccumulate(commandBuffer: cb, value: mlpOut,
                                           prefixValid: &prefixValid, rowBytes: rowBytes)
                }
                // The prefix is the next layer's stream.
                stream = prefix
            }

            if emitHead {
                try encodeHead(commandBuffer: cb, tokens: T, blockCount: blockCount,
                               stream: stream, eps: eps)
            }
            cb.commit()
            committed = true
            cb.waitUntilCompleted()
            if let error = cb.error {
                throw MetalError.libraryCompileFailed(
                    "K3 chunked prefill final command failed "
                        + "(chunkStart=\(chunkStart) tokens=\(T)): \(error)")
            }
            if captureActivationTrace, emitHead {
                let ptr = headNormed.contents().bindMemory(
                    to: Float16.self, capacity: config.hiddenSize)
                lastDiagnosticHeadInput = (0..<config.hiddenSize).map { Float(ptr[$0]) }
                let blockPtr = blocks.contents().bindMemory(
                    to: Float16.self, capacity: allocT * maxBlocks * H)
                let tokenBase = (T - 1) * maxBlocks * H
                lastDiagnosticAttnResBlocks = (0..<blockCount).map { block in
                    let base = tokenBase + block * H
                    return (0..<H).map { Float(blockPtr[base + $0]) }
                }
            }
        } catch {
            unwind()
            throw error
        }
    }

    // MARK: - Layer stages

    private func scoreVectorOffset(layer: Int, site: Int) -> Int {
        (site * config.numLayers + layer) * config.hiddenSize
    }

    private func blitCopy(commandBuffer: MTLCommandBuffer,
                          from source: MTLBuffer, to destination: MTLBuffer,
                          destinationOffset: Int, bytes: Int) {
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else { return }
        encoder.copy(from: source, sourceOffset: 0,
                     to: destination, destinationOffset: destinationOffset, size: bytes)
        encoder.endEncoding()
    }

    private func encodePrefixAccumulate(commandBuffer: MTLCommandBuffer,
                                        value: MTLBuffer, prefixValid: inout Bool,
                                        rowBytes: Int) {
        if prefixValid {
            shared.util.encodeAdd(commandBuffer: commandBuffer, a: prefix, b: value,
                                  out: prefix, n: UInt32(config.hiddenSize) * UInt32(rowBytes
                                      / (config.hiddenSize * MemoryLayout<Float16>.stride)))
        } else {
            blitCopy(commandBuffer: commandBuffer, from: value, to: prefix,
                     destinationOffset: 0, bytes: rowBytes)
            prefixValid = true
        }
    }

    private func encodeKDAPrefill(layer: Int, tokens T: Int,
                                  commandBuffer cb: MTLCommandBuffer) throws {
        let c = config
        let P = c.kdaChannels
        let f16 = MemoryLayout<Float16>.stride
        guard let ordinal = c.kdaOrdinal(layer0: layer) else { return }

        // Projections (NAX / fallback QMM), all reading the normed input.
        try gemm.encode(commandBuffer: cb, view: model.kdaQProj(layer: layer),
                        x: normed, y: kdaQ, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.kdaKProj(layer: layer),
                        x: normed, y: kdaK, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.kdaVProj(layer: layer),
                        x: normed, y: kdaV, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.kdaGProj(layer: layer),
                        x: normed, y: kdaGate, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.kdaFAProj(layer: layer),
                        x: normed, y: kdaFA, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.kdaBProj(layer: layer),
                        x: normed, y: kdaBeta16, tokens: T)

        let convState = state.kdaConvState(layer0: layer)
        let convWeightsOffset = ordinal * 3 * P * c.kdaConvWidth * MemoryLayout<Float>.stride
        kernels.encodeKDAConv(commandBuffer: cb,
                              xq: kdaQ, xk: kdaK, xv: kdaV,
                              weights: shared.convWeights, weightsOffset: convWeightsOffset,
                              convStates: convState.buffer, convStatesOffset: convState.offset,
                              qOut: kdaConvQ, kOut: kdaConvK, vOut: kdaConvV,
                              channels: P, tokens: T)
        kernels.encodeKDAConvState(commandBuffer: cb,
                                   xq: kdaQ, xk: kdaK, xv: kdaV,
                                   convStates: convState.buffer, convStatesOffset: convState.offset,
                                   channels: P, tokens: T)

        // z = f_b(f_a(x)) + dt_bias; beta = widen(b_proj(x)).
        try gemm.encode(commandBuffer: cb, view: model.kdaFBProj(layer: layer),
                        x: kdaFA, y: kdaFB, tokens: T)
        let dtBias = model.kdaDTBias(layer: layer)
        kernels.encodeAddBias(commandBuffer: cb, x: kdaFB,
                              bias: dtBias.buffer, biasOffset: Int(dtBias.offset),
                              out: kdaZ, width: P, tokens: T)
        shared.util.encodeWiden(commandBuffer: cb, x: kdaBeta16, out: kdaBeta32,
                                n: UInt32(T * c.kdaNumHeads))

        let recurrent = state.kdaRecurrentState(layer0: layer)
        let aLog = model.kdaALog(layer: layer)
        kernels.encodeKDASerial(commandBuffer: cb,
                                state: recurrent.buffer, stateOffset: recurrent.offset,
                                q: kdaConvQ, k: kdaConvK, v: kdaConvV, z: kdaZ,
                                betaLogits: kdaBeta32,
                                aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                o: kdaStepOut,
                                numHeads: c.kdaNumHeads, headDim: c.kdaHeadDim, tokens: T)

        let oNorm = model.kdaONorm(layer: layer)
        kernels.encodeKDAONorm(commandBuffer: cb, x: kdaStepOut, gate: kdaGate,
                               weight: oNorm.buffer, weightOffset: Int(oNorm.offset),
                               out: kdaONorm, numHeads: c.kdaNumHeads,
                               headDim: c.kdaHeadDim, tokens: T, eps: Float(c.rmsNormEpsilon))
        try gemm.encode(commandBuffer: cb, view: model.kdaOProj(layer: layer),
                        x: kdaONorm, y: attnOut, tokens: T)
        _ = f16
    }

    private func encodeMLAPrefill(layer: Int, tokens T: Int, chunkStart: Int,
                                  commandBuffer cb: MTLCommandBuffer) throws {
        let c = config
        let f16 = MemoryLayout<Float16>.stride
        guard let ordinal = c.mlaOrdinal(layer0: layer) else { return }
        let L = c.mlaKVLoraRank, R = c.mlaQKRopeHeadDim, N = c.mlaQKNopeHeadDim
        let V = c.mlaVHeadDim, mlaHeads = c.mlaNumHeads

        try gemm.encode(commandBuffer: cb, view: model.mlaQAProj(layer: layer),
                        x: normed, y: mlaQA, tokens: T)
        let qaNorm = model.mlaQANorm(layer: layer)
        kernels.encodeRMSNormF32W(commandBuffer: cb, x: mlaQA,
                                  weight: qaNorm.buffer, weightOffset: Int(qaNorm.offset),
                                  out: mlaQLat, tokens: T, d: c.mlaQLoraRank,
                                  eps: K3ChunkedPrefiller.mlaLoraNormEps)
        try gemm.encode(commandBuffer: cb, view: model.mlaQBProj(layer: layer),
                        x: mlaQLat, y: mlaQ, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.mlaKVAProj(layer: layer),
                        x: normed, y: mlaKVA, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.mlaGProj(layer: layer),
                        x: normed, y: mlaGate, tokens: T)

        let cache = state.mlaCache(layer0: layer)
        let kvNorm = model.mlaKVANorm(layer: layer)
        kernels.encodeMLACacheAppend(commandBuffer: cb, kvA: mlaKVA,
                                     normWeight: kvNorm.buffer,
                                     normWeightOffset: Int(kvNorm.offset),
                                     cache: cache.buffer, cacheOffset: cache.offset,
                                     chunkStart: chunkStart, latent: L, rope: R,
                                     tokens: T, eps: K3ChunkedPrefiller.mlaLoraNormEps)
        kernels.encodeMLAAbsorbQ(commandBuffer: cb, kT: shared.mlaKTPlane,
                                 kTOffset: ordinal * mlaHeads * L * N * f16,
                                 q: mlaQ, qAbs: mlaQAbs,
                                 numHeads: mlaHeads, latent: L, rope: R, nope: N, tokens: T)
        kernels.encodeMLAAttention(commandBuffer: cb, cache: cache.buffer,
                                   cacheOffset: cache.offset, qAbs: mlaQAbs, outLat: mlaOutLat,
                                   numHeads: mlaHeads, latent: L, rope: R,
                                   chunkStart: chunkStart, scale: c.mlaDecodeScale, tokens: T)
        kernels.encodeMLAOutProject(commandBuffer: cb, v: shared.mlaVPlane,
                                    vOffset: ordinal * mlaHeads * V * L * f16,
                                    outLat: mlaOutLat, gate: mlaGate, out: mlaOut,
                                    numHeads: mlaHeads, latent: L, vHead: V, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.mlaOProj(layer: layer),
                        x: mlaOut, y: attnOut, tokens: T)
    }

    /// KimiRMSNorm default eps for the MLA q_a / kv_a layernorms (they are
    /// built without config.rms_norm_eps in the reference model).
    private static let mlaLoraNormEps: Float = 1e-6

    private func encodeDensePrefill(layer: Int, tokens T: Int,
                                    commandBuffer cb: MTLCommandBuffer) throws {
        let c = config
        try gemm.encode(commandBuffer: cb, view: model.denseGateProj(layer: layer),
                        x: normed, y: denseGate, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.denseUpProj(layer: layer),
                        x: normed, y: denseUp, tokens: T)
        shared.util.encodeSiTU(commandBuffer: cb, gate: denseGate, up: denseUp,
                               out: denseH, n: UInt32(T * c.denseMLPIntermediateSize),
                               beta1: Float(c.situGLUGateBeta), beta2: Float(c.situGLUUpBeta))
        try gemm.encode(commandBuffer: cb, view: model.denseDownProj(layer: layer),
                        x: denseH, y: mlpOut, tokens: T)
    }

    // MARK: - Head

    /// Output AttnRes -> final norm -> int8 lm_head for the LAST position of
    /// the last chunk, writing fp32 logits into `shared.logitsBuffer`. The
    /// single-position decode kernels serve here: the head is one row, so
    /// batching it would buy nothing.
    private func encodeHead(commandBuffer cb: MTLCommandBuffer, tokens T: Int,
                            blockCount: Int, stream: MTLBuffer, eps: Float) throws {
        let c = config
        let H = c.hiddenSize
        let f16 = MemoryLayout<Float16>.stride
        shared.attnRes.encode(commandBuffer: cb,
                              blocks: blocks,
                              blocksOffset: (T - 1) * maxBlocks * H * f16,
                              prefix: stream,
                              prefixOffset: (T - 1) * H * f16,
                              scoreVector: shared.scoreVectors,
                              scoreVectorOffset: scoreVectorOffset(layer: 0, site: 2)
                                  * MemoryLayout<Float>.stride,
                              out: headLast,
                              hidden: UInt32(H),
                              numBlocks: UInt32(blockCount),
                              eps: eps)
        shared.rmsNorm.encodeBF16W(commandBuffer: cb,
                                   x: headLast,
                                   weight: model.finalNorm.buffer,
                                   weightOffset: Int(model.finalNorm.offset),
                                   out: headNormed, d: UInt32(H), eps: eps)
        let head = model.lmHead
        shared.lmHead.encode(commandBuffer: cb,
                             weights: head.buffer, weightsOffset: Int(head.offset),
                             scales: head.buffer, scalesOffset: Int(head.scaleOffset),
                             biases: head.buffer, biasesOffset: Int(head.biasOffset),
                             x: headNormed, y: shared.logitsBuffer,
                             m: UInt32(c.vocabSize), n: UInt32(H))
    }

    // MARK: - MoE

    /// Router + down + shared-expert into `cb`, commit + wait for the router
    /// readback, then double-buffered expert tiles and the reduce / up-proj.
    /// Updates `cb` to the buffer holding the block's tail so the caller keeps
    /// encoding subsequent layers into it.
    private func encodeMoEPrefill(layer: Int, tokens T: Int,
                                  commandBuffer cb: inout MTLCommandBuffer,
                                  committed: inout Bool,
                                  prefixValid: inout Bool, rowBytes: Int) throws {
        let c = config
        let H = c.hiddenSize
        let dLatent = c.moeLatentBottleneckSize
        let topK = c.moeTopKExperts

        // Router scores + the shared/down trunk work (no expert dependence).
        shared.util.encodeWiden(commandBuffer: cb, x: normed, out: x32,
                                n: UInt32(T * H))
        let gate = model.routerGate(layer: layer)
        kernels.encodeRouter(commandBuffer: cb, gate: gate.buffer,
                             gateOffset: Int(gate.offset), x: x32, scores: routerScores,
                             tokens: T, numExperts: c.moeNumExperts, hidden: H)
        try gemm.encode(commandBuffer: cb, view: model.routedDownProj(layer: layer),
                        x: normed, y: xLat, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.sharedGateProj(layer: layer),
                        x: normed, y: sharedGate, tokens: T)
        try gemm.encode(commandBuffer: cb, view: model.sharedUpProj(layer: layer),
                        x: normed, y: sharedUp, tokens: T)
        shared.util.encodeSiTU(commandBuffer: cb, gate: sharedGate, up: sharedUp,
                               out: sharedH, n: UInt32(T * c.moeSharedExpertIntermediateSize),
                               beta1: Float(c.situGLUGateBeta), beta2: Float(c.situGLUUpBeta))
        try gemm.encode(commandBuffer: cb, view: model.sharedDownProj(layer: layer),
                        x: sharedH, y: sharedOut, tokens: T)

        cb.commit()
        committed = true
        cb.waitUntilCompleted()
        if let error = cb.error {
            throw MetalError.libraryCompileFailed(
                "K3 chunked prefill pre-MoE command failed "
                    + "(layer=\(layer) tokens=\(T)): \(error)")
        }

        if captureActivationTrace, layer == c.moeLayers0.min() {
            let row = T - 1
            let inputPtr = x32.contents().bindMemory(
                to: Float.self, capacity: T * H) + row * H
            let scorePtr = routerScores.contents().bindMemory(
                to: Float.self, capacity: T * c.moeNumExperts)
                + row * c.moeNumExperts
            lastRouterDiagnostics = try K3ActivationReference.routerDiagnostics(
                model: model, layer: layer, tokenInChunk: row,
                input: Array(UnsafeBufferPointer(start: inputPtr, count: H)),
                actualScores: Array(UnsafeBufferPointer(
                    start: scorePtr, count: c.moeNumExperts)))
        }

        // CPU top-k per position + expert-major grouping into tiles.
        let plan = buildMoEPlan(layer: layer, tokens: T)
        // Seed the decode-time prediction with the last position's routing —
        // exactly what serial replay leaves in the streaming engine after the
        // final prompt token, so the first decoded token prefetches warm.
        let topKCount = c.moeTopKExperts
        streaming.recordRouting(
            layer0: layer,
            experts: plan.pairExpert[((T - 1) * topKCount)..<(T * topKCount)]
                .map(Int.init))

        // Double-buffered expert tiles: pread half h+1 while half h computes.
        // The same-half wait before readExpertTile covers BOTH the pool bytes
        // and that half's pair-record buffer (tile h-2's consumer is done).
        let offsets = K3ExpertSubtensorOffsets.canonical(
            dLatent: UInt32(dLatent), intermediate: UInt32(c.moeExpertIntermediateSize))
        var halfCB: [MTLCommandBuffer?] = [nil, nil]
        for (tileIndex, tile) in plan.tiles.enumerated() {
            let half = tileIndex & 1
            if let prior = halfCB[half] {
                prior.waitUntilCompleted()
                if let error = prior.error {
                    throw MetalError.libraryCompileFailed(
                        "K3 chunked prefill expert tile failed "
                            + "(layer=\(layer) tile=\(max(tileIndex - 2, 0)) "
                            + "tokens=\(T)): \(error)")
                }
            }
            try readExpertTile(layer: layer, experts: tile.experts, half: half)
            writeTilePairs(plan: plan, tile: tile, half: half, into: pairsBuf[half])
            guard let tileCB = context.queue.makeCommandBuffer() else {
                throw MetalError.noQueue
            }
            kernels.encodeMoEPhase1(commandBuffer: tileCB, experts: pool,
                                    pairs: pairsBuf[half], xLat: xLat, h: pairH,
                                    subtensorOffsets: offsets,
                                    expertStride: Int(c.expertStride),
                                    dLatent: dLatent,
                                    intermediate: c.moeExpertIntermediateSize,
                                    tilePairCount: tile.pairIndices.count)
            kernels.encodeMoEPhase2(commandBuffer: tileCB, experts: pool,
                                    pairs: pairsBuf[half], h: pairH, pairOut: pairOut,
                                    subtensorOffsets: offsets,
                                    expertStride: Int(c.expertStride),
                                    dLatent: dLatent,
                                    intermediate: c.moeExpertIntermediateSize,
                                    tilePairCount: tile.pairIndices.count)
            tileCB.label = "k3.prefill.layer\(layer).expertTile\(tileIndex)"
            tileCB.commit()
            halfCB[half] = tileCB
        }
        for (half, tileCB) in halfCB.enumerated() {
            tileCB?.waitUntilCompleted()
            if let error = tileCB?.error {
                throw MetalError.libraryCompileFailed(
                    "K3 chunked prefill final expert tile failed "
                        + "(layer=\(layer) half=\(half) tokens=\(T)): \(error)")
            }
        }

        // Reduce + norm + up-proj + shared add, into the continuing buffer.
        guard let nextCB = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        cb = nextCB
        committed = false
        kernels.encodeMoEReduce(commandBuffer: cb, pairOut: pairOut, yLat: yLat,
                                dLatent: dLatent, topK: topK, tokens: T)
        let rNorm = model.routedExpertNorm(layer: layer)
        kernels.encodeRMSNormF32X(commandBuffer: cb, x: yLat,
                                  weight: rNorm.buffer, weightOffset: Int(rNorm.offset),
                                  out: yLatNormed, tokens: T, d: dLatent,
                                  eps: Float(c.rmsNormEpsilon))
        try gemm.encode(commandBuffer: cb, view: model.routedUpProj(layer: layer),
                        x: yLatNormed, y: mlpOut, tokens: T)
        shared.util.encodeAdd(commandBuffer: cb, a: mlpOut, b: sharedOut, out: mlpOut,
                              n: UInt32(T * H))
        encodePrefixAccumulate(commandBuffer: cb, value: mlpOut,
                               prefixValid: &prefixValid, rowBytes: rowBytes)
    }

    /// CPU top-k per position -> token-major pairs + expert-major tiles.
    private struct MoEPlan {
        var pairExpert: [UInt32]    // [T*topK]
        var pairWeight: [Float]     // [T*topK]
        struct Tile {
            var experts: [Int]          // distinct experts, pool-slot order
            var pairIndices: [Int]      // token-major global pair indices
        }
        var tiles: [Tile]
    }

    private func buildMoEPlan(layer: Int, tokens T: Int) -> MoEPlan {
        let c = config
        let topK = c.moeTopKExperts
        let numExperts = c.moeNumExperts
        let bias = shared.routerBiasByLayer[layer] ?? [Float](repeating: 0, count: numExperts)
        let scorePtr = routerScores.contents().bindMemory(to: Float.self,
                                                          capacity: T * numExperts)

        var pairExpert = [UInt32](repeating: 0, count: T * topK)
        var pairWeight = [Float](repeating: 0, count: T * topK)
        for t in 0..<T {
            let row = Array(UnsafeBufferPointer(start: scorePtr + t * numExperts,
                                                count: numExperts))
            let (indices, weights) = K3Router.selectTopK(scores: row, bias: bias,
                                                         topK: topK)
            for rank in 0..<topK {
                pairExpert[t * topK + rank] = indices[rank]
                pairWeight[t * topK + rank] = weights[rank]
            }
        }

        // Expert-major grouping: sort global pair indices by expert (then by
        // pair for determinism), run-length into experts, chunk the runs into
        // tiles of at most tileSlots distinct experts.
        let order = (0..<(T * topK)).sorted { a, b in
            if pairExpert[a] != pairExpert[b] { return pairExpert[a] < pairExpert[b] }
            return a < b
        }
        var tiles: [MoEPlan.Tile] = []
        var currentExperts: [Int] = []
        var currentPairs: [Int] = []
        var lastExpert = -1
        func flush() {
            if !currentExperts.isEmpty {
                tiles.append(MoEPlan.Tile(experts: currentExperts, pairIndices: currentPairs))
            }
            currentExperts = []
            currentPairs = []
        }
        for p in order {
            let e = Int(pairExpert[p])
            if e != lastExpert {
                if currentExperts.count == tileSlots { flush() }
                currentExperts.append(e)
                lastExpert = e
            }
            currentPairs.append(p)
        }
        flush()
        return MoEPlan(pairExpert: pairExpert, pairWeight: pairWeight, tiles: tiles)
    }

    /// Write one tile's pair records (pool-slot resolved) into the half's
    /// pair buffer.
    private func writeTilePairs(plan: MoEPlan, tile: MoEPlan.Tile, half: Int,
                                into buffer: MTLBuffer) {
        var slotOfExpert: [Int: UInt32] = [:]
        for (slotInHalf, expert) in tile.experts.enumerated() {
            slotOfExpert[expert] = UInt32(half * tileSlots + slotInHalf)
        }
        let ptr = buffer.contents().bindMemory(to: K3PrefillPairMSL.self,
                                               capacity: tile.pairIndices.count)
        let topK = config.moeTopKExperts
        for (i, globalPair) in tile.pairIndices.enumerated() {
            ptr[i] = K3PrefillPairMSL(
                globalPair: UInt32(globalPair),
                token: UInt32(globalPair / topK),
                poolSlot: slotOfExpert[Int(plan.pairExpert[globalPair])] ?? 0,
                weight: plan.pairWeight[globalPair])
        }
    }

    // MARK: - Expert tile I/O

    private func layerFile(for layer0: Int) throws -> K3ExpertLayerFile {
        if let file = layerFiles[layer0] { return file }
        let file = try streaming.expertLayerFileForPrefill(layer0)
        layerFiles[layer0] = file
        return file
    }

    /// Pread a tile's experts into pool half `half` (slots
    /// `half*tileSlots ..`) through the decode streamer's shared bounded and
    /// adaptively tuned I/O scheduler.
    private func readExpertTile(layer: Int, experts: [Int], half: Int) throws {
        let file = try layerFile(for: layer)
        let stride = Int(config.expertStride)
        let base = half * tileSlots
        let destinations = experts.indices.map {
            poolPointer.advanced(by: (base + $0) * stride)
        }
        try streaming.readExpertsForPrefill(
            file: file, experts: experts, destinations: destinations)
    }
}
