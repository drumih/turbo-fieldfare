import Foundation
import Metal
import TurboFieldfareFormat

/// Dispatchers for the batched Stage-E1 prefill kernels in `prefill_k3.metal`.
/// One owner compiles every pipeline off the concatenated K3 library; each
/// `encode*` mirrors the corresponding decode dispatcher's math with an added
/// token dimension. All scratch is caller-owned (`K3ChunkedPrefiller`).
final class K3PrefillKernels {
    private let routerPSO: MTLComputePipelineState
    private let rmsBF16WPSO: MTLComputePipelineState
    private let rmsF32WPSO: MTLComputePipelineState
    private let rmsF32XPSO: MTLComputePipelineState
    private let attnResPSO: MTLComputePipelineState
    private let embedPSO: MTLComputePipelineState
    private let blockAppendPSO: MTLComputePipelineState
    private let kdaConvPSO: MTLComputePipelineState
    private let kdaConvStatePSO: MTLComputePipelineState
    private let kdaSerialPSO: MTLComputePipelineState
    private let kdaONormPSO: MTLComputePipelineState
    private let addBiasPSO: MTLComputePipelineState
    private let mlaAbsorbPSO: MTLComputePipelineState
    private let mlaCacheAppendPSO: MTLComputePipelineState
    private let mlaAttentionPSO: MTLComputePipelineState
    private let mlaOutProjectPSO: MTLComputePipelineState
    private let moePhase1PSO: MTLComputePipelineState
    private let moePhase2PSO: MTLComputePipelineState
    private let moeReducePSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        let device = context.device
        routerPSO = try library.pipeline(device: device, name: "k3_router_prefill")
        rmsBF16WPSO = try library.pipeline(device: device, name: "k3_rmsnorm_bf16w_prefill")
        rmsF32WPSO = try library.pipeline(device: device, name: "k3_rmsnorm_f32w_prefill")
        rmsF32XPSO = try library.pipeline(device: device, name: "k3_rmsnorm_f32x_prefill")
        attnResPSO = try library.pipeline(device: device, name: "k3_attnres_prefill")
        embedPSO = try library.pipeline(device: device, name: "k3_embed_gather_prefill")
        blockAppendPSO = try library.pipeline(device: device, name: "k3_block_append_prefill")
        kdaConvPSO = try library.pipeline(device: device, name: "k3_kda_conv_prefill")
        kdaConvStatePSO = try library.pipeline(device: device, name: "k3_kda_conv_prefill_state")
        kdaSerialPSO = try library.pipeline(device: device, name: "k3_kda_prefill_serial")
        kdaONormPSO = try library.pipeline(device: device, name: "k3_kda_onorm_prefill")
        addBiasPSO = try library.pipeline(device: device, name: "k3_addbias_prefill")
        mlaAbsorbPSO = try library.pipeline(device: device, name: "k3_mla_absorb_q_prefill")
        mlaCacheAppendPSO = try library.pipeline(device: device, name: "k3_mla_cache_append_prefill")
        mlaAttentionPSO = try library.pipeline(device: device, name: "k3_mla_prefill_attention")
        mlaOutProjectPSO = try library.pipeline(device: device, name: "k3_mla_out_project_prefill")
        moePhase1PSO = try library.pipeline(device: device, name: "k3_moe_prefill_phase1")
        moePhase2PSO = try library.pipeline(device: device, name: "k3_moe_prefill_phase2")
        moeReducePSO = try library.pipeline(device: device, name: "k3_moe_prefill_reduce")
    }

    // MARK: - Dispatch helpers

    /// 2D threadgroup grid over (xCount, yCount) with 256-thread groups, for
    /// kernels indexed by `thread_position_in_grid`.
    private func dispatch2D(encoder: MTLComputeCommandEncoder,
                            xCount: Int, yCount: Int, threads: Int = 256) {
        encoder.dispatchThreadgroups(
            MTLSize(width: (xCount + threads - 1) / threads, height: yCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// One SIMD per output row, 8 rows per threadgroup, grid (rows/8, tokens).
    private func dispatchSimdRows(encoder: MTLComputeCommandEncoder,
                                  rows: Int, tokens: Int) {
        let rowsPerTG = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerTG - 1) / rowsPerTG, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerTG, height: 1, depth: 1))
    }

    // MARK: - Router

    /// scores[t][e] = sigmoid(gate[e] . x32[t]); bias added on the CPU.
    func encodeRouter(commandBuffer: MTLCommandBuffer,
                      gate: MTLBuffer, gateOffset: Int = 0,
                      x: MTLBuffer, xOffset: Int = 0,
                      scores: MTLBuffer, scoresOffset: Int = 0,
                      tokens: Int, numExperts: Int, hidden: Int) {
        var t = UInt32(tokens), e = UInt32(numExperts), h = UInt32(hidden)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerPSO)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(scores, offset: scoresOffset, index: 2)
        encoder.setBytes(&t, length: 4, index: 3)
        encoder.setBytes(&e, length: 4, index: 4)
        encoder.setBytes(&h, length: 4, index: 5)
        dispatchSimdRows(encoder: encoder, rows: numExperts, tokens: tokens)
        encoder.endEncoding()
    }

    // MARK: - Norms (one threadgroup per position)

    func encodeRMSNormBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            tokens: Int, d: Int, eps: Float) {
        encodeNorm(commandBuffer: commandBuffer, pipeline: rmsBF16WPSO,
                   x: x, xOffset: xOffset, weight: weight, weightOffset: weightOffset,
                   out: out, outOffset: outOffset, tokens: tokens, d: d, eps: eps)
    }

    func encodeRMSNormF32W(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           weight: MTLBuffer, weightOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           tokens: Int, d: Int, eps: Float) {
        encodeNorm(commandBuffer: commandBuffer, pipeline: rmsF32WPSO,
                   x: x, xOffset: xOffset, weight: weight, weightOffset: weightOffset,
                   out: out, outOffset: outOffset, tokens: tokens, d: d, eps: eps)
    }

    func encodeRMSNormF32X(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           weight: MTLBuffer, weightOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           tokens: Int, d: Int, eps: Float) {
        encodeNorm(commandBuffer: commandBuffer, pipeline: rmsF32XPSO,
                   x: x, xOffset: xOffset, weight: weight, weightOffset: weightOffset,
                   out: out, outOffset: outOffset, tokens: tokens, d: d, eps: eps)
    }

    private func encodeNorm(commandBuffer: MTLCommandBuffer,
                            pipeline: MTLComputePipelineState,
                            x: MTLBuffer, xOffset: Int,
                            weight: MTLBuffer, weightOffset: Int,
                            out: MTLBuffer, outOffset: Int,
                            tokens: Int, d: Int, eps: Float) {
        var dim = UInt32(d), e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&dim, length: 4, index: 3)
        encoder.setBytes(&e, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - AttnRes / embed / block append

    /// out[t] = AttnRes(blocks[t][0..<numBlocks], prefix[t], scoreVec).
    func encodeAttnRes(commandBuffer: MTLCommandBuffer,
                       blocks: MTLBuffer, blocksOffset: Int = 0,
                       prefix: MTLBuffer, prefixOffset: Int = 0,
                       scoreVector: MTLBuffer, scoreVectorOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       tokens: Int, hidden: Int, numBlocks: Int, maxBlocks: Int, eps: Float) {
        precondition(numBlocks <= 15, "attnres prefill staging holds <= 15 vectors")
        var h = UInt32(hidden), b = UInt32(numBlocks), mb = UInt32(maxBlocks), e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(attnResPSO)
        encoder.setBuffer(blocks, offset: blocksOffset, index: 0)
        encoder.setBuffer(prefix, offset: prefixOffset, index: 1)
        encoder.setBuffer(scoreVector, offset: scoreVectorOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: 4, index: 4)
        encoder.setBytes(&b, length: 4, index: 5)
        encoder.setBytes(&mb, length: 4, index: 6)
        encoder.setBytes(&e, length: 4, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// out[t][:] = dequant(table[tokens[t]][:]).
    func encodeEmbedGather(commandBuffer: MTLCommandBuffer,
                           table: MTLBuffer, tableOffset: Int = 0,
                           scales: MTLBuffer, scalesOffset: Int = 0,
                           biases: MTLBuffer, biasesOffset: Int = 0,
                           tokens: MTLBuffer, tokensOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           tokenCount: Int, hidden: Int) {
        var h = UInt32(hidden)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embedPSO)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(tokens, offset: tokensOffset, index: 3)
        encoder.setBuffer(out, offset: outOffset, index: 4)
        encoder.setBytes(&h, length: 4, index: 5)
        dispatch2D(encoder: encoder, xCount: hidden, yCount: tokenCount)
        encoder.endEncoding()
    }

    /// dst[t][sel][:] = src[t][:] (AttnRes block append).
    func encodeBlockAppend(commandBuffer: MTLCommandBuffer,
                           src: MTLBuffer, srcOffset: Int = 0,
                           dst: MTLBuffer, dstOffset: Int = 0,
                           tokens: Int, hidden: Int, slot: Int, maxBlocks: Int) {
        var h = UInt32(hidden), s = UInt32(slot), mb = UInt32(maxBlocks)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(blockAppendPSO)
        encoder.setBuffer(src, offset: srcOffset, index: 0)
        encoder.setBuffer(dst, offset: dstOffset, index: 1)
        encoder.setBytes(&h, length: 4, index: 2)
        encoder.setBytes(&s, length: 4, index: 3)
        encoder.setBytes(&mb, length: 4, index: 4)
        dispatch2D(encoder: encoder, xCount: hidden, yCount: tokens)
        encoder.endEncoding()
    }

    // MARK: - KDA

    /// Batched causal conv + SiLU over the chunk (reads conv history).
    func encodeKDAConv(commandBuffer: MTLCommandBuffer,
                       xq: MTLBuffer, xqOffset: Int = 0,
                       xk: MTLBuffer, xkOffset: Int = 0,
                       xv: MTLBuffer, xvOffset: Int = 0,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       convStates: MTLBuffer, convStatesOffset: Int = 0,
                       qOut: MTLBuffer, qOutOffset: Int = 0,
                       kOut: MTLBuffer, kOutOffset: Int = 0,
                       vOut: MTLBuffer, vOutOffset: Int = 0,
                       channels: Int, tokens: Int) {
        var p = UInt32(channels), t = UInt32(tokens)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(kdaConvPSO)
        encoder.setBuffer(xq, offset: xqOffset, index: 0)
        encoder.setBuffer(xk, offset: xkOffset, index: 1)
        encoder.setBuffer(xv, offset: xvOffset, index: 2)
        encoder.setBuffer(weights, offset: weightsOffset, index: 3)
        encoder.setBuffer(convStates, offset: convStatesOffset, index: 4)
        encoder.setBuffer(qOut, offset: qOutOffset, index: 5)
        encoder.setBuffer(kOut, offset: kOutOffset, index: 6)
        encoder.setBuffer(vOut, offset: vOutOffset, index: 7)
        encoder.setBytes(&p, length: 4, index: 8)
        encoder.setBytes(&t, length: 4, index: 9)
        dispatch2D(encoder: encoder, xCount: 3 * channels, yCount: tokens)
        encoder.endEncoding()
    }

    /// Conv state update (last-3 projections), an ordered pass after the conv.
    func encodeKDAConvState(commandBuffer: MTLCommandBuffer,
                            xq: MTLBuffer, xqOffset: Int = 0,
                            xk: MTLBuffer, xkOffset: Int = 0,
                            xv: MTLBuffer, xvOffset: Int = 0,
                            convStates: MTLBuffer, convStatesOffset: Int = 0,
                            channels: Int, tokens: Int) {
        var p = UInt32(channels), t = UInt32(tokens)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(kdaConvStatePSO)
        encoder.setBuffer(xq, offset: xqOffset, index: 0)
        encoder.setBuffer(xk, offset: xkOffset, index: 1)
        encoder.setBuffer(xv, offset: xvOffset, index: 2)
        encoder.setBuffer(convStates, offset: convStatesOffset, index: 3)
        encoder.setBytes(&p, length: 4, index: 4)
        encoder.setBytes(&t, length: 4, index: 5)
        dispatch2D(encoder: encoder, xCount: 3 * channels, yCount: 1)
        encoder.endEncoding()
    }

    /// Serial KDA recurrence over the chunk (one threadgroup per head).
    func encodeKDASerial(commandBuffer: MTLCommandBuffer,
                         state: MTLBuffer, stateOffset: Int = 0,
                         q: MTLBuffer, qOffset: Int = 0,
                         k: MTLBuffer, kOffset: Int = 0,
                         v: MTLBuffer, vOffset: Int = 0,
                         z: MTLBuffer, zOffset: Int = 0,
                         betaLogits: MTLBuffer, betaLogitsOffset: Int = 0,
                         aLog: MTLBuffer, aLogOffset: Int = 0,
                         o: MTLBuffer, oOffset: Int = 0,
                         numHeads: Int, headDim: Int, tokens: Int) {
        precondition(headDim <= 128, "kda prefill headDim \(headDim) exceeds staging (128)")
        var h = UInt32(numHeads), d = UInt32(headDim), t = UInt32(tokens)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(kdaSerialPSO)
        encoder.setBuffer(state, offset: stateOffset, index: 0)
        encoder.setBuffer(q, offset: qOffset, index: 1)
        encoder.setBuffer(k, offset: kOffset, index: 2)
        encoder.setBuffer(v, offset: vOffset, index: 3)
        encoder.setBuffer(z, offset: zOffset, index: 4)
        encoder.setBuffer(betaLogits, offset: betaLogitsOffset, index: 5)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 6)
        encoder.setBuffer(o, offset: oOffset, index: 7)
        encoder.setBytes(&h, length: 4, index: 8)
        encoder.setBytes(&d, length: 4, index: 9)
        encoder.setBytes(&t, length: 4, index: 10)
        encoder.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Batched KDA gated output norm (one threadgroup per (head, token)).
    func encodeKDAONorm(commandBuffer: MTLCommandBuffer,
                        x: MTLBuffer, xOffset: Int = 0,
                        gate: MTLBuffer, gateOffset: Int = 0,
                        weight: MTLBuffer, weightOffset: Int = 0,
                        out: MTLBuffer, outOffset: Int = 0,
                        numHeads: Int, headDim: Int, tokens: Int, eps: Float) {
        var h = UInt32(numHeads), d = UInt32(headDim), e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(kdaONormPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: 4, index: 4)
        encoder.setBytes(&d, length: 4, index: 5)
        encoder.setBytes(&e, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: numHeads, height: tokens, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// z[t][c] = x[t][c] + bias[c] (KDA z = f_b(f_a(x)) + dt_bias).
    func encodeAddBias(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       bias: MTLBuffer, biasOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       width: Int, tokens: Int) {
        var p = UInt32(width)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addBiasPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&p, length: 4, index: 3)
        dispatch2D(encoder: encoder, xCount: width, yCount: tokens)
        encoder.endEncoding()
    }

    // MARK: - MLA

    /// qAbs[t][h] = [kT_h^T q_nope | q_rope], fp32.
    func encodeMLAAbsorbQ(commandBuffer: MTLCommandBuffer,
                          kT: MTLBuffer, kTOffset: Int = 0,
                          q: MTLBuffer, qOffset: Int = 0,
                          qAbs: MTLBuffer, qAbsOffset: Int = 0,
                          numHeads: Int, latent: Int, rope: Int, nope: Int, tokens: Int) {
        var h = UInt32(numHeads), l = UInt32(latent), r = UInt32(rope)
        var n = UInt32(nope), t = UInt32(tokens)
        let rows = numHeads * (latent + rope)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mlaAbsorbPSO)
        encoder.setBuffer(kT, offset: kTOffset, index: 0)
        encoder.setBuffer(q, offset: qOffset, index: 1)
        encoder.setBuffer(qAbs, offset: qAbsOffset, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&l, length: 4, index: 4)
        encoder.setBytes(&r, length: 4, index: 5)
        encoder.setBytes(&n, length: 4, index: 6)
        encoder.setBytes(&t, length: 4, index: 7)
        dispatchSimdRows(encoder: encoder, rows: rows, tokens: tokens)
        encoder.endEncoding()
    }

    /// Append [norm(latent) | rope] per position to the cache.
    func encodeMLACacheAppend(commandBuffer: MTLCommandBuffer,
                              kvA: MTLBuffer, kvAOffset: Int = 0,
                              normWeight: MTLBuffer, normWeightOffset: Int = 0,
                              cache: MTLBuffer, cacheOffset: Int = 0,
                              chunkStart: Int, latent: Int, rope: Int,
                              tokens: Int, eps: Float) {
        var cs = UInt32(chunkStart), l = UInt32(latent), r = UInt32(rope), e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mlaCacheAppendPSO)
        encoder.setBuffer(kvA, offset: kvAOffset, index: 0)
        encoder.setBuffer(normWeight, offset: normWeightOffset, index: 1)
        encoder.setBuffer(cache, offset: cacheOffset, index: 2)
        encoder.setBytes(&cs, length: 4, index: 3)
        encoder.setBytes(&l, length: 4, index: 4)
        encoder.setBytes(&r, length: 4, index: 5)
        encoder.setBytes(&e, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: tokens, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Causal absorbed attention over the cache (one threadgroup per (head, pos)).
    func encodeMLAAttention(commandBuffer: MTLCommandBuffer,
                            cache: MTLBuffer, cacheOffset: Int = 0,
                            qAbs: MTLBuffer, qAbsOffset: Int = 0,
                            outLat: MTLBuffer, outLatOffset: Int = 0,
                            numHeads: Int, latent: Int, rope: Int,
                            chunkStart: Int, scale: Float, tokens: Int) {
        var h = UInt32(numHeads), l = UInt32(latent), r = UInt32(rope)
        var cs = UInt32(chunkStart), sc = scale
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mlaAttentionPSO)
        encoder.setBuffer(cache, offset: cacheOffset, index: 0)
        encoder.setBuffer(qAbs, offset: qAbsOffset, index: 1)
        encoder.setBuffer(outLat, offset: outLatOffset, index: 2)
        encoder.setBytes(&h, length: 4, index: 3)
        encoder.setBytes(&l, length: 4, index: 4)
        encoder.setBytes(&r, length: 4, index: 5)
        encoder.setBytes(&cs, length: 4, index: 6)
        encoder.setBytes(&sc, length: 4, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: numHeads, height: tokens, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// out[t][h*V+i] = sigmoid(gate) * (v_h . outLat), batched over t.
    func encodeMLAOutProject(commandBuffer: MTLCommandBuffer,
                             v: MTLBuffer, vOffset: Int = 0,
                             outLat: MTLBuffer, outLatOffset: Int = 0,
                             gate: MTLBuffer, gateOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             numHeads: Int, latent: Int, vHead: Int, tokens: Int) {
        var h = UInt32(numHeads), l = UInt32(latent), vh = UInt32(vHead), t = UInt32(tokens)
        let rows = numHeads * vHead
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(mlaOutProjectPSO)
        encoder.setBuffer(v, offset: vOffset, index: 0)
        encoder.setBuffer(outLat, offset: outLatOffset, index: 1)
        encoder.setBuffer(gate, offset: gateOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: 4, index: 4)
        encoder.setBytes(&l, length: 4, index: 5)
        encoder.setBytes(&vh, length: 4, index: 6)
        encoder.setBytes(&t, length: 4, index: 7)
        dispatchSimdRows(encoder: encoder, rows: rows, tokens: tokens)
        encoder.endEncoding()
    }

    // MARK: - MoE

    /// Phase 1 (fused w1/w3 + SiTU) for a tile's pairs.
    func encodeMoEPhase1(commandBuffer: MTLCommandBuffer,
                         experts: MTLBuffer, expertsOffset: Int = 0,
                         pairs: MTLBuffer, pairsOffset: Int = 0,
                         xLat: MTLBuffer, xLatOffset: Int = 0,
                         h: MTLBuffer, hOffset: Int = 0,
                         subtensorOffsets: K3ExpertSubtensorOffsets,
                         expertStride: Int, dLatent: Int, intermediate: Int,
                         tilePairCount: Int) {
        var offsets = subtensorOffsets
        var stride = UInt32(expertStride), dl = UInt32(dLatent)
        var inter = UInt32(intermediate), count = UInt32(tilePairCount)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(moePhase1PSO)
        encoder.setBuffer(experts, offset: expertsOffset, index: 0)
        encoder.setBuffer(pairs, offset: pairsOffset, index: 1)
        encoder.setBuffer(xLat, offset: xLatOffset, index: 2)
        encoder.setBuffer(h, offset: hOffset, index: 3)
        encoder.setBytes(&offsets, length: MemoryLayout<K3ExpertSubtensorOffsets>.stride, index: 4)
        encoder.setBytes(&stride, length: 4, index: 5)
        encoder.setBytes(&dl, length: 4, index: 6)
        encoder.setBytes(&inter, length: 4, index: 7)
        encoder.setBytes(&count, length: 4, index: 8)
        dispatchSimdRows(encoder: encoder, rows: tilePairCount * intermediate, tokens: 1)
        encoder.endEncoding()
    }

    /// Phase 2 (w2 + router weight) for a tile's pairs.
    func encodeMoEPhase2(commandBuffer: MTLCommandBuffer,
                         experts: MTLBuffer, expertsOffset: Int = 0,
                         pairs: MTLBuffer, pairsOffset: Int = 0,
                         h: MTLBuffer, hOffset: Int = 0,
                         pairOut: MTLBuffer, pairOutOffset: Int = 0,
                         subtensorOffsets: K3ExpertSubtensorOffsets,
                         expertStride: Int, dLatent: Int, intermediate: Int,
                         tilePairCount: Int) {
        var offsets = subtensorOffsets
        var stride = UInt32(expertStride), dl = UInt32(dLatent)
        var inter = UInt32(intermediate), count = UInt32(tilePairCount)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(moePhase2PSO)
        encoder.setBuffer(experts, offset: expertsOffset, index: 0)
        encoder.setBuffer(pairs, offset: pairsOffset, index: 1)
        encoder.setBuffer(h, offset: hOffset, index: 2)
        encoder.setBuffer(pairOut, offset: pairOutOffset, index: 3)
        encoder.setBytes(&offsets, length: MemoryLayout<K3ExpertSubtensorOffsets>.stride, index: 4)
        encoder.setBytes(&stride, length: 4, index: 5)
        encoder.setBytes(&dl, length: 4, index: 6)
        encoder.setBytes(&inter, length: 4, index: 7)
        encoder.setBytes(&count, length: 4, index: 8)
        dispatchSimdRows(encoder: encoder, rows: tilePairCount * dLatent, tokens: 1)
        encoder.endEncoding()
    }

    /// Token-major weighted reduce: yLat[t][d] = sum_rank pairOut[t*topK+rank][d].
    func encodeMoEReduce(commandBuffer: MTLCommandBuffer,
                         pairOut: MTLBuffer, pairOutOffset: Int = 0,
                         yLat: MTLBuffer, yLatOffset: Int = 0,
                         dLatent: Int, topK: Int, tokens: Int) {
        var dl = UInt32(dLatent), tk = UInt32(topK)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(moeReducePSO)
        encoder.setBuffer(pairOut, offset: pairOutOffset, index: 0)
        encoder.setBuffer(yLat, offset: yLatOffset, index: 1)
        encoder.setBytes(&dl, length: 4, index: 2)
        encoder.setBytes(&tk, length: 4, index: 3)
        dispatch2D(encoder: encoder, xCount: dLatent, yCount: tokens)
        encoder.endEncoding()
    }
}
