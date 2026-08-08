import Foundation
import TurboFieldfare

/// Full-forward FP32 CPU oracle for a K3 model, for small dims only. This is
/// the Stage-C2 correctness capstone: a straight, independent port of
/// docs/K3_DATAFLOW.md composed from the per-primitive references
/// (`K3KDAReference`, `K3MLAReference`, `K3AttnResReference`,
/// `K3RouterReference`, `K3MoEReference`) — naive MLA expansion, sequential
/// KDA recurrence, CPU MoE over the packed expert blobs. Nothing here touches
/// the GPU code paths.
///
/// Inputs are an in-memory tensor dictionary keyed by the full checkpoint
/// tensor names (the `K3Model` schema names), holding the EXACT fp32 values
/// the GPU pipeline computes with: affine int4/int8 tensors as their exact
/// dequantization (`q*s+b`), bf16 tensors as their exact bf16 values, fp32
/// tensors as stored. The dictionary never rounds through fp16; instead the
/// oracle rounds activations to fp16 at exactly the points where the GPU
/// pipeline stores an fp16 buffer (every trunk GEMV output, the stream and
/// prefix updates, the AttnRes/norm outputs, the MLA cache rows and the
/// fp16 kv_b plane, the SiTU outputs). That mirrors the engine's dtype
/// contract one-to-one, so what remains between oracle and GPU is reduction
/// order only — the fp16-chained tolerance.
///
/// Conv1d biases are intentionally unused, matching the verified C reference
/// (`k3_ops.c k3_shortconv`) and the `k3_kda_conv` kernel contract.
public struct K3ForwardReference {
    public let config: K3ArchConfig
    private let tensors: [String: [Float]]
    private let shapes: [String: (rows: Int, columns: Int)]
    private let expertBlobs: [Int: [[UInt8]]]

    private var position = 0
    /// KDA layer0 -> [H*D*D] recurrent state (fp32, row = key channel).
    private var kdaStates: [Int: [Float]] = [:]
    /// KDA layer0 -> [3*P*(width-1)] conv history (fp32, oldest first).
    private var kdaConvStates: [Int: [Float]] = [:]
    /// MLA layer0 -> cached rows (fp16-valued fp32, [L+R] each).
    private var mlaCaches: [Int: [[Float]]] = [:]

    /// Most recent token's pre-AttnRes inputs at block boundaries. Exposed
    /// so lifecycle tests can pin the official layer-12 append contract
    /// independently of the Metal implementations.
    public private(set) var lastAttnResBoundaryInputs: [Int: [Float]] = [:]
    /// Block list retained at the end of the most recent token forward.
    public private(set) var lastAttnResBlocks: [[Float]] = []

    private let rmsEps: Float
    /// KimiRMSNorm default (q_a/kv_a layernorms ignore config.rms_norm_eps).
    private let loraNormEps: Float = 1e-6

    public init(config: K3ArchConfig,
                tensors: [String: [Float]],
                shapes: [String: (rows: Int, columns: Int)],
                expertBlobs: [Int: [[UInt8]]]) {
        self.config = config
        self.tensors = tensors
        self.shapes = shapes
        self.expertBlobs = expertBlobs
        self.rmsEps = Float(config.rmsNormEpsilon)
        reset()
    }

    public mutating func reset() {
        position = 0
        kdaStates = [:]
        kdaConvStates = [:]
        mlaCaches = [:]
        lastAttnResBoundaryInputs = [:]
        lastAttnResBlocks = []
        let P = config.kdaChannels
        for layer in config.kdaLayers0 {
            kdaStates[layer] = [Float](
                repeating: 0, count: config.kdaStateElementsPerLayer)
            kdaConvStates[layer] = [Float](repeating: 0, count: 3 * P * 3)
        }
        for layer in config.mlaLayers0 {
            mlaCaches[layer] = []
        }
    }

    public var currentPosition: Int { position }

    // MARK: - Tensor access

    private func tensor(_ name: String) -> [Float] {
        guard let value = tensors[name] else {
            preconditionFailure("oracle tensor dictionary is missing \(name)")
        }
        return value
    }

    private func matrix(_ name: String) -> (w: [Float], rows: Int, columns: Int) {
        guard let shape = shapes[name] else {
            preconditionFailure("oracle shape dictionary is missing \(name)")
        }
        return (tensor(name), shape.rows, shape.columns)
    }

    private static func layerPrefix(_ layer0: Int) -> String {
        "language_model.model.layers.\(layer0)"
    }

    // MARK: - Primitives (fp32 math; fp16 rounding at GPU storage points)

    @inline(__always)
    private func r16(_ x: Float) -> Float { Float(Float16(x)) }

    private func r16(_ v: [Float]) -> [Float] { v.map(r16) }

    /// Plain fp32 GEMV over the exact dequantized weights: `y[r] = W[r] . x`.
    /// `x` must already carry the fp16 rounding of the GPU's input buffer.
    private func gemv(_ name: String, x: [Float]) -> [Float] {
        let (w, rows, columns) = matrix(name)
        precondition(x.count == columns, "\(name): x \(x.count) != columns \(columns)")
        var out = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            let base = r * columns
            var acc: Float = 0
            for c in 0..<columns {
                acc += w[base + c] * x[c]
            }
            out[r] = acc
        }
        return out
    }

    /// KimiRMSNorm, fp32-internal: `w * (x * rsqrt(mean(x^2) + eps))`.
    private func rmsNorm(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count)
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1 / (ss / Float(x.count) + eps).squareRoot()
        return zip(x, weight).map { $1 * $0 * inv }
    }

    /// SiTU-GLU elementwise: `(b1*tanh(g/b1)*sigmoid(g)) * (b2*tanh(u/b2))`,
    /// with the kernels' +/-20 tanh-argument clamp.
    private func situ(_ gate: [Float], _ up: [Float]) -> [Float] {
        let b1 = Float(config.situGLUGateBeta)
        let b2 = Float(config.situGLUUpBeta)
        precondition(gate.count == up.count)
        return (0..<gate.count).map { i in
            let g = gate[i]
            let u = up[i]
            let t1 = tanh(max(-20, min(20, g / b1)))
            let sg = 1 / (1 + exp(-g))
            let t2 = tanh(max(-20, min(20, u / b2)))
            return (b1 * t1 * sg) * (b2 * t2)
        }
    }

    /// Fused AttnRes score vector: `norm.weight ⊙ proj.weight`.
    private func scoreVector(_ normName: String, _ projName: String) -> [Float] {
        let n = tensor(normName)
        let p = tensor(projName)
        precondition(n.count == p.count)
        return (0..<n.count).map { n[$0] * p[$0] }
    }

    private func attnRes(blocks: [[Float]], prefix: [Float],
                         scoreVector: [Float]) -> [Float] {
        K3AttnResReference.apply(blocks: blocks, prefix: prefix,
                                 scoreVector: scoreVector, eps: rmsEps)
    }

    // MARK: - Full forward

    /// Replay one token at the current position; returns the fp32 logits
    /// (the head always runs — the oracle has no `emitHead` elision).
    @discardableResult
    public mutating func forward(token: Int32) -> [Float] {
        let c = config
        let H = c.hiddenSize

        // Embedding lookup (no scale factor), fp16-rounded like the gather.
        let emb = matrix("language_model.model.embed_tokens.weight")
        let row = Int(token)
        precondition(row >= 0 && row < emb.rows)
        var x = r16(Array(emb.w[(row * H)..<((row + 1) * H)]))

        var blocks: [[Float]] = []
        var prefix: [Float]?
        lastAttnResBoundaryInputs = [:]
        lastAttnResBlocks = []

        for layer in 0..<c.numLayers {
            let lp = Self.layerPrefix(layer)
            // The official model assigns `prefix_sum = hidden_states`
            // before pre-attn AttnRes. The combined value feeds attention;
            // the incoming value owns the residual lifecycle.
            let incoming = x
            // 1. Pre-attn AttnRes (skipped while the block list is empty).
            if !blocks.isEmpty {
                x = r16(attnRes(
                    blocks: blocks, prefix: x,
                    scoreVector: scoreVector(
                        "\(lp).self_attention_res_norm.weight",
                        "\(lp).self_attention_res_proj.weight")))
            }
            // 2. Boundary append / prefix capture.
            if c.isAttnResBoundary(layer0: layer) {
                blocks.append(incoming)
                lastAttnResBoundaryInputs[layer] = incoming
                prefix = nil
            } else {
                prefix = incoming
            }
            // 3. input_layernorm -> attention.
            let h = r16(rmsNorm(x, weight: tensor("\(lp).input_layernorm.weight"),
                                eps: rmsEps))
            let attnOut: [Float]
            if c.isKDA(layer0: layer) {
                attnOut = kdaAttention(layer: layer, h: h)
            } else {
                attnOut = mlaAttention(layer: layer, h: h)
            }
            // 4. Prefix update (attention).
            prefix = prefix == nil ? attnOut : r16(zip(prefix!, attnOut).map(+))
            // 5. Pre-mlp AttnRes over the post-append block list.
            let hPre = r16(attnRes(
                blocks: blocks, prefix: prefix!,
                scoreVector: scoreVector(
                    "\(lp).mlp_res_norm.weight",
                    "\(lp).mlp_res_proj.weight")))
            // 6. post_attention_layernorm -> MoE / dense.
            let hn = r16(rmsNorm(
                hPre, weight: tensor("\(lp).post_attention_layernorm.weight"),
                eps: rmsEps))
            let mlpOut: [Float]
            if c.isMoE(layer0: layer) {
                mlpOut = moeBlock(layer: layer, h: hn)
            } else {
                mlpOut = denseBlock(layer: layer, h: hn)
            }
            // 7. Prefix update (MLP); the prefix is the next layer's stream.
            prefix = prefix == nil ? mlpOut : r16(zip(prefix!, mlpOut).map(+))
            x = prefix!
        }
        lastAttnResBlocks = blocks

        // Head: output AttnRes -> final norm -> int8 lm_head (fp32 logits).
        x = r16(attnRes(blocks: blocks, prefix: x,
                        scoreVector: scoreVector(
                            "language_model.model.output_attn_res_norm.weight",
                            "language_model.model.output_attn_res_proj.weight")))
        x = r16(rmsNorm(x, weight: tensor("language_model.model.norm.weight"),
                        eps: rmsEps))
        position += 1
        return gemv("language_model.lm_head.weight", x: x)
    }

    // MARK: - KDA attention (docs/K3_DATAFLOW.md "KDA layer")

    private mutating func kdaAttention(layer: Int, h: [Float]) -> [Float] {
        let c = config
        let attn = "\(Self.layerPrefix(layer)).self_attn"
        let P = c.kdaChannels
        let hist = c.kdaConvWidth - 1

        // Projections (int4 trunk), fp16-rounded like the GEMV outputs.
        let q = r16(gemv("\(attn).q_proj.weight", x: h))
        let k = r16(gemv("\(attn).k_proj.weight", x: h))
        let v = r16(gemv("\(attn).v_proj.weight", x: h))
        let gate = r16(gemv("\(attn).g_proj.weight", x: h))
        let fa = r16(gemv("\(attn).f_a_proj.weight", x: h))
        let beta = r16(gemv("\(attn).b_proj.weight", x: h))

        // Depthwise causal conv + SiLU, fp32 states, per projection.
        var convState = kdaConvStates[layer]!
        func conv(_ x: [Float], _ which: Int, _ name: String) -> [Float] {
            let w = matrix(name)
            precondition(w.rows == P && w.columns == c.kdaConvWidth)
            var slice = Array(convState[(which * P * hist)..<((which + 1) * P * hist)])
            let y = K3KDAReference.convStep(x: x, weights: w.w, state: &slice)
            convState.replaceSubrange((which * P * hist)..<((which + 1) * P * hist),
                                      with: slice)
            return y
        }
        let qC = conv(q, 0, "\(attn).q_conv1d.weight")
        let kC = conv(k, 1, "\(attn).k_conv1d.weight")
        let vC = conv(v, 2, "\(attn).v_conv1d.weight")
        kdaConvStates[layer] = convState

        // z = f_b(f_a(x)) + dt_bias (fp32; the f_b output rounds through fp16
        // first, matching the GPU buffer).
        let fb = r16(gemv("\(attn).f_b_proj.weight", x: fa))
        let dtBias = tensor("\(attn).dt_bias")
        let z = zip(fb, dtBias).map(+)

        var recurrent = kdaStates[layer]!
        let o = K3KDAReference.step(state: &recurrent,
                                    q: qC, k: kC, v: vC,
                                    z: z, betaLogits: beta,
                                    aLog: tensor("\(attn).A_log"),
                                    numHeads: c.kdaNumHeads,
                                    headDim: c.kdaHeadDim)
        kdaStates[layer] = recurrent

        let oN = r16(K3KDAReference.outputNorm(o: o, gate: gate,
                                               weight: tensor("\(attn).o_norm.weight"),
                                               eps: rmsEps,
                                               numHeads: c.kdaNumHeads,
                                               headDim: c.kdaHeadDim))
        return r16(gemv("\(attn).o_proj.weight", x: oN))
    }

    // MARK: - MLA attention (docs/K3_DATAFLOW.md "MLA layer")

    private mutating func mlaAttention(layer: Int, h: [Float]) -> [Float] {
        let c = config
        let attn = "\(Self.layerPrefix(layer)).self_attn"
        let L = c.mlaKVLoraRank
        let R = c.mlaQKRopeHeadDim

        let qa = r16(gemv("\(attn).q_a_proj.weight", x: h))
        let qLat = r16(rmsNorm(qa, weight: tensor("\(attn).q_a_layernorm.weight"),
                               eps: loraNormEps))
        let q = r16(gemv("\(attn).q_b_proj.weight", x: qLat))
        let kvA = r16(gemv("\(attn).kv_a_proj_with_mqa.weight", x: h))
        let gate = r16(gemv("\(attn).g_proj.weight", x: h))

        // Cache row: [ kv_a_layernorm(latent) | raw rope ], fp16-valued.
        let row = r16(K3MLAReference.cacheRow(
            kvA: kvA, normWeight: tensor("\(attn).kv_a_layernorm.weight"),
            eps: loraNormEps, latent: L, rope: R))
        mlaCaches[layer]!.append(row)

        // kv_b dequantized exactly, then rounded to fp16 — the GPU's absorb
        // planes are fp16 (encodeKVBExpand over the dequantized buffer).
        let kvB = matrix("\(attn).kv_b_proj.weight")
        let kvB16 = r16(kvB.w)

        let out = K3MLAReference.attention(
            kvB: kvB16, q: q, cache: mlaCaches[layer]!, gate: gate,
            scale: c.mlaDecodeScale,
            numHeads: c.mlaNumHeads, latent: L, rope: R,
            nope: c.mlaQKNopeHeadDim, vHead: c.mlaVHeadDim)
        // k3_mla_out_project writes fp16; o_proj is the int4 trunk.
        return r16(gemv("\(attn).o_proj.weight", x: r16(out)))
    }

    // MARK: - MoE block (docs/K3_DATAFLOW.md "MoE block")

    private func moeBlock(layer: Int, h: [Float]) -> [Float] {
        let c = config
        let moe = "\(Self.layerPrefix(layer)).block_sparse_moe"
        let dLat = c.moeLatentBottleneckSize
        let inter = c.moeExpertIntermediateSize

        // Router (fp32). The GPU kernel widens the fp16 normed input — the
        // same values the oracle carries.
        let gateW = matrix("\(moe).gate.weight")
        let (_, _, indices, weights) = K3RouterReference.route(
            x: h, gate: gateW.w, bias: tensor("\(moe).gate.e_score_correction_bias"),
            numExperts: c.moeNumExperts, d: c.hiddenSize, topK: c.moeTopKExperts)

        let xLat = r16(gemv("\(moe).routed_expert_down_proj.weight", x: h))
        let blobs = indices.map { expertBlobs[layer]![$0] }
        let offsets = K3ExpertSubtensorOffsets.canonical(
            dLatent: UInt32(dLat), intermediate: UInt32(inter))
        let yLat = K3MoEReference.apply(xLat: xLat, blobs: blobs,
                                        offsets: offsets, routingWeights: weights,
                                        dLatent: dLat, intermediate: inter)
        let yN = r16(rmsNorm(yLat, weight: tensor("\(moe).routed_expert_norm.weight"),
                             eps: rmsEps))
        let up = r16(gemv("\(moe).routed_expert_up_proj.weight", x: yN))

        // Shared expert (KimiMLP, SiTU-GLU, full width).
        let sg = r16(gemv("\(moe).shared_experts.gate_proj.weight", x: h))
        let su = r16(gemv("\(moe).shared_experts.up_proj.weight", x: h))
        let sh = r16(situ(sg, su))
        let so = r16(gemv("\(moe).shared_experts.down_proj.weight", x: sh))

        return r16(zip(up, so).map(+))
    }

    // MARK: - Dense MLP (layer 0)

    private func denseBlock(layer: Int, h: [Float]) -> [Float] {
        let mlp = "\(Self.layerPrefix(layer)).mlp"
        let g = r16(gemv("\(mlp).gate_proj.weight", x: h))
        let u = r16(gemv("\(mlp).up_proj.weight", x: h))
        let hh = r16(situ(g, u))
        return r16(gemv("\(mlp).down_proj.weight", x: hh))
    }
}
