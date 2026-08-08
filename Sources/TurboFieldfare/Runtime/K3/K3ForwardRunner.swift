import Foundation
import Metal
import TurboFieldfareFormat

/// Swift wrappers for `dequant_k3.metal` — the small utility kernels the K3
/// forward runner needs beyond the C1 dispatchers: buffer dequant (the
/// per-load MLA kv_b plane expansion), the standalone SiTU-GLU, elementwise
/// add, two RMSNorm dtype variants, fp16→fp32 widening, and the fused
/// bias-add. One dispatcher, raw buffers + offsets, generic pipelines only
/// (see the shader's header for why there are no function constants).
final class K3UtilityKernels {
    private let dequantInt4PSO: MTLComputePipelineState
    private let dequantInt8PSO: MTLComputePipelineState
    private let situPSO: MTLComputePipelineState
    private let addPSO: MTLComputePipelineState
    private let rmsnormF32WPSO: MTLComputePipelineState
    private let rmsnormF32XPSO: MTLComputePipelineState
    private let cvtPSO: MTLComputePipelineState
    private let addBiasPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        dequantInt4PSO = try library.pipeline(
            device: context.device, name: "k3_dequant_int4_buffer")
        dequantInt8PSO = try library.pipeline(
            device: context.device, name: "k3_dequant_int8_buffer")
        situPSO = try library.pipeline(device: context.device, name: "k3_situ_mul")
        addPSO = try library.pipeline(device: context.device, name: "k3_add")
        rmsnormF32WPSO = try library.pipeline(
            device: context.device, name: "k3_rmsnorm_f32w")
        rmsnormF32XPSO = try library.pipeline(
            device: context.device, name: "k3_rmsnorm_f32x")
        cvtPSO = try library.pipeline(device: context.device, name: "k3_cvt_f16_f32")
        addBiasPSO = try library.pipeline(
            device: context.device, name: "k3_add_bias_f32")
    }

    private static let threadsPerThreadgroup = 256

    private func dispatch1D(encoder: MTLComputeCommandEncoder, count: Int) {
        encoder.dispatchThreadgroups(
            MTLSize(width: (count + Self.threadsPerThreadgroup - 1)
                        / Self.threadsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerThreadgroup,
                                           height: 1, depth: 1))
    }

    /// Dequant one affine-g64 resident view to fp16 (`y[r][c] = q*s+b`).
    /// `view` must be int4 or int8 (the `K3TrunkGEMV.format` rule).
    func encodeDequant(commandBuffer: MTLCommandBuffer,
                       view: TensorView,
                       y: MTLBuffer, yOffset: Int = 0) throws {
        let format = try K3TrunkGEMV.format(of: view)
        precondition(format == .int4G64 || format == .int8G64,
                     "buffer dequant only serves affine views")
        let m = view.shape.0
        let n = view.shape.1
        precondition(yOffset % MemoryLayout<Float16>.stride == 0)
        var rows = m
        var cols = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            format == .int4G64 ? dequantInt4PSO : dequantInt8PSO)
        encoder.setBuffer(view.buffer, offset: Int(view.offset), index: 0)
        encoder.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: 1)
        encoder.setBuffer(view.buffer, offset: Int(view.biasOffset), index: 2)
        encoder.setBuffer(y, offset: yOffset, index: 3)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&cols, length: MemoryLayout<UInt32>.stride, index: 5)
        dispatch1D(encoder: encoder, count: Int(m) * Int(n))
        encoder.endEncoding()
    }

    /// SiTU-GLU over separate fp16 gate/up vectors, fp16 out, fp32 math.
    func encodeSiTU(commandBuffer: MTLCommandBuffer,
                    gate: MTLBuffer, gateOffset: Int = 0,
                    up: MTLBuffer, upOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    n: UInt32, beta1: Float, beta2: Float) {
        var count = n
        var b1 = beta1
        var b2 = beta2
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(situPSO)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(up, offset: upOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&b1, length: MemoryLayout<Float>.stride, index: 4)
        encoder.setBytes(&b2, length: MemoryLayout<Float>.stride, index: 5)
        dispatch1D(encoder: encoder, count: Int(n))
        encoder.endEncoding()
    }

    /// `out = a + b`, fp32 math on fp16 io. In-place safe.
    func encodeAdd(commandBuffer: MTLCommandBuffer,
                   a: MTLBuffer, aOffset: Int = 0,
                   b: MTLBuffer, bOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   n: UInt32) {
        var count = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPSO)
        encoder.setBuffer(a, offset: aOffset, index: 0)
        encoder.setBuffer(b, offset: bOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        dispatch1D(encoder: encoder, count: Int(n))
        encoder.endEncoding()
    }

    /// KimiRMSNorm, fp16 x, fp32 weight (MLA q_a_layernorm), fp16 out.
    func encodeRMSNormF32W(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           weight: MTLBuffer, weightOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           d: UInt32, eps: Float) {
        var dim = d
        var e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rmsnormF32WPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&e, length: MemoryLayout<Float>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// KimiRMSNorm over an fp32 x with a bf16 weight (routed_expert_norm on
    /// the fp32 y_lat), fp16 out.
    func encodeRMSNormF32X(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           weight: MTLBuffer, weightOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           d: UInt32, eps: Float) {
        var dim = d
        var e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rmsnormF32XPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&e, length: MemoryLayout<Float>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// fp16 → fp32 widening.
    func encodeWiden(commandBuffer: MTLCommandBuffer,
                     x: MTLBuffer, xOffset: Int = 0,
                     out: MTLBuffer, outOffset: Int = 0,
                     n: UInt32) {
        var count = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(cvtPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(out, offset: outOffset, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        dispatch1D(encoder: encoder, count: Int(n))
        encoder.endEncoding()
    }

    /// `out = float(x) + bias` (KDA z = f_b(f_a(x)) + dt_bias), fp16 x,
    /// fp32 bias/out.
    func encodeAddBias(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       bias: MTLBuffer, biasOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       n: UInt32) {
        var count = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addBiasPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        dispatch1D(encoder: encoder, count: Int(n))
        encoder.endEncoding()
    }
}

/// CPU-side wall-time counters for one runner, nanos, cumulative since init
/// or the last `resetTimings()`. `encodeCommit` is CPU time spent encoding +
/// committing command buffers; `routerWait` is the per-MoE-layer sync point;
/// `expertIO` is time inside `K3ExpertStreaming.beginLayer` (the SSD reads);
/// `tailWait` is the end-of-token wait that makes logits readable.
public struct K3ForwardTimings: Sendable, Equatable {
    public var encodeCommitNanos: UInt64 = 0
    public var routerWaitNanos: UInt64 = 0
    public var expertIONanos: UInt64 = 0
    public var tailWaitNanos: UInt64 = 0
    public var tokensProduced: UInt64 = 0

    public init() {}

    public var totalNanos: UInt64 {
        encodeCommitNanos + routerWaitNanos + expertIONanos + tailWaitNanos
    }
}

/// The K3 per-token decode forward pass (docs/K3_DATAFLOW.md), driving the
/// C1 dispatchers over `K3State` slabs and `K3ExpertStreaming` banks.
///
/// Command-buffer discipline (RealForwardRunner philosophy, correctness
/// first): one command buffer accumulates embed + whole layers until a sync
/// point. The only sync points are (a) each MoE layer's router readback and
/// (b) the end-of-token wait that publishes logits. For an MoE layer the
/// shared-expert + down-projection chain is encoded and committed BEFORE the
/// CPU blocks in `beginLayer`, so that GPU work overlaps the SSD preads; the
/// expert phases then encode into a fresh buffer. Banks are released by
/// draining `pendingEndLayers` after every wait — the waited command buffer
/// was committed after every pending batch's consuming buffer, and the queue
/// completes buffers in commit order, so the bytes are provably idle by the
/// time `endLayer` runs (no reliance on completion-handler timing).
///
/// Per-layer wiring (layer i, 0-based), exactly the dataflow contract:
///   1. pre-attn AttnRes over the block slab (skipped while the block list
///      is empty — the kernel would return the prefix bit-exactly anyway);
///   2. boundary append at i % blockSize == 0 (blit the stream into the
///      block slab, prefix becomes invalid) else blit stream → prefix;
///   3. input_layernorm → KDA or MLA attention → o_proj;
///   4. prefix = prefixValid ? prefix + attn : attn;
///   5. pre-mlp AttnRes over the post-append block list;
///   6. post_attention_layernorm → MoE or dense MLP;
///   7. prefix += mlp (or assign); the prefix is the next layer's stream.
/// Head (when `emitHead`): output AttnRes → final norm → int8 lm_head →
/// fp32 logits in `logitsBuffer`.
///
/// Weight conversions at init (all once per load):
///   - AttnRes score vectors fused on the CPU into one fp32 slab
///     (`norm.weight ⊙ proj.weight` per layer per site + the output site);
///   - KDA conv weights gathered from the three per-projection fp32 matrices
///     into the kernel's `[3][P][4]` slab (per KDA ordinal). The decode conv
///     is bias-free — the verified C reference (`k3_ops.c k3_shortconv`) and
///     the `k3_kda_conv` kernel take no bias, so the checkpoint's conv1d.bias
///     tensors are loaded by the schema but intentionally unused (flagged for
///     the real-checkpoint validation in Stage E);
///   - MLA kv_b dequantized to fp16 and expanded once into the per-layer
///     absorb planes kT [H][L][N] and v [H][V][L] (`encodeKVBExpand` is a
///     per-load layout conversion, per the K3MLA contract).
final class K3ForwardRunner {
    let model: K3Model
    let context: MetalContext
    let config: K3ArchConfig
    let state: K3State
    let streaming: K3ExpertStreaming

    // Dispatchers.
    private let embed: K3Embed
    private let lmHead: K3LMHeadGEMV
    private let trunk: K3TrunkGEMV
    private let kda: K3KDA
    private let mla: K3MLA
    private let attnRes: K3AttnRes
    private let moe: K3MoE
    private let router: K3Router
    private let rmsNorm: RMSNorm
    private let util: K3UtilityKernels

    // Load-time converted weights.
    /// fp32 [(2*numLayers + 1) * hidden]: fused AttnRes score vectors.
    private let scoreVectors: MTLBuffer
    /// fp32 [kdaLayers * 3 * kdaP * convWidth]: gathered conv weights.
    private let convWeights: MTLBuffer
    /// fp16 [mlaLayers * H * L * N] absorb plane (transposed k part).
    private let mlaKTPlane: MTLBuffer
    /// fp16 [mlaLayers * H * V * L] value plane.
    private let mlaVPlane: MTLBuffer
    /// Per-MoE-layer router correction bias, read once from the resident
    /// buffer (fp32 [numExperts]) for the CPU top-k.
    private let routerBiasByLayer: [Int: [Float]]

    // Per-token scratch. Everything is storageModeShared.
    private let streamA: MTLBuffer      // fp16 [H]
    private let streamB: MTLBuffer      // fp16 [H] (AttnRes ping-pong)
    private let hPre: MTLBuffer         // fp16 [H] pre-mlp AttnRes out
    private let normed: MTLBuffer       // fp16 [H] norm outputs
    private let attnOut: MTLBuffer      // fp16 [H] o_proj out
    private let mlpOut: MTLBuffer       // fp16 [H] MLP/MoE out
    private let kdaProj: MTLBuffer      // fp16 [3 * kdaP] q|k|v projections
    private let kdaFA: MTLBuffer        // fp16 [lowRank]
    private let kdaFB16: MTLBuffer      // fp16 [kdaP] f_b out
    private let kdaBeta16: MTLBuffer    // fp16 [kdaHeads] b_proj out
    private let kdaGate: MTLBuffer      // fp16 [kdaP] g_proj out
    private let kdaONorm: MTLBuffer     // fp16 [kdaP] gated-norm out
    private let kdaConvOut: MTLBuffer   // fp32 [3 * kdaP]
    private let kdaZ: MTLBuffer         // fp32 [kdaP]
    private let kdaBeta32: MTLBuffer    // fp32 [kdaHeads]
    private let kdaStepOut: MTLBuffer   // fp32 [kdaP]
    private let mlaQA: MTLBuffer        // fp16 [qLora]
    private let mlaQLat: MTLBuffer      // fp16 [qLora]
    private let mlaQ: MTLBuffer         // fp16 [H*(N+R)]
    private let mlaKVA: MTLBuffer       // fp16 [L+R]
    private let mlaGate: MTLBuffer      // fp16 [H*V] g_proj out
    private let mlaOut: MTLBuffer       // fp16 [H*V]
    private let mlaQAbs: MTLBuffer      // fp32 [H*(L+R)]
    private let mlaOutLat: MTLBuffer    // fp32 [H*L]
    private let h32: MTLBuffer          // fp32 [H] router input
    private let scores: MTLBuffer       // fp32 [numExperts]
    private let routingWeights: MTLBuffer  // fp32 [topK]
    private let slotOffsets: MTLBuffer  // u64 [topK]
    private let xLat: MTLBuffer         // fp16 [dLatent]
    private let moeH: MTLBuffer         // fp32 [topK * intermediate]
    private let yLat: MTLBuffer         // fp32 [dLatent]
    private let yLatNormed: MTLBuffer   // fp16 [dLatent]
    private let sharedGate: MTLBuffer   // fp16 [sharedInter]
    private let sharedUp: MTLBuffer     // fp16 [sharedInter]
    private let sharedH: MTLBuffer      // fp16 [sharedInter]
    private let sharedOut: MTLBuffer    // fp16 [H]
    private let denseGate: MTLBuffer    // fp16 [denseInter]
    private let denseUp: MTLBuffer      // fp16 [denseInter]
    private let denseH: MTLBuffer       // fp16 [denseInter]

    /// fp32 [vocab] logits, valid after `produce(..., emitHead: true)`.
    let logitsBuffer: MTLBuffer

    private(set) var timings = K3ForwardTimings()
    private var captureActivationTrace = false
    private(set) var lastRouterDiagnostics: K3RouterActivationDiagnostics?
    private(set) var lastDiagnosticHeadInput: [Float]?

    /// Batches whose consuming command buffer is committed but not yet known
    /// complete; drained after every wait (see the type docstring).
    private var pendingEndLayers: [K3ExpertBatch] = []

    /// KimiRMSNorm default eps — used by the MLA q_a/kv_a layernorms, which
    /// the reference model constructs WITHOUT config.rms_norm_eps.
    private static let mlaLoraNormEps: Float = 1e-6

    init(model: K3Model,
         context: MetalContext,
         state: K3State,
         streaming: K3ExpertStreaming) throws {
        self.model = model
        self.context = context
        self.config = model.config
        self.state = state
        self.streaming = streaming
        let device = context.device
        let c = model.config

        embed = try K3Embed(context: context)
        lmHead = try K3LMHeadGEMV(context: context)
        trunk = try K3TrunkGEMV(context: context)
        kda = try K3KDA(context: context)
        mla = try K3MLA(context: context)
        attnRes = try K3AttnRes(context: context)
        moe = try K3MoE(context: context)
        router = try K3Router(context: context)
        rmsNorm = try RMSNorm(context: context)
        util = try K3UtilityKernels(context: context)

        func buf(_ elements: Int, _ stride: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(elements, 1) * stride,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }
        let f16 = MemoryLayout<Float16>.stride
        let f32 = MemoryLayout<Float>.stride
        let H = c.hiddenSize
        let kdaP = c.kdaChannels

        scoreVectors = try buf((2 * c.numLayers + 1) * H, f32, "k3.run.attnres_sv")
        convWeights = try buf(c.kdaLayers0.count * 3 * kdaP * c.kdaConvWidth,
                              f32, "k3.run.conv_weights")
        mlaKTPlane = try buf(c.mlaLayers0.count * c.mlaNumHeads * c.mlaKVLoraRank
                                 * c.mlaQKNopeHeadDim, f16, "k3.run.mla_kT")
        mlaVPlane = try buf(c.mlaLayers0.count * c.mlaNumHeads * c.mlaVHeadDim
                                * c.mlaKVLoraRank, f16, "k3.run.mla_v")

        streamA = try buf(H, f16, "k3.run.stream_a")
        streamB = try buf(H, f16, "k3.run.stream_b")
        hPre = try buf(H, f16, "k3.run.h_pre")
        normed = try buf(H, f16, "k3.run.normed")
        attnOut = try buf(H, f16, "k3.run.attn_out")
        mlpOut = try buf(H, f16, "k3.run.mlp_out")
        kdaProj = try buf(3 * kdaP, f16, "k3.run.kda_proj")
        kdaFA = try buf(c.kdaDecayLowRankSize, f16, "k3.run.kda_fa")
        kdaFB16 = try buf(kdaP, f16, "k3.run.kda_fb")
        kdaBeta16 = try buf(c.kdaNumHeads, f16, "k3.run.kda_beta16")
        kdaGate = try buf(kdaP, f16, "k3.run.kda_gate")
        kdaONorm = try buf(kdaP, f16, "k3.run.kda_onorm")
        kdaConvOut = try buf(3 * kdaP, f32, "k3.run.kda_conv_out")
        kdaZ = try buf(kdaP, f32, "k3.run.kda_z")
        kdaBeta32 = try buf(c.kdaNumHeads, f32, "k3.run.kda_beta32")
        kdaStepOut = try buf(kdaP, f32, "k3.run.kda_step_out")
        mlaQA = try buf(c.mlaQLoraRank, f16, "k3.run.mla_qa")
        mlaQLat = try buf(c.mlaQLoraRank, f16, "k3.run.mla_qlat")
        mlaQ = try buf(c.mlaQElements, f16, "k3.run.mla_q")
        mlaKVA = try buf(c.mlaCacheRowElements, f16, "k3.run.mla_kva")
        mlaGate = try buf(c.mlaOutputElements, f16, "k3.run.mla_gate")
        mlaOut = try buf(c.mlaOutputElements, f16, "k3.run.mla_out")
        mlaQAbs = try buf(c.mlaNumHeads * c.mlaCacheRowElements, f32, "k3.run.mla_qabs")
        mlaOutLat = try buf(c.mlaNumHeads * c.mlaKVLoraRank, f32, "k3.run.mla_outlat")
        h32 = try buf(H, f32, "k3.run.h32")
        scores = try buf(c.moeNumExperts, f32, "k3.run.scores")
        routingWeights = try buf(c.moeTopKExperts, f32, "k3.run.routing_w")
        slotOffsets = try buf(c.moeTopKExperts, MemoryLayout<UInt64>.stride,
                              "k3.run.slot_offsets")
        xLat = try buf(c.moeLatentBottleneckSize, f16, "k3.run.x_lat")
        moeH = try buf(c.moeTopKExperts * c.moeExpertIntermediateSize, f32,
                       "k3.run.moe_h")
        yLat = try buf(c.moeLatentBottleneckSize, f32, "k3.run.y_lat")
        yLatNormed = try buf(c.moeLatentBottleneckSize, f16, "k3.run.y_lat_normed")
        sharedGate = try buf(c.moeSharedExpertIntermediateSize, f16, "k3.run.sh_gate")
        sharedUp = try buf(c.moeSharedExpertIntermediateSize, f16, "k3.run.sh_up")
        sharedH = try buf(c.moeSharedExpertIntermediateSize, f16, "k3.run.sh_h")
        sharedOut = try buf(H, f16, "k3.run.sh_out")
        denseGate = try buf(c.denseMLPIntermediateSize, f16, "k3.run.dense_gate")
        denseUp = try buf(c.denseMLPIntermediateSize, f16, "k3.run.dense_up")
        denseH = try buf(c.denseMLPIntermediateSize, f16, "k3.run.dense_h")
        logitsBuffer = try buf(c.vocabSize, f32, "k3.run.logits")

        // ---- Load-time conversions -------------------------------------
        try Self.fuseAttnResScoreVectors(model: model, into: scoreVectors)
        try Self.gatherConvWeights(model: model, into: convWeights)
        var biasByLayer: [Int: [Float]] = [:]
        for layer in c.moeLayers0.sorted() {
            biasByLayer[layer] = Self.readFP32Vector(model.routerCorrectionBias(layer: layer))
        }
        routerBiasByLayer = biasByLayer
        try expandMLAPlanes()
    }

    // MARK: - Load-time conversions

    /// Score-vector slab offset (elements) for a site: 0 = self-attention,
    /// 1 = MLP, 2 = the model-head output site.
    private func scoreVectorOffset(layer: Int, site: Int) -> Int {
        (site * config.numLayers + layer) * config.hiddenSize
    }

    private static func readBF16Vector(_ view: TensorView) -> [Float] {
        let count = Int(view.shape.0)
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        let ptr = base.bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Quantization.bf16ToFloat(ptr[$0]) }
    }

    private static func readFP32Vector(_ view: TensorView) -> [Float] {
        let count = Int(view.shape.0)
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        let ptr = base.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private static func readFP16Vector(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private static func readFP32Buffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// sv[j] = norm.weight[j] * proj.weight[j], fp32, per layer and site.
    private static func fuseAttnResScoreVectors(model: K3Model,
                                                into slab: MTLBuffer) throws {
        let c = model.config
        let H = c.hiddenSize
        let dst = slab.contents().bindMemory(to: Float.self,
                                             capacity: (2 * c.numLayers + 1) * H)
        func fuse(_ norm: TensorView, _ proj: TensorView, _ index: Int) {
            let n = readBF16Vector(norm)
            let p = readBF16Vector(proj)
            let base = index * H
            for j in 0..<H { dst[base + j] = n[j] * p[j] }
        }
        for layer in 0..<c.numLayers {
            fuse(model.attnResNorm(layer: layer), model.attnResProj(layer: layer),
                 layer)
            fuse(model.mlpResNorm(layer: layer), model.mlpResProj(layer: layer),
                 c.numLayers + layer)
        }
        fuse(model.outputAttnResNorm, model.outputAttnResProj, 2 * c.numLayers)
    }

    /// Gather the per-projection fp32 [P][4] conv matrices into the kernel's
    /// [3][P][4] slab (q block, then k, then v), per KDA ordinal.
    private static func gatherConvWeights(model: K3Model,
                                          into slab: MTLBuffer) throws {
        let c = model.config
        let P = c.kdaChannels
        let width = c.kdaConvWidth
        let layerFloats = 3 * P * width
        let dst = slab.contents().bindMemory(
            to: Float.self, capacity: c.kdaLayers0.count * layerFloats)
        for layer in c.kdaLayers0.sorted() {
            guard let ordinal = c.kdaOrdinal(layer0: layer) else { continue }
            let base = ordinal * layerFloats
            for (which, qkv) in ["q", "k", "v"].enumerated() {
                let weights = readFP32Matrix(model.kdaConvWeight(layer: layer, qkv))
                let dstBase = base + which * P * width
                for i in 0..<(P * width) { dst[dstBase + i] = weights[i] }
            }
        }
    }

    private static func readFP32Matrix(_ view: TensorView) -> [Float] {
        let count = Int(view.shape.0) * Int(view.shape.1)
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        let ptr = base.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Per MLA layer: dequant kv_b to fp16, then expand into the absorb
    /// planes. Runs once here; the planes are read-only afterwards.
    private func expandMLAPlanes() throws {
        let c = model.config
        guard !c.mlaLayers0.isEmpty else { return }
        let L = c.mlaKVLoraRank
        let rows = c.mlaKVBRows
        guard let scratch = context.device.makeBuffer(
            length: rows * L * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        scratch.label = "k3.run.mla_kvb_scratch"
        guard let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        for layer in c.mlaLayers0.sorted() {
            guard let ordinal = c.mlaOrdinal(layer0: layer) else { continue }
            try util.encodeDequant(commandBuffer: cb,
                                   view: model.mlaKVBProj(layer: layer),
                                   y: scratch)
            mla.encodeKVBExpand(
                commandBuffer: cb,
                kvB: scratch,
                kT: mlaKTPlane,
                kTOffset: ordinal * c.mlaNumHeads * L * c.mlaQKNopeHeadDim
                    * MemoryLayout<Float16>.stride,
                v: mlaVPlane,
                vOffset: ordinal * c.mlaNumHeads * c.mlaVHeadDim * L
                    * MemoryLayout<Float16>.stride,
                numHeads: UInt32(c.mlaNumHeads), latent: UInt32(L),
                nope: UInt32(c.mlaQKNopeHeadDim), vHead: UInt32(c.mlaVHeadDim))
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.libraryCompileFailed("\(error)") }
    }

    // MARK: - Per-token forward

    func resetTimings() { timings = K3ForwardTimings() }

    /// Run only the first resident activation boundary without mutating token
    /// state.  This is the real-bundle counterpart to the small CPU oracle in
    /// `K3ActivationReference`: the probe is safe before/after generation and
    /// intentionally avoids any SSD expert traffic.
    func activationProbe(token: Int32) throws -> K3ActivationDiagnostics {
        let H = config.hiddenSize
        guard let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        let emb = model.embedding
        embed.encodeGather(commandBuffer: cb,
                           table: emb.buffer, tableOffset: Int(emb.offset),
                           scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                           biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                           out: streamA, tokenID: UInt32(bitPattern: token),
                           d: UInt32(H))
        let weight = model.inputNorm(layer: 0)
        rmsNorm.encodeBF16W(commandBuffer: cb,
                            x: streamA, weight: weight.buffer,
                            weightOffset: Int(weight.offset), out: normed,
                            d: UInt32(H), eps: Float(config.rmsNormEpsilon))
        let projections: [(String, TensorView, MTLBuffer, Int)] = [
            ("q", model.kdaQProj(layer: 0), kdaProj, 0),
            ("k", model.kdaKProj(layer: 0), kdaProj,
             config.kdaChannels * MemoryLayout<Float16>.stride),
            ("v", model.kdaVProj(layer: 0), kdaProj,
             2 * config.kdaChannels * MemoryLayout<Float16>.stride),
            ("g", model.kdaGProj(layer: 0), kdaGate, 0),
            ("f_a", model.kdaFAProj(layer: 0), kdaFA, 0),
            ("b", model.kdaBProj(layer: 0), kdaBeta16, 0),
        ]
        for (_, view, output, offset) in projections {
            try trunk.encode(commandBuffer: cb, view: view, x: normed,
                             y: output, yOffset: offset)
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.libraryCompileFailed("\(error)") }

        let gpuEmbedding = Self.readFP16Vector(streamA, count: H)
        let gpuNorm = Self.readFP16Vector(normed, count: H)
        let referenceEmbedding = K3ActivationReference.fp16(
            try K3ActivationReference.affineRow(emb, row: Int(token)))
        let referenceNorm = K3ActivationReference.rmsNormFP16(
            referenceEmbedding,
            weight: try K3ActivationReference.bf16Vector(weight),
            eps: Float(config.rmsNormEpsilon))
        var kdaComparisons: [K3ActivationComparison] = []
        for (name, view, output, byteOffset) in projections {
            let rows = K3ActivationReference.sampledRows(count: Int(view.shape.0))
            let outputPtr = output.contents().advanced(by: byteOffset)
                .bindMemory(to: Float16.self, capacity: Int(view.shape.0))
            let actual = rows.map { Float(outputPtr[$0]) }
            let reference = try K3ActivationReference.affineGEMVSamples(
                view, x: referenceNorm, rows: rows)
            kdaComparisons.append(K3ActivationReference.compare(
                name: "layer0.kda.\(name)", actual: actual, reference: reference))
        }
        captureActivationTrace = true
        lastRouterDiagnostics = nil
        lastDiagnosticHeadInput = nil
        return K3ActivationDiagnostics(
            token: token,
            embedding: K3ActivationReference.compare(
                name: "embedding", actual: gpuEmbedding, reference: referenceEmbedding),
            layer0InputNorm: K3ActivationReference.compare(
                name: "layer0.input_norm", actual: gpuNorm, reference: referenceNorm),
            layer0KDAProjections: kdaComparisons)
    }

    /// One token through the full stack. On return (with `emitHead`), the
    /// fp32 logits for this position sit in `logitsBuffer`. `position` must
    /// equal the state's cursor; the cursor advances on success.
    func produce(token: Int32, position: Int, emitHead: Bool) throws {
        precondition(position == state.position,
                     "produce position \(position) != state cursor \(state.position)")
        precondition(position < state.maxContext,
                     "produce position \(position) exceeds maxContext")
        let c = config
        let H = c.hiddenSize
        let eps = Float(c.rmsNormEpsilon)
        let rowBytes = H * MemoryLayout<Float16>.stride

        state.beginToken()
        var prefixValid = false

        let tStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var lap0 = tStart
        var encodeNanos: UInt64 = 0
        func lap() {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            encodeNanos &+= now - lap0
            lap0 = now
        }

        guard var cb = context.queue.makeCommandBuffer() else {
            throw MetalError.noQueue
        }
        var committed = false
        /// Error-path unwind: finish and drain so the streaming banks are not
        /// left live. The partial token's state is discarded by `reset()`.
        func unwind() {
            if !committed { cb.commit() }
            cb.waitUntilCompleted()
            drainPendingEndLayers()
        }

        do {
            let emb = model.embedding
            embed.encodeGather(commandBuffer: cb,
                               table: emb.buffer, tableOffset: Int(emb.offset),
                               scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                               biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                               out: streamA,
                               tokenID: UInt32(bitPattern: token),
                               d: UInt32(H))
            var stream = streamA

            for layer in 0..<c.numLayers {
                // The official AttnRes path keeps this incoming stream as
                // `prefix_sum`. Pre-attn AttnRes changes only the attention
                // input; it must not replace the residual prefix or the
                // vector appended at a block boundary.
                let incoming = stream
                // 1. Pre-attn AttnRes (skipped on the empty block list).
                //    `out` must not alias `prefix` or `blocks` (the kernel
                //    reads every input element while writing); pick the idle
                //    half of the stream ping-pong. `stream` may BE the
                //    state's prefix buffer after a layer boundary, so never
                //    derive the output from a stale swap partner.
                if state.attnResBlockCount > 0 {
                    let out = stream === streamA ? streamB : streamA
                    attnRes.encode(commandBuffer: cb,
                                   blocks: state.attnResBlocks,
                                   prefix: stream,
                                   scoreVector: scoreVectors,
                                   scoreVectorOffset: scoreVectorOffset(layer: layer,
                                                                        site: 0)
                                        * MemoryLayout<Float>.stride,
                                   out: out,
                                   hidden: UInt32(H),
                                   numBlocks: UInt32(state.attnResBlockCount),
                                   eps: eps)
                    stream = out
                }
                // 2. Boundary append / prefix capture.
                if c.isAttnResBoundary(layer0: layer) {
                    blitCopy(commandBuffer: cb, from: incoming,
                             to: state.attnResBlocks,
                             destinationOffset: state.attnResBlockCount * rowBytes,
                             bytes: rowBytes)
                    state.recordAttnResBoundary()
                    prefixValid = false
                } else {
                    // `incoming` is the prefix buffer itself whenever the
                    // block list was empty at step 1 (only possible for
                    // non-boundary layer 0 on exotic configs); skip the
                    // self-copy (Metal forbids overlapping blit ranges).
                    if incoming !== state.attnResPrefix {
                        blitCopy(commandBuffer: cb, from: incoming,
                                 to: state.attnResPrefix,
                                 destinationOffset: 0, bytes: rowBytes)
                    }
                    prefixValid = true
                }
                // 3. input_layernorm -> attention -> o_proj (into attnOut).
                rmsNorm.encodeBF16W(commandBuffer: cb,
                                    x: stream,
                                    weight: model.inputNorm(layer: layer).buffer,
                                    weightOffset: Int(model.inputNorm(layer: layer).offset),
                                    out: normed, d: UInt32(H), eps: eps)
                if c.isKDA(layer0: layer) {
                    encodeKDA(layer: layer, commandBuffer: cb)
                } else {
                    encodeMLA(layer: layer, position: position, commandBuffer: cb)
                }
                // 4. Prefix update (attention).
                encodePrefixAccumulate(commandBuffer: cb, value: attnOut,
                                       prefixValid: &prefixValid, rowBytes: rowBytes)
                // 5. Pre-mlp AttnRes over the post-append block list.
                attnRes.encode(commandBuffer: cb,
                               blocks: state.attnResBlocks,
                               prefix: state.attnResPrefix,
                               scoreVector: scoreVectors,
                               scoreVectorOffset: scoreVectorOffset(layer: layer,
                                                                    site: 1)
                                    * MemoryLayout<Float>.stride,
                               out: hPre,
                               hidden: UInt32(H),
                               numBlocks: UInt32(state.attnResBlockCount),
                               eps: eps)
                // 6. post_attention_layernorm -> MoE / dense.
                rmsNorm.encodeBF16W(
                    commandBuffer: cb,
                    x: hPre,
                    weight: model.postAttnNorm(layer: layer).buffer,
                    weightOffset: Int(model.postAttnNorm(layer: layer).offset),
                    out: normed, d: UInt32(H), eps: eps)

                if c.isMoE(layer0: layer) {
                    // Router scores, then the token's only mid-layer sync.
                    util.encodeWiden(commandBuffer: cb, x: normed, out: h32,
                                     n: UInt32(H))
                    let gate = model.routerGate(layer: layer)
                    let bias = model.routerCorrectionBias(layer: layer)
                    router.encodeScores(commandBuffer: cb,
                                        gate: gate.buffer,
                                        gateOffset: Int(gate.offset),
                                        x: h32,
                                        bias: bias.buffer,
                                        biasOffset: Int(bias.offset),
                                        scores: scores,
                                        numExperts: UInt32(c.moeNumExperts),
                                        d: UInt32(H))
                    lap()
                    cb.commit()
                    committed = true
                    let w0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    cb.waitUntilCompleted()
                    timings.routerWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - w0
                    if let error = cb.error {
                        throw MetalError.libraryCompileFailed("\(error)")
                    }
                    drainPendingEndLayers()

                    if captureActivationTrace,
                       layer == c.moeLayers0.min() {
                        let input = Self.readFP32Buffer(h32, count: H)
                        let actualScores = Self.readFP32Buffer(
                            scores, count: c.moeNumExperts)
                        lastRouterDiagnostics = try K3ActivationReference.routerDiagnostics(
                            model: model, layer: layer, tokenInChunk: 0,
                            input: input, actualScores: actualScores)
                    }

                    let selected = readbackTopK(layer: layer)
                    // Overlap: shared-expert + down-proj GPU work commits
                    // BEFORE the CPU blocks on the expert preads.
                    guard let sharedCB = context.queue.makeCommandBuffer() else {
                        throw MetalError.noQueue
                    }
                    encodeMoESharedAndDown(layer: layer, commandBuffer: sharedCB)
                    lap()
                    sharedCB.commit()

                    let io0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let batch = try streaming.beginLayer(layer, actualExperts: selected.indices)
                    timings.expertIONanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - io0
                    writeMoEArguments(batch: batch, weights: selected.weights)

                    guard let nextCB = context.queue.makeCommandBuffer() else {
                        throw MetalError.noQueue
                    }
                    cb = nextCB
                    committed = false
                    encodeMoEExpertPhases(layer: layer, batch: batch,
                                          commandBuffer: cb)
                    // 7. Prefix update (MLP), inside the same buffer.
                    encodePrefixAccumulate(commandBuffer: cb, value: mlpOut,
                                           prefixValid: &prefixValid, rowBytes: rowBytes)
                    pendingEndLayers.append(batch)
                    streaming.recordRouting(layer0: layer, experts: selected.indices)
                } else {
                    encodeDenseMLP(layer: layer, commandBuffer: cb)
                    encodePrefixAccumulate(commandBuffer: cb, value: mlpOut,
                                           prefixValid: &prefixValid, rowBytes: rowBytes)
                }
                // The prefix is the next layer's stream.
                stream = state.attnResPrefix
            }

            if emitHead {
                // Output AttnRes -> final norm -> int8 lm_head (fp32 logits).
                attnRes.encode(commandBuffer: cb,
                               blocks: state.attnResBlocks,
                               prefix: stream,
                               scoreVector: scoreVectors,
                               scoreVectorOffset: scoreVectorOffset(layer: 0, site: 2)
                                    * MemoryLayout<Float>.stride,
                               out: hPre,
                               hidden: UInt32(H),
                               numBlocks: UInt32(state.attnResBlockCount),
                               eps: eps)
                rmsNorm.encodeBF16W(commandBuffer: cb,
                                    x: hPre,
                                    weight: model.finalNorm.buffer,
                                    weightOffset: Int(model.finalNorm.offset),
                                    out: normed, d: UInt32(H), eps: eps)
                let head = model.lmHead
                lmHead.encode(commandBuffer: cb,
                              weights: head.buffer, weightsOffset: Int(head.offset),
                              scales: head.buffer, scalesOffset: Int(head.scaleOffset),
                              biases: head.buffer, biasesOffset: Int(head.biasOffset),
                              x: normed, y: logitsBuffer,
                              m: UInt32(c.vocabSize), n: UInt32(H))
            }
            lap()
            cb.commit()
            committed = true
            let w1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            cb.waitUntilCompleted()
            timings.tailWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - w1
            if let error = cb.error {
                throw MetalError.libraryCompileFailed("\(error)")
            }
            drainPendingEndLayers()
            if captureActivationTrace, emitHead {
                lastDiagnosticHeadInput = Self.readFP16Vector(normed, count: H)
            }
            state.advance()
            timings.encodeCommitNanos &+= encodeNanos
            timings.tokensProduced &+= 1
        } catch {
            unwind()
            throw error
        }
    }

    // MARK: - Layer stages

    private func blitCopy(commandBuffer: MTLCommandBuffer,
                          from source: MTLBuffer, to destination: MTLBuffer,
                          destinationOffset: Int, bytes: Int) {
        guard let encoder = commandBuffer.makeBlitCommandEncoder() else { return }
        encoder.copy(from: source, sourceOffset: 0,
                     to: destination, destinationOffset: destinationOffset,
                     size: bytes)
        encoder.endEncoding()
    }

    /// Step 4/7: `prefix = prefixValid ? prefix + value : value`.
    private func encodePrefixAccumulate(commandBuffer: MTLCommandBuffer,
                                        value: MTLBuffer,
                                        prefixValid: inout Bool,
                                        rowBytes: Int) {
        if prefixValid {
            util.encodeAdd(commandBuffer: commandBuffer,
                           a: state.attnResPrefix, b: value,
                           out: state.attnResPrefix, n: UInt32(config.hiddenSize))
        } else {
            blitCopy(commandBuffer: commandBuffer, from: value,
                     to: state.attnResPrefix, destinationOffset: 0, bytes: rowBytes)
            prefixValid = true
        }
    }

    /// KDA attention from `normed` into `attnOut`
    /// (docs/K3_DATAFLOW.md "KDA layer").
    private func encodeKDA(layer: Int, commandBuffer cb: MTLCommandBuffer) {
        let c = config
        let P = c.kdaChannels
        let f16 = MemoryLayout<Float16>.stride
        guard let ordinal = c.kdaOrdinal(layer0: layer) else { return }

        // Projections: q/k/v/g/f_a/b all read the normed input.
        do {
            try trunk.encode(commandBuffer: cb, view: model.kdaQProj(layer: layer),
                             x: normed, y: kdaProj)
            try trunk.encode(commandBuffer: cb, view: model.kdaKProj(layer: layer),
                             x: normed, y: kdaProj, yOffset: P * f16)
            try trunk.encode(commandBuffer: cb, view: model.kdaVProj(layer: layer),
                             x: normed, y: kdaProj, yOffset: 2 * P * f16)
            try trunk.encode(commandBuffer: cb, view: model.kdaGProj(layer: layer),
                             x: normed, y: kdaGate)
            try trunk.encode(commandBuffer: cb, view: model.kdaFAProj(layer: layer),
                             x: normed, y: kdaFA)
            try trunk.encode(commandBuffer: cb, view: model.kdaBProj(layer: layer),
                             x: normed, y: kdaBeta16)
        } catch {
            preconditionFailure("KDA trunk view rejected: \(error)")
        }

        let convState = state.kdaConvState(layer0: layer)
        kda.encodeConv(commandBuffer: cb,
                       xq: kdaProj, xk: kdaProj, xkOffset: P * f16,
                       xv: kdaProj, xvOffset: 2 * P * f16,
                       weights: convWeights,
                       weightsOffset: ordinal * 3 * P * c.kdaConvWidth
                           * MemoryLayout<Float>.stride,
                       convStates: convState.buffer,
                       convStatesOffset: convState.offset,
                       qOut: kdaConvOut,
                       kOut: kdaConvOut, kOutOffset: P * MemoryLayout<Float>.stride,
                       vOut: kdaConvOut, vOutOffset: 2 * P * MemoryLayout<Float>.stride,
                       channels: UInt32(P))

        // z = f_b(f_a(x)) + dt_bias; beta = widen(b_proj(x)).
        do {
            try trunk.encode(commandBuffer: cb, view: model.kdaFBProj(layer: layer),
                             x: kdaFA, y: kdaFB16)
        } catch {
            preconditionFailure("KDA f_b trunk view rejected: \(error)")
        }
        let dtBias = model.kdaDTBias(layer: layer)
        util.encodeAddBias(commandBuffer: cb, x: kdaFB16,
                           bias: dtBias.buffer, biasOffset: Int(dtBias.offset),
                           out: kdaZ, n: UInt32(P))
        util.encodeWiden(commandBuffer: cb, x: kdaBeta16, out: kdaBeta32,
                         n: UInt32(c.kdaNumHeads))

        let recurrent = state.kdaRecurrentState(layer0: layer)
        let aLog = model.kdaALog(layer: layer)
        kda.encodeStep(commandBuffer: cb,
                       state: recurrent.buffer, stateOffset: recurrent.offset,
                       q: kdaConvOut,
                       k: kdaConvOut, kOffset: P * MemoryLayout<Float>.stride,
                       v: kdaConvOut, vOffset: 2 * P * MemoryLayout<Float>.stride,
                       z: kdaZ,
                       betaLogits: kdaBeta32,
                       aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                       o: kdaStepOut,
                       numHeads: UInt32(c.kdaNumHeads),
                       headDim: UInt32(c.kdaHeadDim))

        let oNorm = model.kdaONorm(layer: layer)
        kda.encodeOutputNorm(commandBuffer: cb,
                             o: kdaStepOut, gate: kdaGate,
                             weight: oNorm.buffer, weightOffset: Int(oNorm.offset),
                             out: kdaONorm,
                             eps: Float(c.rmsNormEpsilon),
                             numHeads: UInt32(c.kdaNumHeads),
                             headDim: UInt32(c.kdaHeadDim))
        do {
            try trunk.encode(commandBuffer: cb, view: model.kdaOProj(layer: layer),
                             x: kdaONorm, y: attnOut)
        } catch {
            preconditionFailure("KDA o_proj trunk view rejected: \(error)")
        }
    }

    /// MLA attention from `normed` into `attnOut`
    /// (docs/K3_DATAFLOW.md "MLA layer"; absorbed decode over the latent cache).
    private func encodeMLA(layer: Int, position: Int, commandBuffer cb: MTLCommandBuffer) {
        let c = config
        let f16 = MemoryLayout<Float16>.stride
        guard let ordinal = c.mlaOrdinal(layer0: layer) else { return }
        let L = c.mlaKVLoraRank
        let R = c.mlaQKRopeHeadDim

        do {
            try trunk.encode(commandBuffer: cb, view: model.mlaQAProj(layer: layer),
                             x: normed, y: mlaQA)
        } catch {
            preconditionFailure("MLA q_a trunk view rejected: \(error)")
        }
        let qaNorm = model.mlaQANorm(layer: layer)
        util.encodeRMSNormF32W(commandBuffer: cb, x: mlaQA,
                               weight: qaNorm.buffer,
                               weightOffset: Int(qaNorm.offset),
                               out: mlaQLat, d: UInt32(c.mlaQLoraRank),
                               eps: Self.mlaLoraNormEps)
        do {
            try trunk.encode(commandBuffer: cb, view: model.mlaQBProj(layer: layer),
                             x: mlaQLat, y: mlaQ)
            try trunk.encode(commandBuffer: cb, view: model.mlaKVAProj(layer: layer),
                             x: normed, y: mlaKVA)
            try trunk.encode(commandBuffer: cb, view: model.mlaGProj(layer: layer),
                             x: normed, y: mlaGate)
        } catch {
            preconditionFailure("MLA trunk view rejected: \(error)")
        }

        let cache = state.mlaCache(layer0: layer)
        let kvNorm = model.mlaKVANorm(layer: layer)
        mla.encodeCacheAppend(commandBuffer: cb,
                              cache: cache.buffer, cacheOffset: cache.offset,
                              position: UInt32(position),
                              kvA: mlaKVA,
                              normWeight: kvNorm.buffer,
                              normWeightOffset: Int(kvNorm.offset),
                              eps: Self.mlaLoraNormEps,
                              latent: UInt32(L), rope: UInt32(R))
        mla.encodeAbsorbQ(commandBuffer: cb,
                          kT: mlaKTPlane,
                          kTOffset: ordinal * c.mlaNumHeads * L * c.mlaQKNopeHeadDim * f16,
                          q: mlaQ, qAbs: mlaQAbs,
                          numHeads: UInt32(c.mlaNumHeads), latent: UInt32(L),
                          rope: UInt32(R), nope: UInt32(c.mlaQKNopeHeadDim))
        mla.encodeAttnDecode(commandBuffer: cb,
                             cache: cache.buffer, cacheOffset: cache.offset,
                             qAbs: mlaQAbs, outLat: mlaOutLat,
                             seqLen: UInt32(position + 1),
                             numHeads: UInt32(c.mlaNumHeads), latent: UInt32(L),
                             rope: UInt32(R), scale: c.mlaDecodeScale)
        mla.encodeOutProject(commandBuffer: cb,
                             v: mlaVPlane,
                             vOffset: ordinal * c.mlaNumHeads * c.mlaVHeadDim * L * f16,
                             outLat: mlaOutLat, gate: mlaGate, out: mlaOut,
                             numHeads: UInt32(c.mlaNumHeads), latent: UInt32(L),
                             vHead: UInt32(c.mlaVHeadDim))
        do {
            try trunk.encode(commandBuffer: cb, view: model.mlaOProj(layer: layer),
                             x: mlaOut, y: attnOut)
        } catch {
            preconditionFailure("MLA o_proj trunk view rejected: \(error)")
        }
    }

    /// Dense SiTU-GLU MLP (layer 0) from `normed` into `mlpOut`.
    private func encodeDenseMLP(layer: Int, commandBuffer cb: MTLCommandBuffer) {
        let c = config
        do {
            try trunk.encode(commandBuffer: cb, view: model.denseGateProj(layer: layer),
                             x: normed, y: denseGate)
            try trunk.encode(commandBuffer: cb, view: model.denseUpProj(layer: layer),
                             x: normed, y: denseUp)
        } catch {
            preconditionFailure("dense MLP trunk view rejected: \(error)")
        }
        util.encodeSiTU(commandBuffer: cb, gate: denseGate, up: denseUp, out: denseH,
                        n: UInt32(c.denseMLPIntermediateSize),
                        beta1: Float(c.situGLUGateBeta),
                        beta2: Float(c.situGLUUpBeta))
        do {
            try trunk.encode(commandBuffer: cb, view: model.denseDownProj(layer: layer),
                             x: denseH, y: mlpOut)
        } catch {
            preconditionFailure("dense down trunk view rejected: \(error)")
        }
    }

    /// MoE work that does NOT need the experts: the latent down-projection
    /// and the full shared-expert MLP. Encoded into a command buffer that
    /// commits before the CPU blocks on the expert reads.
    private func encodeMoESharedAndDown(layer: Int, commandBuffer cb: MTLCommandBuffer) {
        let c = config
        do {
            try trunk.encode(commandBuffer: cb, view: model.routedDownProj(layer: layer),
                             x: normed, y: xLat)
            try trunk.encode(commandBuffer: cb, view: model.sharedGateProj(layer: layer),
                             x: normed, y: sharedGate)
            try trunk.encode(commandBuffer: cb, view: model.sharedUpProj(layer: layer),
                             x: normed, y: sharedUp)
        } catch {
            preconditionFailure("MoE shared/down trunk view rejected: \(error)")
        }
        util.encodeSiTU(commandBuffer: cb, gate: sharedGate, up: sharedUp,
                        out: sharedH,
                        n: UInt32(c.moeSharedExpertIntermediateSize),
                        beta1: Float(c.situGLUGateBeta),
                        beta2: Float(c.situGLUUpBeta))
        do {
            try trunk.encode(commandBuffer: cb, view: model.sharedDownProj(layer: layer),
                             x: sharedH, y: sharedOut)
        } catch {
            preconditionFailure("MoE shared down trunk view rejected: \(error)")
        }
    }

    /// The streamed-expert half of the MoE block: phase1 (fused w1/w3 +
    /// SiTU), phase2 (w2 + weighted reduce), routed_expert_norm, the up
    /// projection, and the shared-expert add — into `mlpOut`.
    private func encodeMoEExpertPhases(layer: Int, batch: K3ExpertBatch,
                                       commandBuffer cb: MTLCommandBuffer) {
        let c = config
        let offsets = K3ExpertSubtensorOffsets.canonical(
            dLatent: UInt32(c.moeLatentBottleneckSize),
            intermediate: UInt32(c.moeExpertIntermediateSize))
        moe.encodePhase1(commandBuffer: cb,
                         experts: batch.buffer,
                         slotOffsets: slotOffsets,
                         xLat: xLat, h: moeH,
                         subtensorOffsets: offsets,
                         dLatent: UInt32(c.moeLatentBottleneckSize),
                         intermediate: UInt32(c.moeExpertIntermediateSize),
                         topK: UInt32(c.moeTopKExperts))
        moe.encodePhase2(commandBuffer: cb,
                         experts: batch.buffer,
                         slotOffsets: slotOffsets,
                         h: moeH,
                         routingWeights: routingWeights,
                         yLat: yLat,
                         subtensorOffsets: offsets,
                         dLatent: UInt32(c.moeLatentBottleneckSize),
                         intermediate: UInt32(c.moeExpertIntermediateSize),
                         topK: UInt32(c.moeTopKExperts))
        let rNorm = model.routedExpertNorm(layer: layer)
        util.encodeRMSNormF32X(commandBuffer: cb, x: yLat,
                               weight: rNorm.buffer,
                               weightOffset: Int(rNorm.offset),
                               out: yLatNormed,
                               d: UInt32(c.moeLatentBottleneckSize),
                               eps: Float(c.rmsNormEpsilon))
        do {
            try trunk.encode(commandBuffer: cb, view: model.routedUpProj(layer: layer),
                             x: yLatNormed, y: mlpOut)
        } catch {
            preconditionFailure("MoE up trunk view rejected: \(error)")
        }
        util.encodeAdd(commandBuffer: cb, a: mlpOut, b: sharedOut, out: mlpOut,
                       n: UInt32(c.hiddenSize))
    }

    // MARK: - Router readback / MoE arguments

    /// Sigmoid scores are already in `scores` (post-wait); select on the CPU.
    private func readbackTopK(layer: Int) -> (indices: [Int], weights: [Float]) {
        let count = config.moeNumExperts
        let ptr = scores.contents().bindMemory(to: Float.self, capacity: count)
        let allScores = Array(UnsafeBufferPointer(start: ptr, count: count))
        let bias = routerBiasByLayer[layer] ?? [Float](repeating: 0, count: count)
        let (indices32, weights) = K3Router.selectTopK(
            scores: allScores, bias: bias, topK: config.moeTopKExperts)
        return (indices32.map(Int.init), weights)
    }

    /// Publish the batch's slot offsets and the router weights into the
    /// shared argument buffers the phase kernels read. The CPU writes land
    /// before the consuming command buffer commits, and the buffers are not
    /// touched by the shared-expert CB committed in between.
    private func writeMoEArguments(batch: K3ExpertBatch, weights: [Float]) {
        let so = slotOffsets.contents().bindMemory(to: UInt64.self,
                                                   capacity: config.moeTopKExperts)
        for i in 0..<config.moeTopKExperts { so[i] = batch.slotOffsets[i] }
        let rw = routingWeights.contents().bindMemory(to: Float.self,
                                                      capacity: config.moeTopKExperts)
        for i in 0..<config.moeTopKExperts { rw[i] = weights[i] }
    }

    private func drainPendingEndLayers() {
        for batch in pendingEndLayers { streaming.endLayer(batch) }
        pendingEndLayers.removeAll(keepingCapacity: true)
    }

    // MARK: - Stage-E1 chunked prefill hook

    /// Read-only products the chunked prefiller reuses: the load-time weight
    /// conversions (fused AttnRes score vectors, gathered KDA conv weights,
    /// MLA absorb planes), the per-layer router correction bias, and the small
    /// single-purpose dispatchers the batched path shares with decode (embed
    /// gather, lm_head, the bf16 RMSNorm used for the head, the elementwise
    /// utility kernels, and the single-position AttnRes used for the head).
    /// The prefiller owns no weights of its own; it reads them through here.
    var prefillShared: K3PrefillShared {
        K3PrefillShared(
            scoreVectors: scoreVectors,
            convWeights: convWeights,
            mlaKTPlane: mlaKTPlane,
            mlaVPlane: mlaVPlane,
            routerBiasByLayer: routerBiasByLayer,
            embed: embed,
            lmHead: lmHead,
            rmsNorm: rmsNorm,
            util: util,
            attnRes: attnRes,
            logitsBuffer: logitsBuffer)
    }
}
