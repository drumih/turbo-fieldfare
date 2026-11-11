import Foundation
import Metal

/// One GDN layer's decode step, driven from the arena: the four input
/// projections, the causal conv with its tail, the per-head qk norm, the delta
/// rule with its persistent FP32 state, the gated output norm and the out
/// projection.
///
/// Every weight the GDN kernels read is bfloat in `gdn.metal` (conv, A_log,
/// dt_bias, the gated-norm weight) and the projections go through BF16GEMV, so
/// the arena's bf16 views serve the whole chain: no GGUF-specific GDN kernel.
package final class GGUFGDNDecodeStage {
    private let gdn: GDN
    private let gemv: BF16GEMV
    private let fused: GGUFExpertGEMV
    private let arena: GGUFResidentArena
    private let mapped: GGUFMappedShards
    private let config: LinearAttentionConfig
    private let hidden: Int

    private let qkvRaw: MTLBuffer
    private let zOut: MTLBuffer
    private let aOut: MTLBuffer
    private let bOut: MTLBuffer
    private let convOut: MTLBuffer
    private let yOut: MTLBuffer
    private let gdnOut: MTLBuffer
    /// The causal conv's tail and the delta rule's FP32 state, one pair per
    /// GDN layer: both are recurrent, so a layer reading another layer's
    /// state is wrong from the first token on.
    private let tails: [Int: MTLBuffer]
    private let states: [Int: MTLBuffer]

    package init(
        context: MetalContext, arch: ArchConfig, arena: GGUFResidentArena,
        mapped: GGUFMappedShards
    ) throws {
        // A llama.cpp GGUF stores `ssm_a` as -exp(A_log) and the value heads in
        // tiled order; the delta kernels compile for that layout.
        self.gdn = try GDN(
            context: context, config: arch.linearAttention, storage: .llamaCppGGUF)
        self.gemv = try BF16GEMV(context: context)
        self.fused = try GGUFExpertGEMV(context: context)
        self.arena = arena
        self.mapped = mapped
        self.config = arch.linearAttention
        self.hidden = arch.hiddenSize

        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: bytes, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed(label) }
            memset(buffer.contents(), 0, bytes)
            return buffer
        }
        let half = 2
        self.qkvRaw = try make(config.qkvDim * half, "gdn qkv")
        let value = config.numVHeads * config.valueHeadDim
        self.zOut = try make(value * half, "gdn z")
        self.aOut = try make(config.numVHeads * half, "gdn a")
        self.bOut = try make(config.numVHeads * half, "gdn b")
        self.convOut = try make(config.qkvDim * half, "gdn conv")
        self.yOut = try make(value * half, "gdn y")
        self.gdnOut = try make(value * half, "gdn out")
        var tails: [Int: MTLBuffer] = [:]
        var states: [Int: MTLBuffer] = [:]
        for layer in 0..<arch.numLayers where arch.fullAttentionLayerMask[layer] == 2 {
            tails[layer] = try make(
                (config.convKernelSize - 1) * config.qkvDim * half, "gdn tail \(layer)")
            states[layer] = try make(
                config.numVHeads * config.valueHeadDim * config.keyHeadDim * 4,
                "gdn state \(layer)")
        }
        self.tails = tails
        self.states = states
    }

    /// Exposed for the verification test: the intermediate rows this stage
    /// produces, so a mismatch can be attributed to one kernel.
    package var debugRows: (qkv: MTLBuffer, conv: MTLBuffer, y: MTLBuffer, gated: MTLBuffer) {
        (qkvRaw, convOut, yOut, gdnOut)
    }

    /// Runs the layer for one token. `x` is the normalized block input; the
    /// result lands in `out` (hidden-wide, half).
    package func encode(
        commandBuffer: MTLCommandBuffer,
        layer: Int,
        x: MTLBuffer,
        xOffset: Int = 0,
        out: MTLBuffer,
        outOffset: Int = 0
    ) throws {
        guard let tail = tails[layer], let state = states[layer] else {
            throw MetalError.invalidState("layer \(layer) is not a GDN layer")
        }
        let prefix = "blk.\(layer)."
        // The fused GEMV entry point takes whole buffers; every caller passes
        // base buffers, so an offset here would be a silent mis-read.
        precondition(xOffset == 0 && outOffset == 0, "fused GEMV takes whole buffers")
        func view(_ suffix: String, rows: Int) throws -> (MTLBuffer, Int) {
            guard let entry = arena.entries[prefix + suffix] else {
                throw MetalError.invalidState("missing \(prefix + suffix) in the arena")
            }
            _ = rows
            return (arena.buffer, Int(entry.offset))
        }
        func fusedGEMV(
            _ name: String, _ y: MTLBuffer, rows: Int
        ) throws {
            guard let weight = try mapped.weight(named: prefix + name) else {
                throw MetalError.invalidState("missing \(prefix + name) in the shards")
            }
            try fused.encode(
                commandBuffer: commandBuffer, weights: weight.buffer,
                weightsOffset: weight.offset, x: x, y: y,
                rows: rows, columns: hidden, type: weight.type)
        }
        try fusedGEMV("attn_qkv.weight", qkvRaw, rows: config.qkvDim)
        try fusedGEMV("attn_gate.weight", zOut, rows: config.valueDim)
        let (aW, aOff) = try view("ssm_alpha.weight", rows: config.numVHeads)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: aW, weightsOffset: aOff,
            x: x, xOffset: xOffset, y: aOut,
            m: UInt32(config.numVHeads), n: UInt32(hidden))
        let (bW, bOff) = try view("ssm_beta.weight", rows: config.numVHeads)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: bW, weightsOffset: bOff,
            x: x, xOffset: xOffset, y: bOut,
            m: UInt32(config.numVHeads), n: UInt32(hidden))

        let (convW, convOff) = try view("ssm_conv1d.weight", rows: config.convKernelSize)
        try gdn.encodeConvDecode(
            commandBuffer: commandBuffer, tail: tail, qkv: qkvRaw,
            convWeight: convW, convWeightOffset: convOff, out: convOut)
        try gdn.encodeQKNorm(commandBuffer: commandBuffer, convOut: convOut)
        let (aLog, aLogOff) = try view("ssm_a", rows: config.numVHeads)
        let (dtBias, dtOff) = try view("ssm_dt.bias", rows: config.numVHeads)
        try gdn.encodeDeltaStepDecode(
            commandBuffer: commandBuffer, convOut: convOut,
            aProj: aOut, bProj: bOut,
            aLog: aLog, aLogOffset: aLogOff,
            dtBias: dtBias, dtBiasOffset: dtOff,
            state: state, y: yOut)
        let (normW, normOff) = try view("ssm_norm.weight", rows: config.valueHeadDim)
        try gdn.encodeGatedNorm(
            commandBuffer: commandBuffer, y: yOut, z: zOut,
            weight: normW, weightOffset: normOff, out: gdnOut)
        guard let outWeight = try mapped.weight(named: prefix + "ssm_out.weight") else {
            throw MetalError.invalidState("missing \(prefix)ssm_out.weight in the shards")
        }
        try fused.encode(
            commandBuffer: commandBuffer, weights: outWeight.buffer,
            weightsOffset: outWeight.offset, x: gdnOut, y: out,
            rows: hidden, columns: config.valueDim, type: outWeight.type)
    }

    /// Clears every layer's conv tail and recurrent state, for a new sequence.
    package func resetState() {
        for buffer in tails.values { memset(buffer.contents(), 0, buffer.length) }
        for buffer in states.values { memset(buffer.contents(), 0, buffer.length) }
    }
}
