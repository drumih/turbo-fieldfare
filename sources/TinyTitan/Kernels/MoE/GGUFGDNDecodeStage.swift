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
    private let arena: GGUFResidentArena
    private let config: LinearAttentionConfig
    private let hidden: Int

    private let qkvRaw: MTLBuffer
    private let zOut: MTLBuffer
    private let aOut: MTLBuffer
    private let bOut: MTLBuffer
    private let convOut: MTLBuffer
    private let yOut: MTLBuffer
    private let gdnOut: MTLBuffer
    private let tail: MTLBuffer
    private let state: MTLBuffer

    package init(
        context: MetalContext, arch: ArchConfig, arena: GGUFResidentArena
    ) throws {
        self.gdn = try GDN(context: context, config: arch.linearAttention)
        self.gemv = try BF16GEMV(context: context)
        self.arena = arena
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
        self.tail = try make((config.convKernelSize - 1) * config.qkvDim * half, "gdn tail")
        self.state = try make(
            config.numVHeads * config.valueHeadDim * config.keyHeadDim * 4, "gdn state")
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
        let prefix = "blk.\(layer)."
        func view(_ suffix: String, rows: Int) throws -> (MTLBuffer, Int) {
            guard let entry = arena.entries[prefix + suffix] else {
                throw MetalError.invalidState("missing \(prefix + suffix) in the arena")
            }
            _ = rows
            return (arena.buffer, Int(entry.offset))
        }
        let (qkvW, qkvOff) = try view("attn_qkv.weight", rows: config.qkvDim)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: qkvW, weightsOffset: qkvOff,
            x: x, xOffset: xOffset, y: qkvRaw,
            m: UInt32(config.qkvDim), n: UInt32(hidden))
        let (zW, zOff) = try view("attn_gate.weight", rows: config.valueDim)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: zW, weightsOffset: zOff,
            x: x, xOffset: xOffset, y: zOut,
            m: UInt32(config.valueDim), n: UInt32(hidden))
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
        let (outW, outOff) = try view("ssm_out.weight", rows: hidden)
        try gemv.encode(
            commandBuffer: commandBuffer, weights: outW, weightsOffset: outOff,
            x: gdnOut, y: out, yOffset: outOffset,
            m: UInt32(hidden), n: UInt32(config.valueDim))
    }
}
