import Foundation
import Metal

/// Kimi Delta Attention single-token decode step. Three encode stages per
/// token: `encodeConv` (depthwise causal conv width 4 + SiLU on projected
/// q/k/v, carrying conv state), `encodeStep` (fused per-head L2 norm +
/// gates + delta-rule recurrence on the fp32 state, output AFTER the
/// update), and `encodeOutputNorm` (per-head RMSNorm + sigmoid gate, fp16
/// out for the o_proj trunk GEMV).
///
/// All state buffers (conv states, recurrent state) are owned by the caller
/// (the Stage-C state manager); this dispatcher only encodes. Conv weights
/// are FP32 [3][P][4] — converted once at load (see kda.metal's header for
/// the no-bias contract).
final class K3KDA {
    static let convWidth = 4
    static let convHist = 3
    static let maxHeadDim = 128

    private static let realNumHeads: UInt32 = 96
    private static let realHeadDim: UInt32 = 128
    private static let realChannels: UInt32 = 96 * 128
    private static let realConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 70, value: .uint32(realNumHeads)),
        MetalFunctionConstant(index: 71, value: .uint32(realHeadDim)),
        MetalFunctionConstant(index: 72, value: .uint32(realChannels)),
        MetalFunctionConstant(index: 73, value: .bool(true)),
    ]

    private let convPSO: MTLComputePipelineState
    private let convSpecializedPSO: MTLComputePipelineState
    private let stepPSO: MTLComputePipelineState
    private let stepSpecializedPSO: MTLComputePipelineState
    private let onormPSO: MTLComputePipelineState
    private let onormSpecializedPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        let library = K3MetalLibrary.shared
        self.convPSO = try library.pipeline(
            device: context.device, name: "k3_kda_conv")
        self.convSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_kda_conv",
            constants: Self.realConstants)
        self.stepPSO = try library.pipeline(
            device: context.device, name: "k3_kda_step")
        self.stepSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_kda_step",
            constants: Self.realConstants)
        self.onormPSO = try library.pipeline(
            device: context.device, name: "k3_kda_onorm")
        self.onormSpecializedPSO = try library.pipeline(
            device: context.device, name: "k3_kda_onorm",
            constants: Self.realConstants)
    }

    /// Depthwise causal conv + SiLU for q/k/v. `xq`/`xk`/`xv` are fp16 [P];
    /// `weights` fp32 [3][P][4] (q|k|v, taps oldest..newest); `convStates`
    /// fp32 [3][P][3] updated in place (history oldest first); outputs fp32
    /// [P] each.
    func encodeConv(commandBuffer: MTLCommandBuffer,
                    xq: MTLBuffer, xqOffset: Int = 0,
                    xk: MTLBuffer, xkOffset: Int = 0,
                    xv: MTLBuffer, xvOffset: Int = 0,
                    weights: MTLBuffer, weightsOffset: Int = 0,
                    convStates: MTLBuffer, convStatesOffset: Int = 0,
                    qOut: MTLBuffer, qOutOffset: Int = 0,
                    kOut: MTLBuffer, kOutOffset: Int = 0,
                    vOut: MTLBuffer, vOutOffset: Int = 0,
                    channels: UInt32) {
        precondition(channels > 0)
        var p = channels
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            channels == Self.realChannels ? convSpecializedPSO : convPSO)
        encoder.setBuffer(xq, offset: xqOffset, index: 0)
        encoder.setBuffer(xk, offset: xkOffset, index: 1)
        encoder.setBuffer(xv, offset: xvOffset, index: 2)
        encoder.setBuffer(weights, offset: weightsOffset, index: 3)
        encoder.setBuffer(convStates, offset: convStatesOffset, index: 4)
        encoder.setBuffer(qOut, offset: qOutOffset, index: 5)
        encoder.setBuffer(kOut, offset: kOutOffset, index: 6)
        encoder.setBuffer(vOut, offset: vOutOffset, index: 7)
        encoder.setBytes(&p, length: MemoryLayout<UInt32>.stride, index: 8)
        let total = 3 * Int(channels)
        let threadsPerThreadgroup = 256
        encoder.dispatchThreadgroups(
            MTLSize(width: (total + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Fused per-head recurrence step. `state` is fp32 [H][D][D] (row = key
    /// channel) updated in place; `q`/`k`/`v` are the fp32 post-conv vectors
    /// [P]; `z` is fp32 [P] with dt_bias already added; `betaLogits` fp32
    /// [H]; `aLog` fp32 [H]; `o` fp32 [P] out.
    func encodeStep(commandBuffer: MTLCommandBuffer,
                    state: MTLBuffer, stateOffset: Int = 0,
                    q: MTLBuffer, qOffset: Int = 0,
                    k: MTLBuffer, kOffset: Int = 0,
                    v: MTLBuffer, vOffset: Int = 0,
                    z: MTLBuffer, zOffset: Int = 0,
                    betaLogits: MTLBuffer, betaLogitsOffset: Int = 0,
                    aLog: MTLBuffer, aLogOffset: Int = 0,
                    o: MTLBuffer, oOffset: Int = 0,
                    numHeads: UInt32, headDim: UInt32) {
        precondition(numHeads > 0)
        precondition(headDim > 0 && headDim <= UInt32(Self.maxHeadDim),
                     "headDim \(headDim) exceeds kernel staging (\(Self.maxHeadDim))")
        var h = numHeads
        var d = headDim
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealConstants(numHeads: numHeads, headDim: headDim)
                ? stepSpecializedPSO : stepPSO)
        encoder.setBuffer(state, offset: stateOffset, index: 0)
        encoder.setBuffer(q, offset: qOffset, index: 1)
        encoder.setBuffer(k, offset: kOffset, index: 2)
        encoder.setBuffer(v, offset: vOffset, index: 3)
        encoder.setBuffer(z, offset: zOffset, index: 4)
        encoder.setBuffer(betaLogits, offset: betaLogitsOffset, index: 5)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 6)
        encoder.setBuffer(o, offset: oOffset, index: 7)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(numHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Per-head gated output norm: fp32 `o` [P] in, fp16 out [P]. `weight`
    /// is fp32 [D] (o_norm.weight); `gate` fp16 [P] (g_proj output).
    func encodeOutputNorm(commandBuffer: MTLCommandBuffer,
                          o: MTLBuffer, oOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          weight: MTLBuffer, weightOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          eps: Float,
                          numHeads: UInt32, headDim: UInt32) {
        precondition(numHeads > 0)
        precondition(headDim > 0 && headDim <= UInt32(Self.maxHeadDim),
                     "headDim \(headDim) exceeds kernel staging (\(Self.maxHeadDim))")
        var h = numHeads
        var d = headDim
        var e = eps
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealConstants(numHeads: numHeads, headDim: headDim)
                ? onormSpecializedPSO : onormPSO)
        encoder.setBuffer(o, offset: oOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        encoder.setBytes(&h, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&e, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(numHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func useRealConstants(numHeads: UInt32, headDim: UInt32) -> Bool {
        numHeads == Self.realNumHeads && headDim == Self.realHeadDim
    }
}
