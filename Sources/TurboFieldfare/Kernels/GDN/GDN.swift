import Foundation
import Metal

/// Gated-DeltaNet linear attention kernels (Qwen 3.6 layers with layer-mask
/// value 2). Wraps the `gdn.metal` module: causal depthwise conv + SiLU,
/// per-head q/k norm with the delta-rule scales folded in, the gated delta
/// recurrence (decode step and chunked prefill), and the gated output norm.
///
/// Buffer layouts follow the projection outputs directly:
///  - `mixed_qkv` / `conv_out` rows: `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`
///  - recurrent state: FP32 `[Hv, Dv, Dk]` (owned by `GDNStateManager`)
///  - conv tail: FP16 `[convKernel - 1, convDim]`
final class GDN {
    private let convDecodePSO: MTLComputePipelineState
    private let convPrefillPSO: MTLComputePipelineState
    private let convTailUpdatePSO: MTLComputePipelineState
    private let qkNormPSO: MTLComputePipelineState
    private let deltaDecodePSO: MTLComputePipelineState
    private let deltaPrefillPSO: MTLComputePipelineState
    private let gatedNormPSO: MTLComputePipelineState
    private let inProjPSO: MTLComputePipelineState
    private let inProjSpecializedPSO: MTLComputePipelineState?

    let config: LinearAttentionConfig

    /// `specializedHiddenSize` compiles a constant-folded variant of the fused
    /// input projection for the decode shape. Measured elsewhere in this
    /// package: an unspecialized INT4 GEMV runs ~102 GB/s against ~141 GB/s
    /// specialized, so the runtime path must not be the only one available.
    init(context: MetalContext, config: LinearAttentionConfig,
         specializedHiddenSize: Int? = nil) throws {
        precondition(config.keyHeadDim > 0 && config.keyHeadDim % 32 == 0,
                     "keyHeadDim must be a positive multiple of 32")
        precondition(config.keyHeadDim / 32 <= 8,
                     "keyHeadDim register tile exceeds the kernel's bound")
        precondition(config.valueHeadDim % 4 == 0,
                     "valueHeadDim must be a multiple of 4")
        precondition(config.numVHeads % config.numKHeads == 0,
                     "numVHeads must be a multiple of numKHeads")
        self.config = config
        self.convDecodePSO = try context.pipeline("gdn_conv_mix_decode")
        self.convPrefillPSO = try context.pipeline("gdn_conv_mix_prefill")
        self.convTailUpdatePSO = try context.pipeline("gdn_conv_tail_update")
        self.qkNormPSO = try context.pipeline("gdn_qk_norm")
        self.deltaDecodePSO = try context.pipeline("gdn_delta_step_decode")
        self.deltaPrefillPSO = try context.pipeline("gdn_delta_step_prefill")
        self.gatedNormPSO = try context.pipeline("gdn_gated_norm")
        self.inProjPSO = try context.pipeline("gdn_in_proj_gemv_simd",
                                              constants: [],
                                              maxTotalThreadsPerThreadgroup: 512)
        if let n = specializedHiddenSize {
            self.inProjSpecializedPSO = try context.pipeline(
                "gdn_in_proj_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 90, value: .uint32(UInt32(config.qkvDim))),
                    MetalFunctionConstant(index: 91, value: .uint32(UInt32(config.valueDim))),
                    MetalFunctionConstant(index: 92, value: .uint32(UInt32(config.numVHeads))),
                    MetalFunctionConstant(index: 93, value: .uint32(UInt32(n))),
                    MetalFunctionConstant(index: 94, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        } else {
            self.inProjSpecializedPSO = nil
        }
    }

    /// Fused `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` INT4 GEMV.
    /// One dispatch over the concatenated row space replaces four, of which two
    /// (a and b, `numVHeads` rows each) were near-empty launches. Bit-identical
    /// to the four separate GEMVs — the per-row body and its operand order are
    /// unchanged.
    func encodeInputProjections(commandBuffer: MTLCommandBuffer,
                                x: MTLBuffer, xOffset: Int = 0,
                                qkv: TensorView, qkvOut: MTLBuffer,
                                z: TensorView, zOut: MTLBuffer,
                                a: TensorView, aOut: MTLBuffer,
                                b: TensorView, bOut: MTLBuffer,
                                hiddenSize: Int) {
        precondition(hiddenSize % Quantization.groupSize == 0,
                     "hiddenSize must be a multiple of \(Quantization.groupSize)")
        // The row body reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte.
        precondition(Int(qkv.offset) % 2 == 0 && Int(z.offset) % 2 == 0 &&
                     Int(a.offset) % 2 == 0 && Int(b.offset) % 2 == 0,
                     "gdn_in_proj_gemv_simd needs 2-aligned weights offsets")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(inProjSpecializedPSO ?? inProjPSO)
        for (slot, view) in [qkv, z, a, b].enumerated() {
            encoder.setBuffer(view.buffer, offset: Int(view.offset), index: slot * 3)
            encoder.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: slot * 3 + 1)
            encoder.setBuffer(view.buffer, offset: Int(view.biasOffset), index: slot * 3 + 2)
        }
        encoder.setBuffer(x, offset: xOffset, index: 12)
        encoder.setBuffer(qkvOut, offset: 0, index: 13)
        encoder.setBuffer(zOut, offset: 0, index: 14)
        encoder.setBuffer(aOut, offset: 0, index: 15)
        encoder.setBuffer(bOut, offset: 0, index: 16)
        var qkvRows = UInt32(config.qkvDim)
        var zRows = UInt32(config.valueDim)
        var abRows = UInt32(config.numVHeads)
        var n = UInt32(hiddenSize)
        encoder.setBytes(&qkvRows, length: MemoryLayout<UInt32>.size, index: 17)
        encoder.setBytes(&zRows, length: MemoryLayout<UInt32>.size, index: 18)
        encoder.setBytes(&abRows, length: MemoryLayout<UInt32>.size, index: 19)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 20)
        let totalRows = config.qkvDim + config.valueDim + 2 * config.numVHeads
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Decode: conv over [tail | current row] with SiLU, shifting the tail in
    /// place. `convWeight` is the BF16 `[convDim, kernel]` tensor view region.
    func encodeConvDecode(commandBuffer: MTLCommandBuffer,
                          tail: MTLBuffer,
                          qkv: MTLBuffer, qkvOffset: Int = 0,
                          convWeight: MTLBuffer, convWeightOffset: Int,
                          out: MTLBuffer, outOffset: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convDecodePSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 1)
        encoder.setBuffer(convWeight, offset: convWeightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch1D(encoder, pipeline: convDecodePSO, threads: config.qkvDim)
        encoder.endEncoding()
    }

    /// Prefill: conv over [tail | chunk rows]; the tail is read-only here.
    func encodeConvPrefill(commandBuffer: MTLCommandBuffer,
                           tail: MTLBuffer,
                           qkvRows: MTLBuffer, qkvRowsOffset: Int = 0,
                           convWeight: MTLBuffer, convWeightOffset: Int,
                           out: MTLBuffer, outOffset: Int = 0,
                           rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convPrefillPSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkvRows, offset: qkvRowsOffset, index: 1)
        encoder.setBuffer(convWeight, offset: convWeightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        var rowCount = UInt32(rows)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 6)
        dispatch2D(encoder, pipeline: convPrefillPSO,
                   width: config.qkvDim, height: rows)
        encoder.endEncoding()
    }

    /// After a prefill chunk: tail := last (kernel-1) raw rows of [tail | chunk].
    func encodeConvTailUpdate(commandBuffer: MTLCommandBuffer,
                              tail: MTLBuffer,
                              qkvRows: MTLBuffer, qkvRowsOffset: Int = 0,
                              rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(convTailUpdatePSO)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkvRows, offset: qkvRowsOffset, index: 1)
        var channels = UInt32(config.qkvDim)
        var taps = UInt32(config.convKernelSize)
        var rowCount = UInt32(rows)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch2D(encoder, pipeline: convTailUpdatePSO,
                   width: config.qkvDim, height: config.convKernelSize - 1)
        encoder.endEncoding()
    }

    /// Per-head no-weight RMS norm over the q and k slices of `convOut`,
    /// in place, with the delta-rule scales folded in. `rows` > 1 for prefill.
    func encodeQKNorm(commandBuffer: MTLCommandBuffer,
                      convOut: MTLBuffer, convOutOffset: Int = 0,
                      rows: Int = 1) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qkNormPSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        var kHeads = UInt32(config.numKHeads)
        var keyDim = UInt32(config.keyHeadDim)
        var rowStride = UInt32(config.qkvDim)
        encoder.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 2 * config.numKHeads, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Decode: one gated delta rule step. `state` is FP32 [Hv, Dv, Dk],
    /// updated in place; `y` receives [Hv * Dv] FP16.
    func encodeDeltaStepDecode(commandBuffer: MTLCommandBuffer,
                               convOut: MTLBuffer, convOutOffset: Int = 0,
                               aProj: MTLBuffer, aProjOffset: Int = 0,
                               bProj: MTLBuffer, bProjOffset: Int = 0,
                               aLog: MTLBuffer, aLogOffset: Int,
                               dtBias: MTLBuffer, dtBiasOffset: Int,
                               state: MTLBuffer,
                               y: MTLBuffer, yOffset: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(deltaDecodePSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(y, offset: yOffset, index: 6)
        setHeadDims(encoder, startingAt: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads,
                    height: config.valueHeadDim / 4,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    /// Prefill: the recurrence runs sequentially over `rows` inside the
    /// kernel; state persists in registers and is written back once.
    func encodeDeltaStepPrefill(commandBuffer: MTLCommandBuffer,
                                convOut: MTLBuffer, convOutOffset: Int = 0,
                                aProj: MTLBuffer, aProjOffset: Int = 0,
                                bProj: MTLBuffer, bProjOffset: Int = 0,
                                aLog: MTLBuffer, aLogOffset: Int,
                                dtBias: MTLBuffer, dtBiasOffset: Int,
                                state: MTLBuffer,
                                y: MTLBuffer, yOffset: Int = 0,
                                rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(deltaPrefillPSO)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(aProj, offset: aProjOffset, index: 1)
        encoder.setBuffer(bProj, offset: bProjOffset, index: 2)
        encoder.setBuffer(aLog, offset: aLogOffset, index: 3)
        encoder.setBuffer(dtBias, offset: dtBiasOffset, index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(y, offset: yOffset, index: 6)
        setHeadDims(encoder, startingAt: 7)
        var rowCount = UInt32(rows)
        var rowStride = UInt32(config.qkvDim)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads,
                    height: config.valueHeadDim / 4,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    /// out = rmsnorm(y; weight) * silu(z), per value head, `rows` rows.
    func encodeGatedNorm(commandBuffer: MTLCommandBuffer,
                         y: MTLBuffer, yOffset: Int = 0,
                         z: MTLBuffer, zOffset: Int = 0,
                         weight: MTLBuffer, weightOffset: Int,
                         out: MTLBuffer, outOffset: Int = 0,
                         rows: Int = 1) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatedNormPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(z, offset: zOffset, index: 1)
        encoder.setBuffer(weight, offset: weightOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var vHeads = UInt32(config.numVHeads)
        var valueDim = UInt32(config.valueHeadDim)
        encoder.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: config.numVHeads, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func setHeadDims(_ encoder: MTLComputeCommandEncoder, startingAt index: Int) {
        var kHeads = UInt32(config.numKHeads)
        var vHeads = UInt32(config.numVHeads)
        var keyDim = UInt32(config.keyHeadDim)
        var valueDim = UInt32(config.valueHeadDim)
        encoder.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: index)
        encoder.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: index + 1)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: index + 2)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: index + 3)
    }

    private func dispatch1D(_ encoder: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func dispatch2D(_ encoder: MTLComputeCommandEncoder,
                            pipeline: MTLComputePipelineState,
                            width: Int, height: Int) {
        let tgWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    }
}
