import Foundation
import Metal

/// Decode-time activation of the quantized GEMV kernels: the role-to-width
/// dispatcher, the primary and head variants, the kernel-split rotation, and
/// the per-kernel ablation switches they are diagnosed with.
///
/// Split out of `RealForwardRunner+Decode.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion:
/// the same bodies, moved so the layer loop reads without the dispatch detail.
/// No signature or behaviour change; the moved declarations were already
/// internal.
extension RealForwardRunner {

    /// Attribution only. TINYTITAN_ABLATE=<name> skips one kernel of the GDN
    /// decode chain so the attention command buffer's GPU time can be
    /// differenced per kernel. Output is wrong while it is set; the timing is
    /// not, because none of these kernels' cost depends on the data.
    /// Environment flags are read once per process. Reading
    /// `ProcessInfo.processInfo.environment` rebuilds a dictionary from
    /// environ every time; at ~20 kernels per layer that was ~800 rebuilds a
    /// token, and on the 35B family's 65 ms token it showed up as the
    /// hottest CPU frames in a decode sample (Swift String indexing).
    static let ablationTarget = ProcessInfo.processInfo.environment["TINYTITAN_ABLATE"]
    func ablated(_ name: String) -> Bool { Self.ablationTarget == name }

    /// Per-kernel GPU attribution for the GDN decode chain.
    ///
    /// TINYTITAN_KERNEL_SPLIT=1 gives every kernel of this chain its own command
    /// buffer, committed in order on the same queue, so `recordKernelGPU` can
    /// time each one rather than the chain as a whole. Data stays valid --
    /// unlike ablation, which feeds garbage downstream and measures NaN
    /// slow-paths and shortened generations instead of the kernel. The split
    /// buffers are collected here and recorded after the layer's tail wait,
    /// where their GPU timestamps exist.
    static let splitKernelTimingFlag =
        ProcessInfo.processInfo.environment["TINYTITAN_KERNEL_SPLIT"] == "1"
    var splitKernelTiming: Bool { Self.splitKernelTimingFlag }

    func rotate(_ cb: inout MTLCommandBuffer, role: String) throws {
        guard splitKernelTiming else { return }
        cb.commit()
        splitTimedBuffers.append((role, cb))
        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
    }

    /// The quantized GEMV for a projection at the width its *role* declares.
    ///
    /// The runner holds one affine dispatcher per width it needs, chosen from
    /// the roles the manifest distinguishes; this picks the right one. A bf16
    /// tensor -- the dense installs' GDN `a`/`b`, or a promoted projection --
    /// takes the bf16 kernel regardless of the role's width, because it carries
    /// no scales or biases to read.
    func encodeRoleGEMV(
        commandBuffer cb: MTLCommandBuffer,
        projection p: TensorView,
        weightBits: Int,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        if p.dtype == 1 {
            try bf16Projection.encode(
                commandBuffer: cb,
                weights: p.buffer,
                weightsOffset: Int(p.offset),
                x: x, xOffset: xOffset,
                y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        // Four-bit weights are always executable (`int4` is unconditional);
        // anything wider needs the affine dispatcher for that width -- the KV
        // role's own when the install declares it differently from the
        // attention slot's, else the attention one.
        if weightBits == 4 {
            try int4.encode(
                commandBuffer: cb,
                weights: p.buffer, weightsOffset: Int(p.offset),
                scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                biases: p.buffer, biasesOffset: Int(p.biasOffset),
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        guard let dispatcher = affineByWidth[weightBits] else {
            throw ModelError.unsupportedArchitecture(
                detail: "no \(weightBits)-bit GEMV is built for a \(m)x\(n) projection")
        }
        try dispatcher.encode(
            commandBuffer: cb,
            weights: p.buffer, weightsOffset: Int(p.offset),
            scales: p.buffer, scalesOffset: Int(p.scaleOffset),
            biases: p.buffer, biasesOffset: Int(p.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: m, n: n)
    }

    func encodePrimaryGEMV(
        commandBuffer cb: MTLCommandBuffer,
        projection p: TensorView,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        // A promoted tensor carries no scales or biases, so it cannot go
        // through the quantized GEMV -- that would read the companions from
        // offset zero, which is the file header, and produce NaN. Decided from
        // the tensor's dtype because promotion is per tensor, not per slot.
        if p.dtype == 1 {
            try bf16Projection.encode(
                commandBuffer: cb,
                weights: p.buffer,
                weightsOffset: Int(p.offset),
                x: x, xOffset: xOffset,
                y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        try encodePrimaryGEMV(
            commandBuffer: cb,
            weights: p.buffer, weightsOffset: Int(p.offset),
            scales: p.buffer, scalesOffset: Int(p.scaleOffset),
            biases: p.buffer, biasesOffset: Int(p.biasOffset),
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: m, n: n)
    }

    func encodePrimaryGEMV(
        commandBuffer cb: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        if let affine {
            try affine.encode(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        } else {
            try int4.encode(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        }
    }

    /// Vocabulary head GEMV. Same shape as `encodePrimaryGEMV` but keyed off
    /// the head's own quantization, which a model may set independently of
    /// the attention slot.
    func encodeHeadGEMV(
        commandBuffer cb: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: UInt32, n: UInt32
    ) throws {
        // A bf16 head has no scales or biases; the bf16 GEMV ignores them.
        if headIsBF16 {
            try bf16Projection.encode(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
            return
        }
        if let affineHead {
            try affineHead.encode(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        } else {
            try int4.encode(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        }
    }

    /// The model's final norm into a caller-chosen buffer and offset.
    ///
    /// The single-row decode writes `normed`; the batched head writes one
    /// row's worth into a shared batch buffer, so the head can read two
    /// contiguous rows with one weight pass.
    func encodeFinalNorm(
        commandBuffer cb: MTLCommandBuffer,
        out: MTLBuffer, outOffset: Int,
        fNorm: TensorView, d: UInt32, eps: Float
    ) throws {
        if let hc = hyperConnection {
            try hc.encodeRead(
                commandBuffer: cb,
                streamsBuffer: hidden,
                hcNorm: fNorm.buffer,
                hcNormOffset: Int(fNorm.offset),
                down: gateWeightsPublic(try model.hcMixerDown()),
                up: gateWeightsPublic(try model.hcMixerUp()),
                blockInput: out, blockInputOffset: outOffset, eps: eps)
        } else {
            try rms.encodeBF16W(
                commandBuffer: cb, x: hidden,
                weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                out: out, outOffset: outOffset, d: d, eps: eps)
        }
    }

    /// Two rows through the LM head, weights read once.
    func encodeHeadTwoRows(
        commandBuffer cb: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        x: MTLBuffer, xOffset: Int,
        y: MTLBuffer, yOffset: Int,
        m: UInt32, n: UInt32
    ) throws {
        // The bf16 head reads both rows in one dispatch through the multi-row
        // GEMV, which keeps `bf16_gemv_simd`'s per-row arithmetic: each row's
        // logits are the bytes the single-row head gives it.
        if headIsBF16 {
            try multiRowGEMV.encode(
                commandBuffer: cb,
                segments: [
                    .init(
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        format: .bf16, rows: Int(m),
                        y: y, yOffset: yOffset, yRowStride: Int(m))
                ],
                x: x, xOffset: xOffset, xRowStride: Int(n),
                activationRows: 2, columns: Int(n))
            return
        }
        if let affineHead {
            try affineHead.encodeTwoRows(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        } else {
            try int4.encodeTwoRows(
                commandBuffer: cb,
                weights: weights, weightsOffset: weightsOffset,
                scales: scales, scalesOffset: scalesOffset,
                biases: biases, biasesOffset: biasesOffset,
                x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                m: m, n: n)
        }
    }

    /// Four rows through the LM head, weights read once. The batched decode at
    /// width 4 uses one of these instead of two two-row launches, which reads
    /// the head's weights once rather than twice.
    func encodeHeadFourRows(
        commandBuffer cb: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        x: MTLBuffer, xOffset: Int,
        y: MTLBuffer, yOffset: Int,
        m: UInt32, n: UInt32
    ) throws {
        // The four-row kernel is only for the affine head. Its int4 sibling was
        // measured at the two-row launch's cost for twice the rows (7.4 ms
        // against 3.8 ms at 248,320 x 2,048), so the 4-bit head keeps two-row
        // pairs: sharing the read further does not pay once dequant + dot is
        // the limit rather than bandwidth.
        guard let affineHead else {
            throw ModelError.internalInconsistency(
                detail: "four-row head requires an affine head")
        }
        headFourRowLaunches += 1
        try affineHead.encodeFourRows(
            commandBuffer: cb,
            weights: weights, weightsOffset: weightsOffset,
            scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset,
            x: x, xOffset: xOffset, y: y, yOffset: yOffset,
            m: m, n: n)
    }

    func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> MTLCommandBuffer? {
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
        return cb
    }
}
